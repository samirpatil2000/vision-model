#!/usr/bin/env python3
"""
Previs CLI: scene description -> physics-planned keyframes -> storyboard + animatic.

Claude (via the `claude` CLI) is the physics/world-context planner; a deterministic solver turns its
decisions into trajectories and timing; Qwen-Image-2.1 only renders.

Examples:
  python previs.py run --scene "Mara knocks a glass off the kitchen table; Jon turns" --tier draft
  python previs.py run --planner fixture --tier draft            # offline plan (no Claude call), real engine
  python previs.py plan --scene-file scene.txt --run-name kitchen
  python previs.py block --run-name kitchen                      # gray-box + physics-approval animatic
  python previs.py render --run-name kitchen --tier final        # reuses frames whose render fingerprint matches
  python previs.py assemble --run-name kitchen                   # animatic at the accepted frames' size
  python previs.py probe                                         # Claude CLI isolation/capability probes
"""
import argparse
import json
import re
import shlex
import signal
import sys
import time
from pathlib import Path

from src import config
from src.config import GenerationDefaults, TIERS


DEFAULT_TIER = "draft"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "scene"


def _tier_name(args) -> str:
    return args.tier or DEFAULT_TIER


def _run_dir(args) -> Path:
    if not args.run_name:
        raise SystemExit("error: --run-name is required for this command")
    return config.PREVIS_OUTPUT_DIR / args.run_name


def _scene_text(args) -> str:
    if args.scene_file:
        return Path(args.scene_file).read_text().strip()
    if args.scene:
        return args.scene.strip()
    raise SystemExit("error: provide --scene or --scene-file")


def _cast_refs(args) -> dict:
    refs = {}
    for item in getattr(args, "cast", []) or []:
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise SystemExit(f"error: --cast expects NAME=IMAGE, got {item!r}")
        if not Path(path).expanduser().is_file():
            raise SystemExit(f"error: --cast image not found: {path}")
        refs[name.strip()] = str(Path(path).expanduser().resolve())
    return refs


def _planner(args):
    if args.planner == "fixture":
        from src.previs.planner.fixture import FixturePlanner
        return FixturePlanner(args.fixture) if args.fixture else FixturePlanner()
    from src.previs.planner.claude_cli import ClaudeCliPlanner
    return ClaudeCliPlanner(model=args.model, effort=args.effort, config_dir=args.claude_config_dir, auth=args.auth)


def _engine(args):
    from src.engine import make_engine
    from src.engine.version import verify_engine
    # Fail before planning (minutes of Claude time) on a wrong engine build; the client checks again on start.
    # Check the binary this run uses, so a missing or stale sd-cli never blocks a server run (and vice versa).
    binary = config.SD_SERVER if args.engine == "server" else config.SD_CLI
    verify_engine(binary, config.pinned_engine_sha(), allow_mismatch=args.allow_engine_mismatch)
    return make_engine(args.engine, allow_mismatch=args.allow_engine_mismatch, defaults=GenerationDefaults(),
                       verbose_log_dir=str(_run_dir(args) / "engine_logs"))


def _sequencer(args, with_planner: bool = False, with_engine: bool = False):
    from src.previs.sequencer import RunConfig, Sequencer
    cfg = RunConfig(
        run_dir=_run_dir(args),
        tier=TIERS[_tier_name(args)],
        defaults=GenerationDefaults(),
        project=args.run_name or "previs",
        blocking_backend=args.blocking,
        use_qa_models=not args.no_qa_models,
        fact_check=args.fact_check,
        auto_approve=not args.approve,
        cast_refs=_cast_refs(args),
    )
    planner = _planner(args) if with_planner else None
    engine = _engine(args) if with_engine else None
    return Sequencer(cfg, planner=planner, engine=engine)


def _banner(title: str) -> None:
    print(f"\n=== {title} ===")


def cmd_plan(args) -> None:
    seq = _sequencer(args, with_planner=True)
    _banner(f"Physics plan ({args.planner})")
    plan = seq.plan(_scene_text(args))
    shots = len(plan.bible.shots)
    kfs = sum(len(s.keyframes) for s in plan.timeline.shots)
    print(f"✓ {shots} shots, {kfs} keyframes -> {seq.run_dir / 'plan.json'}")
    seq.write_run_manifest({"planner": args.planner})


def cmd_solve(args) -> None:
    seq = _sequencer(args, with_planner=True)
    _banner("Solve + validate")
    solved = seq.solve(seq.load_plan())
    print(f"✓ {len(solved.frames)} keyframes, {solved.total_duration_s:.2f} s total")
    for w in solved.warnings:
        print(f"  ⚠ {w}")
    seq.write_run_manifest()


def cmd_block(args) -> None:
    seq = _sequencer(args, with_planner=True)
    plan = seq.load_plan()
    _banner("Solve + validate")
    solved = seq.solve(plan)
    _banner(f"Gray-box blocking ({args.blocking})")
    seq.block(seq._plan, solved, graybox_animatic=not args.skip_graybox)
    print(f"✓ gray-box frames in {seq.run_dir / 'graybox'}"
          + ("" if args.skip_graybox else f"; physics animatic {seq.run_dir / 'graybox' / 'graybox.mp4'}"))
    seq.write_run_manifest()


def render_command(args) -> str:
    """The `render --approved` command that continues this run after gray-box approval."""
    parts = ["python", "previs.py", "render", "--run-name", args.run_name, "--approved", "--tier", _tier_name(args)]
    if args.planner != "claude":
        parts += ["--planner", args.planner] + (["--fixture", args.fixture] if args.fixture else [])
    for flag, default in (("engine", "cli"), ("blocking", "box")):
        if getattr(args, flag) != default:
            parts += [f"--{flag}", getattr(args, flag)]
    for flag in ("no_qa_models", "fact_check", "skip_graybox", "skip_eval", "allow_engine_mismatch"):
        if getattr(args, flag):
            parts.append("--" + flag.replace("_", "-"))
    return " ".join(shlex.quote(p) for p in parts)


def _render_and_assemble(seq, args) -> bool:
    """Solve, block, then render + assemble. False when the run stops for gray-box approval (--approve)."""
    plan = seq.load_plan()
    _banner("Solve + validate")
    solved = seq.solve(plan)
    plan = seq._plan
    _banner(f"Gray-box blocking ({args.blocking})")
    blocking = seq.block(plan, solved, graybox_animatic=not args.skip_graybox)
    if args.approve and not args.approved:
        review = seq.run_dir / "graybox" / ("" if args.skip_graybox else "graybox.mp4")
        print(f"\nReview {review}, then render with:\n  {render_command(args)}")
        seq.write_run_manifest()
        return False
    _banner(f"Render keyframes ({_tier_name(args)}, {args.engine})")
    try:
        frames = seq.render(plan, solved, blocking)
    finally:
        _close_engine(seq)          # free the engine before assembling
    _banner("Assemble")
    out = seq.assemble(plan, solved, frames, size=(seq.cfg.tier.width, seq.cfg.tier.height))
    print(f"✓ storyboard: {seq.run_dir / 'storyboard' / 'storyboard.pdf'}")
    print(f"✓ animatic:   {out['animatic']}")
    seq.write_run_manifest()
    return True


def _evaluate(seq, args) -> None:
    if args.skip_eval:
        return
    _banner("Evaluate")
    import benchmark_previs
    benchmark_previs.main(["--run-dir", str(seq.run_dir)] + (["--no-models"] if args.no_qa_models else []))


def _close_engine(seq) -> None:
    if seq.engine is not None:
        seq.engine.close()


def cmd_render(args) -> None:
    seq = _sequencer(args, with_planner=True, with_engine=True)
    try:
        rendered = _render_and_assemble(seq, args)
    finally:
        _close_engine(seq)          # also when stopping for approval or failing before the render stage
    if rendered:
        _evaluate(seq, args)


def cmd_run(args) -> None:
    seq = _sequencer(args, with_planner=True, with_engine=True)
    print(f"Run directory: {seq.run_dir}  (--run-name {args.run_name})")
    try:
        _banner(f"Physics plan ({args.planner})")
        t0 = time.time()
        plan = seq.plan(_scene_text(args))
        print(f"✓ {len(plan.bible.shots)} shots planned in {time.time() - t0:.1f} s")
        rendered = _render_and_assemble(seq, args)
    finally:
        _close_engine(seq)
    if rendered:
        _evaluate(seq, args)


def cmd_assemble(args) -> None:
    seq = _sequencer(args)
    plan = seq.load_plan()
    from src.previs import physics
    solved = physics.solve(plan, max_duration_s=config.MAX_VIDEO_SECONDS)  # deterministic, so re-derived
    frames_dir = seq.run_dir / "frames"
    frames = {f.kf_id: str(frames_dir / f"{f.kf_id}.png") for f in solved.frames
              if (frames_dir / f"{f.kf_id}.png").is_file()}
    # Animatic at the accepted frames' own size unless --tier is given explicitly.
    size = (TIERS[args.tier].width, TIERS[args.tier].height) if args.tier else None
    _banner("Assemble")
    out = seq.assemble(plan, solved, frames, size=size)
    print(f"✓ animatic: {out['animatic']}")


def cmd_grid(args) -> None:
    """Zero-code baseline B1: one native 2K multi-panel storyboard image generated from the plan."""
    from src.engine.base import GenerationRequest
    seq = _sequencer(args)
    plan = seq.load_plan()
    keyframes = [kf for s in plan.timeline.shots for kf in s.keyframes][: args.panels]
    cols = 3 if len(keyframes) > 4 else 2
    rows = (len(keyframes) + cols - 1) // cols
    chars = "; ".join(f"{c.name}: {c.appearance}, wearing {c.wardrobe}" for c in plan.bible.characters)
    panels = " ".join(f"Panel {i + 1}: {kf.action}" for i, kf in enumerate(keyframes))
    prompt = (f"A {rows}x{cols} storyboard grid of {len(keyframes)} film panels with thin white borders, "
              f"{plan.bible.scene.style}. Same characters in every panel. {chars}. {panels}")
    out = seq.run_dir / "grid" / "grid_baseline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = _engine(args)
    try:
        result = engine.generate(GenerationRequest(prompt=prompt, output_path=str(out), width=args.grid_width,
                                                   height=args.grid_height, seed=42,
                                                   sigmas=list(config.VIGGLE_SIGMAS), tag="grid"))
    finally:
        engine.close()
    print(f"✓ grid baseline {out} ({result.seconds:.1f} s)")


def cmd_probe(args) -> None:
    from src.previs.planner.claude_cli import probe
    work = config.PREVIS_OUTPUT_DIR / "_probe" / time.strftime("%Y%m%d_%H%M%S")
    _banner("Claude CLI probes")
    report = probe(work)
    print(json.dumps(report, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Physics-planned previs on Qwen-Image-2.1")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, scene=False):
        sp.add_argument("--run-name", "-r", default=None, help="Run directory name under output/previs/")
        sp.add_argument("--tier", default=None, choices=sorted(TIERS),
                        help=f"Resolution tier (default {DEFAULT_TIER}; assemble: the accepted frames' size)")
        sp.add_argument("--planner", default="claude", choices=["claude", "fixture"])
        sp.add_argument("--fixture", default=None, help="Fixture plan JSON for --planner fixture")
        sp.add_argument("--model", default="claude-opus-5-5", help="Claude model for planning")
        sp.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
        sp.add_argument("--auth", default="subscription", choices=["subscription", "api_key"])
        sp.add_argument("--claude-config-dir", default=None, help="Dedicated CLAUDE_CONFIG_DIR for pipeline calls")
        sp.add_argument("--engine", default="cli", choices=["cli", "server"])
        sp.add_argument("--blocking", default="box", choices=["box", "blender"])
        sp.add_argument("--no-qa-models", action="store_true", help="Skip CLIP/CLIPSeg QA metrics")
        sp.add_argument("--fact-check", action="store_true", help="Claude (Haiku) yes/no fact QA per keyframe")
        sp.add_argument("--cast", action="append", default=[], metavar="NAME=IMAGE",
                        help="Use IMAGE as the identity reference for the character whose name/id matches NAME (repeatable)")
        sp.add_argument("--approve", action="store_true", help="Stop after the gray-box animatic for approval")
        sp.add_argument("--approved", action="store_true", help="Continue after gray-box approval")
        sp.add_argument("--skip-graybox", action="store_true", help="Skip the 24 fps gray-box animatic")
        sp.add_argument("--skip-eval", action="store_true", help="Do not write eval/ after rendering")
        sp.add_argument("--allow-engine-mismatch", action="store_true",
                        help="Run an engine build other than engine/UPSTREAM_SHA")
        if scene:
            sp.add_argument("--scene", "-s", default=None, help="Scene description text")
            sp.add_argument("--scene-file", "-f", default=None, help="File with the scene description")

    for name, fn, scene in [
        ("run", cmd_run, True), ("plan", cmd_plan, True), ("solve", cmd_solve, False), ("block", cmd_block, False),
        ("render", cmd_render, False), ("assemble", cmd_assemble, False), ("grid", cmd_grid, False),
    ]:
        sp = sub.add_parser(name)
        common(sp, scene=scene)
        sp.set_defaults(func=fn)
        if name == "grid":
            sp.add_argument("--panels", type=int, default=6)
            sp.add_argument("--grid-width", type=int, default=1536)
            sp.add_argument("--grid-height", type=int, default=1024)
    sp = sub.add_parser("probe")
    sp.set_defaults(func=cmd_probe)
    return p


def main(argv=None) -> int:
    # SIGTERM/SIGHUP -> SystemExit, so `finally` blocks run and a resident sd-server is always stopped
    # (also when the terminal closes). An ignored SIGHUP (nohup) stays ignored.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) == signal.SIG_DFL:
        signal.signal(signal.SIGHUP, lambda *_: sys.exit(129))
    args = build_parser().parse_args(argv)
    if args.command in ("run", "plan"):
        if args.planner == "fixture" and not (args.scene or args.scene_file):
            args.scene = "fixture scene"
        if not (args.scene or args.scene_file):
            raise SystemExit("error: provide --scene or --scene-file")
        if not args.run_name:
            source = args.scene or Path(args.scene_file).stem
            args.run_name = _slug(source)[:32] + time.strftime("_%Y%m%d_%H%M%S")
    from src.previs.sequencer import SequencerError
    try:
        args.func(args)
    except SequencerError as exc:      # a stage cannot proceed: the message says what to do, no traceback
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
