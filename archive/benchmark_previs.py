#!/usr/bin/env python3
"""
Previs benchmark: reads one run directory (output/previs/<run>) and writes
eval/metrics.json, eval/report.md and eval/grid.png.

  .venv/bin/python benchmark_previs.py --run-dir output/previs/<run> [--no-models]

--no-models skips the CLIP-based identity and layout metrics (no torch model is loaded).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.previs import eval_metrics as em
from src.previs.assembly import font, format_timecode, load_frame, load_solved, placeholder, probe_video
from src.previs.schema import ScenePlan
from src.previs.state import SceneIndex, build_index
from src.previs.types import BlockingResult, SolvedPlan

GRID_THUMB_W = 320
GRID_PAIRS_PER_ROW = 2


# ---------------------------------------------------------------- run directory readers
def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def latest_version(directory: Path, stem: str) -> Optional[str]:
    """Highest-numbered `<stem>_v<N>.png` in `directory`."""
    best: Tuple[int, Optional[Path]] = (-1, None)
    for path in directory.glob(f"{stem}_v*.png") if directory.is_dir() else []:
        m = re.fullmatch(rf"{re.escape(stem)}_v(\d+)\.png", path.name)
        if m and int(m.group(1)) > best[0]:
            best = (int(m.group(1)), path)
    return str(best[1]) if best[1] else None


def load_blocking(kf_dir: Path, index: SceneIndex) -> Optional[BlockingResult]:
    """BlockingResult rebuilt from graybox/<kf_id>/ (meta.json when present, else derived from id.png)."""
    id_path = kf_dir / "id.png"
    if not id_path.is_file():
        return None
    meta = _read_json(kf_dir / "meta.json", {}) or {}
    id_map = {int(k): v for k, v in (meta.get("id_map") or {}).items()}
    if not id_map:
        id_map = {i + 1: eid for i, eid in enumerate(index.entities)}
    with Image.open(id_path) as img:
        ids = np.asarray(img)
    if ids.ndim == 3:
        ids = ids[..., 0]
    bboxes = {eid: tuple(int(v) for v in box) for eid, box in (meta.get("bboxes") or {}).items()}
    if not bboxes:
        for value, eid in id_map.items():
            ys, xs = np.nonzero(ids == value)
            if len(xs):
                bboxes[eid] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    visible = {eid: float(v) for eid, v in (meta.get("visible_frac") or {}).items()} or {eid: 1.0 for eid in bboxes}
    return BlockingResult(
        blocking_path=str(kf_dir / "blocking.png"), id_path=str(id_path), depth_path=str(kf_dir / "depth.npy"),
        id_map=id_map, bboxes=bboxes, visible_frac=visible,
        screen_velocity={eid: tuple(v) for eid, v in (meta.get("screen_velocity") or {}).items()},
    )


def load_attempts(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Per-attempt records (result.json + qa.json) and the accepted attempt's QA per keyframe.
    frames/manifest.json (written by the sequencer) names the accepted attempt and how many attempts the
    render that produced the frame made; attempt directories beyond that are leftovers and are ignored.
    Without a manifest entry the highest-numbered attempt counts as the accepted one.
    """
    records: List[Dict[str, Any]] = []
    final_qa: Dict[str, Dict[str, Any]] = {}
    manifest = _read_json(run_dir / "frames" / "manifest.json", {}) or {}
    for kf_dir in sorted((run_dir / "keyframes").glob("*")) if (run_dir / "keyframes").is_dir() else []:
        entry = manifest.get(kf_dir.name) if isinstance(manifest.get(kf_dir.name), dict) else {}
        made, chosen = entry.get("attempts"), entry.get("attempt")
        attempts = sorted((p for p in kf_dir.glob("attempt_*") if re.fullmatch(r"attempt_\d+", p.name)),
                          key=lambda p: int(p.name.split("_")[-1]))
        for attempt_dir in attempts:
            n = int(attempt_dir.name.split("_")[-1])
            if isinstance(made, int) and n >= made:
                continue
            result = _read_json(attempt_dir / "result.json")
            qa = _read_json(attempt_dir / "qa.json")
            if result is None and qa is None:
                continue
            record = {"kf_id": kf_dir.name, "attempt": n}
            record.update(result or {})
            if qa is not None:
                record["qa_passed"] = bool(qa.get("passed"))
                if not isinstance(chosen, int) or n == chosen:
                    final_qa[kf_dir.name] = qa
            records.append(record)
    if not records:
        records = list(_read_json(run_dir / "render_log.json", []) or [])
    return records, final_qa


def _memory_records(memory: Any) -> List[Dict[str, Any]]:
    if isinstance(memory, dict):
        return list(memory.get("records", memory.values()))
    return list(memory or [])


# ---------------------------------------------------------------- outputs
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _probe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file() or shutil.which("ffprobe") is None:
        return None
    try:
        return probe_video(path)
    except RuntimeError as exc:
        return {"error": str(exc)}


def build_grid(run_dir: Path, plan: ScenePlan, solved: SolvedPlan, frames: Dict[str, str],
               captions: Dict[str, str], out_path: Path) -> None:
    """Blocking | accepted frame pairs per keyframe with a caption line."""
    aspect = 2.39 if plan.bible.scene.aspect == "2.39:1" else 16 / 9
    tw, th = GRID_THUMB_W, int(round(GRID_THUMB_W / aspect))
    pad, cap_h, head_h = 12, 44, 48
    pair_w = 2 * tw + pad
    kfs = [f for f in sorted(solved.frames, key=lambda f: f.t_s)]
    rows = max(1, -(-len(kfs) // GRID_PAIRS_PER_ROW))
    width = pad + GRID_PAIRS_PER_ROW * (pair_w + 2 * pad)
    height = head_h + rows * (th + cap_h + pad) + pad
    grid = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(grid)
    draw.text((pad, 14), f"{run_dir.name} · blocking | accepted frame", font=font(20), fill=(235, 235, 235))
    for n, f in enumerate(kfs):
        r, c = divmod(n, GRID_PAIRS_PER_ROW)
        x, y = pad + c * (pair_w + 2 * pad), head_h + r * (th + cap_h + pad)
        gray_dir = run_dir / "graybox" / f.kf_id
        gray = load_frame(gray_dir / "blocking_sharp.png", (tw, th)) or load_frame(gray_dir / "blocking.png", (tw, th))
        grid.paste(gray or placeholder((tw, th), f.kf_id, "no blocking"), (x, y))
        grid.paste(load_frame(frames.get(f.kf_id), (tw, th)) or placeholder((tw, th), f.kf_id), (x + tw + pad, y))
        draw.text((x, y + th + 4), f"{f.kf_id}  {f.shot_id}  {format_timecode(f.t_s)}", font=font(15), fill=(235, 235, 235))
        draw.text((x, y + th + 23), captions.get(f.kf_id, ""), font=font(13), fill=(170, 170, 170))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def write_report(path: Path, run_dir: Path, plan: ScenePlan, metrics: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    phys, ident, layout = metrics["physics"], metrics["identity"], metrics["layout"]
    drift, adj, timing, mem = metrics["background_drift"], metrics["adjacent_change"], metrics["timing"], metrics["memory"]
    cov = metrics["coverage"]
    skipped = "skipped (--no-models)"
    lines = [
        f"# Previs Benchmark: `{run_dir.name}`\n",
        f"- **Logline**: {plan.bible.scene.logline}",
        f"- **Solved duration**: {metrics['plan']['duration_s']:.2f} s · {metrics['plan']['shots']} shots · "
        f"{metrics['plan']['keyframes']} keyframes",
        f"- **Frames rendered**: {cov['frames']}/{cov['keyframes']}" + (f" (missing: {', '.join(cov['missing'])})" if cov["missing"] else ""),
        f"- **Models**: {'CLIP/CLIPSeg' if metrics['models'] else 'disabled (--no-models)'}\n",
        "## Summary\n",
        "| Metric | Value | Target |",
        "| :--- | :---: | :---: |",
        f"| Physics invariant rate (checks passed) | `{_fmt(phys['invariant_rate'])}` ({phys['checks_passed']}/{phys['checks']}) | `1.000` |",
        f"| Validator errors / warnings | `{phys['errors']}` / `{phys['warnings']}` | `0` / – |",
        f"| Identity: crop vs sheet (CLIP cosine) | `{skipped if ident.get('skipped') else _fmt(ident.get('sheet_mean'))}` | ≥ `{em.IDENTITY_THRESHOLD}` |",
        f"| Identity: same character across shots | `{skipped if ident.get('skipped') else _fmt(ident.get('cross_shot_mean'))}` | higher is better |",
        f"| Layout IoU (CLIPSeg vs blocking) | `{skipped if layout.get('skipped') else _fmt(layout.get('mean'))}` | ≥ `{em.LAYOUT_THRESHOLD}` |",
        f"| Background drift vs plate (ΔE76) | `{_fmt(drift['mean'], 2)}` | lower is better |",
        f"| Adjacent change within shots (MAD / SSIM) | `{_fmt(adj['mean_mad'], 2)}` / `{_fmt(adj['mean_ssim'])}` | motion without flicker |",
        f"| QA first-pass / final pass rate | `{_fmt(timing['first_pass_rate'])}` / `{_fmt(timing['final_pass_rate'])}` | `1.000` |",
        f"| Fact accuracy (judge) | `{_fmt(metrics['fact_accuracy'].get('accuracy'))}` | ≥ `0.75` |",
        f"| Engine seconds / request (mean, p95) | `{_fmt(timing['mean_s'], 2)}` / `{_fmt(timing['p95_s'], 2)}` | – |",
        f"| Peak RSS (GB) | `{_fmt(timing['peak_rss_gb'], 2)}` | – |",
        f"| Engine fallback warnings | `{timing['fallback_warnings']}` | `0` |",
        "",
        "## Per keyframe\n",
        "| Keyframe | Shot | Time | Attempts | QA | ΔE76 | MAD | SSIM | Identity | Layout |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in rows:
        lines.append(f"| `{row['kf_id']}` | `{row['shot_id']}` | {row['timecode']} | {row['attempts']} | {row['qa']} | "
                     f"{_fmt(row['drift'], 2)} | {_fmt(row['mad'], 2)} | {_fmt(row['ssim'])} | "
                     f"{_fmt(row['identity'])} | {_fmt(row['layout'])} |")
    lines += ["", "## Validator findings\n"]
    if metrics["violations"]:
        lines += ["| Code | Severity | Where | Message |", "| :--- | :---: | :--- | :--- |"]
        for v in metrics["violations"][:50]:
            where = " ".join(x for x in (v.get("shot_id"), v.get("kf_id"), v.get("entity_id")) if x and x != "none")
            message = str(v.get("message", "")).replace("|", "\\|")
            lines.append(f"| `{v.get('code')}` | {v.get('severity', 'error')} | {where or '–'} | {message} |")
    else:
        lines.append("No violations recorded.")
    lines += ["", "## Stage timings\n", "| Stage | Seconds |", "| :--- | :---: |"]
    for stage, secs in (metrics["stage_timings"] or {}).items():
        lines.append(f"| {stage} | {_fmt(secs, 2)} |")
    lines += ["", "## Keyframe memory\n",
              f"- Records: {mem['records']} (approved {mem['approved']}), max ancestry depth {_fmt(mem['max_depth'])}, "
              f"mean QA score {_fmt(mem['mean_score'])}",
              f"- Depth histogram: {json.dumps(mem['depth_histogram'])}",
              f"- Background drift by ancestry depth: {json.dumps(drift['by_depth'])}", "",
              "## Outputs\n"]
    for name, info in metrics["outputs"].items():
        if isinstance(info, dict) and "duration_s" in info:
            lines.append(f"- `{name}`: {info['duration_s']:.2f} s, {info['fps']:.2f} fps, {info['width']}x{info['height']}, {info['frames']} frames")
        else:
            lines.append(f"- `{name}`: {'present' if info else 'missing'}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main
def evaluate(run_dir: Path, use_models: bool) -> Dict[str, Any]:
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"No plan.json in {run_dir}")
    plan = ScenePlan.model_validate_json(plan_path.read_text())
    index = build_index(plan)
    if (run_dir / "solved.json").is_file():
        solved = load_solved(run_dir / "solved.json")
    else:
        from src.previs import physics
        solved = physics.solve(plan)
    keyframes = sorted(solved.frames, key=lambda f: f.t_s)

    # 1. Inputs from the run directory.
    frames = {f.kf_id: str(run_dir / "frames" / f"{f.kf_id}.png") for f in keyframes
              if (run_dir / "frames" / f"{f.kf_id}.png").is_file()}
    shot_of = {f.kf_id: f.shot_id for f in keyframes}
    blocking = {f.kf_id: b for f in keyframes if (b := load_blocking(run_dir / "graybox" / f.kf_id, index)) is not None}
    plates = {f.kf_id: p for f in keyframes if (p := latest_version(run_dir / "assets" / "plates", f.location_id))}
    sheets = {ch: p for ch in index.characters if (p := latest_version(run_dir / "assets" / "cast", f"{ch}_full"))}
    memory = _memory_records(_read_json(run_dir / "memory.json", []))
    ancestry = {r["kf_id"]: int(r.get("ancestry_depth", 0)) for r in memory if isinstance(r, dict) and "kf_id" in r}
    validation = _read_json(run_dir / "validation.json", {}) or {}
    violations = validation.get("violations", []) if isinstance(validation, dict) else list(validation)
    attempts, final_qa = load_attempts(run_dir)
    masks = {}
    for kf_id, blk in blocking.items():
        per_entity = em.entity_masks(blk)
        masks[kf_id] = np.any(np.stack(list(per_entity.values())), axis=0) if per_entity else None

    # 2. Metrics.
    embedder = em.ClipEmbedder() if use_models else None
    identity = em.identity_consistency(frames, plan, index, blocking, sheets, embedder=embedder)
    layout = em.layout_iou(frames, plan, index, blocking, segmenter=embedder.segmenter if embedder else None)
    drift = em.background_drift(frames, plates, masks, ancestry=ancestry)
    adjacent = em.adjacent_change(frames, shot_of)
    fact_scores = {kf: qa["scores"]["fact_accuracy"] for kf, qa in final_qa.items()
                   if isinstance(qa.get("scores"), dict) and "fact_accuracy" in qa["scores"]}
    metrics: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": use_models,
        "plan": {"shots": len(plan.bible.shots), "keyframes": len(keyframes), "characters": len(plan.bible.characters),
                 "props": len(plan.bible.props), "duration_s": solved.total_duration_s, "warnings": solved.warnings},
        "coverage": {"keyframes": len(keyframes), "frames": len(frames), "blocking": len(blocking),
                     "missing": [f.kf_id for f in keyframes if f.kf_id not in frames]},
        "physics": em.physics_invariant_rate(violations, n_keyframes=len(keyframes)),
        "violations": [em.as_dict(v) for v in violations],
        "identity": identity,
        "layout": layout,
        "background_drift": drift,
        "adjacent_change": adjacent,
        "fact_accuracy": {"accuracy": em.mean_or_none(fact_scores.values()), "per_keyframe": fact_scores,
                          "source": "qa.json scores.fact_accuracy"},
        "timing": em.timing_summary(attempts),
        "memory": em.memory_summary(memory),
        "stage_timings": (_read_json(run_dir / "run.json", {}) or {}).get("timings", {}),
        "outputs": {
            "storyboard/storyboard.pdf": (run_dir / "storyboard" / "storyboard.pdf").is_file(),
            "animatic/animatic.mp4": _probe(run_dir / "animatic" / "animatic.mp4"),
            "graybox/graybox.mp4": _probe(run_dir / "graybox" / "graybox.mp4"),
        },
    }

    # 3. Per-keyframe rows for the report and grid captions.
    adj_by_to = {p["to"]: p for p in adjacent["pairs"]}
    rows = []
    for f in keyframes:
        tries = [a for a in attempts if a.get("kf_id") == f.kf_id]
        qa = final_qa.get(f.kf_id)
        ident = (identity.get("per_keyframe") or {}).get(f.kf_id, {})
        lay = (layout.get("per_keyframe") or {}).get(f.kf_id, {})
        rows.append({
            "kf_id": f.kf_id, "shot_id": f.shot_id, "timecode": format_timecode(f.t_s), "attempts": len(tries),
            "qa": ("pass" if qa.get("passed") else "fail") if qa else "–",
            "drift": drift["per_keyframe"].get(f.kf_id),
            "mad": adj_by_to.get(f.kf_id, {}).get("mad"), "ssim": adj_by_to.get(f.kf_id, {}).get("ssim"),
            "identity": em.mean_or_none(ident.values()), "layout": em.mean_or_none(lay.values()),
        })
    metrics["per_keyframe"] = rows
    return {"metrics": metrics, "plan": plan, "solved": solved, "frames": frames, "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a previs run directory")
    parser.add_argument("--run-dir", required=True, help="output/previs/<run>")
    parser.add_argument("--no-models", action="store_true", help="skip CLIP identity/layout metrics")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()

    print("==========================================================")
    print(" Previs Benchmark")
    print("==========================================================")
    print(f" Run dir: {run_dir}")
    print(f" Models:  {'off' if args.no_models else 'CLIP/CLIPSeg'}")
    print("==========================================================")

    result = evaluate(run_dir, use_models=not args.no_models)
    metrics, rows = result["metrics"], result["rows"]
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(eval_dir / "report.md", run_dir, result["plan"], metrics, rows)
    captions = {r["kf_id"]: f"dE76 {_fmt(r['drift'], 1)} · MAD {_fmt(r['mad'], 1)} · SSIM {_fmt(r['ssim'], 2)} · "
                            f"tries {r['attempts']} · QA {r['qa']}" for r in rows}
    build_grid(run_dir, result["plan"], result["solved"], result["frames"], captions, eval_dir / "grid.png")

    phys = metrics["physics"]
    print(f"\n[✓] Physics invariant rate {phys['invariant_rate']:.3f} ({phys['errors']} errors, {phys['warnings']} warnings)")
    print(f"    Frames {metrics['coverage']['frames']}/{metrics['coverage']['keyframes']}, "
          f"background ΔE76 {_fmt(metrics['background_drift']['mean'], 2)}")
    print(f"Report:  {eval_dir / 'report.md'}")
    print(f"Metrics: {eval_dir / 'metrics.json'}")
    print(f"Grid:    {eval_dir / 'grid.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
