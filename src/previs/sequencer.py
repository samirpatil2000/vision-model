"""
Previs sequencer: scene description -> physics plan -> solved timeline -> gray-box -> keyframes -> storyboard/animatic.

Stages (each writes into the run directory, so any stage can be resumed or re-run on its own):
  1. plan      Claude physics pass (call A bible + call B timeline) or a fixture plan.
  2. solve     Deterministic solver + validator, with at most `max_repairs` planner repair rounds.
  3. block     Camera solve + gray-box blocking per keyframe, plus the 24 fps physics-approval animatic.
  4. render    Cast/plate assets, then per keyframe: conditioning -> engine -> static lock -> QA -> retakes.
               An accepted frame is reused on a re-run only while its render fingerprint still matches.
  5. assemble  Storyboard pages/PDF and the hold animatic.
"""
import dataclasses
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src import config
from src.config import GenerationDefaults, Tier
from src.engine.base import EngineClient
from src.previs import seeds
from src.previs.eval_metrics import decided_fact_accuracy
from src.previs.schema import Keyframe, ScenePlan, Shot, Timeline
from src.previs.types import BlockingResult, FrameState, SolvedPlan, Violation

FINGERPRINT_VERSION = 1
FRAMES_MANIFEST = "manifest.json"       # frames/manifest.json: render fingerprint + QA summary per accepted frame
FINGERPRINT_FLOAT_DIGITS = 4            # solver floats are rounded so float noise never invalidates a frame
QUALITY_SCORE_PREFIXES = ("identity", "layout")
FACT_ACCURACY_MIN = 0.75                # share of "yes" among the judge's decided (yes/no) answers


class SequencerError(RuntimeError):
    """Raised when a stage cannot proceed (invalid plan after repairs, missing inputs)."""
    pass


@dataclass
class RunConfig:
    run_dir: Path
    tier: Tier
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)
    project: str = "previs"
    max_repairs: int = 2
    blocking_backend: str = "box"          # "box" or "blender"
    use_qa_models: bool = True
    fact_check: bool = False               # Claude yes/no fact QA per keyframe (Haiku via claude CLI)
    auto_approve: bool = True              # False: stop after the gray-box animatic for human approval
    cast_refs: Dict[str, str] = field(default_factory=dict)   # name/id fragment -> user image for that character


def match_cast_ref(ch, cast_refs: Dict[str, str]) -> Optional[str]:
    """The user image for a planned character: a cast key that matches its id or name (case-insensitive,
    ignoring spaces, hyphens and underscores; e.g. "co-commander" matches ch_co_commander / "Co-Commander")."""
    norm = lambda text: "".join(c for c in text.lower() if c.isalnum())
    targets = (norm(ch.id[len("ch_"):]), norm(ch.name))
    for key, path in cast_refs.items():
        k = norm(key)
        if k and any(k == t or k in t for t in targets):
            return path
    return None


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2))


# ---------------------------------------------------------------- render fingerprints
def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, FINGERPRINT_FLOAT_DIGITS) + 0.0      # + 0.0 folds -0.0 into 0.0
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def sha256_json(obj: Any) -> str:
    """sha256 of the canonical JSON of `obj` (sorted keys, floats rounded)."""
    text = json.dumps(_round_floats(_to_jsonable(obj)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_fingerprint(plan: ScenePlan, index, frame: FrameState, shot: Shot, keyframe: Keyframe,
                       visible_ids: Sequence[str], tier: Tier, defaults: GenerationDefaults, project: str,
                       prev_frame_sha: str) -> Dict[str, Any]:
    """
    What an accepted keyframe is rendered from; a frame on disk is reused only when every field matches.
      tier      tier name and pixel size;
      plan      sha256 of the keyframe's slice of the plan: scene, shot, keyframe, location, the definitions of the
                on-screen entities, the focus/active events, and the solved frame (entity states + camera,
                without the edited/physical timestamps, so a timing shift elsewhere does not invalidate it);
      seed      the attempt-0 seed (seed policy over project, shot and keyframe);
      sampling  sha256 of the sampling settings (steps, cfg, sampler, scheduler, sigmas, init noise, extra flags);
      prev      sha256 of the accepted previous frame of the same shot ("" for the first keyframe), so a retake
                invalidates every later frame of that shot (static-lock pixels and the shot anchor come from it).
    """
    location = index.locations.get(frame.location_id) or index.locations.get(shot.location_id)
    event_ids = sorted({frame.focus_event_id, *frame.active_event_ids} & set(index.events))
    solved_frame = {k: v for k, v in dataclasses.asdict(frame).items() if k not in ("t_s", "t_physical_s")}
    plan_slice = {
        "scene": plan.bible.scene.model_dump(mode="json"),
        "shot": shot.model_dump(mode="json"),
        "keyframe": keyframe.model_dump(mode="json"),
        "location": location.model_dump(mode="json") if location is not None else None,
        "entities": {eid: index.entities[eid].model_dump(mode="json") for eid in sorted(set(visible_ids))
                     if eid in index.entities},
        "events": {eid: index.events[eid].model_dump(mode="json") for eid in event_ids},
        "frame": solved_frame,
    }
    sampling = {
        "steps": defaults.steps, "cfg_scale": defaults.cfg_scale, "sampler": defaults.sampler,
        "scheduler": defaults.scheduler, "sigmas": list(config.VIGGLE_SIGMAS) if defaults.use_custom_sigmas else None,
        "init_noise": defaults.init_noise, "extra_model_flags": list(defaults.extra_model_flags),
    }
    return {
        "version": FINGERPRINT_VERSION,
        "tier": f"{tier.name}:{tier.width}x{tier.height}",
        "plan": sha256_json(plan_slice),
        "seed": seeds.seed_for(project, shot.id, frame.kf_id, 0),
        "sampling": sha256_json(sampling),
        "prev": prev_frame_sha,
    }


STALE_REASONS = {
    "version": "fingerprint format changed", "tier": "tier changed", "plan": "plan changed",
    "seed": "seed policy changed", "sampling": "sampling settings changed",
    "prev": "an earlier frame of the shot was re-rendered",
}


def stale_reasons(frame_path: Path, stored: Optional[Dict[str, Any]], fingerprint: Dict[str, Any]) -> List[str]:
    """Why the frame at `frame_path` cannot be reused for `fingerprint` (empty list: reuse it)."""
    if not Path(frame_path).is_file():
        return ["no accepted frame"]
    if not stored:
        return ["no render fingerprint (rendered by an older version)"]
    return [f"tier {stored.get('tier')} -> {value}" if key == "tier" else STALE_REASONS.get(key, key)
            for key, value in fingerprint.items() if stored.get(key) != value]


def read_frames_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}                            # unreadable manifest: every frame re-renders
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


# ---------------------------------------------------------------- attempt selection
def quality_score(scores: Dict[str, float]) -> float:
    """Mean of the identity and layout scores. Copy, lock and fact scores are pass/fail gates, not quality."""
    values = [float(v) for k, v in scores.items() if k.split(":", 1)[0] in QUALITY_SCORE_PREFIXES]
    return float(np.mean(values)) if values else 0.0


def attempt_rank(qa_result) -> Tuple[bool, float, int]:
    """Sort key for retakes (higher is better): passed first, then identity+layout quality, then fewer failures."""
    return bool(qa_result.passed), quality_score(qa_result.scores), -len(qa_result.failures)


# ---------------------------------------------------------------- repair triage
def _violation_key(v: Violation) -> Tuple[str, str, str, str, str]:
    return v.code, v.message, v.shot_id, v.kf_id, v.entity_id


def bible_errors(plan: ScenePlan, errors: Sequence[Violation]) -> List[Violation]:
    """
    The `errors` no repair round can fix, because they live in the scene bible: the ones the validator still
    reports when the timeline is removed (bible references, heights/sizes/marks/lenses out of range, target
    duration, the 180-degree rule, one location per setup, the "init" states). Repairs replace shot timelines only.
    """
    from src.previs import validator

    bible_only = ScenePlan(bible=plan.bible, timeline=Timeline(invariants=[], shots=[]))
    no_motion = SolvedPlan(frames=[], events={}, shots=[], total_duration_s=0.0)
    persistent = {_violation_key(v) for v in validator.validate(bible_only, no_motion) if v.severity == "error"}
    return [v for v in errors if _violation_key(v) in persistent]


def _violation_list(violations: Sequence[Violation], limit: int = 8) -> str:
    more = f"; ... and {len(violations) - limit} more" if len(violations) > limit else ""
    return "; ".join(f"{v.code}: {v.message}" for v in violations[:limit]) + more


class Sequencer:
    """Runs the previs stages against one run directory."""

    def __init__(self, cfg: RunConfig, planner=None, engine: Optional[EngineClient] = None):
        self.cfg = cfg
        self.planner = planner
        self.engine = engine
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------ 1. plan
    def plan(self, scene_text: str) -> ScenePlan:
        if self.planner is None:
            raise SequencerError("No planner configured")
        t0 = time.time()
        (self.run_dir / "scene.txt").write_text(scene_text)
        plan = self.planner.plan(scene_text, self.run_dir)
        self.save_plan(plan)
        self.timings["plan_s"] = round(time.time() - t0, 2)
        return plan

    def save_plan(self, plan: ScenePlan) -> None:
        (self.run_dir / "plan.json").write_text(plan.model_dump_json(indent=2))

    def load_plan(self) -> ScenePlan:
        path = self.run_dir / "plan.json"
        if not path.exists():
            raise SequencerError(f"No plan at {path}; run the plan stage first")
        return ScenePlan.model_validate_json(path.read_text())

    # ------------------------------------------------------------ 2. solve + validate (+ repair)
    def replan_command(self) -> str:
        """The `plan` command that rebuilds this run's plan (scene bible included) from its scene text."""
        scene = self.run_dir / "scene.txt"
        source = f"--scene-file {scene}" if scene.is_file() else '--scene "<scene description>"'
        return f"python previs.py plan --run-name {self.run_dir.name} {source}"

    def solve(self, plan: ScenePlan) -> SolvedPlan:
        from src.previs import physics, validator

        t0 = time.time()
        for round_idx in range(self.cfg.max_repairs + 1):
            solved = physics.solve(plan, max_duration_s=config.MAX_VIDEO_SECONDS)
            violations = validator.validate(plan, solved)
            errors = [v for v in violations if v.severity == "error"]
            write_json(self.run_dir / "validation.json", {"round": round_idx, "violations": violations})
            if not errors:
                break
            # Repairs replace shot timelines only, so a defect in the bible would burn every round unchanged.
            in_bible = bible_errors(plan, errors)
            if in_bible:
                raise SequencerError(
                    f"Plan has {len(in_bible)} error(s) in the scene bible (of {len(errors)} in total), which "
                    f"timeline repairs cannot fix: {_violation_list(in_bible)}. Re-plan the scene "
                    f"(`{self.replan_command()}`) or correct the bible in {self.run_dir / 'plan.json'}, then solve again."
                )
            if self.planner is None or round_idx == self.cfg.max_repairs:
                raise SequencerError(
                    f"Plan still has {len(errors)} physics/continuity errors after {round_idx} repair round(s): "
                    + _violation_list(errors)
                )
            shot_ids = validator.repair_shot_set(plan, errors)
            print(f"  ↻ repair round {round_idx + 1}: {len(errors)} errors in shots {', '.join(shot_ids)}")
            repaired = self.planner.repair(plan, validator.format_for_repair(errors), shot_ids, self.run_dir)
            if repaired.model_dump_json() == plan.model_dump_json():
                raise SequencerError(
                    f"Repair round {round_idx + 1} returned the plan unchanged; it still has {len(errors)} "
                    f"physics/continuity errors: {_violation_list(errors)}"
                )
            plan = repaired
            self.save_plan(plan)
        write_json(self.run_dir / "solved.json", solved)
        self.timings["solve_s"] = round(time.time() - t0, 2)
        self._plan, self._solved = plan, solved
        return solved

    # ------------------------------------------------------------ 3. camera + blocking + gray-box animatic
    def _renderer(self):
        if self.cfg.blocking_backend == "blender":
            from src.previs.blocking.blender_builder import BlenderRenderer
            renderer = BlenderRenderer(self.cfg.tier.width, self.cfg.tier.height)
            if not renderer.is_available():
                raise SequencerError(f"Blender not available at {config.BLENDER}")
            return renderer
        from src.previs.blocking.box_renderer import BoxRenderer
        return BoxRenderer(self.cfg.tier.width, self.cfg.tier.height)

    def block(self, plan: ScenePlan, solved: SolvedPlan, graybox_animatic: bool = True) -> Dict[str, BlockingResult]:
        from src.previs import camera, validator
        from src.previs.state import build_index

        t0 = time.time()
        index = build_index(plan)
        camera.solve_cameras(plan, solved, index, self.cfg.tier.width, self.cfg.tier.height)
        renderer = self._renderer()
        blocking: Dict[str, BlockingResult] = {}
        for frame in solved.frames:
            out_dir = self.run_dir / "graybox" / frame.kf_id
            out_dir.mkdir(parents=True, exist_ok=True)
            blocking[frame.kf_id] = renderer.render(frame, plan, index, out_dir)
        write_json(self.run_dir / "solved.json", solved)  # now includes cameras

        visibility = [v for v in validator.validate(plan, solved, blocking) if v.code == "subject_visible"]
        if visibility:
            write_json(self.run_dir / "visibility_warnings.json", visibility)
            for v in visibility:
                print(f"  ⚠ {v.kf_id}: {v.message}")

        if graybox_animatic:
            from src.previs.assembly.graybox import build_graybox_animatic
            build_graybox_animatic(self.run_dir, plan, solved, index, self.cfg.tier.width, self.cfg.tier.height)
        self.timings["block_s"] = round(time.time() - t0, 2)
        return blocking

    # ------------------------------------------------------------ 4. render keyframes
    def render(self, plan: ScenePlan, solved: SolvedPlan, blocking: Dict[str, BlockingResult]) -> Dict[str, str]:
        from src.previs import conditioning, qa
        from src.previs.assets import AssetBank
        from src.previs.memory import KeyframeMemory, MemoryRecord
        from src.previs.state import build_index
        from src.processing.compositor import StrictPixelCompositor

        if self.engine is None:
            raise SequencerError("No engine configured")
        t0 = time.time()
        index = build_index(plan)
        bible = plan.bible
        assets = AssetBank(self.run_dir, self.engine, self.cfg.tier, bible.scene.style, self.cfg.project,
                           defaults=self.cfg.defaults)
        memory_path = self.run_dir / "memory.json"
        # Continuity memory is rebuilt in keyframe order, so a reference is always a frame accepted earlier in
        # this pass (never a stale frame, or a later one, left by an earlier render). Saved records only carry
        # score/ancestry/approval over for frames that are reused.
        saved_memory = KeyframeMemory.load(memory_path) if memory_path.exists() else KeyframeMemory()
        memory = KeyframeMemory()
        plan_kf_ids = [f.kf_id for f in solved.frames]

        def save_memory() -> None:
            merged = KeyframeMemory()
            merged.records = list(memory.records) + [
                r for r in saved_memory.records if r.kf_id in plan_kf_ids and memory.get(r.kf_id) is None]
            merged.save(memory_path)

        gate = qa.QualityGate(use_models=self.cfg.use_qa_models)
        compositor = StrictPixelCompositor(strict_mode=True)
        judge = None
        if self.cfg.fact_check:
            from src.previs.planner.claude_cli import ClaudeCliJudge
            judge = ClaudeCliJudge()

        # 4a. Assets: one full-body sheet per character, one plate per location.
        sheets: Dict[str, str] = {}
        for ch in bible.characters:
            ref = match_cast_ref(ch, self.cfg.cast_refs)
            full_body, _face = assets.use_reference(ch, ref) if ref else assets.ensure_character(ch)
            sheets[ch.id] = full_body
        plates = {loc.id: assets.ensure_plate(loc, bible.scene.time_of_day) for loc in bible.locations}

        frames_dir = self.run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        manifest_path = frames_dir / FRAMES_MANIFEST
        manifest = read_frames_manifest(manifest_path)
        accepted: Dict[str, str] = {}
        results_log: List[Dict[str, Any]] = []
        prev_by_shot: Dict[str, str] = {}

        for frame in solved.frames:
            shot = index.shots[frame.shot_id]
            keyframe = index.keyframe_by_id[frame.kf_id]
            prev_kf_id = conditioning.previous_keyframe_in_shot(index, frame.kf_id)
            prev_frame = solved.frame(prev_kf_id) if prev_kf_id else None
            prev_blocking = blocking.get(prev_kf_id) if prev_kf_id else None
            visible = conditioning.on_screen_ids(frame, blocking[frame.kf_id], shot.subject_id)
            prev_accepted = prev_by_shot.get(frame.shot_id)
            fingerprint = render_fingerprint(plan, index, frame, shot, keyframe, visible, self.cfg.tier,
                                             self.cfg.defaults, self.cfg.project,
                                             file_sha256(prev_accepted) if prev_accepted else "")
            existing = frames_dir / f"{frame.kf_id}.png"
            stale = stale_reasons(existing, manifest.get(frame.kf_id), fingerprint)
            if not stale:
                accepted[frame.kf_id] = str(existing)
                prev_by_shot[frame.shot_id] = str(existing)
                memory.add(saved_memory.get(frame.kf_id) or MemoryRecord(   # memory.json lost: neutral record
                    kf_id=frame.kf_id, shot_id=frame.shot_id, setup_id=shot.setup_id,
                    location_id=frame.location_id, camera_yaw=frame.camera.yaw_deg if frame.camera else 0.0,
                    path=str(existing), ancestry_depth=1, score=0.0, approved=False,
                ))
                continue
            if existing.is_file():
                print(f"  ↻ {frame.kf_id}: re-rendering ({'; '.join(stale)})")
            kf_dir = self.run_dir / "keyframes" / frame.kf_id
            shutil.rmtree(kf_dir, ignore_errors=True)   # attempts of a superseded render must not mix with these
            best = None
            for attempt in range(self.cfg.defaults.max_retakes + 1):
                attempt_dir = kf_dir / f"attempt_{attempt}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                cond = conditioning.build(
                    plan, index, frame, shot, keyframe, blocking[frame.kf_id], assets, memory,
                    self.cfg.tier, self.cfg.defaults, attempt, attempt_dir, prev_by_shot.get(frame.shot_id),
                    prev_frame=prev_frame, prev_blocking=prev_blocking,
                )
                cond.request.tag = f"{frame.kf_id}_attempt_{attempt}"
                print(f"  ▶ {frame.kf_id} attempt {attempt}: {cond.request.prompt[:90]}...")
                result = self.engine.generate(cond.request)
                write_json(attempt_dir / "result.json", result)
                output_path = result.image_path

                # Static camera: everything outside the change mask is copied bit-exact from the previous frame.
                if cond.static_lock and cond.lock_mask_path and prev_by_shot.get(frame.shot_id):
                    original = Image.open(prev_by_shot[frame.shot_id]).convert("RGB")
                    candidate = np.array(Image.open(output_path).convert("RGB").resize(original.size))
                    allowed = (np.array(Image.open(cond.lock_mask_path).convert("L").resize(original.size)) > 127).astype(np.uint8) * 255
                    locked, _ = compositor.composite(original, candidate, allowed, feather_pixels=4)
                    output_path = str(attempt_dir / "output_locked.png")
                    locked.save(output_path)

                qa_result = gate.check(
                    output_path, frame, blocking[frame.kf_id], plan, index,
                    {cid: sheets[cid] for cid in visible if cid in sheets},
                    prev_by_shot.get(frame.shot_id),
                    cond.lock_mask_path if cond.static_lock else None,
                    prev_by_shot.get(frame.shot_id) if cond.static_lock else None,
                    prev_blocking=prev_blocking, prev_frame=prev_frame,
                )
                qa_record: Dict[str, Any] = dataclasses.asdict(qa_result)
                if judge is not None and qa_result.passed:
                    questions = qa.fact_questions(plan, index, frame, visible)
                    if questions:
                        try:
                            answers = judge.facts(output_path, questions, attempt_dir)
                        except Exception as exc:  # optional QA signal: a CLI/judge failure never aborts the render
                            error = f"{type(exc).__name__}: {exc}"
                            write_json(attempt_dir / "facts.json", {"questions": questions, "error": error})
                            qa_record["facts_error"] = error
                            print(f"    ⚠ fact check unavailable ({error[:120]}); using the model-free QA result")
                        else:
                            write_json(attempt_dir / "facts.json", {"questions": questions, "answers": answers})
                            # "unsure" (the judge cannot tell, e.g. the thing is out of frame) is no evidence
                            # either way, so only decided answers count; with none, facts do not gate the frame.
                            accuracy = decided_fact_accuracy(answers)
                            if accuracy is None:
                                print("    ⚠ fact check undecided (every answer unsure); using the model-free QA result")
                            else:
                                qa_result.scores["fact_accuracy"] = accuracy
                                if accuracy < FACT_ACCURACY_MIN:
                                    qa_result.passed = False
                                    qa_result.failures.append("facts")
                            qa_record = dataclasses.asdict(qa_result)
                            if accuracy is None:
                                qa_record["facts_undecided"] = True
                write_json(attempt_dir / "qa.json", qa_record)
                results_log.append({"kf_id": frame.kf_id, "attempt": attempt, "seconds": result.seconds,
                                    "peak_rss_bytes": result.peak_rss_bytes, "qa_passed": qa_result.passed})
                rank = attempt_rank(qa_result)
                if best is None or rank > best["rank"]:
                    best = {"rank": rank, "attempt": attempt, "path": output_path, "cond": cond, "qa": qa_result}
                if qa_result.passed:
                    break
                print(f"    ✗ QA failed ({', '.join(qa_result.failures)}); retaking")

            shutil.copyfile(best["path"], existing)
            accepted[frame.kf_id] = str(existing)
            prev_by_shot[frame.shot_id] = str(existing)
            cam = frame.camera
            score = quality_score(best["qa"].scores)
            memory.add(MemoryRecord(
                kf_id=frame.kf_id, shot_id=frame.shot_id, setup_id=shot.setup_id, location_id=frame.location_id,
                camera_yaw=cam.yaw_deg if cam else 0.0, path=str(existing),
                ancestry_depth=best["cond"].ancestry_depth + 1, score=score, approved=False,
            ))
            manifest[frame.kf_id] = {**fingerprint, "image": file_sha256(existing), "attempt": best["attempt"],
                                     "attempts": attempt + 1, "qa_passed": bool(best["qa"].passed), "score": score}
            write_json(manifest_path, manifest)
            save_memory()

        write_json(manifest_path, {k: manifest[k] for k in plan_kf_ids if k in manifest})
        save_memory()
        write_json(self.run_dir / "render_log.json", results_log)
        write_json(self.run_dir / "assets.json", {"cast_sheets": sheets, "plates": plates})
        self.timings["render_s"] = round(time.time() - t0, 2)
        return accepted

    # ------------------------------------------------------------ 5. assemble
    def assemble(self, plan: ScenePlan, solved: SolvedPlan, frames: Dict[str, str],
                 size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Storyboard + animatic. `size` is the animatic size; None uses the accepted frames' own size."""
        from src.previs.assembly.animatic import build_animatic
        from src.previs.assembly.storyboard import build_storyboard

        t0 = time.time()
        pages = build_storyboard(self.run_dir, plan, solved, frames)
        width, height = size if size is not None else (None, None)
        animatic = build_animatic(self.run_dir, plan, solved, frames, fps=config.FPS, width=width, height=height)
        self.timings["assemble_s"] = round(time.time() - t0, 2)
        return {"storyboard_pages": pages, "animatic": animatic}

    def write_run_manifest(self, extra: Optional[Dict[str, Any]] = None) -> None:
        from src.engine.version import read_build_info

        manifest = {
            "engine_build_info": read_build_info(),
            "tier": dataclasses.asdict(self.cfg.tier),
            "defaults": dataclasses.asdict(self.cfg.defaults),
            "project": self.cfg.project,
            "blocking_backend": self.cfg.blocking_backend,
            "timings": self.timings,
            "engine_pinned_sha": config.pinned_engine_sha() if config.UPSTREAM_SHA_FILE.exists() else None,
        }
        if extra:
            manifest.update(extra)
        path = self.run_dir / "run.json"
        if path.exists():
            previous = json.loads(path.read_text())
            # Stage timings accumulate across invocations (`run --approve`, then `render --approved`).
            manifest["timings"] = {**(previous.get("timings") or {}), **self.timings}
            previous.update(manifest)
            manifest = previous
        write_json(path, manifest)
