import dataclasses
import json
import shlex
import shutil
import signal
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import benchmark_previs
import previs
from src import config
from src.config import Tier
from src.engine.base import EngineClient, GenerationRequest, GenerationResult
from src.previs import eval_metrics as em
from src.previs import physics
from src.previs.schema import ScenePlan
from src.previs.state import build_index
from src.previs.types import BlockingResult, Violation

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
W, H = 96, 64
PLATE_RGB = (180, 170, 150)
# id value -> (entity, box, colour) for the synthetic blocking + frames
BOXES = {1: ("ch_mara", (10, 10, 30, 60), (212, 160, 23)), 2: ("ch_jon", (60, 8, 80, 60), (85, 107, 47)),
         3: ("pr_table", (35, 40, 58, 56), (139, 90, 43))}


def solid(path: Path, rgb, size=(W, H)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path)
    return str(path)


def id_map_image(path: Path) -> np.ndarray:
    ids = np.zeros((H, W), dtype=np.uint8)
    for value, (_eid, (x0, y0, x1, y1), _rgb) in BOXES.items():
        ids[y0:y1, x0:x1] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids).save(path)
    return ids


def frame_with_entities(path: Path, background=PLATE_RGB) -> str:
    arr = np.zeros((H, W, 3), dtype=np.uint8) + np.array(background, dtype=np.uint8)
    for _eid, (x0, y0, x1, y1), rgb in BOXES.values():
        arr[y0:y1, x0:x1] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return str(path)


def mean_colour_embedder(images):
    """Deterministic stand-in for CLIP: the normalised mean colour of each image."""
    feats = np.array([np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(axis=0) for im in images])
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


class OracleSegmenter:
    """Returns the id-mask of whichever entity the prompt names (perfect layout)."""

    def __init__(self, index, ids):
        self.index, self.ids = index, ids
        self.prompts = []

    def segment(self, image, prompt):
        self.prompts.append(prompt)
        value = next(v for v, (eid, _b, _c) in BOXES.items() if em.entity_prompt(self.index, eid) == prompt)

        class Result:
            target_mask = (self.ids == value).astype(np.uint8) * 255
        return Result()


class TestPixelMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rgb_to_lab_reference_values(self):
        lab = em.rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)

    def test_ssim_identity_and_noise(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (64, 64)).astype(np.float64)
        self.assertAlmostEqual(em.ssim(a, a), 1.0, places=6)
        light = em.ssim(a, np.clip(a + rng.normal(0, 10, a.shape), 0, 255))
        heavy = em.ssim(a, np.clip(a + rng.normal(0, 60, a.shape), 0, 255))
        self.assertGreater(light, heavy)
        smooth = np.tile(np.linspace(0, 255, 64), (64, 1))
        self.assertLess(em.ssim(smooth, np.clip(smooth + rng.normal(0, 40, smooth.shape), 0, 255)), 0.3)

    def test_background_drift_ignores_entities_and_groups_by_depth(self):
        plate = solid(self.tmp / "plate.png", PLATE_RGB)
        ids = id_map_image(self.tmp / "id.png")
        frames = {"kf_a": frame_with_entities(self.tmp / "a.png"),
                  "kf_b": frame_with_entities(self.tmp / "b.png", background=(170, 170, 150))}
        masks = {"kf_a": ids > 0, "kf_b": ids > 0}
        out = em.background_drift(frames, plate, masks, ancestry={"kf_a": 0, "kf_b": 1})
        self.assertAlmostEqual(out["per_keyframe"]["kf_a"], 0.0, places=6)
        self.assertGreater(out["per_keyframe"]["kf_b"], 1.0)
        self.assertEqual(set(out["by_depth"]), {"0", "1"})
        # Without masks the entities count as drift.
        self.assertGreater(em.background_drift({"kf_a": frames["kf_a"]}, plate, {})["mean"], 5.0)
        # Per-keyframe plates; missing plates are skipped.
        self.assertEqual(em.background_drift(frames, {"kf_a": plate}, masks)["per_keyframe"], {"kf_a": 0.0})

    def test_adjacent_change_only_within_shots(self):
        frames = {"kf_1": solid(self.tmp / "1.png", (10, 10, 10)), "kf_2": solid(self.tmp / "2.png", (20, 10, 10)),
                  "kf_3": solid(self.tmp / "3.png", (200, 200, 200))}
        out = em.adjacent_change(frames, {"kf_1": "sh_a", "kf_2": "sh_a", "kf_3": "sh_b"})
        self.assertEqual([(p["from"], p["to"]) for p in out["pairs"]], [("kf_1", "kf_2")])
        self.assertAlmostEqual(out["mean_mad"], 10 / 3, places=3)
        self.assertEqual(len(em.adjacent_change(frames)["pairs"]), 2)


class TestModelMetricsWithFakes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = ScenePlan.model_validate_json(FIXTURE.read_text())
        self.index = build_index(self.plan)
        self.ids = id_map_image(self.tmp / "id.png")
        id_map = {v: eid for v, (eid, _b, _c) in BOXES.items()}
        boxes = {eid: box for eid, box, _c in BOXES.values()}
        self.blocking = {kf: BlockingResult(blocking_path="", id_path=str(self.tmp / "id.png"), depth_path="",
                                            id_map=id_map, bboxes=boxes, visible_frac={e: 1.0 for e in boxes})
                         for kf in ("kf_wide_1", "kf_jon_1")}
        self.frames = {kf: frame_with_entities(self.tmp / f"{kf}.png") for kf in self.blocking}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_identity_skipped_without_models(self):
        out = em.identity_consistency(self.frames, self.plan, self.index, self.blocking, {})
        self.assertTrue(out["skipped"])
        self.assertTrue(em.layout_iou(self.frames, self.plan, self.index, self.blocking)["skipped"])

    def test_identity_scores_sheets_and_cross_shot(self):
        sheets = {"ch_mara": solid(self.tmp / "mara.png", BOXES[1][2]), "ch_jon": solid(self.tmp / "jon.png", (0, 0, 255))}
        out = em.identity_consistency(self.frames, self.plan, self.index, self.blocking, sheets,
                                      embedder=mean_colour_embedder)
        self.assertFalse(out["skipped"])
        self.assertEqual(out["crops"], 4)
        mara, jon = out["per_character"]["ch_mara"], out["per_character"]["ch_jon"]
        self.assertGreater(mara["sheet_mean"], jon["sheet_mean"])
        self.assertGreater(mara["sheet_mean"], 0.95)
        # Same character in two different shots -> high cross-shot similarity.
        self.assertGreater(mara["cross_shot_mean"], 0.95)
        self.assertEqual(set(out["per_keyframe"]), {"kf_wide_1", "kf_jon_1"})

    def with_quadruped(self, ch_id: str):
        data = json.loads(FIXTURE.read_text())
        for ch in data["bible"]["characters"]:
            if ch["id"] == ch_id:
                ch.update(body="quadruped", appearance="Small brown terrier with a short wiry coat", wardrobe="red collar")
        plan = ScenePlan.model_validate(data)
        return plan, build_index(plan)

    def test_identity_skips_quadrupeds(self):
        plan, index = self.with_quadruped("ch_jon")
        sheets = {"ch_mara": solid(self.tmp / "mara.png", BOXES[1][2]), "ch_jon": solid(self.tmp / "jon.png", (0, 0, 255))}
        out = em.identity_consistency(self.frames, plan, index, self.blocking, sheets, embedder=mean_colour_embedder)
        self.assertEqual(out["crops"], 2)                                  # Mara in both keyframes, no dog crops
        self.assertEqual(set(out["per_character"]), {"ch_mara"})
        self.assertEqual(out["skipped_characters"], ["ch_jon"])
        self.assertTrue(all(set(scores) == {"ch_mara"} for scores in out["per_keyframe"].values()))
        self.assertEqual(out["pass_rate"], 1.0)                             # the dog vs its sheet no longer fails it

    def test_layout_prompt_for_a_quadruped_describes_the_animal(self):
        from src.previs.qa import segmentation_prompt
        _plan, index = self.with_quadruped("ch_jon")
        prompt = em.entity_prompt(index, "ch_jon")
        self.assertNotIn("person", prompt)
        self.assertIn("terrier", prompt)
        self.assertEqual(prompt, segmentation_prompt(index, "ch_jon"))     # same text the QA gate segments with
        self.assertEqual(em.entity_prompt(index, "ch_mara"),
                         "a person wearing mustard yellow knit sweater and dark blue jeans")

    def test_layout_iou_with_oracle_segmenter(self):
        seg = OracleSegmenter(self.index, self.ids)
        out = em.layout_iou(self.frames, self.plan, self.index, self.blocking, segmenter=seg)
        self.assertEqual(out["pairs"], 6)
        self.assertEqual(out["mean"], 1.0)
        self.assertIn("a person wearing mustard yellow knit sweater and dark blue jeans", seg.prompts)
        self.assertIn("kitchen table", seg.prompts)


class TestSummaries(unittest.TestCase):
    def test_physics_invariant_rate(self):
        violations = [
            Violation(code="custody", message="two holders", kf_id="kf_a"),
            {"code": "custody", "message": "again", "kf_id": "kf_b", "severity": "error"},
            {"code": "subject_visible", "message": "small", "kf_id": "kf_a", "severity": "warning"},
        ]
        out = em.physics_invariant_rate(violations, n_keyframes=4)
        self.assertEqual((out["errors"], out["warnings"]), (2, 1))
        self.assertEqual(out["failed_checks"], ["custody"])
        self.assertEqual(out["checks_passed"], len(em.CHECK_CODES) - 1)
        self.assertAlmostEqual(out["clean_keyframe_rate"], 0.5)
        self.assertEqual(em.physics_invariant_rate([])["invariant_rate"], 1.0)
        self.assertEqual(em.physics_invariant_rate([{"code": "new_check", "message": ""}])["checks"],
                         len(em.CHECK_CODES) + 1)

    def test_fact_accuracy(self):
        out = em.fact_accuracy({"kf_a": ["yes", "yes"], "kf_b": ["no"]})
        self.assertAlmostEqual(out["accuracy"], 2 / 3, places=3)
        self.assertEqual(out["per_keyframe"], {"kf_a": 1.0, "kf_b": 0.0})
        self.assertIsNone(em.fact_accuracy([])["accuracy"])

    def test_fact_accuracy_excludes_unsure_answers(self):
        out = em.fact_accuracy(["yes", "no", "Yes", "unsure"])
        self.assertAlmostEqual(out["accuracy"], 2 / 3, places=3)       # 2 yes of 3 decided, not 2 of 4
        self.assertEqual((out["questions"], out["unsure"]), (4, 1))
        self.assertIsNone(em.fact_accuracy(["unsure", "unsure"])["accuracy"])   # nothing decided
        per_kf = em.fact_accuracy({"kf_a": ["yes", "unsure"], "kf_b": ["unsure"]})
        self.assertEqual(per_kf["per_keyframe"], {"kf_a": 1.0, "kf_b": None})
        self.assertEqual(per_kf["accuracy"], 1.0)
        self.assertIsNone(em.decided_fact_accuracy(["unsure"]))
        self.assertEqual(em.decided_fact_accuracy(["no", "unsure", "unsure"]), 0.0)

    def test_timing_summary(self):
        results = [
            {"kf_id": "kf_a", "attempt": 0, "seconds": 10.0, "peak_rss_bytes": 2 * 1024 ** 3, "qa_passed": False,
             "phase_seconds": {"sampling": 6.0}, "fallback_warnings": ["Flash Attention unavailable"]},
            {"kf_id": "kf_a", "attempt": 1, "seconds": 12.0, "peak_rss_bytes": 3 * 1024 ** 3, "qa_passed": True,
             "phase_seconds": {"sampling": 8.0}},
            {"kf_id": "kf_b", "attempt": 0, "seconds": 8.0, "qa_passed": True},
        ]
        out = em.timing_summary(results)
        self.assertEqual(out["requests"], 3)
        self.assertEqual(out["total_s"], 30.0)
        self.assertEqual(out["peak_rss_gb"], 3.0)
        self.assertEqual(out["phase_mean_s"], {"sampling": 7.0})
        self.assertEqual(out["fallback_warnings"], 1)
        self.assertEqual(out["mean_attempts"], 1.5)
        self.assertEqual(out["first_pass_rate"], 0.5)
        self.assertEqual(out["final_pass_rate"], 1.0)
        self.assertIsNone(em.timing_summary([])["mean_s"])

    def test_memory_summary_accepts_saved_json_shape(self):
        saved = {"records": [
            {"kf_id": "kf_a", "ancestry_depth": 1, "score": 0.8, "approved": False, "location_id": "lo_k"},
            {"kf_id": "kf_b", "ancestry_depth": 2, "score": 0.6, "approved": True, "location_id": "lo_k"},
        ]}
        out = em.memory_summary(saved)
        self.assertEqual(out["records"], 2)
        self.assertEqual(out["approved"], 1)
        self.assertEqual(out["depth_histogram"], {"1": 1, "2": 1})
        self.assertEqual(out["max_depth"], 2)
        self.assertEqual(em.memory_summary(saved["records"])["records"], 2)


def make_run_dir(root: Path, plan: ScenePlan, solved) -> Path:
    """Synthetic run directory following the sequencer layout."""
    run = root / "fixture_run"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(plan.model_dump_json(indent=2))
    (run / "solved.json").write_text(json.dumps(dataclasses.asdict(solved)))
    (run / "validation.json").write_text(json.dumps({"round": 0, "violations": [
        {"code": "subject_visible", "message": "subject small | partly hidden", "shot_id": "sh_glass",
         "kf_id": "kf_glass_2", "entity_id": "pr_glass", "severity": "warning"}]}))
    (run / "run.json").write_text(json.dumps({"timings": {"plan_s": 1.0, "render_s": 30.0}}))
    solid(run / "assets" / "plates" / "lo_kitchen_v1.png", (0, 0, 0))
    solid(run / "assets" / "plates" / "lo_kitchen_v2.png", PLATE_RGB)
    solid(run / "assets" / "cast" / "ch_mara_full_v1.png", BOXES[1][2])
    records = []
    log = []
    for n, f in enumerate(solved.frames):
        kf_dir = run / "graybox" / f.kf_id
        id_map_image(kf_dir / "id.png")
        (kf_dir / "meta.json").write_text(json.dumps({"id_map": {str(v): e for v, (e, _b, _c) in BOXES.items()}}))
        solid(kf_dir / "blocking.png", (120, 120, 120))
        if f.kf_id != "kf_jon_2":
            frame_with_entities(run / "frames" / f"{f.kf_id}.png")
        for attempt in range(2 if f.kf_id == "kf_glass_2" else 1):
            attempt_dir = run / "keyframes" / f.kf_id / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True)
            passed = not (f.kf_id == "kf_glass_2" and attempt == 0)
            (attempt_dir / "result.json").write_text(json.dumps({
                "image_path": "x.png", "seconds": 20.0 + n, "command": ["sd-cli"], "engine_commit": "2988060",
                "peak_rss_bytes": 9 * 1024 ** 3, "phase_seconds": {"sampling": 15.0}, "fallback_warnings": []}))
            (attempt_dir / "qa.json").write_text(json.dumps({
                "passed": passed, "scores": {"fact_accuracy": 1.0 if passed else 0.5}, "failures": [] if passed else ["facts"]}))
            log.append({"kf_id": f.kf_id, "attempt": attempt, "seconds": 20.0 + n, "qa_passed": passed})
        records.append({"kf_id": f.kf_id, "shot_id": f.shot_id, "setup_id": "su_x", "location_id": f.location_id,
                        "camera_yaw": 0.0, "path": "", "ancestry_depth": f.order_in_shot - 1, "score": 0.7,
                        "approved": False})
    (run / "memory.json").write_text(json.dumps({"records": records}))
    (run / "render_log.json").write_text(json.dumps(log))
    return run


class TestBenchmarkCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = ScenePlan.model_validate_json(FIXTURE.read_text())
        self.solved = physics.solve(self.plan)
        self.run = make_run_dir(self.tmp, self.plan, self.solved)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_models_run_writes_metrics_report_and_grid(self):
        with redirect_stdout(StringIO()):
            code = benchmark_previs.main(["--run-dir", str(self.run), "--no-models"])
        self.assertEqual(code, 0)
        eval_dir = self.run / "eval"
        metrics = json.loads((eval_dir / "metrics.json").read_text())
        self.assertFalse(metrics["models"])
        self.assertTrue(metrics["identity"]["skipped"])
        self.assertTrue(metrics["layout"]["skipped"])
        self.assertEqual(metrics["coverage"], {"keyframes": 7, "frames": 6, "blocking": 7, "missing": ["kf_jon_2"]})
        self.assertEqual(metrics["physics"]["errors"], 0)
        self.assertEqual(metrics["physics"]["warnings"], 1)
        self.assertEqual(metrics["physics"]["invariant_rate"], 1.0)
        # Latest plate version is used and entity pixels are masked out -> no drift.
        self.assertEqual(metrics["background_drift"]["mean"], 0.0)
        self.assertEqual(set(metrics["background_drift"]["by_depth"]), {"0", "1", "2"})
        self.assertEqual(metrics["timing"]["requests"], 8)
        self.assertAlmostEqual(metrics["timing"]["first_pass_rate"], 6 / 7, places=3)
        self.assertEqual(metrics["timing"]["final_pass_rate"], 1.0)
        self.assertEqual(metrics["fact_accuracy"]["accuracy"], 1.0)
        self.assertEqual(metrics["memory"]["records"], 7)
        self.assertEqual(metrics["stage_timings"], {"plan_s": 1.0, "render_s": 30.0})
        self.assertEqual(len(metrics["per_keyframe"]), 7)
        self.assertEqual(len(metrics["adjacent_change"]["pairs"]), 3)   # wide 1->2, glass 1->2->3; jon_2 missing

        report = (eval_dir / "report.md").read_text()
        self.assertIn("# Previs Benchmark: `fixture_run`", report)
        self.assertIn("| Physics invariant rate (checks passed) | `1.000`", report)
        self.assertIn("skipped (--no-models)", report)
        self.assertIn("`kf_glass_2`", report)
        self.assertIn("subject small \\| partly hidden", report)

        with Image.open(eval_dir / "grid.png") as grid:
            self.assertGreater(grid.size[0], 4 * benchmark_previs.GRID_THUMB_W)
            self.assertGreater(grid.size[1], 4 * 180)

    def test_missing_solved_json_is_recomputed(self):
        (self.run / "solved.json").unlink()
        result = benchmark_previs.evaluate(self.run, use_models=False)
        self.assertAlmostEqual(result["metrics"]["plan"]["duration_s"], self.solved.total_duration_s)

    def test_load_blocking_derives_bboxes_from_id_map(self):
        blk = benchmark_previs.load_blocking(self.run / "graybox" / "kf_wide_1", build_index(self.plan))
        self.assertEqual(blk.id_map[1], "ch_mara")
        self.assertEqual(blk.bboxes["ch_mara"], BOXES[1][1])
        self.assertEqual(blk.visible_frac["pr_table"], 1.0)
        self.assertIsNone(benchmark_previs.load_blocking(self.run / "graybox" / "kf_missing", build_index(self.plan)))

    def test_latest_version_picks_highest_number(self):
        self.assertTrue(benchmark_previs.latest_version(self.run / "assets" / "plates", "lo_kitchen").endswith("_v2.png"))
        self.assertIsNone(benchmark_previs.latest_version(self.run / "assets" / "plates", "lo_garden"))

    def test_attempts_follow_the_frames_manifest(self):
        # A retake passed at attempt 0; attempt_1/attempt_2 failed in an earlier render and were left behind.
        kf_dir = self.run / "keyframes" / "kf_glass_3"
        shutil.rmtree(kf_dir, ignore_errors=True)
        for n, passed in ((0, True), (1, False), (2, False)):
            attempt = kf_dir / f"attempt_{n}"
            attempt.mkdir(parents=True)
            (attempt / "result.json").write_text(json.dumps({"seconds": 10.0 + n}))
            (attempt / "qa.json").write_text(json.dumps({"passed": passed, "scores": {}, "failures": []}))
        (self.run / "frames" / "manifest.json").write_text(json.dumps(
            {"kf_glass_3": {"tier": "draft:512x288", "attempt": 0, "attempts": 1, "qa_passed": True}}))
        records, final_qa = benchmark_previs.load_attempts(self.run)
        self.assertEqual([r["attempt"] for r in records if r["kf_id"] == "kf_glass_3"], [0])
        self.assertTrue(final_qa["kf_glass_3"]["passed"])

    def test_final_qa_is_the_accepted_attempt(self):
        # Every retake failed and the sequencer accepted attempt 1 (best identity/layout), not the last one.
        kf_dir = self.run / "keyframes" / "kf_glass_3"
        shutil.rmtree(kf_dir, ignore_errors=True)
        for n in range(3):
            attempt = kf_dir / f"attempt_{n}"
            attempt.mkdir(parents=True)
            (attempt / "qa.json").write_text(json.dumps(
                {"passed": False, "scores": {"identity:ch_mara": 0.1 * (n + 1)}, "failures": [f"attempt{n}"]}))
        (self.run / "frames" / "manifest.json").write_text(json.dumps(
            {"kf_glass_3": {"attempt": 1, "attempts": 3, "qa_passed": False}}))
        records, final_qa = benchmark_previs.load_attempts(self.run)
        self.assertEqual(len([r for r in records if r["kf_id"] == "kf_glass_3"]), 3)
        self.assertEqual(final_qa["kf_glass_3"]["failures"], ["attempt1"])


class InitEchoEngine(EngineClient):
    """Writes the init image back (or flat grey) at the requested size: no weights, real files."""

    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.init_image:
            img = Image.open(request.init_image).convert("RGB").resize((request.width, request.height))
        else:
            img = Image.new("RGB", (request.width, request.height), (128, 128, 128))
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(request.output_path)
        return GenerationResult(image_path=request.output_path, seconds=0.01, command=["fake"], engine_commit="fake")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
class TestPrevisCliEval(unittest.TestCase):
    """README flow: `run --approve` stops after the gray-box; `render --approved` renders and refreshes eval/."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.engines = []
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}
        self.patches = [
            mock.patch.object(config, "PREVIS_OUTPUT_DIR", self.tmp),
            mock.patch.object(previs, "_engine", lambda args: self.engines.append(InitEchoEngine()) or self.engines[-1]),
            mock.patch.dict(config.TIERS, {"mini": Tier("mini", 256, 160, (256, 128), (192, 128))}),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv: str) -> str:
        buf = StringIO()
        with redirect_stdout(buf):
            previs.main(list(argv))
        return buf.getvalue()

    def test_approve_then_render_writes_a_current_eval_report(self):
        out = self.run_cli("run", "--planner", "fixture", "--approve", "--run-name", "kitchen", "--tier", "mini",
                           "--no-qa-models", "--skip-graybox")
        run_dir = self.tmp / "kitchen"
        self.assertFalse((run_dir / "eval" / "report.md").exists())        # nothing rendered yet: no eval
        self.assertTrue(self.engines[0].closed)                              # no resident engine left running
        follow_up = next(line.strip() for line in out.splitlines() if "previs.py render" in line)
        argv = shlex.split(follow_up)[2:]                                    # drop "python previs.py"
        self.assertEqual(argv[:4], ["render", "--run-name", "kitchen", "--approved"])
        self.assertEqual(argv[argv.index("--tier") + 1], "mini")
        self.assertEqual(argv[argv.index("--planner") + 1], "fixture")

        self.run_cli(*argv)
        report = (run_dir / "eval" / "report.md").read_text()
        self.assertIn("**Frames rendered**: 7/7", report)
        timings = json.loads((run_dir / "run.json").read_text())["timings"]
        self.assertIn("plan_s", timings)                                     # kept from the `run` invocation
        self.assertIn("render_s", timings)
        self.assertIn("| plan_s |", report)


if __name__ == "__main__":
    unittest.main()
