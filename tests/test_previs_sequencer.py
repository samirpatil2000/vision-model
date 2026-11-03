import argparse
import contextlib
import dataclasses
import io
import json
import re
import shlex
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Tuple
from unittest import mock

import numpy as np
from PIL import Image

import benchmark_previs
import previs
from src import config
from src.config import GenerationDefaults, TIERS, Tier
from src.engine.base import EngineClient, GenerationRequest, GenerationResult
from src.previs import conditioning, physics, qa, validator
from src.previs.planner.fixture import DEFAULT_FIXTURE, FixturePlanner
from src.previs.schema import ScenePlan
from src.previs.sequencer import RunConfig, Sequencer, SequencerError, attempt_rank, bible_errors, quality_score
from src.previs.types import BlockingResult, SolvedPlan

REPO_ROOT = Path(__file__).resolve().parent.parent
# Small tiers keep the gray-box pass cheap; the fingerprint logic does not depend on the size.
MINI = Tier("mini", 256, 160, (256, 128), (192, 128))
SMALL = Tier("small", 320, 192, (320, 160), (256, 160))


class FakeEngine(EngineClient):
    """Returns the init image (or a flat frame) slightly perturbed, so QA and locking have real pixels."""

    def __init__(self):
        self.requests = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if request.init_image:
            img = Image.open(request.init_image).convert("RGB").resize((request.width, request.height))
        else:
            img = Image.new("RGB", (request.width, request.height), (128, 128, 128))
        arr = np.array(img).astype(np.int16)
        arr = np.clip(arr + self.offset(request), 0, 255).astype(np.uint8)
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(request.output_path)
        return GenerationResult(image_path=request.output_path, seconds=0.01, command=["fake"], engine_commit="fake")

    def offset(self, request: GenerationRequest) -> int:
        return (request.seed % 7) - 3


class NoisyEngine(FakeEngine):
    """Like FakeEngine, but every call shifts the pixels differently (a real engine is not bit-reproducible here)."""

    def offset(self, request: GenerationRequest) -> int:
        return 9 * (len(self.requests) % 5) - 18


def keyframe_ids_rendered(engine: FakeEngine):
    """Keyframes the engine was asked to render (asset requests excluded)."""
    return sorted({Path(r.output_path).parent.parent.name for r in engine.requests if "/keyframes/" in r.output_path})


def make_sequencer(run_dir: Path, tier: Tier = MINI, engine=None, retakes: int = 0) -> Sequencer:
    cfg = RunConfig(run_dir=run_dir, tier=tier, defaults=GenerationDefaults(max_retakes=retakes), use_qa_models=False)
    return Sequencer(cfg, planner=FixturePlanner(), engine=engine or FakeEngine())


_TEMPLATES: Dict[str, Tuple[Path, SolvedPlan, Dict[str, BlockingResult]]] = {}
_TEMPLATE_ROOT = []


def tearDownModule():
    for root in _TEMPLATE_ROOT:
        shutil.rmtree(root, ignore_errors=True)


def graybox_template(tier: Tier):
    """The fixture plan solved and blocked once per tier (blocking dominates the test time)."""
    if tier.name not in _TEMPLATES:
        if not _TEMPLATE_ROOT:
            _TEMPLATE_ROOT.append(Path(tempfile.mkdtemp(prefix="previs_graybox_")))
        root = _TEMPLATE_ROOT[0] / tier.name
        seq = make_sequencer(root, tier)
        with contextlib.redirect_stdout(io.StringIO()):
            solved = seq.solve(seq.plan("fixture scene"))
            blocking = seq.block(seq._plan, solved, graybox_animatic=False)
        _TEMPLATES[tier.name] = (root, solved, blocking)
    return _TEMPLATES[tier.name]


def render_run(seq: Sequencer):
    """
    The render stage on seq.run_dir, quietly: the fixture plan.json (first time only) and the tier's gray-box
    are copied in, as `plan` + `block` would write them. Edits to plan.json that do not move anything
    (actions, wardrobe) keep the cached solve valid. Returns the accepted frames.
    """
    root, solved, blocking = graybox_template(seq.cfg.tier)
    if not (seq.run_dir / "plan.json").exists():
        shutil.copyfile(root / "plan.json", seq.run_dir / "plan.json")
    graybox = seq.run_dir / "graybox"
    shutil.copytree(root / "graybox", graybox, dirs_exist_ok=True)
    local = {kf: dataclasses.replace(b, blocking_path=str(graybox / kf / Path(b.blocking_path).name),
                                     id_path=str(graybox / kf / Path(b.id_path).name),
                                     depth_path=str(graybox / kf / Path(b.depth_path).name))
             for kf, b in blocking.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        return seq.render(seq.load_plan(), solved, local)


class TestSequencerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fixture_run_produces_frames_storyboard_and_animatic(self):
        cfg = RunConfig(run_dir=self.tmp / "run", tier=TIERS["draft"],
                        defaults=GenerationDefaults(max_retakes=0), use_qa_models=False)
        engine = FakeEngine()
        seq = Sequencer(cfg, planner=FixturePlanner(), engine=engine)

        plan = seq.plan("fixture scene")
        solved = seq.solve(plan)
        self.assertLessEqual(solved.total_duration_s, 30.0)
        blocking = seq.block(seq._plan, solved, graybox_animatic=False)
        self.assertEqual(set(blocking), {f.kf_id for f in solved.frames})

        frames = seq.render(seq._plan, solved, blocking)
        self.assertEqual(set(frames), {f.kf_id for f in solved.frames})
        for path in frames.values():
            self.assertTrue(Path(path).exists())

        # Every keyframe request carries the cast sheet first and stays within the ref budget.
        keyframe_requests = [r for r in engine.requests if r.output_path.endswith(".png") and "/keyframes/" in r.output_path]
        self.assertEqual(len(keyframe_requests), len(solved.frames))
        for req in keyframe_requests:
            self.assertLessEqual(len(req.ref_images), 3)
            self.assertLessEqual(len(req.prompt.split()), 120)

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not installed")
        out = seq.assemble(seq._plan, solved, frames)
        self.assertTrue(Path(out["animatic"]).exists())
        self.assertTrue((cfg.run_dir / "storyboard" / "storyboard.pdf").exists())
        validation = json.loads((cfg.run_dir / "validation.json").read_text())
        self.assertFalse([v for v in validation["violations"] if v["severity"] == "error"])


class TestResumeFingerprint(unittest.TestCase):
    """A frame on disk is reused only while its render fingerprint (tier, plan slice, seed, previous frame) matches."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = self.tmp / "run"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unchanged_rerun_reuses_every_frame(self):
        render_run(make_sequencer(self.run_dir))
        again = FakeEngine()
        render_run(make_sequencer(self.run_dir, engine=again))
        self.assertEqual(keyframe_ids_rendered(again), [])

    def test_frames_without_a_fingerprint_are_rerendered(self):
        render_run(make_sequencer(self.run_dir))
        (self.run_dir / "frames" / "manifest.json").unlink(missing_ok=True)   # run from before fingerprints existed
        again = FakeEngine()
        render_run(make_sequencer(self.run_dir, engine=again))
        self.assertEqual(len(keyframe_ids_rendered(again)), 7)

    def test_tier_change_rerenders_frames_at_the_new_size(self):
        render_run(make_sequencer(self.run_dir, MINI))
        bigger = FakeEngine()
        frames = render_run(make_sequencer(self.run_dir, SMALL, engine=bigger))
        self.assertEqual(len(keyframe_ids_rendered(bigger)), len(frames))
        for path in frames.values():
            with Image.open(path) as img:
                self.assertEqual(img.size, (SMALL.width, SMALL.height))
        manifest = json.loads((self.run_dir / "frames" / "manifest.json").read_text())
        self.assertEqual({e["tier"] for e in manifest.values()}, {"small:320x192"})

    def test_replanned_keyframe_and_later_frames_of_its_shot_rerender(self):
        render_run(make_sequencer(self.run_dir))
        plan_path = self.run_dir / "plan.json"
        data = json.loads(plan_path.read_text())
        for shot in data["timeline"]["shots"]:
            for kf in shot["keyframes"]:
                if kf["id"] == "kf_glass_2":
                    kf["action"] = "The glass hangs in mid-air above the tiles, spinning slowly"
        plan_path.write_text(json.dumps(data))
        engine = NoisyEngine()
        render_run(make_sequencer(self.run_dir, engine=engine))
        # kf_glass_3 is invalidated because the frame before it in the same shot was re-rendered.
        self.assertEqual(keyframe_ids_rendered(engine), ["kf_glass_2", "kf_glass_3"])

    def test_wardrobe_change_rerenders_frames_that_show_the_character(self):
        render_run(make_sequencer(self.run_dir))
        plan_path = self.run_dir / "plan.json"
        data = json.loads(plan_path.read_text())
        for ch in data["bible"]["characters"]:
            if ch["id"] == "ch_jon":
                ch["wardrobe"] = "a bright yellow raincoat and rubber boots"
        plan_path.write_text(json.dumps(data))
        engine = FakeEngine()
        render_run(make_sequencer(self.run_dir, engine=engine))
        rendered = keyframe_ids_rendered(engine)
        self.assertIn("kf_jon_1", rendered)
        self.assertIn("kf_jon_2", rendered)
        self.assertNotIn("kf_glass_1", rendered)

    def test_retake_rerenders_later_frames_of_the_shot_and_keeps_the_lock_exact(self):
        render_run(make_sequencer(self.run_dir, engine=NoisyEngine()))
        frames_dir = self.run_dir / "frames"
        (frames_dir / "kf_wide_1.png").unlink()                       # retake the shot's first keyframe
        engine = NoisyEngine()
        render_run(make_sequencer(self.run_dir, engine=engine))
        self.assertEqual(keyframe_ids_rendered(engine), ["kf_wide_1", "kf_wide_2"])

        request = json.loads((self.run_dir / "keyframes" / "kf_wide_2" / "attempt_0" / "request.json").read_text())
        if not request["static_lock"]:
            return                     # camera moved between the keyframes: no pixel lock to check
        first = np.asarray(Image.open(frames_dir / "kf_wide_1.png").convert("RGB"))
        second = np.asarray(Image.open(frames_dir / "kf_wide_2.png").convert("RGB"))
        outside = np.asarray(Image.open(request["mask_image"]).convert("L")) == 0
        self.assertTrue(np.array_equal(first[outside], second[outside]))

    def test_retake_ignores_stale_later_frames_as_continuity(self):
        render_run(make_sequencer(self.run_dir))
        attempt = self.run_dir / "keyframes" / "kf_wide_1" / "attempt_0" / "request.json"
        fresh_source = json.loads(attempt.read_text())["continuity_source"]
        # Approve kf_wide_2 (depth 0), so the old memory would offer it as the same-setup reference.
        memory_path = self.run_dir / "memory.json"
        saved = json.loads(memory_path.read_text())
        for record in saved["records"]:
            if record["kf_id"] == "kf_wide_2":
                record.update(approved=True, ancestry_depth=0)
        memory_path.write_text(json.dumps(saved))
        (self.run_dir / "frames" / "kf_wide_1.png").unlink()
        render_run(make_sequencer(self.run_dir))
        # Same reference as in a fresh run: never kf_wide_2, which comes after kf_wide_1.
        self.assertEqual(fresh_source, "plate")
        self.assertEqual(json.loads(attempt.read_text())["continuity_source"], fresh_source)
        records = json.loads(memory_path.read_text())["records"]
        self.assertEqual([r["kf_id"] for r in records][:2], ["kf_wide_1", "kf_wide_2"])
        self.assertTrue(records[1]["approved"])                 # the reused frame keeps its approval

    def test_retake_clears_attempts_of_the_superseded_render(self):
        failing = {"on": True}
        original_check = qa.QualityGate.check

        def check(gate, image_path, frame, *args, **kwargs):
            result = original_check(gate, image_path, frame, *args, **kwargs)
            if failing["on"] and frame.kf_id == "kf_jon_2":
                return qa.QAResult(passed=False, scores=dict(result.scores), failures=["identity:ch_jon"])
            return result

        with mock.patch.object(qa.QualityGate, "check", check):
            render_run(make_sequencer(self.run_dir, retakes=2))
            kf_dir = self.run_dir / "keyframes" / "kf_jon_2"
            self.assertEqual(sorted(p.name for p in kf_dir.iterdir()), ["attempt_0", "attempt_1", "attempt_2"])
            failing["on"] = False
            (self.run_dir / "frames" / "kf_jon_2.png").unlink()
            render_run(make_sequencer(self.run_dir, retakes=2))
        self.assertEqual(sorted(p.name for p in kf_dir.iterdir()), ["attempt_0"])
        records, final_qa = benchmark_previs.load_attempts(self.run_dir)
        self.assertEqual(len([r for r in records if r["kf_id"] == "kf_jon_2"]), 1)
        self.assertTrue(final_qa["kf_jon_2"]["passed"])


class TestRunConfigDefaults(unittest.TestCase):
    def test_assets_use_the_run_generation_defaults(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            engine = FakeEngine()
            cfg = RunConfig(run_dir=tmp / "run", tier=MINI, use_qa_models=False,
                            defaults=GenerationDefaults(max_retakes=0, steps=3, use_custom_sigmas=False))
            render_run(Sequencer(cfg, planner=FixturePlanner(), engine=engine))
            assets = [r for r in engine.requests if "/assets/" in r.output_path]
            self.assertTrue(assets)
            self.assertEqual({(r.steps, r.sigmas and tuple(r.sigmas)) for r in engine.requests}, {(3, None)})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestAttemptSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rank_ignores_copy_lock_and_fact_scores(self):
        diluted = qa.QAResult(False, {"identity:ch_a": 0.2, "lock": 1.0, "copy:pr_x": 1.0, "fact_accuracy": 1.0},
                              ["identity:ch_a"])
        better = qa.QAResult(False, {"identity:ch_a": 0.5, "layout:ch_a": 0.5, "copy:pr_x": 0.1}, ["copy:pr_x"])
        self.assertAlmostEqual(quality_score(diluted.scores), 0.2)
        self.assertAlmostEqual(quality_score(better.scores), 0.5)
        self.assertGreater(attempt_rank(better), attempt_rank(diluted))
        passed = qa.QAResult(True, {"identity:ch_a": 0.1}, [])
        self.assertGreater(attempt_rank(passed), attempt_rank(better))

    def test_best_failed_attempt_is_the_one_with_best_identity_and_layout(self):
        scripted = [  # kf_wide_1 attempts 0..2; all fail. Mean of ALL scores would pick attempt 0 (0.8).
            qa.QAResult(False, {"identity:ch_mara": 0.2, "lock": 1.0, "copy:pr_glass": 1.0, "copy:pr_table": 1.0},
                        ["identity:ch_mara"]),
            qa.QAResult(False, {"identity:ch_mara": 0.5, "layout:ch_mara": 0.5, "copy:pr_glass": 0.1},
                        ["copy:pr_glass"]),
            qa.QAResult(False, {"identity:ch_mara": 0.1}, ["identity:ch_mara"]),
        ]

        def check(gate, image_path, frame, *args, **kwargs):
            if frame.kf_id != "kf_wide_1":
                return qa.QAResult(True, {}, [])
            return scripted[int(re.search(r"attempt_(\d+)", str(image_path)).group(1))]

        run_dir = self.tmp / "run"
        with mock.patch.object(qa.QualityGate, "check", check):
            render_run(make_sequencer(run_dir, engine=NoisyEngine(), retakes=2))
        accepted = (run_dir / "frames" / "kf_wide_1.png").read_bytes()
        outputs = [(run_dir / "keyframes" / "kf_wide_1" / f"attempt_{n}" / "output.png").read_bytes() for n in range(3)]
        self.assertEqual(len(set(outputs)), 3)
        self.assertEqual(accepted, outputs[1])
        entry = json.loads((run_dir / "frames" / "manifest.json").read_text())["kf_wide_1"]
        self.assertEqual((entry["attempt"], entry["attempts"], entry["qa_passed"]), (1, 3, False))


class TestQaPreviousFrame(unittest.TestCase):
    def test_qa_reuses_the_solved_previous_frame(self):
        """The render stage hands QA the solved previous keyframe, so QA never re-solves physics and cameras."""
        tmp = Path(tempfile.mkdtemp())
        try:
            seen = []
            original_check = qa.QualityGate.check

            def check(gate, image_path, frame, *args, **kwargs):
                seen.append((frame.kf_id, kwargs.get("prev_frame")))
                return original_check(gate, image_path, frame, *args, **kwargs)

            with mock.patch.object(qa.QualityGate, "check", check), \
                    mock.patch.object(physics, "solve", wraps=physics.solve) as solve:
                render_run(make_sequencer(tmp / "run"))
            self.assertEqual(solve.call_count, 0)
            previous = {"kf_wide_2": "kf_wide_1", "kf_glass_2": "kf_glass_1", "kf_glass_3": "kf_glass_2",
                        "kf_jon_2": "kf_jon_1"}
            self.assertEqual({kf: prev.kf_id for kf, prev in seen if prev is not None}, previous)
            self.assertTrue(all(prev.camera is not None for _kf, prev in seen if prev is not None))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def fixture_data() -> dict:
    return json.loads(DEFAULT_FIXTURE.read_text())


class ScriptedPlanner(FixturePlanner):
    """Plans `data`; each repair returns `fixed` (or the plan unchanged) and is recorded."""

    def __init__(self, data: dict, fixed: dict = None):
        super().__init__()
        self.data, self.fixed, self.repairs = data, fixed, []

    def plan(self, scene_text, work_dir):
        return ScenePlan.model_validate(self.data)

    def repair(self, plan, violations_text, shot_ids, work_dir):
        self.repairs.append(shot_ids)
        return ScenePlan.model_validate(self.fixed) if self.fixed is not None else plan


class TestRepairTriage(unittest.TestCase):
    """Repairs replace shot timelines only: bible defects fail fast, and an unchanged repair stops the loop."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def solve(self, planner: ScriptedPlanner, max_repairs: int = 2):
        seq = Sequencer(RunConfig(run_dir=self.tmp / "cafe", tier=MINI, max_repairs=max_repairs), planner=planner)
        with contextlib.redirect_stdout(io.StringIO()):
            return seq.solve(seq.plan("a scene"))

    @staticmethod
    def errors_of(data: dict):
        plan = ScenePlan.model_validate(data)
        return plan, [v for v in validator.validate(plan) if v.severity == "error"]

    def test_bible_errors_are_the_ones_without_a_timeline_cause(self):
        bible_cases = {
            "lens": lambda d: d["bible"]["shots"][0].update(lens_mm=5.0),
            "mark": lambda d: d["bible"]["marks"][0].update(x_m=99.0),
            "height": lambda d: d["bible"]["characters"][0].update(height_m=3.5),
            "line_side": lambda d: d["bible"]["shots"][2].update(line_side="b"),
        }
        for name, edit in bible_cases.items():
            data = fixture_data()
            edit(data)
            plan, errors = self.errors_of(data)
            with self.subTest(case=name):
                found = bible_errors(plan, errors)
                self.assertTrue(found)
                self.assertTrue({v.code for v in found} <= {"ranges", "line_side"})
        timeline_cases = {   # same codes as bible defects (ref_integrity, ranges), but a repair can fix them
            "cause": lambda d: d["timeline"]["shots"][1]["events"][0].update(cause_event_id="ev_nope"),
            "progress": lambda d: d["timeline"]["shots"][0]["keyframes"][0].update(move_progress=1.5),
        }
        for name, edit in timeline_cases.items():
            data = fixture_data()
            edit(data)
            plan, errors = self.errors_of(data)
            with self.subTest(case=name):
                self.assertTrue(errors)
                self.assertEqual(bible_errors(plan, errors), [])

    def test_bible_error_fails_fast_with_a_replan_hint(self):
        data = fixture_data()
        data["bible"]["shots"][2]["line_side"] = "b"          # 180-degree rule broken in the bible
        planner = ScriptedPlanner(data, fixed=fixture_data())
        with self.assertRaises(SequencerError) as ctx:
            self.solve(planner)
        self.assertEqual(planner.repairs, [])                  # no repair round spent on it
        message = str(ctx.exception)
        self.assertIn("scene bible", message)
        self.assertIn("line_side", message)
        self.assertIn("python previs.py plan --run-name cafe --scene-file", message)

    def test_timeline_error_is_still_repaired(self):
        data = fixture_data()
        data["timeline"]["shots"][1]["events"][0]["cause_event_id"] = "ev_nope"
        planner = ScriptedPlanner(data, fixed=fixture_data())
        solved = self.solve(planner)
        self.assertEqual(len(planner.repairs), 1)
        self.assertEqual(len(solved.frames), 7)

    def test_unchanged_repair_stops_after_one_round(self):
        data = fixture_data()
        data["timeline"]["shots"][1]["events"][0]["cause_event_id"] = "ev_nope"
        planner = ScriptedPlanner(data)                        # repair returns the plan as it was
        with self.assertRaises(SequencerError) as ctx:
            self.solve(planner, max_repairs=3)
        self.assertEqual(len(planner.repairs), 1)
        self.assertIn("unchanged", str(ctx.exception))


class TestFactCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def render_with_answers(self, answer_for):
        """Render with a fake judge; answer_for(question_index) gives each answer."""
        judge = mock.Mock()
        judge.return_value.facts.side_effect = lambda path, questions, work_dir: [answer_for(i) for i in range(len(questions))]
        seq = make_sequencer(self.tmp / "run")
        seq.cfg.fact_check = True
        with mock.patch("src.previs.planner.claude_cli.ClaudeCliJudge", judge):
            render_run(seq)
        keyframes = self.tmp / "run" / "keyframes"
        qa_records = {p.parent.parent.name: json.loads(p.read_text()) for p in keyframes.glob("*/attempt_0/qa.json")}
        facts = {p.parent.parent.name: json.loads(p.read_text()) for p in keyframes.glob("*/attempt_0/facts.json")}
        return qa_records, facts

    def test_unsure_answers_do_not_count_against_the_frame(self):
        qa_records, facts = self.render_with_answers(lambda i: "yes" if i == 0 else "unsure")
        self.assertTrue(any(len(f["questions"]) >= 2 for f in facts.values()))   # old rule: 1/n < 0.75
        for kf_id in facts:
            self.assertTrue(qa_records[kf_id]["passed"], kf_id)
            self.assertEqual(qa_records[kf_id]["scores"]["fact_accuracy"], 1.0)

    def test_all_unsure_answers_leave_facts_ungated(self):
        qa_records, facts = self.render_with_answers(lambda i: "unsure")
        self.assertTrue(facts)
        for kf_id in facts:
            record = qa_records[kf_id]
            self.assertTrue(record["passed"], kf_id)
            self.assertNotIn("fact_accuracy", record["scores"])
            self.assertNotIn("facts", record["failures"])
            self.assertTrue(record["facts_undecided"])

    def test_decided_no_answers_still_fail_the_frame(self):
        qa_records, facts = self.render_with_answers(lambda i: "no" if i == 0 else "unsure")
        self.assertTrue(facts)
        for kf_id in facts:
            self.assertFalse(qa_records[kf_id]["passed"], kf_id)
            self.assertIn("facts", qa_records[kf_id]["failures"])

    def test_judge_failure_does_not_abort_the_render(self):
        judge = mock.Mock()
        judge.return_value.facts.side_effect = RuntimeError("claude CLI timed out")
        seq = make_sequencer(self.tmp / "run")
        seq.cfg.fact_check = True
        with mock.patch("src.previs.planner.claude_cli.ClaudeCliJudge", judge):
            frames = render_run(seq)
        self.assertEqual(len(frames), 7)
        self.assertTrue(judge.return_value.facts.called)
        qa_files = list((self.tmp / "run" / "keyframes").glob("*/attempt_0/qa.json"))
        errors = [json.loads(p.read_text()).get("facts_error") for p in qa_files]
        self.assertTrue(any(e and "timed out" in e for e in errors))


class TestRunManifest(unittest.TestCase):
    def test_stage_timings_accumulate_across_invocations(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            first = make_sequencer(tmp)
            first.timings = {"plan_s": 120.0, "block_s": 2.0}
            first.write_run_manifest()
            second = make_sequencer(tmp)
            second.timings = {"block_s": 3.0, "render_s": 40.0}
            second.write_run_manifest()
            timings = json.loads((tmp / "run.json").read_text())["timings"]
            self.assertEqual(timings, {"plan_s": 120.0, "block_s": 3.0, "render_s": 40.0})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPrevisCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = mock.patch.object(config, "PREVIS_OUTPUT_DIR", self.tmp)
        self.out.start()
        self.engine = mock.patch.object(previs, "_engine", lambda args: FakeEngine())
        self.engine.start()
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}

    def tearDown(self):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
        self.engine.stop()
        self.out.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            previs.main(list(argv))
        return buf.getvalue()

    def test_plan_without_scene_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            previs.main(["plan"])
        self.assertIn("--scene", str(ctx.exception.code))

    def test_block_honours_skip_graybox(self):
        with contextlib.redirect_stdout(io.StringIO()):
            make_sequencer(self.tmp / "kitchen").plan("fixture scene")
        with mock.patch.object(Sequencer, "block") as block:
            self.run_cli("block", "--run-name", "kitchen", "--planner", "fixture", "--skip-graybox")
        self.assertIs(block.call_args.kwargs.get("graybox_animatic"), False)

    def test_allow_engine_mismatch_reaches_the_engine_client(self):
        self.engine.stop()
        try:
            for engine in ("cli", "server"):
                args = argparse.Namespace(allow_engine_mismatch=True, engine=engine, run_name="kitchen")
                with mock.patch("src.engine.version.verify_engine") as verify, \
                        mock.patch("src.engine.make_engine") as make_engine:
                    previs._engine(args)
                self.assertTrue(verify.call_args.kwargs["allow_mismatch"])
                self.assertEqual(make_engine.call_args.args, (engine,))
                # The clients verify the build again when they start; without this they refuse it.
                self.assertIs(make_engine.call_args.kwargs.get("allow_mismatch"), True)
        finally:
            self.engine.start()

    def test_bible_error_exits_with_the_replan_command(self):
        run_dir = self.tmp / "cafe"
        run_dir.mkdir()
        data = fixture_data()
        data["bible"]["shots"][0]["lens_mm"] = 5.0
        (run_dir / "plan.json").write_text(json.dumps(data))
        (run_dir / "scene.txt").write_text("a cafe scene")
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli("solve", "--run-name", "cafe", "--planner", "fixture")
        message = str(ctx.exception.code)
        self.assertTrue(message.startswith("error: Plan has 1 error(s) in the scene bible"), message)
        self.assertIn(f"python previs.py plan --run-name cafe --scene-file {run_dir / 'scene.txt'}", message)

    def test_engine_precheck_verifies_the_binary_the_run_uses(self):
        self.engine.stop()
        try:
            for engine, binary in (("cli", config.SD_CLI), ("server", config.SD_SERVER)):
                args = argparse.Namespace(allow_engine_mismatch=False, engine=engine, run_name="kitchen")
                with mock.patch("src.engine.version.verify_engine") as verify, mock.patch("src.engine.make_engine"):
                    previs._engine(args)
                with self.subTest(engine=engine):
                    self.assertEqual(Path(verify.call_args.args[0]), Path(binary))
        finally:
            self.engine.start()

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "no SIGHUP on this platform")
    def test_sighup_stops_cleanly_unless_ignored(self):
        parser = mock.Mock()
        parser.parse_args.return_value = argparse.Namespace(command="probe", func=lambda args: None)
        with mock.patch.object(previs, "build_parser", return_value=parser):
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            previs.main([])
            handler = signal.getsignal(signal.SIGHUP)
            self.assertTrue(callable(handler))
            with self.assertRaises(SystemExit):
                handler(signal.SIGHUP, None)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)                   # nohup
            previs.main([])
            self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)


class TestReadmePrevisDocs(unittest.TestCase):
    def setUp(self):
        self.readme = (REPO_ROOT / "README.md").read_text()

    def test_init_noise_default_and_retake_step_match_the_code(self):
        row = next(line for line in self.readme.splitlines() if line.startswith("| `INIT_NOISE`"))
        documented = float(re.search(r"\| `([0-9.]+)` \|", row).group(1))
        code_default = re.search(r'"INIT_NOISE", "([0-9.]+)"', (REPO_ROOT / "src" / "config.py").read_text()).group(1)
        self.assertEqual(documented, float(code_default))
        self.assertIn(f"+{conditioning.NOISE_STEP_PER_ATTEMPT:g} per retake", row)
        self.assertIn(f"capped at {conditioning.NOISE_CAP:g}", row)

    def test_body_seat_and_repair_rules_match_the_code(self):
        from typing import get_args
        from src.previs.schema import Body
        from src.previs.sequencer import STALE_REASONS
        text = " ".join(self.readme.split())
        rules = text[text.index("**Animals and seats.**"):text.index("**Repairs.**")]
        self.assertIn("Character.body", rules)
        for body in get_args(Body):
            self.assertIn(f'`"{body}"`', rules)
        seat = float(re.search(r"entity at most ([0-9.]+) m tall", rules).group(1))
        self.assertEqual(seat, physics.SEAT_MAX_M)
        listed = set(re.search(r"every pose except ((?:`\w+`(?:, | and )?)+)", rules).group(1).replace("`", "")
                     .replace(" and ", ", ").split(", "))
        self.assertEqual(listed, set(physics.ON_TOP_POSES) - {"none"})
        repairs = text[text.index("**Repairs.**"):]
        self.assertIn(f"at most {RunConfig.__dataclass_fields__['max_repairs'].default} repair rounds", repairs)
        self.assertIn(f"re-rendering ({STALE_REASONS['plan']})", text)

    def test_every_documented_previs_command_parses(self):
        text = self.readme.replace("\\\n", " ")
        commands = [line.split("#")[0].strip() for line in text.splitlines()
                    if line.strip().startswith(("python previs.py", "python benchmark_previs.py"))]
        self.assertGreaterEqual(len(commands), 6)
        parser = previs.build_parser()
        for command in commands:
            argv = shlex.split(command.replace("<run>", "kitchen"))
            with self.subTest(command=command):
                if argv[1] == "benchmark_previs.py":
                    self.assertEqual(argv[2], "--run-dir")
                    continue
                args = parser.parse_args(argv[2:])
                if args.command in ("render", "assemble", "block", "solve"):
                    self.assertEqual(args.run_name, "kitchen")


if __name__ == "__main__":
    unittest.main()


class TestCastReferences(unittest.TestCase):
    def test_match_by_name_or_id_fragment(self):
        from src.previs.schema import Character
        from src.previs.sequencer import match_cast_ref
        ch = Character(id="ch_co_commander", name="Co-Commander Rustam", body="human", height_m=1.8,
                       appearance="a", wardrobe="w", palette_hex=["#000000", "#ffffff"])
        self.assertEqual(match_cast_ref(ch, {"co-commander": "/a.png"}), "/a.png")
        self.assertEqual(match_cast_ref(ch, {"Rustam": "/b.png"}), "/b.png")
        self.assertIsNone(match_cast_ref(ch, {"fighter": "/c.png"}))

    def test_user_reference_replaces_generated_sheet(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "hero.png"
            Image.new("RGB", (300, 600), (200, 50, 50)).save(src)
            seq = make_sequencer(tmp / "run")
            seq.cfg.cast_refs = {"mara": str(src)}
            engine = seq.engine
            render_run(seq)
            asset_requests = [r.output_path for r in engine.requests if "/assets/cast/" in r.output_path]
            self.assertFalse([p for p in asset_requests if "ch_mara_full" in p])     # not generated
            self.assertTrue((tmp / "run" / "assets" / "cast" / "ch_mara_full_ref.png").is_file())
            self.assertTrue([p for p in asset_requests if "ch_jon_full" in p])       # others still generated
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
