import copy
import dataclasses
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src.previs import physics, qa
from src.previs.qa import QAResult, QualityGate, fact_questions
from src.previs.schema import ScenePlan
from src.previs.state import build_index
from src.previs.types import BlockingResult, CameraState
from src.segmentation.base import SegmentationResult

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
W, H = 128, 96
CAM = CameraState(position=(0.2, 0.78, 3.8), yaw_deg=115.8, pitch_deg=0.0, roll_deg=0.0, lens_mm=28.0,
                  vfov_deg=46.4, width=W, height=H, look_at=(2.0, 0.78, 2.93))


def quadruped_plan() -> ScenePlan:
    """The fixture with Jon replaced by a dog (Character.body == "quadruped")."""
    data = json.loads(FIXTURE.read_text())
    for ch in data["bible"]["characters"]:
        if ch["id"] == "ch_jon":
            ch.update(name="Biscuit", body="quadruped", height_m=0.5, wardrobe="red nylon collar",
                      appearance="medium-sized scruffy terrier mix, wiry tan and white coat, floppy ears")
    return ScenePlan.model_validate(data)
ID_MAP = {1: "ch_mara", 2: "ch_jon", 3: "pr_table", 4: "pr_glass"}
RECTS = {
    # entity -> (x0, y0, x1, y1) at the previous / current keyframe
    "kf_wide_1": {"ch_mara": (10, 10, 40, 90), "pr_table": (50, 50, 100, 80), "pr_glass": (60, 40, 66, 46)},
    "kf_wide_2": {"ch_mara": (10, 10, 40, 90), "pr_table": (50, 50, 100, 80), "pr_glass": (100, 40, 106, 46)},
}


def write_blocking(folder: Path, rects) -> BlockingResult:
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.zeros((H, W), dtype=np.uint8)
    values = {eid: v for v, eid in ID_MAP.items()}
    for eid, (x0, y0, x1, y1) in rects.items():
        ids[y0:y1, x0:x1] = values[eid]
    Image.fromarray(ids, mode="L").save(folder / "id.png")
    Image.new("RGB", (W, H), (150, 150, 150)).save(folder / "blocking.png")
    bboxes = {eid: (x0, y0, x1 - 1, y1 - 1) for eid, (x0, y0, x1, y1) in rects.items()}
    return BlockingResult(blocking_path=str(folder / "blocking.png"), id_path=str(folder / "id.png"),
                          depth_path=str(folder / "depth.npy"), id_map=dict(ID_MAP), bboxes=bboxes,
                          visible_frac={eid: 1.0 for eid in rects})


class _QACase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = ScenePlan.model_validate_json(FIXTURE.read_text())
        cls.index = build_index(cls.plan)
        cls.solved = physics.solve(cls.plan)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prev_blocking = write_blocking(self.tmp / "graybox" / "kf_wide_1", RECTS["kf_wide_1"])
        self.blocking = write_blocking(self.tmp / "graybox" / "kf_wide_2", RECTS["kf_wide_2"])
        rng = np.random.default_rng(1)
        self.prev = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        self.prev_path = self.save(self.prev, "prev.png")
        self.frame = self.solved.frame("kf_wide_2")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save(self, arr, name) -> str:
        path = self.tmp / name
        Image.fromarray(arr).save(path)
        return str(path)

    def check(self, gate, out, plan=None, index=None, **kw):
        args = dict(sheet_paths={}, prev_path=self.prev_path, lock_mask_path=None, original_for_lock=None)
        args.update(kw)
        return gate.check(out, self.frame, self.blocking, plan or self.plan, index or self.index, args["sheet_paths"],
                          args["prev_path"], args["lock_mask_path"], args["original_for_lock"],
                          **{k: kw[k] for k in ("prev_blocking", "prev_frame") if k in kw})


class TestCopyAndLock(_QACase):
    def test_copied_frame_fails_for_moved_entity_only(self):
        out = self.save(self.prev.copy(), "out.png")
        result = self.check(QualityGate(use_models=False), out, prev_blocking=self.prev_blocking)
        self.assertIsInstance(result, QAResult)
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["copy:pr_glass"])      # Mara and the table did not move
        self.assertEqual(result.scores["copy:pr_glass"], 0.0)

    def test_changed_pixels_pass(self):
        arr = self.prev.copy()
        arr[40:50, 100:110] = 255 - arr[40:50, 100:110]
        result = self.check(QualityGate(use_models=False), self.save(arr, "out.png"), prev_blocking=self.prev_blocking)
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.scores["copy:pr_glass"], 1.0)

    def test_previous_blocking_read_from_sibling_directory(self):
        out = self.save(self.prev.copy(), "out.png")
        result = self.check(QualityGate(use_models=False), out)
        self.assertEqual(result.failures, ["copy:pr_glass"])

    def _occlusion_case(self, cam_prev, cam_now):
        """Mara walks in front of the (unchanged) table; only Mara's pixels are regenerated."""
        rects_prev = {"pr_table": (20, 50, 110, 80), "ch_mara": (0, 10, 15, 90)}     # drawn in order: Mara in front
        rects_now = {"pr_table": (20, 50, 110, 80), "ch_mara": (20, 10, 75, 90)}
        prev_blocking = write_blocking(self.tmp / "graybox" / "kf_wide_1", rects_prev)
        self.blocking = write_blocking(self.tmp / "graybox" / "kf_wide_2", rects_now)
        # The table's id-mask centroid shifts by 27.5 px although the table did not move.
        out = self.prev.copy()
        x0, y0, x1, y1 = rects_now["ch_mara"]
        out[y0:y1, x0:x1] = 255 - out[y0:y1, x0:x1]
        prev_frame = copy.deepcopy(self.solved.frame("kf_wide_1"))
        self.frame = copy.deepcopy(self.solved.frame("kf_wide_2"))
        prev_frame.camera, self.frame.camera = cam_prev, cam_now
        return self.check(QualityGate(use_models=False), self.save(out, "out.png"), prev_blocking=prev_blocking,
                          prev_frame=prev_frame)

    def test_occluded_static_entity_is_not_a_copy(self):
        # Review: under a static camera, an unchanged table partly hidden by a mover failed copy:pr_table on every
        # attempt (its pixels are locked to the previous frame by design).
        result = self._occlusion_case(CAM, CAM)
        self.assertTrue(result.passed, result.failures)
        self.assertNotIn("copy:pr_table", result.scores)
        self.assertEqual(result.scores["copy:ch_mara"], 1.0)               # the mover is still checked

    def test_unchanged_entity_is_still_checked_when_the_camera_moves(self):
        # A pan moves every entity on screen: a copied frame must still fail for them.
        moved = dataclasses.replace(CAM, yaw_deg=CAM.yaw_deg + 10.0)
        result = self._occlusion_case(CAM, moved)
        self.assertIn("copy:pr_table", result.failures)

    def test_first_keyframe_of_shot_has_no_copy_check(self):
        out = self.save(self.prev.copy(), "out.png")
        self.frame = self.solved.frame("kf_wide_1")
        result = self.check(QualityGate(use_models=False), out)
        self.assertTrue(result.passed)
        self.assertEqual(result.scores, {})

    def test_lock_exact_outside_mask(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[32:64, 96:128] = 255
        mask_path = self.save(mask, "mask.png")
        arr = self.prev.copy()
        arr[40:50, 100:110] = 0                                   # inside the mask: allowed
        out = self.save(arr, "out.png")
        gate = QualityGate(use_models=False)
        result = self.check(gate, out, lock_mask_path=mask_path, original_for_lock=self.prev_path)
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.scores["lock"], 1.0)
        arr[5, 5, 0] ^= 1                                          # one bit outside the mask
        result = self.check(gate, self.save(arr, "out2.png"), lock_mask_path=mask_path,
                            original_for_lock=self.prev_path)
        self.assertIn("lock", result.failures)
        self.assertLess(result.scores["lock"], 1.0)

    def test_lock_size_mismatch_fails(self):
        small = self.save(np.zeros((H // 2, W // 2, 3), dtype=np.uint8), "small.png")
        mask = self.save(np.zeros((H, W), dtype=np.uint8), "mask.png")
        result = self.check(QualityGate(use_models=False), self.prev_path, lock_mask_path=mask,
                            original_for_lock=small, prev_path=None)
        self.assertEqual(result.failures, ["lock"])


class TestModelChecks(_QACase):
    def test_use_models_false_never_loads_models(self):
        sheet = self.save(np.full((64, 32, 3), 200, dtype=np.uint8), "sheet.png")
        with mock.patch("src.segmentation.clipseg_segmenter.CLIPSegSegmenter", side_effect=AssertionError):
            result = self.check(QualityGate(use_models=False), self.prev_path, sheet_paths={"ch_mara": sheet},
                                prev_path=None)
        self.assertTrue(result.passed)
        self.assertFalse([k for k in result.scores if k.startswith(("identity", "layout"))])

    def test_identity_and_layout_logic(self):
        sheet = self.save(np.full((64, 32, 3), 200, dtype=np.uint8), "sheet.png")
        gate = QualityGate(use_models=True, identity_threshold=0.6, layout_threshold=0.25)

        def fake_embed(images):
            # Crops of Mara (30x80 bbox + padding) vs the 32x64 sheet: orthogonal -> cosine 0.
            return np.array([[1.0, 0.0] if im.size == (32, 64) else [0.0, 1.0] for im in images])

        class FakeSegmenter:
            prompts = []

            def segment(self, image, prompt):
                self.prompts.append(prompt)
                mask = np.zeros((H, W), dtype=np.uint8)
                if prompt == "kitchen table":
                    mask[50:80, 50:100] = 255                     # exact match
                return SegmentationResult(target_mask=mask, confidence=1.0, bounding_box=(0, 0, 0, 0))

        seg = FakeSegmenter()
        with mock.patch.object(QualityGate, "embed", side_effect=fake_embed), \
                mock.patch.object(QualityGate, "segmenter", return_value=seg):
            result = self.check(gate, self.prev_path, sheet_paths={"ch_mara": sheet}, prev_path=None)
        self.assertEqual(result.scores["identity:ch_mara"], 0.0)
        self.assertIn("identity:ch_mara", result.failures)
        self.assertEqual(result.scores["layout:pr_table"], 1.0)
        self.assertEqual(result.scores["layout:ch_mara"], 0.0)
        self.assertIn("layout:ch_mara", result.failures)
        self.assertNotIn("layout:pr_glass", result.scores)        # too small on screen for a meaningful IoU
        self.assertIn("person wearing mustard yellow knit sweater and dark blue jeans", seg.prompts)

    def test_quadruped_skips_identity_and_segments_by_appearance(self):
        # E2E #5: a dog's crop was scored against a human-style sheet and segmented as "person wearing <collar>".
        plan = quadruped_plan()
        index = build_index(plan)
        self.blocking = write_blocking(self.tmp / "graybox" / "kf_wide_2",
                                       {"ch_mara": (10, 10, 40, 90), "ch_jon": (60, 50, 110, 80)})
        sheet = self.save(np.full((64, 32, 3), 200, dtype=np.uint8), "sheet.png")
        embedded = []

        def fake_embed(images):
            embedded.extend(im.size for im in images)
            return np.array([[1.0, 0.0] for _ in images])

        class FakeSegmenter:
            prompts = []

            def segment(self, image, prompt):
                self.prompts.append(prompt)
                return SegmentationResult(target_mask=np.zeros((H, W), dtype=np.uint8), confidence=1.0,
                                          bounding_box=(0, 0, 0, 0))

        seg = FakeSegmenter()
        with mock.patch.object(QualityGate, "embed", side_effect=fake_embed), \
                mock.patch.object(QualityGate, "segmenter", return_value=seg):
            result = self.check(QualityGate(use_models=True), self.prev_path, plan=plan, index=index,
                                sheet_paths={"ch_mara": sheet, "ch_jon": sheet}, prev_path=None)
        self.assertIn("identity:ch_mara", result.scores)
        self.assertNotIn("identity:ch_jon", result.scores)
        self.assertEqual(len(embedded), 2)                          # Mara's crop + the sheet; never the dog
        self.assertEqual(qa.segmentation_prompt(index, "ch_jon"), "medium-sized scruffy terrier mix")
        self.assertIn("medium-sized scruffy terrier mix", seg.prompts)
        self.assertFalse([p for p in seg.prompts if "collar" in p])

    def test_identity_needs_head_and_torso_in_view(self):
        # Camera owner: full-body visible_frac lets a character whose head and torso are hidden (cafe kf_cup_2:
        # Leo 0.30, Ana 0.33 key-part visible) be identity-checked on a crop of legs or a sliver, which fails
        # against the full-body sheet whatever the model painted. Gate identity on camera.visible_fractions.
        sheet = self.save(np.full((64, 32, 3), 200, dtype=np.uint8), "sheet.png")
        self.frame = dataclasses.replace(self.solved.frame("kf_wide_2"), camera=CAM)

        def fake_embed(images):
            return np.array([[1.0, 0.0] if im.size == (32, 64) else [0.0, 1.0] for im in images])

        class FakeSegmenter:
            def segment(self, image, prompt):
                return SegmentationResult(target_mask=np.zeros((H, W), dtype=np.uint8), confidence=1.0,
                                          bounding_box=(0, 0, 0, 0))

        def run(key_visible):
            with mock.patch.object(QualityGate, "embed", side_effect=fake_embed), \
                    mock.patch.object(QualityGate, "segmenter", return_value=FakeSegmenter()), \
                    mock.patch("src.previs.camera.visible_fractions", return_value={"ch_mara": key_visible}) as vis:
                result = self.check(QualityGate(use_models=True), self.prev_path, sheet_paths={"ch_mara": sheet},
                                    prev_path=None)
            return result, vis.call_args

        for hidden in (0.2, None):                               # mostly hidden / too small to measure
            result, call = run(hidden)
            self.assertNotIn("identity:ch_mara", result.scores, hidden)
            self.assertNotIn("identity:ch_mara", result.failures, hidden)
            self.assertIn("layout:ch_mara", result.scores)       # layout still gates her
            self.assertEqual(call.args[2], ["ch_mara"])           # humans only (the table is not rendered for this)
            self.assertTrue(call.kwargs["key_parts_only"])
        result, _ = run(0.9)
        self.assertEqual(result.scores["identity:ch_mara"], 0.0)
        self.assertIn("identity:ch_mara", result.failures)
        # Without a solved camera nothing is filtered (and no render is made).
        self.frame = self.solved.frame("kf_wide_2")
        with mock.patch("src.previs.camera.visible_fractions", side_effect=AssertionError):
            self.assertEqual(qa.identifiable_characters(self.frame, self.plan, self.index, ["ch_mara", "pr_table"]),
                             ["ch_mara", "pr_table"])

    def test_identifiable_characters_with_real_renders(self):
        from src.previs import camera
        solved = physics.solve(self.plan)
        camera.solve_cameras(self.plan, solved, self.index, 256, 144)
        frame = solved.frame("kf_wide_2")
        visible = ["ch_mara", "ch_jon", "pr_table"]
        self.assertEqual(qa.identifiable_characters(frame, self.plan, self.index, visible), visible)
        # Jon steps between the camera and Mara: her head and torso are hidden, her identity is not checked.
        frame = copy.deepcopy(frame)
        cam, mara, jon = np.array(frame.camera.position), frame.entities["ch_mara"], frame.entities["ch_jon"]
        x, z = cam[[0, 2]] + 0.45 * (np.array(mara.position)[[0, 2]] - cam[[0, 2]])
        jon.position, jon.support, jon.in_motion = (float(x), 0.0, float(z)), "ground", False
        self.assertEqual(qa.identifiable_characters(frame, self.plan, self.index, visible), ["ch_jon", "pr_table"])

    @unittest.skipUnless(os.environ.get("RUN_QA_MODEL_TESTS") == "1", "set RUN_QA_MODEL_TESTS=1 to load CLIPSeg")
    def test_real_clipseg_and_clip(self):
        gate = QualityGate(use_models=True)
        t0 = time.time()
        emb = gate.embed([Image.open(self.prev_path)])
        self.assertEqual(emb.shape[0], 1)
        self.assertAlmostEqual(float(np.linalg.norm(emb[0])), 1.0, places=4)
        result = self.check(gate, self.prev_path, sheet_paths={"ch_mara": self.prev_path}, prev_path=None)
        self.assertIn("layout:pr_table", result.scores)
        print(f"\n  CLIPSeg/CLIP QA in {time.time() - t0:.2f}s: {result.scores}")


class TestFactQuestions(_QACase):
    def test_questions_from_state(self):
        falling = self.solved.frame("kf_glass_2")
        qs = fact_questions(self.plan, self.index, falling, ["pr_glass", "pr_table"])
        self.assertIn("Is the glass of water in mid-air?", qs)
        self.assertIn("Is the glass of water roughly half full?", qs)
        self.assertFalse([q for q in qs if "kitchen table" in q])   # fixed props have no support question
        reaction = self.solved.frame("kf_jon_2")
        qs = fact_questions(self.plan, self.index, reaction, ["ch_jon", "pr_glass"])
        self.assertIn("Is the person wearing olive green flannel shirt and grey chinos visible in the image?", qs)
        self.assertIn("Does the person wearing olive green flannel shirt and grey chinos look surprised?", qs)
        self.assertIn("Is the glass of water shattered into pieces?", qs)
        self.assertIn("Have the contents of the glass of water spilled out?", qs)
        self.assertIn("Is the glass of water on the floor?", qs)
        self.assertFalse([q for q in qs if "empty" in q or "mustard" in q])
        start = self.solved.frame("kf_wide_1")
        qs = fact_questions(self.plan, self.index, start, list(start.entities))
        self.assertIn("Is the glass of water resting on the kitchen table?", qs)
        self.assertIn("Does the person wearing mustard yellow knit sweater and dark blue jeans look focused?", qs)
        self.assertLessEqual(len(qs), qa.MAX_FACT_QUESTIONS)
        for q in qs:
            self.assertTrue(q[0].isupper() and q.endswith("?"), q)

    def test_questions_never_use_bible_names(self):
        # Review: the judge sees only the image, so "Is Mara holding the glass?" cannot be answered.
        names = re.compile(r"\b(Mara|Jon|Biscuit)\b")
        frame = copy.deepcopy(self.solved.frame("kf_wide_1"))
        frame.entities["ch_mara"].holding_id = "pr_glass"
        for plan in (self.plan, quadruped_plan()):
            index = build_index(plan)
            for f in list(self.solved.frames) + [frame]:
                for q in fact_questions(plan, index, f, list(f.entities)):
                    self.assertIsNone(names.search(q), q)
        qs = fact_questions(self.plan, self.index, frame, list(frame.entities))
        self.assertIn("Is the person wearing mustard yellow knit sweater and dark blue jeans holding "
                      "the glass of water?", qs)

    def test_quadruped_questions(self):
        plan = quadruped_plan()
        index = build_index(plan)
        reaction = self.solved.frame("kf_jon_2")                     # the dog is "surprised"
        qs = fact_questions(plan, index, reaction, ["ch_jon"])
        self.assertIn("Is the animal (medium-sized scruffy terrier mix) visible in the image?", qs)
        self.assertFalse([q for q in qs if "look" in q])            # no human expression question
        self.assertEqual(qa.visual_label(index, "pr_glass"), "the glass of water")

    def test_visual_label_trims_dangling_words(self):
        data = json.loads(FIXTURE.read_text())
        data["bible"]["characters"][0]["wardrobe"] = "Crisp white shirt and black trousers with a black waist apron and a name tag"
        index = build_index(ScenePlan.model_validate(data))
        self.assertEqual(qa.visual_label(index, "ch_mara"),
                         "the person wearing crisp white shirt and black trousers with a black waist apron")
        self.assertEqual(qa.segmentation_prompt(index, "ch_mara"), "person wearing crisp white shirt and black trousers")


if __name__ == "__main__":
    unittest.main()
