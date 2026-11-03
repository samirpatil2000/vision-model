import copy
import dataclasses
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src import config
from src.config import TIERS, GenerationDefaults
from src.engine.base import EngineClient, GenerationRequest, GenerationResult
from src.previs import conditioning, physics, seeds
from src.previs.assets import AssetBank
from src.previs.memory import KeyframeMemory, MemoryRecord
from src.previs.schema import ScenePlan
from src.previs.state import build_index
from src.previs.types import BlockingResult, CameraState

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
TIER = TIERS["draft"]
# A fixed camera for the fixture's static wide shot (what a solver that pins static shots would give).
STATIC_CAM = CameraState(position=(0.2, 0.78, 3.8), yaw_deg=115.8, pitch_deg=0.0, roll_deg=0.0, lens_mm=28.0,
                         vfov_deg=46.4, width=TIER.width, height=TIER.height, look_at=(2.0, 0.78, 2.93))
# Human-only wording that must never describe a quadruped.
HUMAN_WORDS = re.compile(r"expression|smil|scowl|wide-eyed|wincing|kneel|\bhands?\b|\barms?\b|person|wearing")


def image_size(path):
    with Image.open(path) as img:
        return img.size


def load_fixture() -> ScenePlan:
    return ScenePlan.model_validate_json(FIXTURE.read_text())


def quadruped_fixture() -> ScenePlan:
    """The fixture with Jon replaced by a dog (Character.body == "quadruped")."""
    data = json.loads(FIXTURE.read_text())
    for ch in data["bible"]["characters"]:
        if ch["id"] == "ch_jon":
            ch.update(name="Biscuit", body="quadruped", height_m=0.5, wardrobe="red nylon collar",
                      appearance="medium-sized scruffy terrier mix, wiry tan and white coat, floppy ears, dark eyes")
    return ScenePlan.model_validate(data)


def with_event_kind(index, event_id: str, kind: str, **update):
    """A copy of `index` whose event `event_id` has another kind (the solved frames are reused as they are)."""
    events = dict(index.events)
    events[event_id] = events[event_id].model_copy(update={"kind": kind, **update})
    return dataclasses.replace(index, events=events)


class FakeEngine(EngineClient):
    """Writes a flat image whose colour depends on the seed; records every request."""

    def __init__(self):
        self.requests = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (request.width, request.height), (190, 190, 190))
        shade = 40 + request.seed % 80
        w, h = request.width, request.height
        img.paste((shade, shade // 2, 90), (w // 3, h // 8, 2 * w // 3, 7 * h // 8))
        img.save(request.output_path)
        return GenerationResult(image_path=request.output_path, seconds=0.0, command=["fake"], engine_commit="fake")


def synthetic_blocking(frame, index, out_dir: Path, size=(TIER.width, TIER.height)) -> BlockingResult:
    """A pseudo-perspective box render: one rectangle per present entity, far to near, with an id map."""
    w, h = size
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.full((h, w, 3), 170, dtype=np.uint8)
    ids = np.zeros((h, w), dtype=np.uint8)
    id_map = {i + 1: eid for i, eid in enumerate(index.entities)}
    order = sorted(frame.entities.values(), key=lambda ef: -ef.position[2])
    for ef in order:
        if not ef.present:
            continue
        x, y, z = ef.position
        sx, sy, _ = ef.size
        bottom = h * 0.95 - z * h * 0.15 - y * h * 0.3
        top = bottom - max(sy * h * 0.3, 6)
        half = max(sx / 5.0 * w / 2, 3)
        x0, x1 = int(max(0, x / 5.0 * w - half)), int(min(w, x / 5.0 * w + half))
        y0, y1 = int(max(0, top)), int(min(h, bottom))
        value = next(v for v, eid in id_map.items() if eid == ef.entity_id)
        ids[y0:y1, x0:x1] = value
        colour = tuple(int(ef.color_hex[i:i + 2], 16) for i in (1, 3, 5))
        rgb[y0:y1, x0:x1] = colour
    Image.fromarray(rgb).save(out_dir / "blocking.png")
    Image.fromarray(ids, mode="L").save(out_dir / "id.png")
    np.save(out_dir / "depth.npy", np.zeros((h, w), dtype=np.float32))
    bboxes, visible = {}, {}
    for value, eid in id_map.items():
        box = conditioning.bbox_of(ids == value)
        if box is not None:
            bboxes[eid] = box
            visible[eid] = 1.0
    return BlockingResult(blocking_path=str(out_dir / "blocking.png"), id_path=str(out_dir / "id.png"),
                          depth_path=str(out_dir / "depth.npy"), id_map=id_map, bboxes=bboxes, visible_frac=visible)


class TestSeeds(unittest.TestCase):
    def test_deterministic_and_in_range(self):
        a = seeds.seed_for("proj", "sh_wide", "kf_wide_1", 0)
        self.assertEqual(a, seeds.seed_for("proj", "sh_wide", "kf_wide_1", 0))
        self.assertTrue(0 <= a < 2 ** 31)
        others = {seeds.seed_for("proj", "sh_wide", "kf_wide_1", n) for n in range(1, 5)}
        self.assertNotIn(a, others)
        self.assertNotEqual(a, seeds.seed_for("other", "sh_wide", "kf_wide_1", 0))


class TestPromptCompiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = load_fixture()
        cls.index = build_index(cls.plan)
        cls.solved = physics.solve(cls.plan)

    def _prompt(self, frame, **kw):
        shot = self.index.shots[frame.shot_id]
        kf = self.index.keyframe_by_id[frame.kf_id]
        return conditioning.compile_prompt(self.plan, self.index, frame, shot, kf, list(frame.entities), **kw)

    def test_physics_phrase_table_has_no_numbers_or_causal_clauses(self):
        banned = re.compile(r"\d|about to|because|caus|due to|therefore|as a result|so that|\bm/s\b")
        for group, table in conditioning.PHYSICS_PHRASES.items():
            for key, phrase in table.items():
                self.assertIsNone(banned.search(phrase), f"{group}.{key}: {phrase}")

    def test_phrase_table_covers_every_enum_value(self):
        from typing import get_args
        from src.previs.schema import DepictedPhase, Expression, Flag, Pose, Secondary, WeightCue
        for group, enum in (("phase", DepictedPhase), ("expression", Expression), ("pose", Pose),
                            ("secondary", Secondary), ("weight_cue", WeightCue), ("flags", Flag),
                            ("pose_quadruped", Pose), ("expression_quadruped", Expression)):
            self.assertEqual(set(get_args(enum)), set(conditioning.PHYSICS_PHRASES[group]), group)

    def test_quadruped_tables_have_no_human_words(self):
        for group in ("pose_quadruped", "expression_quadruped", "weight_cue"):
            for key, phrase in conditioning.PHYSICS_PHRASES[group].items():
                self.assertIsNone(HUMAN_WORDS.search(phrase), f"{group}.{key}: {phrase}")

    def test_quadruped_is_described_as_an_animal(self):
        # E2E #5: the dog was described with human pose/expression words, so Qwen painted a person.
        plan = quadruped_fixture()
        index = build_index(plan)
        frame = copy.deepcopy(physics.solve(plan).frame("kf_jon_2"))     # Jon (now the dog) is surprised
        shot, kf = index.shots[frame.shot_id], index.keyframe_by_id[frame.kf_id]
        dog = frame.entities["ch_jon"]
        for pose in ("stand", "sit", "kneel", "lie", "walk", "run", "fall", "crouch", "brace", "reach"):
            for expression in ("surprised", "happy", "pained", "neutral", "focused"):
                dog.pose, dog.expression = pose, expression
                sentence = conditioning._character_sentence(index, frame, "ch_jon", 12, 1, {}, TIER.width)
                self.assertTrue(sentence.startswith("Biscuit, a four-legged animal, medium-sized scruffy terrier mix"),
                                sentence)
                self.assertIsNone(HUMAN_WORDS.search(sentence), sentence)
        dog.pose, dog.expression = "sit", "surprised"
        prompt = conditioning.compile_prompt(plan, index, frame, shot, kf, ["ch_jon"], bboxes={},
                                             image_width=TIER.width)
        self.assertIn("as in the first image, sitting on its haunches, ears pricked up.", prompt)
        self.assertNotIn("startled", prompt)
        # Held objects are carried in the mouth, never "in hand".
        dog.pose, dog.holding_id = "carry", "pr_glass"
        clause = conditioning._pose_clause(index, dog)
        self.assertEqual(clause, "carrying the glass of water in its mouth")
        dog.pose = "stand"
        self.assertEqual(conditioning._pose_clause(index, dog), "standing on all four legs, holding the glass of water in its mouth")
        # Humans are unchanged.
        frame.entities["ch_mara"].expression = "surprised"
        self.assertIn("startled wide-eyed expression",
                      conditioning._character_sentence(index, frame, "ch_mara", 12, 1, {}, TIER.width))

    def test_every_fixture_keyframe_fits_120_words(self):
        bboxes = {"ch_mara": (40, 30, 150, 280), "ch_jon": (400, 20, 480, 250)}
        for frame in self.solved.frames:
            for continuity in (None, 2):
                prompt = self._prompt(frame, bboxes=bboxes, image_width=TIER.width, continuity_image=continuity)
                self.assertLessEqual(len(prompt.split()), 120, prompt)
                physics_words = sum(len(p.split()) for p in conditioning.physics_phrases(
                    self.index, frame, list(frame.entities)))
                self.assertLessEqual(physics_words, 25)

    def test_prompt_structure(self):
        frame = self.solved.frame("kf_wide_2")
        bboxes = {"ch_mara": (40, 30, 150, 280), "ch_jon": (400, 20, 480, 250)}
        prompt = self._prompt(frame, bboxes=bboxes, image_width=TIER.width, continuity_image=2)
        self.assertIn("Wide shot, eye-level angle, 28mm lens.", prompt)
        self.assertIn("as in the first image", prompt)
        self.assertIn("left third of frame", prompt)      # Mara
        self.assertIn("right third of frame", prompt)     # Jon
        self.assertIn("off balance", prompt)
        self.assertIn("instant of contact", prompt)
        self.assertIn("Same set and lighting as the second image.", prompt)
        self.assertTrue(prompt.startswith("Cinematic 35mm film still"))
        # Subject first.
        self.assertLess(prompt.index("Mara,"), prompt.index("Jon,"))
        without = self._prompt(frame, bboxes=bboxes, image_width=TIER.width)
        self.assertNotIn("Same set and lighting", without)

    def test_angle_phrase_follows_the_solved_camera(self):
        # The camera solver may lift a shot above its planned angle to see the subject (cafe kf_cup_1: a 'low'
        # insert filmed from pitch -15), clamp it above the floor, or tilt it. The prompt must describe the
        # init image's camera, not shot.angle.
        frame = copy.deepcopy(self.solved.frame("kf_glass_1"))
        shot = self.index.shots[frame.shot_id]
        self.assertEqual(shot.angle, "low")

        def phrase(pitch, height=1.0, roll=0.0):
            frame.camera = dataclasses.replace(STATIC_CAM, pitch_deg=pitch, roll_deg=roll,
                                               position=(0.2, height, 3.8))
            prompt = self._prompt(frame, bboxes={}, image_width=TIER.width)
            return re.search(r"Insert shot, (.*?), \d+mm lens\.", prompt).group(1)

        self.assertEqual(phrase(-15.0), "high angle looking down")          # lifted 30 deg above 'low'
        self.assertEqual(phrase(15.0, height=0.65), "low angle looking up")
        self.assertEqual(phrase(3.0, height=1.2), "eye-level angle")       # lifted 15 deg above 'low'
        self.assertEqual(phrase(-8.4, height=0.15), "ground-level angle")  # clamped above a glass on the floor
        self.assertEqual(phrase(-80.0, height=3.0), "overhead top-down angle")
        self.assertEqual(phrase(-55.0, height=3.0), "high angle looking down")
        self.assertEqual(phrase(0.0, roll=12.0), "dutch tilted angle")
        self.assertEqual(phrase(-25.0, roll=-12.0), "dutch tilted high angle looking down")
        self.assertEqual(phrase(0.5, roll=0.8), "eye-level angle")         # handheld shake is not dutch
        # Without a solved camera the planned angle is all there is.
        frame.camera = None
        self.assertIn("Insert shot, low angle looking up, 50mm lens.", self._prompt(frame, bboxes={}, image_width=TIER.width))

    def test_angle_phrase_agrees_with_the_real_camera_solver(self):
        from src.previs import camera
        solved = physics.solve(self.plan)
        camera.solve_cameras(self.plan, solved, self.index, TIER.width, TIER.height)
        for frame in solved.frames:
            prompt = self._prompt(frame, bboxes={}, image_width=TIER.width)
            cam = frame.camera
            if "low angle looking up" in prompt:
                self.assertGreaterEqual(cam.pitch_deg, conditioning.LOW_PITCH_DEG, frame.kf_id)
            if "high angle looking down" in prompt:
                self.assertLessEqual(cam.pitch_deg, conditioning.HIGH_PITCH_DEG, frame.kf_id)
            if "eye-level angle" in prompt:
                self.assertLess(abs(cam.pitch_deg), 10.0, frame.kf_id)
                self.assertGreaterEqual(cam.position[1], conditioning.GROUND_LEVEL_M, frame.kf_id)

    def test_physics_phrases_follow_state(self):
        falling = self.solved.frame("kf_glass_2")
        phrases = conditioning.physics_phrases(self.index, falling, ["pr_glass", "pr_table"])
        self.assertIn("the glass of water falling through the air", phrases)
        self.assertIn("a detached shadow on the floor below", phrases)
        impact = self.solved.frame("kf_glass_3")
        phrases = conditioning.physics_phrases(self.index, impact, ["pr_glass"])
        self.assertEqual(phrases[0], "the glass of water hitting the floor")
        self.assertIn("sharp shards scattered across the floor", phrases)
        prompt = self._prompt(impact, bboxes={}, image_width=TIER.width)
        self.assertIn("The glass of water is shattered into pieces and spilled.", prompt)
        self.assertNotIn("empty", prompt)    # a shattered container has no fill level
        # Invisible entities are not described.
        self.assertEqual(conditioning.physics_phrases(self.index, falling, ["pr_table"]), [])
        # Kind-specific phase phrase: a fall's release tips over the edge.
        release = self.solved.frame("kf_glass_1")
        self.assertEqual(conditioning.physics_phrases(self.index, release, ["pr_glass"])[0],
                         "the glass of water tipping over the edge")

    def test_holding_and_weight_cue_in_character_clause(self):
        import copy
        frame = copy.deepcopy(self.solved.frame("kf_wide_1"))
        mara = frame.entities["ch_mara"]
        mara.pose, mara.holding_id, mara.weight_cue = "carry", "pr_glass", "straining"
        prompt = self._prompt(frame, bboxes={}, image_width=TIER.width)
        self.assertIn("carrying the glass of water, straining with effort", prompt)
        mara.pose = "stand"
        self.assertIn("standing, holding the glass of water", self._prompt(frame, bboxes={}, image_width=TIER.width))

    def test_crowded_frame_is_compressed_not_overflowing(self):
        data = json.loads(FIXTURE.read_text())
        base = data["bible"]["characters"][0]
        for n in range(6):
            ch = dict(base, id=f"ch_extra_{n}", name=f"Extra{n}",
                      appearance="tall person with very long braided silver hair, freckles, round glasses, "
                                 "a scar across the left cheek and a weathered face")
            data["bible"]["characters"].append(ch)
            state = dict(data["bible"]["initial_states"][0], entity_id=ch["id"])
            data["bible"]["initial_states"].append(state)
        plan = ScenePlan.model_validate(data)
        index = build_index(plan)
        solved = physics.solve(plan)
        frame = solved.frame("kf_wide_1")
        prompt = conditioning.compile_prompt(plan, index, frame, index.shots["sh_wide"],
                                             index.keyframe_by_id["kf_wide_1"], list(frame.entities),
                                             continuity_image=2)
        self.assertLessEqual(len(prompt.split()), 120)
        self.assertIn("Mara,", prompt)    # the subject survives compression

    def test_crowded_frame_drops_the_least_prominent_characters_first(self):
        # Real cafe kf_cup_2 (insert of the falling cup): Leo, Ana (a sliver at the edge) and the dog (whole, mid-frame)
        # did not fit, and the dog was dropped only because it comes last in the plan, so nothing told the model
        # that the red four-legged proxy is an animal (E2E #5: it paints a person).
        data = json.loads(FIXTURE.read_text())
        base = data["bible"]["characters"][0]
        for n in range(6):
            ch = dict(base, id=f"ch_extra_{n}", name=f"Extra{n}",
                      appearance="tall person with very long braided silver hair, freckles, round glasses, "
                                 "a scar across the left cheek and a weathered face")
            data["bible"]["characters"].append(ch)
            data["bible"]["initial_states"].append(dict(data["bible"]["initial_states"][0], entity_id=ch["id"]))
        plan = ScenePlan.model_validate(data)
        index = build_index(plan)
        frame = physics.solve(plan).frame("kf_wide_1")
        shot = index.shots["sh_wide"]
        self.assertEqual((shot.subject_id, shot.secondary_id), ("ch_mara", "ch_jon"))
        # The last extra fills the frame; the others (and Jon, the secondary) are small.
        bboxes = {"ch_mara": (10, 10, 60, 200), "ch_jon": (470, 100, 480, 120), "ch_extra_5": (150, 20, 400, 280)}
        bboxes.update({f"ch_extra_{n}": (100 + 10 * n, 200, 102 + 11 * n, 210) for n in range(5)})   # Extra0 smallest
        prompt = conditioning.compile_prompt(plan, index, frame, shot, index.keyframe_by_id["kf_wide_1"],
                                             list(frame.entities), bboxes=bboxes, image_width=TIER.width,
                                             continuity_image=2)
        self.assertLessEqual(len(prompt.split()), 120)
        self.assertIn("Mara,", prompt)                         # the subject is never dropped
        self.assertIn("Extra5,", prompt)                       # the most prominent extra is kept ...
        self.assertIn("Jon,", prompt)                          # ... and so is the secondary
        self.assertNotIn("Extra0,", prompt)                    # tiny extras go first
        dropped = [n for n in range(5) if f"Extra{n}," not in prompt]
        self.assertTrue(dropped)
        # Without screen sizes the order is unchanged: the last in prompt order goes first.
        no_boxes = conditioning.compile_prompt(plan, index, frame, shot, index.keyframe_by_id["kf_wide_1"],
                                               list(frame.entities), bboxes={}, image_width=TIER.width,
                                               continuity_image=2)
        self.assertNotIn("Extra5,", no_boxes)
        self.assertIn("Mara,", no_boxes)

    def _phrases_at(self, index, frame, phase):
        frame = copy.deepcopy(frame)
        frame.depicted_phase = phase
        return conditioning.physics_phrases(index, frame, list(frame.entities))

    def test_held_mover_is_never_on_a_surface(self):
        # Review: a lifted/caught object at rest/aftermath/impact was described "at rest on the floor".
        index = with_event_kind(self.index, "ev_knock", "lift")
        frame = copy.deepcopy(self.solved.frame("kf_wide_2"))
        glass, mara = frame.entities["pr_glass"], frame.entities["ch_mara"]
        glass.support, glass.support_id = "held", "ch_mara"
        mara.pose, mara.holding_id = "lift", "pr_glass"
        for phase in ("rest", "aftermath", "impact", "rebound"):
            phrases = self._phrases_at(index, frame, phase)
            self.assertFalse([p for p in phrases if "floor" in p or "ground" in p], (phase, phrases))
            self.assertIn("the glass of water held in a firm grip", phrases, phase)
        # Phases that do not place the mover on a surface keep their wording.
        self.assertIn("the glass of water at the instant of contact", self._phrases_at(index, frame, "contact"))
        # A catch at impact meets the hand, not the floor, whatever the solved support at that instant.
        catch = with_event_kind(self.index, "ev_knock", "catch")
        for support in ("airborne", "held"):
            glass.support = support
            phrases = self._phrases_at(catch, frame, "impact")
            self.assertEqual(phrases[0], "the glass of water at the instant of the catch", support)
            self.assertFalse([p for p in phrases if "hitting" in p], phrases)

    def test_self_moving_character_is_not_at_rest_on_the_floor(self):
        # Review: a walk_to depicted at "rest" gave "Mara at rest on the floor", inviting a lying pose.
        frame = self.solved.frame("kf_wide_2")
        for kind in ("walk_to", "run_to", "enter", "stand_up", "gesture"):
            index = with_event_kind(self.index, "ev_knock", kind)
            for phase in ("rest", "aftermath", "impact"):
                phrases = self._phrases_at(index, frame, phase)
                self.assertFalse([p for p in phrases if p.startswith("Mara")], (kind, phase, phrases))
        sit = self._phrases_at(with_event_kind(self.index, "ev_knock", "sit_down"), frame, "rest")
        self.assertIn("Mara settled on the floor", sit)
        # A character who falls (not a self-moving kind) still hits the floor.
        fell = self._phrases_at(with_event_kind(self.index, "ev_knock", "fall", target_id="ch_mara"), frame, "impact")
        self.assertIn("Mara hitting the floor", fell)

    def _plan_with_flagged_props(self, n: int, name: str = "ceramic mug"):
        data = json.loads(FIXTURE.read_text())
        glass = next(p for p in data["bible"]["props"] if p["id"] == "pr_glass")
        state = next(s for s in data["bible"]["initial_states"] if s["entity_id"] == "pr_glass")
        for i in range(n):
            data["bible"]["props"].append(dict(glass, id=f"pr_extra_{i}", name=f"{name} {i}" if n > 1 else name))
            data["bible"]["initial_states"].append(dict(state, entity_id=f"pr_extra_{i}", flags=[]))
        plan = ScenePlan.model_validate(data)
        frame = copy.deepcopy(physics.solve(plan).frame("kf_wide_2"))
        for i in range(n):
            frame.entities[f"pr_extra_{i}"].flags = ["cracked", "stained", "wet"]
        return plan, build_index(plan), frame

    def test_flag_sentences_do_not_crowd_out_characters_or_focus_physics(self):
        # Review: six flagged props dropped Jon (the shot's secondary) and the focus physics phrase.
        plan, index, frame = self._plan_with_flagged_props(6)
        shot, kf = index.shots["sh_wide"], index.keyframe_by_id["kf_wide_2"]
        bboxes = {"ch_mara": (40, 30, 150, 280), "ch_jon": (400, 20, 480, 250)}
        prompt = conditioning.compile_prompt(plan, index, frame, shot, kf, list(frame.entities), bboxes=bboxes,
                                             image_width=TIER.width, continuity_image=2)
        self.assertLessEqual(len(prompt.split()), 120, prompt)
        self.assertIn("Mara,", prompt)
        self.assertIn("Jon,", prompt)
        self.assertIn("The glass of water at the instant of contact.", prompt)
        self.assertIn("is cracked, stained and wet.", prompt)          # some flags are still stated
        self.assertTrue(prompt.endswith("Same set and lighting as the second image."), prompt)

    def test_hard_cut_keeps_continuity_sentence(self):
        # Review: the last-resort hard cut could truncate the continuity sentence.
        long_name = " ".join(["enormous"] * 130) + " mug"
        plan, index, frame = self._plan_with_flagged_props(1, name=long_name)
        frame.entities["ch_mara"].pose, frame.entities["ch_mara"].holding_id = "carry", "pr_extra_0"
        prompt = conditioning.compile_prompt(plan, index, frame, index.shots["sh_wide"],
                                             index.keyframe_by_id["kf_wide_2"], list(frame.entities),
                                             bboxes={}, image_width=TIER.width, continuity_image=2)
        self.assertLessEqual(len(prompt.split()), 120)
        self.assertTrue(prompt.endswith("Same set and lighting as the second image."), prompt[-80:])

    def test_compact_keeps_whole_clauses(self):
        text = "woman in her thirties, shoulder-length dark curly hair, warm brown skin, oval face"
        self.assertEqual(conditioning.compact(text, 12),
                         "woman in her thirties, shoulder-length dark curly hair, warm brown skin")
        self.assertEqual(conditioning.compact(text, 2), "woman")             # never ends on a dangling "in"

    def test_compact_never_ends_mid_phrase_on_a_function_word(self):
        # Real cafe plan, most compressed level: "Ana, woman in her late as in the first image" and
        # "Small street cafe terrace on, at dusk" (a cut first clause ended on "late" / "on").
        ana = "woman in her late twenties, long straight black hair, olive skin, high cheekbones, slim build"
        cafe = "small street cafe terrace on grey stone pavement, round dark green metal bistro tables"
        self.assertEqual(conditioning.compact(ana, 4), "woman in her late twenties")   # one word over: kept whole
        self.assertEqual(conditioning.compact(cafe, 5), "small street cafe terrace")
        self.assertEqual(conditioning.compact("a cup with a chipped rim and a saucer", 4), "a cup")
        self.assertEqual(conditioning.compact("tall person with very long braided silver hair", 4), "tall person")
        self.assertEqual(conditioning.compact("the", 1), "the")                        # never empty
        self.assertEqual(conditioning.compact(ana, 12), "woman in her late twenties, long straight black hair, olive skin")


class TestMasks(unittest.TestCase):
    def test_snap_to_grid(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[17, 40] = True
        snapped = conditioning.snap_to_grid(mask, 16)
        self.assertEqual(int(snapped.sum()), 256)
        self.assertTrue(snapped[16:32, 32:48].all())
        odd = conditioning.snap_to_grid(np.ones((20, 20), dtype=bool), 16)
        self.assertEqual(odd.shape, (20, 20))
        self.assertTrue(odd.all())

    def test_changed_entities_between_wide_keyframes(self):
        plan = load_fixture()
        solved = physics.solve(plan)
        changed = conditioning.changed_entities(solved.frame("kf_wide_1"), solved.frame("kf_wide_2"))
        self.assertEqual(set(changed), {"ch_mara", "pr_glass"})
        self.assertEqual(conditioning.changed_entities(solved.frame("kf_jon_2"), solved.frame("kf_jon_2")), [])


class _RunDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = load_fixture()
        self.index = build_index(self.plan)
        self.solved = physics.solve(self.plan)
        for frame in self.solved.frames:                 # the static wide shot keeps one camera
            if frame.shot_id == "sh_wide":
                frame.camera = STATIC_CAM
        self.engine = FakeEngine()
        self.assets = AssetBank(self.tmp, self.engine, TIER, self.plan.bible.scene.style, "test")
        for ch in self.plan.bible.characters:
            self.assets.ensure_character(ch)
        self.memory = KeyframeMemory()
        self.blocking = {f.kf_id: synthetic_blocking(f, self.index, self.tmp / "graybox" / f.kf_id)
                         for f in self.solved.frames}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, kf_id, attempt=0, prev=None, defaults=None, **kw):
        """conditioning.build like the sequencer calls it: the previous solved frame (with its camera) is passed."""
        frame = self.solved.frame(kf_id)
        out = self.tmp / "keyframes" / kf_id / f"attempt_{attempt}"
        prev_kf_id = conditioning.previous_keyframe_in_shot(self.index, kf_id)
        if "prev_frame" not in kw and prev_kf_id is not None:
            kw["prev_frame"] = self.solved.frame(prev_kf_id)
        return conditioning.build(
            self.plan, self.index, frame, self.index.shots[frame.shot_id], self.index.keyframe_by_id[kf_id],
            self.blocking[kf_id], self.assets, self.memory, TIER, defaults or GenerationDefaults(init_noise=0.75),
            attempt, out, prev, **kw)

    def accepted_prev(self):
        prev = self.tmp / "prev.png"
        Image.new("RGB", (TIER.width, TIER.height), (10, 20, 30)).save(prev)
        return str(prev)

    def accept(self, kf_id, path, depth=1):
        frame = self.solved.frame(kf_id)
        shot = self.index.shots[frame.shot_id]
        self.memory.add(MemoryRecord(kf_id, frame.shot_id, shot.setup_id, frame.location_id, 0.0, str(path),
                                     depth, 0.9, False))


class TestBuild(_RunDirCase):
    def test_first_keyframe_uses_blocking_init_and_refs(self):
        cond = self.build("kf_wide_1")
        req = cond.request
        out = Path(req.output_path).parent
        self.assertFalse(cond.static_lock)
        self.assertIsNone(req.mask_image)
        self.assertEqual(image_size(req.init_image), (TIER.width, TIER.height))
        np.testing.assert_array_equal(np.array(Image.open(req.init_image)),
                                      np.array(Image.open(self.blocking["kf_wide_1"].blocking_path)))
        # Cast sheet first, then the plate as continuity (nothing accepted yet).
        self.assertEqual(len(req.ref_images), 2)
        self.assertTrue(req.ref_images[0].endswith("assets/sheets/kf_wide_1_cast.png"))
        self.assertEqual(image_size(req.ref_images[0]), TIER.sheet_size)
        self.assertEqual(image_size(req.ref_images[1]), TIER.continuity_size)
        self.assertEqual(cond.continuity_source, "plate")
        self.assertEqual(cond.ancestry_depth, 0)
        self.assertIn("Same set and lighting as the second image.", req.prompt)
        self.assertLessEqual(len(req.prompt.split()), 120)
        self.assertAlmostEqual(req.strength, 0.75)
        self.assertEqual(req.sigmas, list(config.VIGGLE_SIGMAS))
        self.assertEqual(req.seed, seeds.seed_for("test", "sh_wide", "kf_wide_1", 0))
        self.assertEqual((req.width, req.height), (TIER.width, TIER.height))
        for name in ("init.png", "continuity.png", "request.json"):
            self.assertTrue((out / name).exists(), name)
        record = json.loads((out / "request.json").read_text())
        self.assertEqual(record["prompt"], req.prompt)
        self.assertFalse(record["static_lock"])

    def test_retake_noise_schedule_and_custom_sigmas_off(self):
        # build() pins init_noise=0.75; each retake adds conditioning.NOISE_STEP_PER_ATTEMPT, capped at 0.9.
        self.assertAlmostEqual(self.build("kf_wide_1", attempt=2).request.strength, 0.75 + 2 * conditioning.NOISE_STEP_PER_ATTEMPT)
        self.assertAlmostEqual(self.build("kf_wide_1", attempt=5).request.strength, 0.9)
        req = self.build("kf_wide_1", attempt=1, defaults=GenerationDefaults(use_custom_sigmas=False)).request
        self.assertIsNone(req.sigmas)
        self.assertNotEqual(req.seed, self.build("kf_wide_1", attempt=0).request.seed)

    def test_static_camera_lock_mask(self):
        # Accept kf_wide_1 (a distinctive frame) then condition kf_wide_2 on the same static setup.
        prev = self.tmp / "frames" / "kf_wide_1.png"
        prev.parent.mkdir(parents=True)
        rng = np.random.default_rng(0)
        prev_arr = rng.integers(0, 256, (TIER.height, TIER.width, 3), dtype=np.uint8)
        Image.fromarray(prev_arr).save(prev)
        self.accept("kf_wide_1", prev)

        cond = self.build("kf_wide_2", prev=str(prev))
        req = cond.request
        self.assertTrue(cond.static_lock)
        self.assertEqual(cond.lock_mask_path, req.mask_image)
        self.assertEqual(cond.continuity_source, "shot_anchor")
        mask = np.array(Image.open(req.mask_image))
        self.assertEqual(mask.shape, (TIER.height, TIER.width))
        self.assertTrue(set(np.unique(mask)) <= {0, 255})
        # Snapped to the 16 px grid: every 16x16 cell is uniform.
        cells = mask.reshape(TIER.height // 16, 16, TIER.width // 16, 16)
        self.assertTrue((cells.min(axis=(1, 3)) == cells.max(axis=(1, 3))).all())
        on = mask > 0
        # Covers the changed entities (Mara, the glass) at both keyframes...
        for kf in ("kf_wide_1", "kf_wide_2"):
            for eid in ("ch_mara", "pr_glass"):
                emask = conditioning.entity_mask(self.blocking[kf], eid)
                self.assertTrue(on[emask].all(), f"{eid}@{kf}")
        # ...and nothing beyond their dilated, grid-snapped footprint (Jon is unchanged and untouched).
        allowed = np.zeros_like(on)
        for kf in ("kf_wide_1", "kf_wide_2"):
            for eid in ("ch_mara", "pr_glass"):
                allowed |= conditioning.entity_mask(self.blocking[kf], eid)
        allowed = conditioning.snap_to_grid(conditioning.dilate(allowed, 16), 16)
        self.assertFalse((on & ~allowed).any())
        self.assertFalse(on[conditioning.entity_mask(self.blocking["kf_wide_2"], "ch_jon")].any())
        # Init: previous frame outside the mask, blocking proxy inside it.
        init = np.array(Image.open(req.init_image).convert("RGB"))
        proxy = np.array(Image.open(self.blocking["kf_wide_2"].blocking_path).convert("RGB"))
        np.testing.assert_array_equal(init[~on], prev_arr[~on])
        np.testing.assert_array_equal(init[on], proxy[on])
        record = json.loads((Path(req.output_path).parent / "request.json").read_text())
        self.assertEqual(set(record["changed_entities"]), {"ch_mara", "pr_glass"})

    def test_explicit_prev_frame_and_blocking_match_derived(self):
        prev = self.accepted_prev()
        calls = []

        def pinned_cameras(plan, solved, index, width, height):
            calls.append((width, height))
            for frame in solved.frames:
                frame.camera = STATIC_CAM if frame.shot_id == "sh_wide" else None

        # The derived previous frame is re-solved, cameras included (at the current frame's resolution).
        with mock.patch("src.previs.camera.solve_cameras", side_effect=pinned_cameras):
            cond = self.build("kf_wide_2", prev=prev, prev_frame=None)
        self.assertEqual(calls, [(TIER.width, TIER.height)])
        self.assertTrue(cond.static_lock)
        derived = np.array(Image.open(cond.request.mask_image))
        explicit = np.array(Image.open(self.build(
            "kf_wide_2", attempt=1, prev=prev, prev_frame=self.solved.frame("kf_wide_1"),
            prev_blocking=self.blocking["kf_wide_1"]).request.mask_image))
        np.testing.assert_array_equal(derived, explicit)

    def test_moving_camera_never_locks(self):
        cond = self.build("kf_glass_2", prev=self.accepted_prev())   # tilt_down
        self.assertFalse(cond.static_lock)
        self.assertIsNone(cond.request.mask_image)

    def _assert_unlocked(self, cond, kf_id):
        self.assertFalse(cond.static_lock)
        self.assertIsNone(cond.request.mask_image)
        self.assertIsNone(cond.lock_mask_path)
        self.assertFalse((Path(cond.request.output_path).parent / "mask.png").exists())
        np.testing.assert_array_equal(np.array(Image.open(cond.request.init_image).convert("RGB")),
                                      np.array(Image.open(self.blocking[kf_id].blocking_path).convert("RGB")))

    def test_static_shot_whose_solved_camera_moved_does_not_lock(self):
        # Review (high): the lock was decided from shot.move alone, but the solver re-aims a static shot to follow
        # its subject. Copying the previous frame's background under a moved camera misplaces the set.
        moved = dataclasses.replace(STATIC_CAM, position=(1.117, 0.788, 3.8), yaw_deg=117.12)
        self.solved.frame("kf_wide_2").camera = moved
        self._assert_unlocked(self.build("kf_wide_2", prev=self.accepted_prev()), "kf_wide_2")

    def test_static_shot_with_millimetre_camera_drift_does_not_lock(self):
        # The unmodified fixture drifts about 9 mm in height between its two static keyframes.
        x, y, z = STATIC_CAM.position
        self.solved.frame("kf_wide_2").camera = dataclasses.replace(STATIC_CAM, position=(x, y + 0.0092, z))
        self._assert_unlocked(self.build("kf_wide_2", prev=self.accepted_prev()), "kf_wide_2")
        self.solved.frame("kf_wide_2").camera = dataclasses.replace(STATIC_CAM, pitch_deg=0.2)
        self._assert_unlocked(self.build("kf_wide_2", attempt=1, prev=self.accepted_prev()), "kf_wide_2")

    def test_static_shot_without_solved_cameras_does_not_lock(self):
        for frame in self.solved.frames:
            frame.camera = None
        self._assert_unlocked(self.build("kf_wide_2", prev=self.accepted_prev()), "kf_wide_2")

    def test_previous_solved_frame(self):
        frame = self.solved.frame("kf_wide_2")
        self.assertIsNone(conditioning.previous_solved_frame(self.plan, self.index, self.solved.frame("kf_wide_1")))
        given = self.solved.frame("kf_wide_1")
        self.assertIs(conditioning.previous_solved_frame(self.plan, self.index, frame, given), given)
        # A previous frame without the camera the current one has gets the re-solved camera.
        bare = dataclasses.replace(given, camera=None)

        def pinned_cameras(plan, solved, index, width, height):
            for f in solved.frames:
                f.camera = STATIC_CAM

        with mock.patch("src.previs.camera.solve_cameras", side_effect=pinned_cameras):
            filled = conditioning.previous_solved_frame(self.plan, self.index, frame, bare)
        self.assertEqual(filled.camera, STATIC_CAM)
        self.assertEqual(filled.entities, given.entities)
        self.assertIsNone(bare.camera)                                 # the caller's frame is not mutated

    def test_cameras_match(self):
        self.assertTrue(conditioning.cameras_match(STATIC_CAM, dataclasses.replace(STATIC_CAM)))
        self.assertTrue(conditioning.cameras_match(STATIC_CAM, dataclasses.replace(STATIC_CAM, yaw_deg=115.8 + 360.0)))
        self.assertTrue(conditioning.cameras_match(
            STATIC_CAM, dataclasses.replace(STATIC_CAM, position=(0.2 + 1e-6, 0.78, 3.8))))
        for change in ({"position": (0.2, 0.781, 3.8)}, {"yaw_deg": 115.85}, {"pitch_deg": -0.05},
                       {"roll_deg": 0.1}, {"lens_mm": 35.0}, {"vfov_deg": 40.0}, {"width": TIER.width * 2}):
            self.assertFalse(conditioning.cameras_match(STATIC_CAM, dataclasses.replace(STATIC_CAM, **change)), change)
        self.assertFalse(conditioning.cameras_match(None, STATIC_CAM))
        self.assertFalse(conditioning.cameras_match(None, None))

    def test_no_characters_means_no_sheet(self):
        frame = self.solved.frame("kf_glass_2")
        self.blocking["kf_glass_2"].visible_frac = {"pr_glass": 0.8, "pr_table": 0.5}
        cond = self.build("kf_glass_2")
        self.assertEqual(len(cond.request.ref_images), 1)              # plate only
        self.assertIn("Same set and lighting as the first image.", cond.request.prompt)
        self.assertNotIn("as in the first image,", cond.request.prompt)
        self.assertIn("glass of water", cond.request.prompt)
        self.assertIsNotNone(frame)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plate = self._img("plate.png")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _img(self, name):
        path = self.tmp / name
        Image.new("RGB", (32, 32)).save(path)
        return str(path)

    def _rec(self, kf, shot, setup, yaw=0.0, depth=1, score=0.5, approved=False, location="lo_a"):
        return MemoryRecord(kf, shot, setup, location, yaw, self._img(f"{kf}.png"), depth, score, approved)

    def test_rules(self):
        m = KeyframeMemory()
        self.assertEqual(m.continuity_for("sh_1", 1, "su_1", "lo_a", 0.0, self.plate), (self.plate, "plate", 0))
        self.assertEqual(m.continuity_for("sh_1", 1, "su_1", "lo_a", 0.0, None), (None, "none", 0))
        m.add(self._rec("kf_1_1", "sh_1", "su_1"))
        m.add(self._rec("kf_1_2", "sh_1", "su_1"))
        # Inside a shot: the anchor (first accepted frame of the shot).
        path, source, depth = m.continuity_for("sh_1", 3, "su_1", "lo_a", 0.0, self.plate)
        self.assertEqual((Path(path).name, source, depth), ("kf_1_1.png", "shot_anchor", 1))
        # First keyframe of a new shot on the same setup: the latest frame of that setup.
        path, source, _ = m.continuity_for("sh_3", 1, "su_1", "lo_a", 90.0, self.plate)
        self.assertEqual((Path(path).name, source), ("kf_1_2.png", "setup"))
        # Other setup, same location: nearest yaw, then best score.
        m.add(self._rec("kf_2_1", "sh_2", "su_2", yaw=50.0, score=0.9))
        m.add(self._rec("kf_4_1", "sh_4", "su_4", yaw=95.0, score=0.2))
        m.add(self._rec("kf_5_1", "sh_5", "su_5", yaw=100.0, score=0.6))
        path, source, _ = m.continuity_for("sh_9", 1, "su_9", "lo_a", 92.0, self.plate)
        self.assertEqual((Path(path).name, source), ("kf_5_1.png", "location"))
        # Too far round the set -> plate; other location -> plate.
        self.assertEqual(m.continuity_for("sh_9", 1, "su_9", "lo_a", -120.0, self.plate)[1], "plate")
        self.assertEqual(m.continuity_for("sh_9", 1, "su_9", "lo_b", 92.0, self.plate)[1], "plate")

    def test_depth_limit_and_approval(self):
        m = KeyframeMemory()
        m.add(self._rec("kf_1_1", "sh_1", "su_1", depth=2))
        self.assertEqual(m.continuity_for("sh_1", 2, "su_1", "lo_a", 0.0, self.plate)[1], "plate")
        m.approve("kf_1_1")
        self.assertEqual(m.continuity_for("sh_1", 2, "su_1", "lo_a", 0.0, self.plate)[1:], ("shot_anchor", 0))
        m.add(self._rec("kf_1_2", "sh_1", "su_1", depth=5, approved=True))
        self.assertEqual(m.get("kf_1_2").ancestry_depth, 0)

    def test_retake_replaces_and_roundtrip(self):
        m = KeyframeMemory()
        m.add(self._rec("kf_1_1", "sh_1", "su_1", score=0.1))
        m.add(self._rec("kf_2_1", "sh_2", "su_1"))
        m.add(self._rec("kf_1_1", "sh_1", "su_1", score=0.8))
        self.assertEqual([r.kf_id for r in m.records], ["kf_2_1", "kf_1_1"])
        path = self.tmp / "memory.json"
        m.save(path)
        loaded = KeyframeMemory.load(path)
        self.assertEqual(loaded.records, m.records)

    def test_load_tolerates_unknown_and_missing_fields(self):
        # Integrator: KeyframeMemory.load did MemoryRecord(**r), so a memory.json carrying any new key (a render
        # fingerprint) or lacking one crashed every later render with a TypeError/KeyError.
        path = self.tmp / "memory.json"
        newer = {"kf_id": "kf_1_1", "shot_id": "sh_1", "setup_id": "su_1", "location_id": "lo_a",
                 "camera_yaw": 10, "path": self._img("kf_1_1.png"), "ancestry_depth": 0, "score": 0.7,
                 "approved": True, "fingerprint": {"plan": "abc", "tier": "draft:512x288"}, "attempt": 2}
        older = {"kf_id": "kf_1_2", "shot_id": "sh_1", "setup_id": "su_1", "location_id": "lo_a",
                 "path": self._img("kf_1_2.png")}
        path.write_text(json.dumps({"records": [newer, older]}))
        m = KeyframeMemory.load(path)
        self.assertEqual([r.kf_id for r in m.records], ["kf_1_1", "kf_1_2"])
        first, second = m.records
        self.assertEqual((first.camera_yaw, first.ancestry_depth, first.score, first.approved), (10.0, 0, 0.7, True))
        self.assertEqual(first.extra, {"fingerprint": {"plan": "abc", "tier": "draft:512x288"}, "attempt": 2})
        self.assertEqual((second.camera_yaw, second.ancestry_depth, second.score, second.approved, second.extra),
                         (0.0, 1, 0.0, False, {}))
        self.assertEqual(m.continuity_for("sh_1", 2, "su_1", "lo_a", 0.0, self.plate)[1:], ("shot_anchor", 0))
        # Unknown keys survive a save by this version (flat, as written), and a reload is identical.
        m.save(path)
        saved = json.loads(path.read_text())["records"]
        self.assertEqual(saved[0], {**newer, "camera_yaw": 10.0})
        self.assertNotIn("extra", saved[1])
        self.assertEqual(KeyframeMemory.load(path).records, m.records)
        # The positional constructor (sequencer, tests) still works without `extra`.
        self.assertEqual(MemoryRecord("k", "s", "u", "l", 0.0, "p", 1, 0.0, False).extra, {})
        # A record that cannot be used (no path) is a clear error, not a TypeError.
        path.write_text(json.dumps({"records": [{"kf_id": "kf_x", "shot_id": "sh_1"}]}))
        with self.assertRaisesRegex(ValueError, "kf_x lacks setup_id, location_id, path"):
            KeyframeMemory.load(path)


class TestAssetBank(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = load_fixture()
        self.engine = FakeEngine()
        self.bank = AssetBank(self.tmp, self.engine, TIER, "film still", "proj")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_character_reference_cached_by_content(self):
        mara = self.plan.bible.characters[0]
        full, face = self.bank.ensure_character(mara)
        self.assertTrue(full.endswith("assets/cast/ch_mara_full_v1.png"))
        self.assertTrue(face.endswith("assets/cast/ch_mara_face_v1.png"))
        self.assertEqual(image_size(full), (512, 768))
        self.assertEqual(image_size(face)[0], image_size(face)[1])
        req = self.engine.requests[0]
        for words in ("Full-body character reference", "neutral standing pose", "front view",
                      "plain light grey background", "even studio light", "mustard yellow", "film still"):
            self.assertIn(words, req.prompt)
        self.assertEqual(self.bank.ensure_character(mara), (full, face))
        self.assertEqual(len(self.engine.requests), 1)
        # A new bank over the same run dir reuses the manifest.
        again = AssetBank(self.tmp, self.engine, TIER, "film still", "proj")
        self.assertEqual(again.ensure_character(mara)[0], full)
        self.assertEqual(len(self.engine.requests), 1)
        # Changed content -> next version, old file kept.
        changed = mara.model_copy(update={"wardrobe": "red raincoat"})
        full2, _ = again.ensure_character(changed)
        self.assertTrue(full2.endswith("ch_mara_full_v2.png"))
        self.assertTrue(Path(full).exists())
        self.assertEqual(len(self.engine.requests), 2)

    def test_plate_and_cast_sheet(self):
        loc = self.plan.bible.locations[0]
        plate = self.bank.ensure_plate(loc, "day")
        self.assertTrue(plate.endswith("assets/plates/lo_kitchen_v1.png"))
        self.assertEqual(image_size(plate), (TIER.width, TIER.height))
        self.assertIn("no people", self.engine.requests[-1].prompt)
        for ch in self.plan.bible.characters:
            self.bank.ensure_character(ch)
        n = len(self.engine.requests)
        sheet = self.bank.cast_sheet(["ch_mara", "ch_jon"], TIER.sheet_size, name="kf_wide_1_cast")
        self.assertTrue(sheet.endswith("assets/sheets/kf_wide_1_cast.png"))
        self.assertEqual(image_size(sheet), TIER.sheet_size)
        self.assertEqual(len(self.engine.requests), n)            # composed, not generated
        corner = Image.open(sheet).convert("RGB").getpixel((0, 0))
        self.assertTrue(all(abs(c - s) <= 2 for c, s in zip(corner, (214, 214, 214))))
        with self.assertRaises(ValueError):
            self.bank.cast_sheet([], TIER.sheet_size)

    def test_cast_sheet_from_manifest_in_new_session(self):
        self.bank.ensure_character(self.plan.bible.characters[1])
        fresh = AssetBank(self.tmp, self.engine, TIER, "film still", "proj")
        sheet = fresh.cast_sheet(["ch_jon"], TIER.sheet_size)
        self.assertTrue(Path(sheet).exists())
        with self.assertRaises(KeyError):
            fresh.cast_sheet(["ch_nobody"], TIER.sheet_size)

    def test_version_for_flags_edits_current_version(self):
        glass = self.plan.bible.props[1]
        base = self.bank.ensure_prop(glass)
        self.assertEqual(self.bank.version_for_flags("pr_glass", []), base)
        cracked = self.bank.version_for_flags("pr_glass", ["cracked"])
        edit = self.engine.requests[-1]
        self.assertEqual(edit.ref_images, [base])
        self.assertIn("Same glass of water as in the first image, now cracked", edit.prompt)
        self.assertTrue(cracked.endswith("assets/props/pr_glass_v2.png"))
        n = len(self.engine.requests)
        self.assertEqual(self.bank.version_for_flags("pr_glass", ["cracked"]), cracked)
        self.assertEqual(len(self.engine.requests), n)
        both = self.bank.version_for_flags("pr_glass", ["shattered", "cracked"])
        edit = self.engine.requests[-1]
        self.assertEqual(edit.ref_images, [cracked])                 # edited from the current version
        self.assertIn("now shattered into pieces", edit.prompt)
        self.assertNotIn("cracked", edit.prompt)
        self.assertNotEqual(both, cracked)
        # Cached by (entity, sorted flags), whatever version chain produced it.
        n = len(self.engine.requests)
        self.assertEqual(self.bank.version_for_flags("pr_glass", ["cracked", "shattered", "cracked"]), both)
        self.assertEqual(len(self.engine.requests), n)
        with self.assertRaises(KeyError):
            AssetBank(self.tmp, self.engine, TIER, "s", "p").version_for_flags("pr_table", ["dented"])

    def test_quadruped_reference_is_a_side_view_of_the_animal(self):
        # E2E #5: the dog's reference asked for "one person ... neutral standing pose, front view".
        plan = quadruped_fixture()
        dog = next(ch for ch in plan.bible.characters if ch.id == "ch_jon")
        full, face = self.bank.ensure_character(dog)
        prompt = self.engine.requests[-1].prompt
        for words in ("one animal", "medium-sized scruffy terrier mix", "red nylon collar", "side view",
                      "standing on all four legs", "plain light grey background", "film still"):
            self.assertIn(words, prompt)
        for words in ("person", "neutral standing pose", "front view", "arms", "head to feet"):
            self.assertNotIn(words, prompt)
        w, h = image_size(full)
        self.assertGreater(w, h)                                       # landscape for a side view
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertEqual(image_size(face)[0], image_size(face)[1])
        # Humans keep the front-view portrait reference; a mixed sheet composes both.
        self.bank.ensure_character(plan.bible.characters[0])
        human = self.engine.requests[-1].prompt
        self.assertIn("neutral standing pose", human)
        sheet = self.bank.cast_sheet(["ch_mara", "ch_jon"], TIER.sheet_size, name="mixed")
        self.assertEqual(image_size(sheet), TIER.sheet_size)
        # A new session reads the body plan from the manifest.
        fresh = AssetBank(self.tmp, self.engine, TIER, "film still", "proj")
        self.assertTrue(fresh._is_quadruped("ch_jon"))
        self.assertFalse(fresh._is_quadruped("ch_mara"))
        self.assertTrue(Path(fresh.cast_sheet(["ch_jon"], TIER.sheet_size, name="dog")).exists())
        # State versions are edited as an animal, not a person.
        self.bank.version_for_flags("ch_jon", ["wet"])
        self.assertIn("Same animal as in the first image, now wet", self.engine.requests[-1].prompt)

    def test_rejected_asset_is_retaken_then_fails(self):
        with mock.patch.object(AssetBank, "approve", side_effect=[False, True]):
            self.bank.ensure_plate(self.plan.bible.locations[0], "night")
        self.assertEqual(len(self.engine.requests), 2)
        self.assertNotEqual(self.engine.requests[0].seed, self.engine.requests[1].seed)
        with mock.patch.object(AssetBank, "approve", return_value=False):
            with self.assertRaises(RuntimeError):
                self.bank.ensure_prop(self.plan.bible.props[0])


class TestWithRealBlocking(unittest.TestCase):
    """Conditioning over the real camera + box renderer, when those modules are present."""

    def test_fixture_requests(self):
        try:
            from src.previs import camera
            from src.previs.blocking.box_renderer import BoxRenderer
        except ImportError:
            self.skipTest("camera/blocking modules not available")
        tmp = Path(tempfile.mkdtemp())
        try:
            plan = load_fixture()
            index = build_index(plan)
            solved = physics.solve(plan)
            camera.solve_cameras(plan, solved, index, TIER.width, TIER.height)
            renderer = BoxRenderer(TIER.width, TIER.height)
            engine = FakeEngine()
            assets = AssetBank(tmp, engine, TIER, plan.bible.scene.style, "test")
            for ch in plan.bible.characters:
                assets.ensure_character(ch)
            memory = KeyframeMemory()
            prev_by_shot = {}
            for frame in solved.frames:
                blocking = renderer.render(frame, plan, index, tmp / "graybox" / frame.kf_id)
                shot = index.shots[frame.shot_id]
                cond = conditioning.build(plan, index, frame, shot, index.keyframe_by_id[frame.kf_id], blocking,
                                          assets, memory, TIER, GenerationDefaults(), 0,
                                          tmp / "keyframes" / frame.kf_id, prev_by_shot.get(frame.shot_id))
                self.assertLessEqual(len(cond.request.prompt.split()), 120)
                self.assertLessEqual(len(cond.request.ref_images), 3)
                if cond.static_lock:
                    mask = np.array(Image.open(cond.lock_mask_path))
                    cells = mask.reshape(TIER.height // 16, 16, TIER.width // 16, 16)
                    self.assertTrue((cells.min(axis=(1, 3)) == cells.max(axis=(1, 3))).all())
                    prev_kf = conditioning.previous_keyframe_in_shot(index, frame.kf_id)
                    self.assertTrue(conditioning.cameras_match(solved.frame(prev_kf).camera, frame.camera))
                result = engine.generate(cond.request)
                prev_by_shot[frame.shot_id] = result.image_path
                memory.add(MemoryRecord(frame.kf_id, frame.shot_id, shot.setup_id, frame.location_id,
                                        frame.camera.yaw_deg, result.image_path, cond.ancestry_depth + 1, 0.5, False))
            # The static wide shot locks exactly when the solver kept its camera still.
            still = conditioning.cameras_match(solved.frame("kf_wide_1").camera, solved.frame("kf_wide_2").camera)
            self.assertEqual((tmp / "keyframes" / "kf_wide_2" / "mask.png").exists(), still)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
