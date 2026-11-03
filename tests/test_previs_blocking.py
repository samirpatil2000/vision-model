"""Blocking: mannequin geometry, the numpy box renderer, and the Blender backend (spec, ids, guarded real run)."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src.previs import camera, physics
from src.previs.blocking import blender_builder, build_scene, entity_aabb, mannequin
from src.previs.blocking.blender_builder import BlenderRenderer, decode_ids, id_rgb
from src.previs.blocking.box_renderer import BoxRenderer
from src.previs.schema import ScenePlan
from src.previs.state import build_index
from src.previs.types import EntityFrame

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
W, H = 240, 140
POSES = ["none", "stand", "walk", "run", "sit", "crouch", "kneel", "lie", "reach", "lift", "carry", "throw",
         "push", "pull", "fall", "brace"]


def _person(pose: str = "stand", yaw: float = 0.0, holding: str = "none", cue: str = "none", height: float = 1.75,
            base_y: float = 0.0, support: str = "ground", shape: str = "character"):
    return EntityFrame(entity_id="ch_mara", kind="character", present=True, position=(1.0, base_y, 2.0), yaw_deg=yaw,
                       rotation_deg=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), pose=pose, expression="neutral",
                       holding_id=holding, gaze_id="none", support=support, support_id="none", flags=[],
                       weight_cue=cue, secondary=[], size=(0.45, height, 0.3), color_hex="#d4a017",
                       secondary_color_hex="#1f3a5f", shape=shape)


def _dog(pose: str = "stand", yaw: float = 0.0, height: float = 0.5):
    return _person(pose, yaw=yaw, height=height, shape="quadruped")


def _parts(segs):
    return {n: (np.array(a), np.array(b), r) for a, b, r, n in segs}


def _lowest(segs) -> float:
    return min(min(a[1], b[1]) - r for a, b, r, _ in segs)


def _extent(segs):
    pts = np.array([np.array(e) + d for p0, p1, r, _ in segs for e in (p0, p1)
                    for d in ((0, r, 0), (0, -r, 0), (r, 0, 0), (-r, 0, 0), (0, 0, r), (0, 0, -r))])
    return pts.min(axis=0), pts.max(axis=0)


class TestMannequin(unittest.TestCase):
    def test_every_pose_builds_the_full_body(self):
        for pose in POSES:
            segs = mannequin.segments(_person(pose))
            self.assertEqual(len(segs), 18, pose)   # 15 body capsules + hair cap + 2 eyes
            self.assertTrue(np.isfinite(np.array([s[0] + s[1] for s in segs])).all(), pose)
            self.assertTrue(all(s[2] > 0 for s in segs), pose)

    def test_standing_height_and_ground_contact(self):
        lo, hi = _extent(mannequin.segments(_person("stand", height=1.75)))
        self.assertAlmostEqual(hi[1], 1.75, delta=0.05)
        self.assertAlmostEqual(lo[1], 0.0, delta=0.04)

    def test_lying_and_falling_bodies_are_not_upright(self):
        for pose, max_height in (("lie", 0.35), ("fall", 1.0)):
            lo, hi = _extent(mannequin.segments(_person(pose)))
            self.assertLess(hi[1] - lo[1], max_height * 1.75, pose)

    def test_yaw_turns_the_face(self):
        def part(segs, name):
            return next((np.array(a) + np.array(b)) / 2 for a, b, _, n in segs if n == name)
        for yaw, axis in ((0.0, (0, 0, 1)), (90.0, (1, 0, 0)), (180.0, (0, 0, -1))):
            segs = mannequin.segments(_person(yaw=yaw))
            facing = part(segs, "nose") - part(segs, "head")
            self.assertGreater(float(facing @ np.array(axis)), 0.05, yaw)

    def test_held_objects_sit_in_the_right_hand(self):
        for pose in ("stand", "carry", "reach", "throw"):
            ef = _person(pose, yaw=30.0, holding="pr_glass")
            hand = next(np.array(b) for a, b, _, n in mannequin.segments(ef) if n == "lower_arm_r")
            np.testing.assert_allclose(hand, physics.hand_position(ef, "right"), atol=1e-9, err_msg=pose)

    def test_seated_kneeling_and_lying_arms_keep_their_length_and_stay_above_the_floor(self):
        # Regression: sit hands were offset by the pelvis drop twice (forearms 28 cm into the floor, arms
        # stretched to 0.92 m), and a lying body's forward hand offset turned into -y once pitched flat.
        reach = (mannequin.UPPER_ARM + mannequin.FOREARM) * 1.75
        for pose, base_y in (("sit", 0.0), ("sit", 0.45), ("crouch", 0.0), ("kneel", 0.0), ("lie", 0.0)):
            segs = mannequin.segments(_person(pose, base_y=base_y))
            parts = _parts(segs)
            self.assertGreater(_lowest(segs), -0.01, (pose, base_y))
            for side in ("l", "r"):
                upper, lower = parts[f"upper_arm_{side}"], parts[f"lower_arm_{side}"]
                arm = np.linalg.norm(upper[1] - upper[0]) + np.linalg.norm(lower[1] - lower[0])
                self.assertLessEqual(arm, reach + 1e-6, (pose, base_y, side))
            if pose == "sit":                             # hands rest on the lap, not beside the shins
                hand_y = parts["lower_arm_r"][1][1]
                self.assertGreater(hand_y, parts["hips"][0][1], base_y)
                self.assertLess(hand_y, parts["shoulders"][0][1], base_y)

    def test_sitting_with_the_feet_on_the_floor_uses_the_seat(self):
        # Base on the floor (feet), seat given: thighs at the seat, shins down to the floor, not a floor sit.
        chair = mannequin.segments(_person("sit", support="on_entity"), seat_y=0.45)
        floor = mannequin.segments(_person("sit"))
        legacy = mannequin.segments(_person("sit", base_y=0.45))            # base on the seat itself
        hips = lambda segs: _parts(segs)["hips"][0][1]                    # noqa: E731
        self.assertAlmostEqual(hips(chair), 0.45 + 0.04 * 1.75, places=6)
        self.assertAlmostEqual(hips(chair), hips(legacy), places=6)
        self.assertLess(hips(floor), 0.2)
        self.assertLess(_parts(chair)["lower_leg_r"][1][1], 0.1)          # ankles near the floor
        self.assertGreater(_lowest(chair), -0.01)
        tall = mannequin.segments(_person("sit", support="on_entity"), seat_y=0.85)   # chair box incl. backrest
        self.assertLessEqual(hips(tall), mannequin.SEAT_MAX * 1.75 + 0.04 * 1.75 + 1e-6)

    def test_weight_cue_leans_the_upper_body(self):
        head = lambda cue: next(np.array(a) for a, b, _, n in mannequin.segments(_person(cue=cue)) if n == "head")
        self.assertGreater(head("leaning")[2], head("none")[2] + 0.1)     # forward is +z at yaw 0
        self.assertLess(head("recoil")[2], head("none")[2] - 0.1)


class TestQuadruped(unittest.TestCase):
    """Character.body == "quadruped": a four-legged proxy, not a small human."""

    def test_every_pose_builds_four_legs_head_and_tail(self):
        legs = {f"{part}_{leg}" for leg in ("fl", "fr", "rl", "rr") for part in ("upper_leg", "lower_leg", "paw")}
        for pose in POSES:
            segs = mannequin.segments(_dog(pose))
            names = {s[3] for s in segs}
            self.assertTrue(legs | {"torso", "neck", "head", "snout", "tail", "eye_l", "eye_r"} <= names, pose)
            self.assertFalse({"shoulders", "hair", "upper_arm_l", "foot_r"} & names, pose)
            self.assertTrue(np.isfinite(np.array([s[0] + s[1] for s in segs])).all(), pose)
            self.assertTrue(all(s[2] > 0 for s in segs), pose)
            self.assertGreater(_lowest(segs), -0.005, pose)                 # nothing sinks into the ground

    def test_standing_body_is_horizontal_at_shoulder_height_with_the_head_forward(self):
        h = 0.5
        parts = _parts(mannequin.segments(_dog("stand", height=h)))
        rump, chest, _ = parts["torso"]
        self.assertLess(abs(rump[1] - chest[1]), 0.1 * h)                  # horizontal spine
        self.assertTrue(0.55 * h < (rump[1] + chest[1]) / 2 < 0.8 * h)
        _, back = _extent([s for s in mannequin.segments(_dog("stand", height=h)) if s[3] in ("torso", "chest")])
        self.assertAlmostEqual(back[1], h, delta=0.1 * h)                   # withers at the shoulder height
        self.assertAlmostEqual(_lowest(mannequin.segments(_dog("stand", height=h))), 0.0, delta=0.005)   # on the ground
        head = (parts["head"][0] + parts["head"][1]) / 2
        self.assertGreater(head[2] - (rump[2] + chest[2]) / 2, 0.5 * h)    # head forward (+z at yaw 0)
        self.assertGreater(head[1], chest[1])
        self.assertLess(min(parts["tail"][0][2], parts["tail"][1][2]), rump[2])   # tail behind
        for leg in mannequin.Q_LEGS:                                        # legs under the body
            self.assertLess(parts[f"paw_{leg}"][1][1], 0.1 * h, leg)

    def test_yaw_turns_the_whole_body(self):
        for yaw, axis in ((0.0, (0, 0, 1)), (90.0, (1, 0, 0)), (-115.0, (-0.906, 0, -0.423))):
            parts = _parts(mannequin.segments(_dog(yaw=yaw)))
            facing = parts["snout"][1] - (parts["torso"][0] + parts["torso"][1]) / 2
            facing[1] = 0.0
            self.assertGreater(float(facing @ np.array(axis)) / np.linalg.norm(facing), 0.95, yaw)

    def test_poses_change_the_silhouette(self):
        height = lambda pose: _extent(mannequin.segments(_dog(pose)))[1][1]           # noqa: E731
        spine_rise = lambda pose: (lambda t: t[1][1] - t[0][1])(_parts(mannequin.segments(_dog(pose)))["torso"])  # noqa: E731
        self.assertLess(_parts(mannequin.segments(_dog("lie")))["torso"][1][1], 0.3 * 0.5)    # belly down
        self.assertGreater(spine_rise("sit"), 0.25 * 0.5)                   # chest up, haunches down
        self.assertLess(height("fall"), 0.5 * 0.5)                         # on its side
        paws = lambda pose: np.array([_parts(mannequin.segments(_dog(pose)))[f"paw_{leg}"][1]  # noqa: E731
                                      for leg in mannequin.Q_LEGS])
        self.assertGreater(np.abs(paws("walk") - paws("stand")).max(), 0.05)
        self.assertGreater(np.abs(paws("run") - paws("walk")).max(), 0.05)
        self.assertEqual(mannequin.quadruped_pose("kneel"), "lie")
        self.assertEqual(mannequin.quadruped_pose("carry"), "stand")

    def test_standing_limbs_keep_their_lengths(self):
        h = 0.5
        for pose in ("stand", "walk", "run", "brace", "fall"):
            parts = _parts(mannequin.segments(_dog(pose, height=h)))
            for leg in mannequin.Q_LEGS:
                l1, l2 = mannequin.Q_FORE if leg.startswith("f") else mannequin.Q_HIND
                for name, length in ((f"upper_leg_{leg}", l1), (f"lower_leg_{leg}", l2)):
                    a, b, _ = parts[name]
                    self.assertAlmostEqual(float(np.linalg.norm(b - a)), length * h, delta=0.01 * h, msg=(pose, name))


class TestQuadrupedScene(unittest.TestCase):
    """A plan whose second character is a dog: primitives, colours, id map and visible_frac."""

    @classmethod
    def setUpClass(cls):
        plan = ScenePlan.model_validate_json(FIXTURE.read_text())
        dog = next(c for c in plan.bible.characters if c.id == "ch_jon")
        dog.body, dog.height_m, dog.palette_hex = "quadruped", 0.5, ["#c8a165", "#f0e6d2"]
        cls.plan = plan
        cls.index = build_index(plan)
        cls.solved = physics.solve(plan)
        camera.solve_cameras(plan, cls.solved, cls.index, W, H)

    def test_scene_draws_the_dog_proxy_in_its_palette(self):
        frame = self.solved.frame("kf_wide_1")
        scene = build_scene(frame, self.plan, self.index)
        prims = [p for p in scene.primitives if p.entity_id == "ch_jon"]
        parts = {p.part: p for p in prims}
        self.assertIn("tail", parts)
        self.assertNotIn("shoulders", parts)
        self.assertEqual(scene.entities["ch_jon"].shape, mannequin.QUADRUPED)
        self.assertEqual(frame.entities["ch_jon"].shape, mannequin.QUADRUPED)  # physics reports the body plan too
        np.testing.assert_allclose(parts["torso"].color, (0xc8 / 255, 0xa1 / 255, 0x65 / 255))
        np.testing.assert_allclose(parts["paw_fl"].color, (0xf0 / 255, 0xe6 / 255, 0xd2 / 255))
        self.assertFalse(parts["eye_l"].occluder)
        humans = {p.part for p in scene.primitives if p.entity_id == "ch_mara"}
        self.assertIn("shoulders", humans)                                    # people stay human
        lo, hi = entity_aabb(scene.entities["ch_jon"])
        self.assertLess(hi[1], 0.5 * 1.35)                                    # dog-sized, head included

    def test_render_ids_and_visibility_cover_the_dog(self):
        frame = self.solved.frame("kf_wide_1")
        with tempfile.TemporaryDirectory() as tmp:
            res = BoxRenderer(W, H).render(frame, self.plan, self.index, tmp)
            ids = np.array(Image.open(res.id_path))
        value = next(k for k, v in res.id_map.items() if v == "ch_jon")
        ys, xs = np.nonzero(ids == value)
        self.assertGreater(len(xs), 20)
        self.assertEqual(res.bboxes["ch_jon"], (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        self.assertGreater(res.visible_frac["ch_jon"], 0.5)
        spec = blender_builder.build_spec(build_scene(frame, self.plan, self.index), frame.camera, W, H, Path("/tmp/x"))
        self.assertTrue(any(o["entity_id"] == "ch_jon" for o in spec["objects"]))   # same primitives in Blender


class TestBoxRenderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = ScenePlan.model_validate_json(FIXTURE.read_text())
        cls.index = build_index(cls.plan)
        cls.solved = physics.solve(cls.plan)
        camera.solve_cameras(cls.plan, cls.solved, cls.index, W, H)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.renderer = BoxRenderer(W, H, blur_px=2)
        cls.results = {f.kf_id: cls.renderer.render(f, cls.plan, cls.index, Path(cls.tmp.name) / f.kf_id)
                       for f in cls.solved.frames}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_and_id_map_consistency(self):
        for kf, res in self.results.items():
            out = Path(res.id_path).parent
            for name in ("blocking.png", "blocking_sharp.png", "id.png", "depth.npy", "meta.json"):
                self.assertTrue((out / name).is_file(), f"{kf}/{name}")
            ids = np.array(Image.open(res.id_path))
            depth = np.load(res.depth_path)
            self.assertEqual(ids.shape, (H, W))
            self.assertEqual(ids.dtype, np.uint8)
            self.assertEqual(depth.shape, (H, W))
            self.assertTrue(set(np.unique(ids)) <= {0} | set(res.id_map), kf)
            self.assertEqual(res.id_map, {k + 1: eid for k, eid in enumerate(self.index.entities)})
            self.assertTrue(np.isfinite(depth[ids > 0]).all(), kf)
            for value, eid in res.id_map.items():
                ys, xs = np.nonzero(ids == value)
                if len(xs):
                    self.assertEqual(res.bboxes[eid], (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), kf)
                    self.assertGreater(res.visible_frac[eid], 0.0, kf)
                else:
                    self.assertNotIn(eid, res.bboxes)
                    self.assertEqual(res.visible_frac[eid], 0.0)
                self.assertTrue(0.0 <= res.visible_frac[eid] <= 1.0)
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["kf_id"], kf)
            self.assertEqual(meta["id_map"], {str(k): v for k, v in res.id_map.items()})

    def test_visible_fraction_reflects_occlusion(self):
        wide = self.results["kf_wide_1"]
        self.assertLess(wide.visible_frac["ch_mara"], 0.9)       # Mara stands behind the table
        self.assertGreater(wide.visible_frac["ch_mara"], 0.3)
        self.assertAlmostEqual(wide.visible_frac["pr_table"], 1.0, places=3)
        self.assertAlmostEqual(wide.visible_frac["ch_jon"], 1.0, places=2)
        for kf in ("kf_glass_1", "kf_glass_2", "kf_glass_3"):
            self.assertGreater(self.results[kf].visible_frac["pr_glass"], 0.5, kf)
        for kf in ("kf_jon_1", "kf_jon_2"):
            self.assertGreater(self.results[kf].visible_frac["ch_jon"], 0.05, kf)

    def test_render_is_deterministic_and_render_image_matches(self):
        frame = self.solved.frame("kf_glass_2")
        with tempfile.TemporaryDirectory() as tmp:
            again = self.renderer.render(frame, self.plan, self.index, tmp)
            first = Path(self.results["kf_glass_2"].id_path).parent
            for name in ("id.png", "blocking_sharp.png"):
                np.testing.assert_array_equal(np.array(Image.open(first / name)), np.array(Image.open(Path(tmp) / name)))
            np.testing.assert_array_equal(np.load(first / "depth.npy"), np.load(again.depth_path))
        image = self.renderer.render_image(frame, self.plan, self.index)
        self.assertEqual(image.size, (W, H))
        np.testing.assert_array_equal(np.array(image), np.array(Image.open(first / "blocking_sharp.png")))

    def test_airborne_glass_has_arc_droplets_and_a_detached_shadow(self):
        frame = self.solved.frame("kf_glass_2")
        scene = build_scene(frame, self.plan, self.index)
        parts = {p.part for p in scene.primitives if p.entity_id == "pr_glass"}
        self.assertTrue({"liquid_arc", "droplets", "body"} <= parts)
        cam = frame.camera
        lit = self.renderer.rasterize(scene, cam)
        no_glass = replace(scene, primitives=[p for p in scene.primitives if p.entity_id != "pr_glass"])
        bare = self.renderer.rasterize(no_glass, cam)
        x, _, z = frame.entities["pr_glass"].position
        (u, v), = camera.project(cam, np.array([(x, 0.0, z)]))[0]
        u, v = int(u), int(v)
        self.assertEqual(lit.ids[v, u], 0)                                    # floor pixel, not the glass
        self.assertLess(lit.color[v, u].sum(), 0.85 * bare.color[v, u].sum())  # but shaded by it
        column = lit.ids[:v, u]
        glass_rows = np.nonzero(column == scene.id_of("pr_glass"))[0]
        self.assertTrue(len(glass_rows) and v - glass_rows.max() > 5)         # a gap: the shadow is detached

    def test_shattered_glass_draws_shards_and_puddle_not_the_intact_vessel(self):
        impact = build_scene(self.solved.frame("kf_glass_3"), self.plan, self.index)
        parts = {p.part for p in impact.primitives if p.entity_id == "pr_glass"}
        self.assertTrue({"shards", "puddle", "splash"} <= parts)
        self.assertNotIn("body", parts)
        later = build_scene(self.solved.frame("kf_jon_2"), self.plan, self.index)
        later_parts = {p.part for p in later.primitives if p.entity_id == "pr_glass"}
        self.assertTrue({"shards", "puddle"} <= later_parts)
        self.assertNotIn("splash", later_parts)                               # transient: gone after the event
        shards = lambda s: [p.center for p in s.primitives if p.part == "shards"]
        self.assertEqual(shards(impact), shards(later))                       # debris never moves

    def test_full_glass_shows_its_liquid_level(self):
        scene = build_scene(self.solved.frame("kf_wide_1"), self.plan, self.index)
        liquid = [p for p in scene.primitives if p.entity_id == "pr_glass" and p.part == "liquid"]
        self.assertEqual(len(liquid), 1)
        self.assertAlmostEqual(liquid[0].half[1] * 2, 0.12 * 0.85, places=6)


class TestBlenderBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = ScenePlan.model_validate_json(FIXTURE.read_text())
        cls.index = build_index(cls.plan)
        cls.solved = physics.solve(cls.plan)
        camera.solve_cameras(cls.plan, cls.solved, cls.index, W, H)

    def test_id_colours_round_trip_with_noise(self):
        rng = np.random.default_rng(1)
        values = np.arange(0, blender_builder.MAX_IDS + 1)
        rgb = np.array([id_rgb(int(v)) for v in values], dtype=float)[None, :, :]
        noisy = np.clip(rgb + rng.integers(-20, 21, rgb.shape), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(decode_ids(noisy, int(values.max()))[0], values)
        self.assertEqual(decode_ids(np.array([[id_rgb(9)]], dtype=np.uint8), 4)[0, 0], 0)   # unknown -> background

    def test_coordinate_conversion_and_camera_matrix(self):
        conv = blender_builder.to_blender(np.eye(3))
        self.assertAlmostEqual(np.linalg.det(conv), 1.0)                     # a rotation, no mirror
        np.testing.assert_allclose(conv[1], (0, 0, 1))                       # our +y (up) -> Blender +Z
        cam = self.solved.frame("kf_wide_1").camera
        m = np.array(blender_builder.camera_matrix(cam))
        np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
        forward = camera.camera_basis(cam)[2]
        np.testing.assert_allclose(-m[:3, 2], blender_builder.to_blender(forward)[0], atol=1e-9)
        np.testing.assert_allclose(m[:3, 3], blender_builder.to_blender(np.array(cam.position))[0], atol=1e-9)

    def test_spec_meshes_are_closed_and_outward(self):
        scene = build_scene(self.solved.frame("kf_glass_2"), self.plan, self.index)
        spec = blender_builder.build_spec(scene, self.solved.frame("kf_glass_2").camera, W, H, Path("/tmp/x"))
        entity_objects = [o for o in spec["objects"] if o["entity_id"]]
        self.assertEqual({o["entity_id"] for o in entity_objects}, {p.entity_id for p in scene.primitives if p.alpha >= 1})
        for prim in scene.primitives[:20]:
            verts, faces, smooth = blender_builder.primitive_mesh(prim)
            self.assertEqual(len(faces), len(smooth))
            centre = verts.mean(axis=0)
            for face in faces:                                               # every face normal points outward
                p = verts[face]
                normal = sum(np.cross(p[i], p[(i + 1) % len(face)]) for i in range(len(face)))
                self.assertGreaterEqual(float(normal @ (p.mean(axis=0) - centre)), -1e-12)

    def test_command_is_headless_factory_startup_without_autoexec(self):
        cmd = BlenderRenderer(W, H, blender=Path("/opt/blender")).command(Path("s.py"), Path("scene.json"))
        self.assertEqual(cmd[:4], ["/opt/blender", "-b", "--factory-startup", "--disable-autoexec"])
        self.assertIn("--python", cmd)
        self.assertEqual(cmd[-2:], ["--", "scene.json"])
        self.assertFalse({"-y", "--enable-autoexec"} & set(cmd))

    def test_render_decodes_blender_outputs(self):
        frame = self.solved.frame("kf_wide_1")
        box = BoxRenderer(W, H)
        expected = box.rasterize(build_scene(frame, self.plan, self.index), frame.camera)

        def fake_blender(cmd, **kwargs):
            self.assertIn("timeout", kwargs)
            spec = json.loads(Path(cmd[-1]).read_text())
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
            for value in np.unique(expected.ids):
                rgb[expected.ids == value] = id_rgb(int(value))
            Image.fromarray(rgb).save(spec["outputs"]["ids"])
            Image.fromarray(np.full((H, W, 4), 200, dtype=np.uint8)).save(spec["outputs"]["color"])
            Image.fromarray(np.full((H, W, 4), 120, dtype=np.uint8)).save(spec["outputs"]["init"])
            np.save(spec["outputs"]["depth"], expected.depth.astype(np.float32))
            return mock.Mock(returncode=0, stdout="PREVIS_BLENDER_OK\n", stderr="")

        renderer = BlenderRenderer(W, H, blender=Path(__file__))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(renderer, "is_available", return_value=True), \
                mock.patch.object(blender_builder.subprocess, "run", side_effect=fake_blender):
            res = renderer.render(frame, self.plan, self.index, tmp)
            np.testing.assert_array_equal(np.array(Image.open(res.id_path)), expected.ids)
            self.assertEqual(sorted(os.listdir(tmp)), ["blocking.png", "blocking_sharp.png", "depth.npy", "id.png", "meta.json"])
            # blocking.png (the image-model init) is the grid-free pass, blocking_sharp.png the gridded one.
            self.assertNotEqual(np.array(Image.open(Path(tmp) / "blocking.png")).tolist(),
                                np.array(Image.open(Path(tmp) / "blocking_sharp.png")).tolist())
        self.assertAlmostEqual(res.visible_frac["pr_table"], 1.0, places=3)

    def test_spec_flags_the_grid_and_the_script_renders_a_grid_free_init(self):
        frame = self.solved.frame("kf_wide_1")
        spec = blender_builder.build_spec(build_scene(frame, self.plan, self.index), frame.camera, W, H, Path("/tmp/x"))
        grid = [o for o in spec["objects"] if o.get("grid")]
        self.assertTrue(grid)
        self.assertTrue(all("_grid" in o["name"] and not o["entity_id"] for o in grid))
        self.assertTrue(all(not o.get("grid") for o in spec["objects"] if "_grid" not in o["name"]))
        self.assertIn("init", spec["outputs"])
        script = blender_builder.BLENDER_SCRIPT
        self.assertIn('spec["outputs"]["init"]', script)
        self.assertIn("hide_render = True", script)

    def test_failed_blender_run_raises_with_log_tail(self):
        renderer = BlenderRenderer(W, H, blender=Path(__file__))
        failed = mock.Mock(returncode=1, stdout="", stderr="Traceback\nboom")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(renderer, "is_available", return_value=True), \
                mock.patch.object(blender_builder.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(blender_builder.BlenderError, "boom"):
                renderer.render(self.solved.frame("kf_wide_1"), self.plan, self.index, tmp)

    @unittest.skipUnless(os.environ.get("RUN_BLENDER_TESTS") == "1", "set RUN_BLENDER_TESTS=1 to render with Blender")
    def test_real_blender_matches_box_renderer(self):
        renderer = BlenderRenderer(384, 224)
        if not renderer.is_available():
            self.skipTest(f"Blender not found at {renderer.blender}")
        frame = self.solved.frame("kf_glass_2")
        frame = replace(frame, camera=replace(frame.camera, width=384, height=224))
        with tempfile.TemporaryDirectory() as tmp:
            blender = renderer.render(frame, self.plan, self.index, Path(tmp) / "blender")
            box = BoxRenderer(384, 224).render(frame, self.plan, self.index, Path(tmp) / "box")
            ids_b, ids_x = np.array(Image.open(blender.id_path)), np.array(Image.open(box.id_path))
            depth_b, depth_x = np.load(blender.depth_path), np.load(box.depth_path)
            init = np.array(Image.open(Path(tmp) / "blender" / "blocking.png")).astype(int)
            sharp = np.array(Image.open(Path(tmp) / "blender" / "blocking_sharp.png")).astype(int)
        self.assertGreater((np.abs(init - sharp).sum(axis=2) > 0).sum(), 100)     # the init has no grid lines
        for value, eid in box.id_map.items():
            a, b = ids_b == value, ids_x == value
            if b.sum() >= 50:
                self.assertGreater((a & b).sum() / (a | b).sum(), 0.9, eid)
        both = np.isfinite(depth_b) & np.isfinite(depth_x)
        self.assertGreater(both.mean(), 0.99)
        self.assertLess(float(np.median(np.abs(depth_b[both] - depth_x[both]))), 0.01)


if __name__ == "__main__":
    unittest.main()
