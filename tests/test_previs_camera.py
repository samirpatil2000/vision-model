"""Camera solver + pinhole projection (geometry plus low-res box renders; no models)."""
import dataclasses
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.previs import camera, conditioning, physics
from src.previs.blocking import build_scene, entity_aabb, mannequin
from src.previs.blocking.box_renderer import BoxRenderer
from src.previs.schema import ScenePlan
from src.previs.state import build_index
from src.previs.types import CameraState, EntityFrame, FrameState

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
W, H = 768, 448


def _fixture() -> ScenePlan:
    return ScenePlan.model_validate_json(FIXTURE.read_text())


TRAY = dict(id="pr_tray", name="serving tray", shape="cylinder", size_x_m=0.4, size_y_m=0.02, size_z_m=0.3,
            color_hex="#b8b8b8", movable=True)
CUP = dict(id="pr_cup", name="coffee cup", shape="cylinder", size_x_m=0.09, size_y_m=0.07, size_z_m=0.09,
           color_hex="#f2efe8", movable=True)


def _open_plan(*extra_props) -> ScenePlan:
    """The fixture moved to a 60 x 60 m exterior so no wall ever clamps the camera."""
    data = _fixture().model_dump()
    data["bible"]["locations"][0].update(kind="exterior", width_m=60.0, depth_m=60.0)
    data["bible"]["props"] += [dict(p) for p in extra_props]
    return ScenePlan.model_validate(data)


def _character(eid: str, pos, yaw: float = 0.0, height: float = 1.8, pose: str = "stand") -> EntityFrame:
    return EntityFrame(entity_id=eid, kind="character", present=True, position=tuple(pos), yaw_deg=yaw,
                       rotation_deg=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), pose=pose, expression="neutral",
                       holding_id="none", gaze_id="none", support="ground", support_id="none", flags=[],
                       weight_cue="none", secondary=[], size=(0.45, height, 0.3), color_hex="#556b2f",
                       secondary_color_hex="#808080", shape="character")


def _prop(eid: str, pos, size, support: str = "ground", support_id: str = "none") -> EntityFrame:
    return EntityFrame(entity_id=eid, kind="prop", present=True, position=tuple(float(v) for v in pos), yaw_deg=0.0,
                       rotation_deg=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), pose="none", expression="none",
                       holding_id="none", gaze_id="none", support=support, support_id=support_id, flags=[],
                       weight_cue="none", secondary=[], size=tuple(size), color_hex="#f2efe8",
                       secondary_color_hex="#f2efe8", shape="cylinder")


def _visible(plan: ScenePlan, index, frame: FrameState, cam: CameraState, entity_id: str) -> float:
    """Visible / drawn-alone pixels of an entity in a low-res box render (independent of the solver)."""
    renderer = BoxRenderer(192, 112)
    scene = build_scene(frame, plan, index)
    rast = renderer.rasterize(scene, dataclasses.replace(cam, width=192, height=112), grid=False)
    alone = int(rast.alone[entity_id].sum()) if entity_id in rast.alone else 0
    return float((rast.ids == scene.id_of(entity_id)).sum()) / alone if alone else 0.0


def _frame(entities, kf_id: str = "kf_wide_1", shot_id: str = "sh_wide") -> FrameState:
    return FrameState(kf_id=kf_id, shot_id=shot_id, order_in_shot=1, t_s=0.0, t_physical_s=0.0,
                      location_id="lo_kitchen", entities={e.entity_id: e for e in entities},
                      focus_event_id="none", depicted_phase="none", active_event_ids=[])


def _cam(yaw=0.0, pitch=0.0, roll=0.0, pos=(0.0, 1.0, 0.0), lens=50.0) -> CameraState:
    return CameraState(position=pos, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, lens_mm=lens,
                       vfov_deg=camera.vertical_fov_deg(lens), width=W, height=H, look_at=(0.0, 0.0, 0.0))


class TestProjection(unittest.TestCase):
    def test_vertical_fov_uses_36x24_sensor(self):
        self.assertAlmostEqual(camera.vertical_fov_deg(50.0), math.degrees(2 * math.atan(12.0 / 50.0)))
        self.assertGreater(camera.vertical_fov_deg(24.0), camera.vertical_fov_deg(85.0))

    def test_point_straight_ahead_lands_on_image_centre_with_positive_depth(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            cam = _cam(yaw=rng.uniform(-180, 180), pitch=rng.uniform(-80, 80), roll=rng.uniform(-20, 20),
                       pos=tuple(rng.uniform(-5, 5, 3)))
            _, _, forward = camera.camera_basis(cam)
            uv, depth = camera.project(cam, np.array(cam.position) + forward * 3.7)
            np.testing.assert_allclose(uv[0], (W / 2, H / 2), atol=1e-6)
            self.assertAlmostEqual(depth[0], 3.7, places=6)

    def test_image_axes_and_points_behind(self):
        cam = _cam(yaw=30.0, pitch=10.0, roll=0.0)
        right, up, forward = camera.camera_basis(cam)
        base = np.array(cam.position) + forward * 4.0
        uv, depth = camera.project(cam, np.stack([base + right * 0.5, base + up * 0.5, np.array(cam.position) - forward]))
        self.assertGreater(uv[0, 0], W / 2)          # +x right
        self.assertLess(uv[1, 1], H / 2)             # +y down in the image
        self.assertLess(depth[2], 0.0)
        self.assertTrue(np.isnan(uv[2]).all())

    def test_yaw_convention(self):
        # Yaw 0 looks along +z; yaw 90 looks along +x; image right is -x when looking along +z.
        np.testing.assert_allclose(camera.camera_basis(_cam(yaw=0.0))[2], (0, 0, 1), atol=1e-12)
        np.testing.assert_allclose(camera.camera_basis(_cam(yaw=90.0))[2], (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(camera.camera_basis(_cam(yaw=0.0))[0], (-1, 0, 0), atol=1e-12)


class TestFraming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = _open_plan()
        cls.index = build_index(cls.plan)
        cls.subject = _character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.8)
        cls.frame = _frame([cls.subject])

    def _shot(self, **update):
        base = dict(size="ms", angle="eye", frame_side="center", move="static", secondary_id="none", lens_mm=35.0,
                    subject_id="ch_mara")
        base.update(update)
        return self.index.shots["sh_wide"].model_copy(update=base)

    def _solve(self, shot, frame=None, **kw):
        frame = frame or self.frame
        return camera.solve_camera(shot, frame, self.index, W, H, frames=[frame], **kw)

    def test_framed_height_matches_shot_size(self):
        for size, framed in (("ws", 5.0), ("ms", 0.95), ("mcu", 0.6), ("cu", 0.38), ("ecu", 0.18)):
            cam = self._solve(self._shot(size=size))
            centre = np.array(cam.look_at)
            uv, depth = camera.project(cam, np.stack([centre + (0, framed / 2, 0), centre - (0, framed / 2, 0)]))
            self.assertAlmostEqual(uv[0, 1], 0.0, delta=1.0, msg=size)
            self.assertAlmostEqual(uv[1, 1], H, delta=1.0, msg=size)
            self.assertTrue((depth > 0).all())

    def _subject_bbox_height(self, cam):
        pts = np.array([np.array(end) + (0, dy, 0) for p0, p1, r, _ in mannequin.segments(self.subject)
                        for end in (p0, p1) for dy in (r, -r)])
        uv, _ = camera.project(cam, pts)
        return uv[:, 1].min(), uv[:, 1].max()

    def test_full_shot_fits_the_body_with_headroom(self):
        top, bottom = self._subject_bbox_height(self._solve(self._shot(size="fs")))
        self.assertAlmostEqual((bottom - top) / H, 1.0 / camera.FS_FACTOR, delta=0.05)
        self.assertGreater(top, 0.0)
        self.assertLess(bottom, H)

    def test_medium_and_close_shots_keep_the_head_near_the_top(self):
        for size, max_top in (("ms", 0.12), ("cu", 0.12)):
            top, bottom = self._subject_bbox_height(self._solve(self._shot(size=size)))
            self.assertGreater(top / H, -0.05, size)        # fixed-metre sizes may graze a tall head
            self.assertLess(top / H, max_top, size)
            self.assertGreater(bottom, H, size)             # the body continues below the frame

    def test_frame_side_puts_the_subject_on_a_third(self):
        aim = np.array([30.0, 0.72 * self.subject.size[1], 30.0])
        for side, expected in (("left", 1 / 3), ("center", 1 / 2), ("right", 2 / 3)):
            uv, _ = camera.project(self._solve(self._shot(size="ms", frame_side=side)), aim)
            self.assertAlmostEqual(uv[0, 0] / W, expected, delta=0.01, msg=side)

    def test_low_and_high_angles(self):
        low = self._solve(self._shot(size="ms", angle="low"))
        high = self._solve(self._shot(size="ms", angle="high"))
        self.assertGreater(low.pitch_deg, 5.0)
        self.assertLess(high.pitch_deg, -20.0)
        self.assertGreaterEqual(low.position[1], camera.MIN_CAMERA_HEIGHT_M)
        self.assertEqual(self._solve(self._shot(angle="dutch")).roll_deg, camera.DUTCH_ROLL_DEG)


class TestPlacementAndMoves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = _open_plan()
        cls.index = build_index(cls.plan)

    def _pair(self, subject="ch_mara", secondary="ch_jon", **update):
        update = dict(dict(size="ws", angle="eye", frame_side="center", move="static", subject_id=subject,
                           secondary_id=secondary, lens_mm=28.0), **update)
        return self.index.shots["sh_wide"].model_copy(update=update)

    def _two(self):
        return _frame([_character("ch_mara", (28.0, 0.0, 30.0), yaw=90.0, height=1.68),
                       _character("ch_jon", (32.0, 0.0, 30.0), yaw=-90.0, height=1.8)])

    def _side(self, cam):
        # Sign of the camera's position relative to the canonical jon -> mara line (z = 30 here).
        return math.copysign(1.0, cam.position[2] - 30.0)

    def test_line_side_is_a_property_of_the_pair(self):
        frame = self._two()
        solve = lambda shot: camera.solve_camera(shot, frame, self.index, W, H, frames=[frame])
        a1 = solve(self._pair("ch_mara", "ch_jon", line_side="a"))
        b1 = solve(self._pair("ch_mara", "ch_jon", line_side="b"))
        a2 = solve(self._pair("ch_jon", "ch_mara", line_side="a", size="mcu", lens_mm=50.0))
        self.assertNotEqual(self._side(a1), self._side(b1))
        self.assertEqual(self._side(a1), self._side(a2))   # swapping subject/secondary never crosses the line

    def test_two_shot_frames_both_characters(self):
        frame = self._two()
        cam = camera.solve_camera(self._pair(line_side="a"), frame, self.index, W, H, frames=[frame])
        heads = np.array([(28.0, 1.5, 30.0), (32.0, 1.6, 30.0)])
        uv, depth = camera.project(cam, heads)
        self.assertTrue((depth > 0).all())
        self.assertTrue(((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).all())

    def test_moves_by_progress(self):
        frame = _frame([_character("ch_mara", (30.0, 0.0, 30.0), height=1.8)])
        shot = lambda move: self.index.shots["sh_wide"].model_copy(update=dict(
            size="ms", angle="eye", frame_side="center", move=move, secondary_id="none", subject_id="ch_mara", lens_mm=24.0))
        solve = lambda move, p: camera.solve_camera(shot(move), frame, self.index, W, H, frames=[frame], move_progress=p)
        static = solve("static", 1.0)
        dist = lambda c: float(np.linalg.norm(np.array(c.position) - np.array(c.look_at)))
        self.assertAlmostEqual(dist(solve("dolly_in", 1.0)), dist(static) * (1 - camera.DOLLY_IN), places=4)
        self.assertAlmostEqual(dist(solve("dolly_out", 0.5)), dist(static) * (1 + camera.DOLLY_OUT * 0.5), places=4)
        self.assertAlmostEqual(solve("pan_right", 1.0).yaw_deg, static.yaw_deg - camera.PAN_DEG, places=4)
        self.assertAlmostEqual(solve("tilt_up", 0.5).pitch_deg, static.pitch_deg + camera.TILT_DEG * 0.5, places=4)
        orbit = solve("orbit_left", 1.0)
        self.assertAlmostEqual(dist(orbit), dist(static), places=4)
        self.assertAlmostEqual(camera._wrap(orbit.yaw_deg - static.yaw_deg), -camera.ORBIT_DEG, places=3)
        crane = solve("crane_up", 1.0)
        self.assertAlmostEqual(crane.position[1] - static.position[1], camera.CRANE_M, places=4)
        self.assertLess(crane.pitch_deg, static.pitch_deg)
        # Trucking slides sideways without re-aiming, but only as far as the subject stays in the safe frame
        # (a 1.5 m slide at an MS on 24 mm would push the subject out of frame).
        truck = solve("truck_right", 1.0)
        right, _, _ = camera.camera_basis(static)
        slide = np.array(truck.position) - static.position
        self.assertGreater(float(slide @ right), 0.2)
        self.assertLessEqual(float(slide @ right), camera.TRUCK_M + 1e-6)
        np.testing.assert_allclose(slide, right * float(slide @ right), atol=1e-4)
        self.assertAlmostEqual(truck.yaw_deg, static.yaw_deg, places=4)
        uv, _ = camera.project(truck, np.array(static.look_at))
        self.assertLess(abs(uv[0, 0] - W / 2), camera.SAFE_FRAME * W + 1.0)
        self.assertEqual(solve("static", 0.0), solve("static", 1.0))

    def test_handheld_is_deterministic_per_keyframe(self):
        frame = _frame([_character("ch_mara", (30.0, 0.0, 30.0))])
        shot = self.index.shots["sh_wide"].model_copy(update=dict(move="handheld", secondary_id="none", size="ms"))
        a = camera.solve_camera(shot, frame, self.index, W, H, frames=[frame])
        b = camera.solve_camera(shot, frame, self.index, W, H, frames=[frame])
        other = _frame([_character("ch_mara", (30.0, 0.0, 30.0))], kf_id="kf_wide_2")
        c = camera.solve_camera(shot, other, self.index, W, H, frames=[frame])
        self.assertEqual(a, b)
        self.assertNotEqual(a.position, c.position)
        self.assertLess(abs(camera._wrap(a.yaw_deg - c.yaw_deg)), 2 * camera.HANDHELD_ANGLE_DEG + 1e-9)


class TestFixtureCameras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = _fixture()
        cls.index = build_index(cls.plan)
        cls.solved = physics.solve(cls.plan)
        camera.solve_cameras(cls.plan, cls.solved, cls.index, W, H)

    def test_every_keyframe_gets_a_camera_inside_the_room(self):
        loc = self.index.locations["lo_kitchen"]
        for f in self.solved.frames:
            x, y, z = f.camera.position
            self.assertTrue(camera.WALL_MARGIN_M - 1e-9 <= x <= loc.width_m - camera.WALL_MARGIN_M + 1e-9, f.kf_id)
            self.assertTrue(camera.WALL_MARGIN_M - 1e-9 <= z <= loc.depth_m - camera.WALL_MARGIN_M + 1e-9, f.kf_id)
            self.assertGreaterEqual(y, camera.MIN_CAMERA_HEIGHT_M - 1e-9)

    def test_wide_shot_sees_both_characters_and_the_table(self):
        f = self.solved.frame("kf_wide_1")
        pts = np.array([f.entities["ch_mara"].position, f.entities["ch_jon"].position, f.entities["pr_table"].position])
        pts[:, 1] = 0.8
        uv, depth = camera.project(f.camera, pts)
        self.assertTrue((depth > 0).all())
        self.assertTrue(((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).all())

    def test_mcu_puts_jon_on_the_right_third(self):
        for kf in ("kf_jon_1", "kf_jon_2"):
            f = self.solved.frame(kf)
            head = next((np.array(a) + np.array(b)) / 2 for a, b, _, part in mannequin.segments(f.entities["ch_jon"])
                        if part == "head")
            uv, _ = camera.project(f.camera, head)
            self.assertAlmostEqual(uv[0, 0] / W, 2 / 3, delta=0.04, msg=kf)

    def test_shot_keeps_one_side_and_solving_is_deterministic(self):
        yaws = [self.solved.frame(k).camera.yaw_deg for k in ("kf_glass_1", "kf_glass_2", "kf_glass_3")]
        self.assertLess(max(abs(camera._wrap(y - yaws[0])) for y in yaws), 45.0)
        again = physics.solve(self.plan)
        camera.solve_cameras(self.plan, again, build_index(self.plan), W, H)
        for a, b in zip(self.solved.frames, again.frames):
            self.assertEqual(a.camera, b.camera)

    def test_sampled_frames_use_the_shot_side(self):
        shot = self.index.shots["sh_glass"]
        key = self.solved.frame("kf_glass_2")
        sampled = physics.sample_frame(self.plan, self.solved, key.t_s)
        cam = camera.solve_camera(shot, sampled, self.index, W, H)
        self.assertAlmostEqual(cam.yaw_deg, key.camera.yaw_deg, places=3)


def _solid_contains(scene, point, entity_ids=None) -> list:
    """Solid primitives (optionally of some entities) whose volume contains the point."""
    hits = []
    for p in scene.primitives:
        if not p.occluder or p.alpha < 1.0 or (entity_ids is not None and p.entity_id not in entity_ids):
            continue
        if p.kind == "capsule":
            a, b = np.array(p.p0), np.array(p.p1)
            s = min(max(float((point - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-12), 0.0), 1.0)
            inside = np.linalg.norm(point - (a + (b - a) * s)) < p.radius
        else:
            local = (point - np.array(p.center)) @ np.array(p.axes).T
            inside = bool(np.all(np.abs(local) <= np.array(p.half)))
        if inside:
            hits.append(f"{p.entity_id}/{p.part}")
    return hits


class TestHeldSubjects(unittest.TestCase):
    """A prop in someone's hands is filmed from the holder's front, never through the holder's body."""

    def test_glass_in_hand_is_seen_from_the_holders_front(self):
        plan = _open_plan()
        index = build_index(plan)
        shot = index.shots["sh_glass"].model_copy(update=dict(
            subject_id="pr_glass", secondary_id="none", size="insert", angle="eye", lens_mm=35.0, move="static"))
        for pose in ("carry", "stand", "lift"):
            mara = _character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.68, pose=pose)
            mara = dataclasses.replace(mara, holding_id="pr_glass")
            hand = np.array(physics.hand_position(mara, "right"))
            glass = _prop("pr_glass", hand - (0.0, 0.06, 0.0), (0.07, 0.12, 0.07), support="held", support_id="ch_mara")
            frame = _frame([mara, glass])
            cam = camera.solve_camera(shot, frame, index, W, H, frames=[frame])
            self.assertGreater(cam.position[2] - 30.0, 0.2, pose)              # in front of her (she faces +z)
            self.assertGreater(_visible(plan, index, frame, cam, "pr_glass"), 0.7, pose)

    def test_cup_on_a_carried_tray_is_not_hidden_by_the_waiter(self):
        # The cafe insert: a cup on the near edge of a tray a waiter carries, low angle, long lens.
        plan = _open_plan(TRAY, CUP)
        index = build_index(plan)
        waiter = dataclasses.replace(_character("ch_jon", (30.0, 0.0, 30.0), yaw=90.0, pose="carry"), holding_id="pr_tray")
        tray_at = np.array(physics.hand_position(waiter, "right")) - (0.0, 0.01, 0.0)
        tray = _prop("pr_tray", tray_at, (0.4, 0.02, 0.3), support="held", support_id="ch_jon")
        cup = _prop("pr_cup", tray_at + (-0.1, 0.02, 0.0), (0.09, 0.07, 0.09), support="on_entity", support_id="pr_tray")
        frame = _frame([waiter, tray, cup])
        shot = index.shots["sh_glass"].model_copy(update=dict(
            subject_id="pr_cup", secondary_id="none", size="insert", angle="low", lens_mm=65.0, move="static"))
        cam = camera.solve_camera(shot, frame, index, W, H, frames=[frame])
        self.assertGreater(_visible(plan, index, frame, cam, "pr_cup"), 0.9)


class TestSubjectBulk(unittest.TestCase):
    """The camera never ends up inside (or closer than the near plane to) the subject itself."""

    @classmethod
    def setUpClass(cls):
        cls.plan = _fixture()
        cls.index = build_index(cls.plan)
        cls.frame = physics.solve(cls.plan).frame("kf_wide_1")

    def test_close_shots_of_a_big_prop_frame_wider_instead_of_entering_it(self):
        scene = build_scene(self.frame, self.plan, self.index)
        for size, lens in (("ecu", 24.0), ("cu", 24.0), ("cu", 35.0), ("mcu", 24.0)):
            shot = self.index.shots["sh_glass"].model_copy(update=dict(
                subject_id="pr_table", secondary_id="none", size=size, angle="eye", lens_mm=lens, move="static"))
            cam = camera.solve_camera(shot, self.frame, self.index, W, H, frames=[self.frame])
            self.assertEqual(_solid_contains(scene, np.array(cam.position), {"pr_table"}), [], (size, lens))
            self.assertGreater(_visible(self.plan, self.index, self.frame, cam, "pr_table"), 0.9, (size, lens))

    def test_insert_of_a_tiny_prop_keeps_it_beyond_the_near_plane(self):
        data = self.plan.model_dump()
        for prop in data["bible"]["props"]:
            if prop["id"] == "pr_glass":
                prop.update(size_x_m=0.008, size_y_m=0.008, size_z_m=0.008)
        plan = ScenePlan.model_validate(data)
        index = build_index(plan)
        frame = dataclasses.replace(self.frame, entities=dict(self.frame.entities))
        frame.entities["pr_glass"] = dataclasses.replace(frame.entities["pr_glass"], size=(0.008, 0.008, 0.008))
        lo, hi = entity_aabb(frame.entities["pr_glass"])
        for lens in (24.0, 50.0):
            shot = index.shots["sh_glass"].model_copy(update=dict(
                subject_id="pr_glass", secondary_id="none", size="insert", angle="eye", lens_mm=lens, move="static"))
            cam = camera.solve_camera(shot, frame, index, W, H, frames=[frame])
            gap = float(np.linalg.norm(np.array(cam.position) - (lo + hi) / 2.0)) - float(np.linalg.norm(hi - lo)) / 2.0
            self.assertGreater(gap, camera.NEAR_M, lens)
            self.assertGreater(_visible(plan, index, frame, cam, "pr_glass"), 0.9, lens)


class TestDisplacingMoves(unittest.TestCase):
    """Crane / truck / orbit stop short of solids and of losing the subject."""

    @classmethod
    def setUpClass(cls):
        cls.plan = _fixture()
        cls.index = build_index(cls.plan)
        cls.frame = physics.solve(cls.plan).frame("kf_wide_1")
        cls.scene = build_scene(cls.frame, cls.plan, cls.index)

    def _shot(self, **update):
        base = dict(angle="eye", frame_side="center", secondary_id="none", lens_mm=35.0)
        return self.index.shots["sh_wide"].model_copy(update=dict(base, **update))

    def test_crane_down_never_enters_a_table_or_the_subject(self):
        for update in (dict(subject_id="ch_mara", size="mcu", move="crane_down"),
                       dict(subject_id="ch_jon", size="cu", angle="overhead", move="crane_down")):
            for p in (0.5, 1.0):
                cam = camera.solve_camera(self._shot(**update), self.frame, self.index, W, H, frames=[self.frame],
                                          move_progress=p)
                self.assertEqual(_solid_contains(self.scene, np.array(cam.position)), [], (update, p))

    def test_truck_keeps_a_tight_subject_in_frame(self):
        centre = (lambda lo, hi: (lo + hi) / 2.0)(*entity_aabb(self.frame.entities["pr_glass"]))
        for move in ("truck_left", "truck_right"):
            for p in (0.5, 1.0):
                cam = camera.solve_camera(self._shot(subject_id="pr_glass", size="insert", move=move), self.frame,
                                          self.index, W, H, frames=[self.frame], move_progress=p)
                uv, depth = camera.project(cam, centre)
                self.assertGreater(depth[0], camera.NEAR_M, (move, p))
                self.assertTrue(0.0 < uv[0, 0] < W and 0.0 < uv[0, 1] < H, (move, p, uv))


class TestVisibilityInTheLoop(unittest.TestCase):
    """The shot side is checked against a low-res id render of what the camera actually sees."""

    def setUp(self):
        camera._SIDE_CACHE.clear()
        self.addCleanup(camera._SIDE_CACHE.clear)
        self.plan = _open_plan()
        self.index = build_index(self.plan)
        self.shot = self.index.shots["sh_wide"].model_copy(update=dict(
            subject_id="ch_mara", secondary_id="none", size="ms", angle="eye", move="static", lens_mm=40.0,
            frame_side="center", line_side="a"))

    def test_a_person_half_in_front_of_the_subject_moves_the_camera(self):
        # Jon stands between the nominal camera (25 deg off Mara's facing) and Mara, offset so that the
        # analytic sight lines (aim point, head top) pass beside him while he hides a third of her.
        toward = np.array([math.sin(math.radians(25.0)), 0.0, math.cos(math.radians(25.0))])
        side = np.array([toward[2], 0.0, -toward[0]])
        mara = _character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.78)
        jon = _character("ch_jon", np.array([30.0, 0.0, 30.0]) + toward * 0.6 + side * 0.25, yaw=180.0, height=1.66)
        frame = _frame([mara, jon])
        cam = camera.solve_camera(self.shot, frame, self.index, W, H, frames=[frame])
        self.assertGreater(_visible(self.plan, self.index, frame, cam, "ch_mara"), 0.9)

    def test_scan_follows_the_render_deterministically(self):
        # A stand-in render says the subject is hidden whenever the camera is on the +x side of her.
        frame = _frame([_character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.78)])
        free = camera.solve_camera(self.shot, frame, self.index, W, H, frames=[frame])
        self.assertGreater(free.position[0], 30.0)                  # nominal: 25 deg toward +x
        camera._SIDE_CACHE.clear()
        hidden_on_plus_x = lambda scene, cam, ids, key_parts_only=True: {
            e: (0.1 if cam.position[0] > 30.0 else 1.0) for e in ids}
        with mock.patch.object(camera, "visible_fractions", side_effect=hidden_on_plus_x):
            first = camera.solve_camera(self.shot, frame, self.index, W, H, frames=[frame])
            camera._SIDE_CACHE.clear()
            second = camera.solve_camera(self.shot, frame, self.index, W, H, frames=[frame])
        self.assertLess(first.position[0], 30.0)
        self.assertEqual(first, second)

    def test_legs_behind_furniture_do_not_count_as_hidden(self):
        plan = _open_plan()
        index = build_index(plan)
        mara = _character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.78)
        low_table = dataclasses.replace(_prop("pr_table", (30.0, 0.0, 30.6), (1.4, 0.6, 0.4)), shape="box")
        frame = _frame([mara, low_table])
        cam = camera.solve_camera(self.shot.model_copy(update=dict(size="fs")), frame, index, W, H, frames=[frame])
        scene = build_scene(frame, plan, index)
        key = camera.visible_fractions(scene, cam, ["ch_mara"])["ch_mara"]
        whole = camera.visible_fractions(scene, cam, ["ch_mara"], key_parts_only=False)["ch_mara"]
        self.assertGreater(key, 0.9)
        self.assertLess(whole, key)


class TestCharacterFraming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = _open_plan()
        cls.index = build_index(cls.plan)

    def test_headroom_holds_on_tight_shots_and_through_a_dolly_in(self):
        leo = _character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.78)
        frame = _frame([leo])
        segments = mannequin.segments(leo)
        top = max(max(p0[1], p1[1]) + r for p0, p1, r, part in segments if part in ("head", "hair"))
        head = next((np.array(a) + np.array(b)) / 2.0 for a, b, _, part in segments if part == "head")
        for size in ("fs", "ms", "mcu", "cu"):
            for move, p in (("static", 0.0), ("dolly_in", 1.0)):
                shot = self.index.shots["sh_wide"].model_copy(update=dict(
                    subject_id="ch_mara", secondary_id="none", size=size, angle="eye", move=move, lens_mm=40.0,
                    frame_side="center"))
                cam = camera.solve_camera(shot, frame, self.index, W, H, frames=[frame], move_progress=p)
                uv, _ = camera.project(cam, np.array([head[0], top, head[2]]))
                self.assertGreater(uv[0, 1] / H, 0.05, (size, move))      # whole head plus headroom in frame
                self.assertLess(uv[0, 1] / H, 0.25, (size, move))

    def test_line_side_behind_the_face_still_shows_the_face_side(self):
        # The cafe catch: Ana faces -z; the line to the cup she catches puts side "a" behind her, where no
        # 3/4-front view exists. The camera takes the most frontal azimuth the side allows (a profile).
        ana = _character("ch_mara", (30.0, 0.0, 30.0), yaw=180.0, height=1.66)
        cup = _prop("pr_glass", (29.32, 0.62, 29.65), (0.07, 0.12, 0.07), support="airborne")
        frame = _frame([ana, cup])
        shot = self.index.shots["sh_wide"].model_copy(update=dict(
            subject_id="ch_mara", secondary_id="pr_glass", size="ms", angle="eye", move="static", lens_mm=50.0,
            frame_side="right", line_side="a"))
        cam = camera.solve_camera(shot, frame, self.index, W, H, frames=[frame])
        to_cam = np.array(cam.position) - np.array(ana.position)
        face = np.array([math.sin(math.radians(180.0)), 0.0, math.cos(math.radians(180.0))])
        angle = math.degrees(math.acos(float(to_cam[[0, 2]] @ face[[0, 2]]) / np.linalg.norm(to_cam[[0, 2]])))
        self.assertLess(angle, 90.0)
        normal = camera._line_normal(shot, frame)                   # still on side "a" of the line
        self.assertGreater(float(to_cam[[0, 2]] @ normal[[0, 2]]), 0.0)


CHAIR = dict(id="pr_chair", name="wooden chair", shape="box", size_x_m=0.45, size_y_m=0.46, size_z_m=0.45,
             color_hex="#5a4632", movable=True)


def _drawn_head(scene, entity_id: str):
    """(centre, top) of the head blocking actually draws for a character (from the scene's primitives)."""
    caps = [p for p in scene.primitives if p.entity_id == entity_id and p.part == "head"]
    centre = (np.array(caps[0].p0) + np.array(caps[0].p1)) / 2.0
    top = max(max(p.p0[1], p.p1[1]) + p.radius for p in scene.primitives
              if p.entity_id == entity_id and p.part in ("head", "hair", "ear_l", "ear_r"))
    return centre, top


class TestFramingUsesDrawnGeometry(unittest.TestCase):
    """The camera frames what blocking draws: a quadruped's body, a seated character on its seat."""

    def _shot(self, index, **update):
        base = dict(secondary_id="none", angle="eye", move="static", lens_mm=50.0, frame_side="center", line_side="a")
        return index.shots["sh_wide"].model_copy(update=dict(base, **update))

    def test_close_shot_of_a_dog_frames_the_dogs_head(self):
        # The solved frame of a quadruped still says shape "character"; build_scene draws it on four legs
        # with the head forward at shoulder height (a 0.5 m human would have its head straight above the base).
        data = _open_plan().model_dump()
        for ch in data["bible"]["characters"]:
            if ch["id"] == "ch_jon":
                ch.update(body="quadruped", height_m=0.5)
        plan = ScenePlan.model_validate(data)
        index = build_index(plan)
        dog = _character("ch_jon", (30.0, 0.0, 30.0), yaw=0.0, height=0.5)
        frame = _frame([dog])
        scene = build_scene(frame, plan, index)
        head, top = _drawn_head(scene, "ch_jon")
        for size in ("cu", "mcu"):
            cam = camera.solve_camera(self._shot(index, subject_id="ch_jon", size=size), frame, index, W, H,
                                      frames=[frame])
            uv, depth = camera.project(cam, np.stack([head, np.array([head[0], top, head[2]])]))
            self.assertGreater(depth[0], 0.0, size)
            self.assertLess(abs(uv[0, 0] / W - 0.5), 0.15, (size, uv[0]))      # the head, centred
            self.assertTrue(0.15 < uv[0, 1] / H < 0.85, (size, uv[0]))
            self.assertGreater(uv[1, 1] / H, 0.0, size)                          # the top of the head in frame

    def test_tight_shot_of_a_seated_character_keeps_her_head_in_frame(self):
        # Physics keeps a seated character's base (feet) on the floor; blocking draws her hips on the chair.
        plan = _open_plan(CHAIR)
        index = build_index(plan)
        chair = dataclasses.replace(_prop("pr_chair", (30.0, 0.0, 29.9), (0.45, 0.46, 0.45)), shape="box")
        ana = dataclasses.replace(_character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.66, pose="sit"),
                                  support="on_entity", support_id="pr_chair")
        frame = _frame([ana, chair])
        scene = build_scene(frame, plan, index)
        head, top = _drawn_head(scene, "ch_mara")
        self.assertGreater(head[1], 1.0)                                         # drawn on the chair, not the floor
        for size in ("ms", "mcu", "cu"):
            cam = camera.solve_camera(self._shot(index, subject_id="ch_mara", size=size), frame, index, W, H,
                                      frames=[frame])
            uv, _ = camera.project(cam, np.array([head[0], top, head[2]]))
            self.assertGreater(uv[0, 1] / H, 0.05, size)                         # whole head plus headroom
            self.assertLess(uv[0, 1] / H, 0.3, size)
        # A two-shot with the seated character as the secondary frames both heads as drawn.
        jon = _character("ch_jon", (31.6, 0.0, 30.0), yaw=-90.0, height=1.8)
        pair = _frame([ana, chair, jon])
        scene = build_scene(pair, plan, index)
        cam = camera.solve_camera(self._shot(index, subject_id="ch_jon", secondary_id="ch_mara", size="ws", lens_mm=28.0),
                                  pair, index, W, H, frames=[pair])
        uv, depth = camera.project(cam, np.stack([_drawn_head(scene, "ch_mara")[0], _drawn_head(scene, "ch_jon")[0]]))
        self.assertTrue((depth > 0).all())
        self.assertTrue(((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).all(), uv)


class TestStaticShotsAreLockedOff(unittest.TestCase):
    """A static shot has ONE camera for all its keyframes (conditioning's static lock needs identical
    cameras), framing the union of its subject over them."""

    def setUp(self):
        camera._SIDE_CACHE.clear()
        self.addCleanup(camera._SIDE_CACHE.clear)

    def test_fixture_wide_keyframes_share_one_camera(self):
        plan = _fixture()
        index = build_index(plan)
        solved = physics.solve(plan)
        camera.solve_cameras(plan, solved, index, W, H)
        first, second = solved.frame("kf_wide_1"), solved.frame("kf_wide_2")
        self.assertEqual(first.camera, second.camera)
        self.assertTrue(conditioning.cameras_match(first.camera, second.camera))
        # Animatic frames in between get the same camera (no drift between the keyframes either).
        sampled = physics.sample_frame(plan, solved, (first.t_s + second.t_s) / 2.0)
        self.assertEqual(camera.solve_camera(index.shots["sh_wide"], sampled, index, W, H), first.camera)
        # Moving shots still follow their subject per keyframe.
        self.assertNotEqual(solved.frame("kf_jon_1").camera, solved.frame("kf_jon_2").camera)

    def test_a_subject_that_moves_in_a_static_shot_stays_framed_by_one_camera(self):
        plan = _open_plan()
        index = build_index(plan)
        shot = index.shots["sh_wide"].model_copy(update=dict(
            subject_id="ch_mara", secondary_id="none", size="ms", angle="eye", move="static", lens_mm=35.0,
            frame_side="center", line_side="a"))
        frames = [dataclasses.replace(_frame([_character("ch_mara", (30.0 + dx, 0.0, 30.0), yaw=0.0, height=1.7)],
                                             kf_id=f"kf_wide_{k}"), order_in_shot=k, t_s=float(k))
                  for k, dx in ((1, 0.0), (2, 1.5))]
        cams = [camera.solve_camera(shot, f, index, W, H, frames=frames) for f in frames]
        self.assertEqual(cams[0], cams[1])
        for f in frames:                                    # each keyframe's head (and its top) is in frame
            head = next((np.array(a) + np.array(b)) / 2.0 for a, b, _, part in mannequin.segments(f.entities["ch_mara"])
                        if part == "head")
            uv, depth = camera.project(cams[0], np.stack([head, head + (0.0, 0.12, 0.0)]))
            self.assertTrue((depth > 0).all(), f.kf_id)
            self.assertTrue(((uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)).all(), (f.kf_id, uv))

    def test_a_still_subject_keeps_its_single_keyframe_framing(self):
        plan = _open_plan()
        index = build_index(plan)
        shot = index.shots["sh_wide"].model_copy(update=dict(
            subject_id="ch_mara", secondary_id="none", size="mcu", angle="eye", move="static", lens_mm=50.0))
        one = _frame([_character("ch_mara", (30.0, 0.0, 30.0), yaw=0.0, height=1.7)])
        two = dataclasses.replace(one, kf_id="kf_wide_2", order_in_shot=2, t_s=1.0)
        alone = camera.solve_camera(shot, one, index, W, H, frames=[one])
        camera._SIDE_CACHE.clear()
        pinned = camera.solve_camera(shot, two, index, W, H, frames=[one, two])
        np.testing.assert_allclose(pinned.position, alone.position, atol=1e-5)
        self.assertAlmostEqual(pinned.yaw_deg, alone.yaw_deg, places=4)
        self.assertAlmostEqual(pinned.pitch_deg, alone.pitch_deg, places=4)


if __name__ == "__main__":
    unittest.main()
