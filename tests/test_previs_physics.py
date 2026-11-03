import copy
import json
import math
import unittest
from pathlib import Path

from src.previs import physics
from src.previs.schema import ScenePlan
from src.previs.types import EntityFrame

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
FIXTURE_DATA = json.loads(FIXTURE.read_text())
TABLE_TOP = 0.75
FLOOR_GLASS = (3.4, 0.0, 1.75)


def fixture_data() -> dict:
    return copy.deepcopy(FIXTURE_DATA)


def timeline_of(data: dict, shot_id: str) -> dict:
    return next(tl for tl in data["timeline"]["shots"] if tl["shot_id"] == shot_id)


def event_of(data: dict, event_id: str) -> dict:
    return next(ev for tl in data["timeline"]["shots"] for ev in tl["events"] if ev["id"] == event_id)


def shot_of(data: dict, shot_id: str) -> dict:
    return next(s for s in data["bible"]["shots"] if s["id"] == shot_id)


def make_event(event_id: str, start_kf_id: str, kind: str, **overrides) -> dict:
    event = {
        "justification": "synthetic test event", "id": event_id, "start_kf_id": start_kf_id, "kind": kind,
        "actor_id": "none", "target_id": "none", "cause_event_id": "none", "contact_part": "none",
        "dir_ref": "none", "strength": "moderate", "outcome_support_id": "none", "outcome_mark_id": "none",
        "outcome_rest": "none", "outcome_flag": "none",
    }
    event.update(overrides)
    return event


def solve(data: dict, **kw):
    plan = ScenePlan.model_validate(data)
    return plan, physics.solve(plan, **kw)


def with_crate(friction: str, strength: str) -> dict:
    """Fixture plus a 4 kg crate Mara shoves 1 m across the floor at kf_wide_1."""
    data = fixture_data()
    bible = data["bible"]
    bible["props"].append({"id": "pr_crate", "name": "crate", "shape": "box", "size_x_m": 0.5, "size_y_m": 0.5,
                           "size_z_m": 0.5, "color_hex": "#996633", "movable": True})
    bible["marks"] += [{"id": "mk_crate", "location_id": "lo_kitchen", "name": "crate", "x_m": 1.5, "z_m": 1.0},
                       {"id": "mk_crate_end", "location_id": "lo_kitchen", "name": "crate end", "x_m": 0.5, "z_m": 1.0}]
    bible["phys_bodies"].append({"entity_id": "pr_crate", "regime": "rigid", "material": "wood", "mass_class": "m",
                                 "friction": friction, "restitution": "low", "failure_mode": "none",
                                 "container_fill": "none"})
    init = copy.deepcopy(next(s for s in bible["initial_states"] if s["entity_id"] == "pr_table"))
    init.update(entity_id="pr_crate", mark_id="mk_crate")
    bible["initial_states"].append(init)
    timeline_of(data, "sh_wide")["events"].append(make_event(
        "ev_shove", "kf_wide_1", "push", actor_id="ch_mara", target_id="pr_crate", strength=strength,
        dir_ref="away_from_actor", outcome_mark_id="mk_crate_end", outcome_rest="rest"))
    return data


def make_state(kf_id: str, entity_id: str, **overrides) -> dict:
    state = {
        "kf_id": kf_id, "entity_id": entity_id, "present": True, "mark_id": "none", "offset_x_m": 0.0,
        "offset_z_m": 0.0, "facing_id": "none", "pose": "stand", "expression": "neutral", "holding_id": "none",
        "gaze_id": "none", "support": "ground", "support_id": "none", "flags": [], "weight_cue": "none",
        "secondary": [],
    }
    state.update(overrides)
    return state


def make_keyframe(kf_id: str, order: int, focus: str = "none", phase: str = "none") -> dict:
    return {"id": kf_id, "order": order, "beat_id": "bt_react", "action": "test", "dialogue": "",
            "move_progress": 0.0, "focus_event_id": focus, "depicted_phase": phase}


def multi_event_plan() -> dict:
    """
    Fixture plus two shots exercising chained events: Jon walks to the table, lifts a ball, tosses it to
    Mara who catches it and places it on the table; then a kid enters, runs to the table and exits.
    """
    data = fixture_data()
    bible = data["bible"]
    bible["marks"].append({"id": "mk_by_table", "location_id": "lo_kitchen", "name": "by table", "x_m": 3.4, "z_m": 1.4})
    bible["marks"].append({"id": "mk_door", "location_id": "lo_kitchen", "name": "door", "x_m": 0.3, "z_m": 2.0})
    bible["props"].append({"id": "pr_ball", "name": "ball", "shape": "sphere", "size_x_m": 0.2, "size_y_m": 0.2,
                           "size_z_m": 0.2, "color_hex": "#ff0000", "movable": True})
    bible["characters"].append({"id": "ch_kid", "name": "Kid", "height_m": 1.2, "appearance": "a child",
                                "wardrobe": "t-shirt", "palette_hex": ["#00ff00", "#0000ff"]})
    bible["phys_bodies"] += [
        {"entity_id": "pr_ball", "regime": "rigid", "material": "rubber", "mass_class": "s", "friction": "med",
         "restitution": "high", "failure_mode": "none", "container_fill": "none"},
        {"entity_id": "ch_kid", "regime": "articulated", "material": "flesh", "mass_class": "m", "friction": "high",
         "restitution": "dead", "failure_mode": "none", "container_fill": "none"}]
    bible["initial_states"] += [
        make_state("init", "pr_ball", mark_id="mk_table", offset_x_m=0.6, support="on_entity", support_id="pr_table",
                   pose="none", expression="none"),
        make_state("init", "ch_kid", present=False)]
    for shot_id, order, subject, count, duration in (("sh_play", 4, "ch_jon", 5, "medium"), ("sh_kid", 5, "ch_kid", 3, "short")):
        shot = copy.deepcopy(shot_of(data, "sh_wide"))
        shot.update(id=shot_id, order=order, setup_id="su_play", subject_id=subject, keyframe_count=count,
                    duration_class=duration, secondary_id="ch_mara" if subject == "ch_jon" else "none")
        bible["shots"].append(shot)
    data["timeline"]["shots"].append({
        "shot_id": "sh_play",
        "keyframes": [make_keyframe("kf_play_1", 1, "ev_walk", "in_progress"), make_keyframe("kf_play_2", 2, "ev_lift", "contact"),
                      make_keyframe("kf_play_3", 3, "ev_throw", "apex"), make_keyframe("kf_play_4", 4, "ev_catch", "rest"),
                      make_keyframe("kf_play_5", 5, "ev_place", "release")],
        "events": [
            make_event("ev_walk", "kf_play_1", "walk_to", actor_id="ch_jon", outcome_mark_id="mk_by_table", outcome_rest="rest"),
            make_event("ev_lift", "kf_play_2", "lift", actor_id="ch_jon", target_id="pr_ball", outcome_rest="held",
                       cause_event_id="ev_walk"),
            make_event("ev_throw", "kf_play_3", "throw", actor_id="ch_jon", target_id="pr_ball", strength="gentle",
                       outcome_rest="held", outcome_support_id="ch_mara"),
            make_event("ev_catch", "kf_play_4", "catch", actor_id="ch_mara", target_id="pr_ball", outcome_rest="held",
                       cause_event_id="ev_throw"),
            make_event("ev_place", "kf_play_5", "place", actor_id="ch_mara", target_id="pr_ball", outcome_rest="rest",
                       outcome_support_id="pr_table")],
        "states": [
            make_state("kf_play_1", "ch_jon", mark_id="mk_counter", pose="walk", facing_id="mk_by_table"),
            make_state("kf_play_2", "ch_jon", mark_id="mk_by_table", pose="reach", facing_id="pr_ball"),
            make_state("kf_play_3", "ch_jon", mark_id="mk_by_table", pose="throw", facing_id="ch_mara"),
            make_state("kf_play_4", "ch_mara", mark_id="mk_mara", pose="carry", facing_id="ch_jon", holding_id="pr_ball")],
        "cause_links": [{"id": "cl_throw_catch", "cause_event_id": "ev_throw", "effect_event_id": "ev_catch",
                         "kind": "causes"}]})
    data["timeline"]["shots"].append({
        "shot_id": "sh_kid",
        "keyframes": [make_keyframe("kf_kid_1", 1, "ev_enter", "in_progress"), make_keyframe("kf_kid_2", 2, "ev_run", "rest"),
                      make_keyframe("kf_kid_3", 3, "ev_exit", "in_progress")],
        "events": [
            make_event("ev_enter", "kf_kid_1", "enter", actor_id="ch_kid", outcome_mark_id="mk_door", outcome_rest="rest"),
            make_event("ev_run", "kf_kid_2", "run_to", actor_id="ch_kid", outcome_mark_id="mk_by_table", outcome_rest="rest"),
            make_event("ev_exit", "kf_kid_3", "exit", actor_id="ch_kid")],
        "states": [], "cause_links": []})
    return data


def long_plan(extra_shots: int = 7) -> dict:
    """Fixture plus copies of sh_jon (5 s 'long' holds each), well over 30 s at full holds."""
    data = fixture_data()
    base_shot = shot_of(data, "sh_jon")
    base_tl = timeline_of(data, "sh_jon")
    for n in range(2, 2 + extra_shots):
        shot = copy.deepcopy(base_shot)
        shot.update(id=f"sh_jon{n}", order=10 + n, duration_class="long")
        data["bible"]["shots"].append(shot)
        tl = copy.deepcopy(base_tl)
        tl.update(shot_id=f"sh_jon{n}", events=[], cause_links=[])
        for kf in tl["keyframes"]:
            kf["id"] = kf["id"].replace("kf_jon_", f"kf_jon{n}_")
            kf.update(focus_event_id="none", depicted_phase="none")
        for st in tl["states"]:
            st["kf_id"] = st["kf_id"].replace("kf_jon_", f"kf_jon{n}_")
        data["timeline"]["shots"].append(tl)
    return data


class TestClosedForms(unittest.TestCase):
    def test_fall_time(self):
        self.assertAlmostEqual(physics.fall_time(0.75), math.sqrt(2 * 0.75 / 9.81), places=12)
        self.assertAlmostEqual(physics.fall_time(4.905), 1.0, places=9)
        self.assertEqual(physics.fall_time(0.0), 0.0)

    def test_throw_back_solve_lands_on_target(self):
        release, landing = (0.0, 1.4, 0.0), (2.0, 0.0, 1.0)
        sol = physics.solve_throw(release, landing, "gentle", 0.5)
        self.assertTrue(sol.feasible, sol.note)
        self.assertEqual(sol.angle_deg, physics.THROW_ANGLE_DEG["gentle"])
        t = sol.flight_time_s
        end = (release[0] + sol.velocity[0] * t, release[1] + sol.velocity[1] * t - 0.5 * physics.G * t * t,
               release[2] + sol.velocity[2] * t)
        for got, want in zip(end, landing):
            self.assertAlmostEqual(got, want, places=9)

    def test_throw_feasibility_depends_on_strength(self):
        release, far = (0.0, 1.4, 0.0), (12.0, 0.0, 0.0)
        weak = physics.solve_throw(release, far, "gentle", 0.5)
        self.assertFalse(weak.feasible)
        self.assertIn("at most", weak.note)
        strong = physics.solve_throw(release, far, "violent", 0.5)
        self.assertTrue(strong.feasible, strong.note)
        self.assertLessEqual(strong.speed_ms, physics.strength_speed_range("violent", 0.5)[1] + 1e-9)

    def test_slide_distance_vs_friction(self):
        for friction in ("low", "med", "high"):
            v0 = physics.slide_speed(1.0, friction)
            self.assertAlmostEqual(v0, math.sqrt(2 * physics.FRICTION_MU[friction] * physics.G))
            self.assertAlmostEqual(physics.slide_distance(v0, friction), 1.0)
        self.assertLess(physics.slide_speed(1.0, "low"), physics.slide_speed(1.0, "high"))
        self.assertGreater(physics.slide_distance(2.0, "high", roll=True), physics.slide_distance(2.0, "high"))

    def test_hand_position_by_pose(self):
        def frame(pose: str, yaw: float) -> EntityFrame:
            return EntityFrame(
                entity_id="ch_a", kind="character", present=True, position=(1.0, 0.0, 1.0), yaw_deg=yaw,
                rotation_deg=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), pose=pose, expression="neutral",
                holding_id="none", gaze_id="none", support="ground", support_id="none", flags=[], weight_cue="none",
                secondary=[], size=(0.45, 2.0, 0.3), color_hex="#000000", secondary_color_hex="#000000",
                shape="character")
        reach, carry, throw = (physics.hand_position(frame(p, 0.0)) for p in ("reach", "carry", "throw"))
        self.assertAlmostEqual(reach[1], 0.55 * 2.0)
        self.assertAlmostEqual(reach[2], 1.0 + 0.55)
        self.assertAlmostEqual(carry[1], 0.45 * 2.0)
        self.assertAlmostEqual(carry[2], 1.0 + 0.30)
        self.assertAlmostEqual(throw[1], 0.85 * 2.0)
        turned = physics.hand_position(frame("reach", 90.0))   # facing +x
        self.assertAlmostEqual(turned[0], 1.0 + 0.55)
        left = physics.hand_position(frame("carry", 0.0), side="left")
        self.assertAlmostEqual(left[0] - 1.0, -(carry[0] - 1.0))


class TestSolveFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.solved = solve(fixture_data())

    def test_solves_cleanly_within_limit(self):
        s = self.solved
        self.assertEqual(s.warnings, [])
        self.assertLessEqual(s.total_duration_s, 30.0)
        self.assertEqual([f.kf_id for f in s.frames],
                         ["kf_wide_1", "kf_wide_2", "kf_glass_1", "kf_glass_2", "kf_glass_3", "kf_jon_1", "kf_jon_2"])
        times = [f.t_s for f in s.frames]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])), times)
        self.assertEqual(s.shots[0].start_t_s, 0.0)
        for a, b in zip(s.shots, s.shots[1:]):
            self.assertAlmostEqual(a.end_t_s, b.start_t_s)
        self.assertAlmostEqual(s.shots[-1].end_t_s, s.total_duration_s)
        self.assertTrue(all(ev.feasible for ev in s.events.values()))
        for f in s.frames:
            self.assertEqual(set(f.entities), {"ch_mara", "ch_jon", "pr_table", "pr_glass"})

    def test_glass_airborne_mid_fall(self):
        glass = self.solved.frame("kf_glass_2").entities["pr_glass"]
        self.assertGreater(glass.position[1], 0.05)
        self.assertLess(glass.position[1], TABLE_TOP)
        self.assertEqual(glass.support, "airborne")
        self.assertTrue(glass.in_motion)
        self.assertLess(glass.velocity[1], 0.0)
        self.assertEqual(len(glass.trajectory), physics.TRAJECTORY_SAMPLES)
        self.assertAlmostEqual(glass.trajectory[0][1], TABLE_TOP)
        self.assertAlmostEqual(glass.trajectory[-1][1], 0.0, places=6)
        self.assertNotEqual(glass.rotation_deg, (0.0, 0.0, 0.0))   # tumbling
        self.assertIn("ev_fall", self.solved.frame("kf_glass_2").active_event_ids)

    def test_glass_at_rest_shattered_on_floor(self):
        frame = self.solved.frame("kf_glass_3")
        glass = frame.entities["pr_glass"]
        for got, want in zip(glass.position, FLOOR_GLASS):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(glass.support, "ground")
        self.assertFalse(glass.in_motion)
        self.assertTrue({"shattered", "spilled"} <= set(glass.flags))
        self.assertEqual(set(frame.active_event_ids), {"ev_shatter", "ev_spill"})

    def test_fall_time_and_causal_chain(self):
        ev = self.solved.events
        t_fall = math.sqrt(2 * TABLE_TOP / physics.G)
        self.assertAlmostEqual(ev["ev_fall"].phase_times["impact"], t_fall, places=9)
        f1, f3 = self.solved.frame("kf_glass_1"), self.solved.frame("kf_glass_3")
        self.assertAlmostEqual(f3.t_physical_s - f1.t_physical_s, t_fall, places=9)
        # Effects start at their causes' trigger instants; Jon reacts after the shatter.
        self.assertAlmostEqual(ev["ev_fall"].start_t_physical, ev["ev_knock"].end_t_physical, places=9)
        self.assertAlmostEqual(ev["ev_shatter"].start_t_physical, ev["ev_fall"].start_t_physical + t_fall, places=9)
        self.assertAlmostEqual(ev["ev_spill"].start_t_physical, ev["ev_shatter"].start_t_physical, places=9)
        self.assertGreaterEqual(ev["ev_turn"].start_t_physical,
                                ev["ev_shatter"].start_t_physical + physics.REACTION_S - 1e-9)
        # The fall starts where the push left the glass: on the table's edge.
        start = self.solved.frame("kf_glass_1").entities["pr_glass"].position
        self.assertAlmostEqual(start[0], 3.2, places=6)
        self.assertAlmostEqual(start[1], TABLE_TOP, places=6)

    def test_held_props_and_yaw(self):
        jon = self.solved.frame("kf_jon_2").entities["ch_jon"]
        glass = self.solved.frame("kf_jon_2").entities["pr_glass"]
        self.assertAlmostEqual(jon.yaw_deg, physics.yaw_toward(jon.position, glass.position), places=6)
        table = self.solved.frame("kf_wide_1").entities["pr_glass"]
        self.assertAlmostEqual(table.position[1], TABLE_TOP)


class TestSolveVariants(unittest.TestCase):
    def test_time_scale_stretches_edited_time_only(self):
        gaps = {}
        for scale in ("slow_2x", "slow_4x", "real"):
            data = fixture_data()
            shot_of(data, "sh_glass")["time_scale"] = scale
            _, solved = solve(data)
            f1, f3 = solved.frame("kf_glass_1"), solved.frame("kf_glass_3")
            gaps[scale] = (f3.t_s - f1.t_s, f3.t_physical_s - f1.t_physical_s)
        self.assertAlmostEqual(gaps["slow_4x"][0], 2 * gaps["slow_2x"][0], places=9)
        self.assertAlmostEqual(gaps["slow_4x"][1], gaps["slow_2x"][1], places=9)
        self.assertAlmostEqual(gaps["real"][1], gaps["slow_2x"][1], places=9)
        # At real speed the 0.39 s fall is ramped so its three keyframes stay 0.25 s apart on screen.
        self.assertAlmostEqual(gaps["real"][0], 2 * physics.MIN_KF_GAP_S, places=9)

    def test_long_cut_is_compressed_to_limit(self):
        data = long_plan()
        plan = ScenePlan.model_validate(data)
        uncompressed = physics._schedule(physics._prepare(plan), 1.0)
        self.assertGreater(uncompressed.total, 30.0)
        solved = physics.solve(plan)
        self.assertLessEqual(solved.total_duration_s, 30.0 + 1e-6)
        self.assertGreater(solved.total_duration_s, 29.0)        # holds shrink only as much as needed
        self.assertFalse(any(w.startswith("duration") for w in solved.warnings))
        times = [f.t_s for f in solved.frames]
        self.assertTrue(all(b - a >= physics.MIN_KF_GAP_S - 1e-9 for a, b in zip(times, times[1:])))
        # Physics-determined spacing is untouched by compression.
        f1, f3 = solved.frame("kf_glass_1"), solved.frame("kf_glass_3")
        self.assertAlmostEqual(f3.t_s - f1.t_s, 2 * math.sqrt(2 * TABLE_TOP / physics.G), places=9)

    def test_too_long_even_compressed_warns(self):
        _, solved = solve(fixture_data(), max_duration_s=1.0)
        self.assertGreater(solved.total_duration_s, 1.0)
        self.assertTrue(any(w.startswith("duration:") for w in solved.warnings))

    def test_slide_feasibility_and_duration_follow_friction(self):
        _, low = solve(with_crate("low", "moderate"))
        _, high = solve(with_crate("high", "moderate"))
        self.assertTrue(low.events["ev_shove"].feasible, low.events["ev_shove"].note)
        self.assertFalse(high.events["ev_shove"].feasible)
        self.assertIn("N*s", high.events["ev_shove"].note)
        _, high_hard = solve(with_crate("high", "hard"))
        self.assertTrue(high_hard.events["ev_shove"].feasible, high_hard.events["ev_shove"].note)
        # Same 1 m: low friction slides longer (t = v0 / (mu g) = sqrt(2 d / (mu g))).
        t_low = low.events["ev_shove"].phase_times["impact"]
        t_high = high_hard.events["ev_shove"].phase_times["impact"]
        self.assertAlmostEqual(t_low, math.sqrt(2 * 1.0 / (0.15 * physics.G)), places=9)
        self.assertAlmostEqual(t_high, math.sqrt(2 * 1.0 / (0.8 * physics.G)), places=9)
        crate = low.frame("kf_wide_2").entities["pr_crate"]
        self.assertAlmostEqual(crate.position[0], 0.5, places=6)

    def test_plan_level_throw_feasibility(self):
        def throw_plan(strength: str) -> dict:
            data = fixture_data()
            next(b for b in data["bible"]["phys_bodies"] if b["entity_id"] == "pr_glass")["mass_class"] = "m"
            event_of(data, "ev_fall").update(kind="throw", actor_id="ch_mara", strength=strength,
                                              outcome_mark_id="mk_counter_wall")
            return data
        _, weak = solve(throw_plan("gentle"))
        self.assertFalse(weak.events["ev_fall"].feasible)
        _, strong = solve(throw_plan("violent"))
        self.assertTrue(strong.events["ev_fall"].feasible, strong.events["ev_fall"].note)
        apex = strong.events["ev_fall"].phase_times["apex"]
        self.assertGreater(apex, 0.0)

    def test_fixed_body_cannot_be_pushed(self):
        data = fixture_data()
        event_of(data, "ev_knock")["target_id"] = "pr_table"
        _, solved = solve(data)
        self.assertFalse(solved.events["ev_knock"].feasible)
        self.assertIn("fixed", solved.events["ev_knock"].note)

    def test_free_fall_cannot_drift_across_the_room(self):
        data = fixture_data()
        event_of(data, "ev_fall")["outcome_mark_id"] = "mk_mara"
        _, solved = solve(data)
        self.assertFalse(solved.events["ev_fall"].feasible)
        self.assertIn("free fall", solved.events["ev_fall"].note)


class TestChainedEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.solved = solve(multi_event_plan())

    def test_all_events_feasible_and_sequenced(self):
        ev = self.solved.events
        self.assertEqual(self.solved.warnings, [])
        self.assertTrue(all(e.feasible for e in ev.values()), {k: e.note for k, e in ev.items() if not e.feasible})
        walk_motion = ev["ev_walk"].phase_times["impact"]
        self.assertGreaterEqual(ev["ev_lift"].start_t_physical, ev["ev_walk"].start_t_physical + walk_motion)
        self.assertGreaterEqual(ev["ev_throw"].start_t_physical, ev["ev_lift"].start_t_physical + 0.8)
        self.assertAlmostEqual(ev["ev_catch"].start_t_physical,
                               ev["ev_throw"].start_t_physical + ev["ev_throw"].phase_times["impact"], places=9)
        self.assertGreaterEqual(ev["ev_run"].start_t_physical, ev["ev_enter"].start_t_physical + 0.5)

    def test_states_along_the_chain(self):
        throw = self.solved.frame("kf_play_3").entities["pr_ball"]
        self.assertEqual(throw.support, "airborne")
        self.assertTrue(throw.in_motion)
        self.assertGreater(throw.velocity[1], -1.0)              # near the apex
        caught = self.solved.frame("kf_play_4")
        ball, mara = caught.entities["pr_ball"], caught.entities["ch_mara"]
        self.assertEqual((ball.support, ball.support_id), ("held", "ch_mara"))
        for got, want in zip(ball.position, physics.hand_position(mara)):
            self.assertAlmostEqual(got, want, places=9)
        placed = self.solved.frame("kf_kid_2").entities["pr_ball"]
        self.assertEqual((placed.support, placed.support_id), ("on_entity", "pr_table"))
        self.assertAlmostEqual(placed.position[1], TABLE_TOP)
        kid = self.solved.frame("kf_kid_1").entities["ch_kid"]
        self.assertTrue(kid.present)
        self.assertEqual(kid.pose, "walk")

    def test_validates_clean(self):
        from src.previs import validator
        self.assertEqual(validator.validate(self.plan, self.solved), [])

    def test_continuous_for_present_entities(self):
        dt = 1.0 / 240.0
        for shot in self.solved.shots[3:]:
            prev = None
            for i in range(int((shot.end_t_s - shot.start_t_s) / dt)):
                cur = physics.sample_frame(self.plan, self.solved, shot.start_t_s + i * dt)
                if prev is not None:
                    for eid, ef in cur.entities.items():
                        if ef.present and prev.entities[eid].present:
                            step = math.dist(ef.position, prev.entities[eid].position)
                            self.assertLess(step, 0.05, f"{eid} jumps {step:.3f} m at t={cur.t_s:.4f}s")
                prev = cur


def add_shot(data: dict, shot_id: str, order: int, subject: str, keyframes: list, events=(), states=(),
             secondary: str = "none", duration: str = "medium") -> None:
    """Append a shot (a copy of sh_wide's camera) and its timeline to a plan."""
    shot = copy.deepcopy(shot_of(data, "sh_wide"))
    shot.update(id=shot_id, order=order, setup_id="su_" + shot_id[3:], subject_id=subject,
                keyframe_count=len(keyframes), duration_class=duration, secondary_id=secondary)
    data["bible"]["shots"].append(shot)
    data["timeline"]["shots"].append({"shot_id": shot_id, "keyframes": list(keyframes), "events": list(events),
                                      "states": list(states), "cause_links": []})


def with_seat(seat_height: float, jon_pose: str = "", body: str = "human") -> dict:
    """
    Fixture plus a fixed seat prop (size_y = seat_height, e.g. a chair whose backrest is 0.85 m) at mk_seat
    and a mark in front of it. With jon_pose, Jon rests on the seat in that pose for the whole scene.
    """
    data = fixture_data()
    bible = data["bible"]
    bible["props"].append({"id": "pr_seat", "name": "seat", "shape": "box", "size_x_m": 0.45,
                           "size_y_m": seat_height, "size_z_m": 0.45, "color_hex": "#664422", "movable": False})
    bible["phys_bodies"].append({"entity_id": "pr_seat", "regime": "rigid", "material": "wood", "mass_class": "fixed",
                                 "friction": "high", "restitution": "low", "failure_mode": "none",
                                 "container_fill": "none"})
    bible["marks"] += [{"id": "mk_seat", "location_id": "lo_kitchen", "name": "seat", "x_m": 1.6, "z_m": 2.0},
                       {"id": "mk_front", "location_id": "lo_kitchen", "name": "front of seat", "x_m": 1.6, "z_m": 2.4}]
    bible["initial_states"].append(make_state("init", "pr_seat", mark_id="mk_seat", pose="none", expression="none"))
    next(c for c in bible["characters"] if c["id"] == "ch_jon")["body"] = body
    if jon_pose:
        seated = {"mark_id": "mk_seat", "pose": jon_pose, "support": "on_entity", "support_id": "pr_seat"}
        next(s for s in bible["initial_states"] if s["entity_id"] == "ch_jon").update(seated)
        for st in timeline_of(data, "sh_jon")["states"]:
            st.update(seated)
    return data


def stand_up_plan(seat_height: float, phase: str) -> dict:
    """Jon, seated, stands up and steps to mk_front; the keyframe showing it is at `phase`, with no snapshot."""
    data = with_seat(seat_height, "sit")
    add_shot(data, "sh_up", 4, "ch_jon", [make_keyframe("kf_up_1", 1, "ev_up", "in_progress"),
                                          make_keyframe("kf_up_2", 2, "ev_up", phase)],
             events=[make_event("ev_up", "kf_up_1", "stand_up", actor_id="ch_jon", outcome_mark_id="mk_front",
                                outcome_rest="rest")])
    add_shot(data, "sh_after", 5, "ch_jon", [make_keyframe("kf_after_1", 1), make_keyframe("kf_after_2", 2)])
    return data


def sit_down_plan(seat_height: float, phase: str) -> dict:
    """Jon walks from the counter to mk_front and sits down on the seat; the sitting keyframe is at `phase`."""
    data = with_seat(seat_height)
    add_shot(data, "sh_sit", 4, "ch_jon",
             [make_keyframe("kf_sit_1", 1, "ev_walk", "in_progress"), make_keyframe("kf_sit_2", 2, "ev_walk", "rest"),
              make_keyframe("kf_sit_3", 3, "ev_sit", "in_progress"), make_keyframe("kf_sit_4", 4, "ev_sit", phase)],
             events=[make_event("ev_walk", "kf_sit_1", "walk_to", actor_id="ch_jon", outcome_mark_id="mk_front",
                                outcome_rest="rest"),
                     make_event("ev_sit", "kf_sit_3", "sit_down", actor_id="ch_jon", target_id="pr_seat",
                                outcome_support_id="pr_seat", outcome_mark_id="mk_seat", outcome_rest="rest",
                                cause_event_id="ev_walk")],
             states=[make_state("kf_sit_1", "ch_jon", mark_id="mk_counter", pose="walk", facing_id="mk_seat"),
                     make_state("kf_sit_2", "ch_jon", mark_id="mk_front", pose="stand", facing_id="ch_mara")],
             duration="long")
    add_shot(data, "sh_after", 5, "ch_jon", [make_keyframe("kf_after_1", 1), make_keyframe("kf_after_2", 2)])
    return data


def assert_continuous(test: unittest.TestCase, plan, solved, shot_ids, entity_id: str, max_step: float = 0.05) -> None:
    dt = 1.0 / 240.0
    for shot in solved.shots:
        if shot.shot_id not in shot_ids:
            continue
        prev = None
        for i in range(int((shot.end_t_s - shot.start_t_s) / dt)):
            cur = physics.sample_frame(plan, solved, shot.start_t_s + i * dt).entities[entity_id].position
            if prev is not None:
                test.assertLess(math.dist(cur, prev), max_step, f"{entity_id} jumps in {shot.shot_id}")
            prev = cur


class TestSeating(unittest.TestCase):
    """One rule: a person seated on a low support keeps its feet on the floor; standing or lying on it does not."""

    def jon_y(self, data: dict) -> list:
        plan, solved = solve(data)
        from src.previs import validator
        self.assertEqual(validator.validate(plan, solved), [])
        return [f.entities["ch_jon"].position[1] for f in solved.frames]

    def test_seated_on_chair_keeps_feet_on_floor(self):
        # A chair's size_y includes its backrest (0.85 m in the real cafe plan): a seated person must not float there.
        for pose in ("sit", "kneel", "crouch", "reach"):
            for y in self.jon_y(with_seat(0.85, pose)):
                self.assertAlmostEqual(y, 0.0, places=9, msg=pose)
        _, solved = solve(with_seat(0.85, "sit"))
        jon = solved.frame("kf_jon_2").entities["ch_jon"]
        self.assertAlmostEqual(physics.hand_position(jon)[1], 0.30 * 1.8)        # hand at lap height

    def test_standing_lying_or_tall_supports_rest_on_top(self):
        for seat, pose in ((0.5, "stand"), (0.5, "lie"), (1.2, "sit")):
            for y in self.jon_y(with_seat(seat, pose)):
                self.assertAlmostEqual(y, seat, places=9, msg=(seat, pose))

    def test_animals_rest_on_top(self):
        for y in self.jon_y(with_seat(0.45, "sit", body="quadruped")):
            self.assertAlmostEqual(y, 0.45, places=9)

    def test_stand_up_ends_on_floor(self):
        # 0.46 m chair: seated on the floor level throughout. 1.1 m bar stool: seated on top, then down to the floor.
        for seat in (0.46, 1.1):
            for phase in ("aftermath", "rest"):
                plan, solved = solve(stand_up_plan(seat, phase))
                self.assertTrue(solved.events["ev_up"].feasible)
                for kf_id in ("kf_up_2", "kf_after_1", "kf_after_2"):
                    jon = solved.frame(kf_id).entities["ch_jon"]
                    self.assertAlmostEqual(jon.position[1], 0.0, places=9, msg=(seat, phase, kf_id))
                    self.assertAlmostEqual(jon.position[2], 2.4, places=9)
                    self.assertEqual((jon.support, jon.pose), ("ground", "stand"))
                self.assertAlmostEqual(solved.frame("kf_jon_2").entities["ch_jon"].position[1],
                                       0.0 if seat < 1.0 else seat, places=9)
                assert_continuous(self, plan, solved, {"sh_up", "sh_after"}, "ch_jon")

    def test_sit_down_ends_at_the_seat_rule_height(self):
        for seat, want in ((0.85, 0.0), (1.1, 1.1)):
            for phase in ("impact", "aftermath", "rest"):
                plan, solved = solve(sit_down_plan(seat, phase))
                self.assertTrue(all(e.feasible for e in solved.events.values()))
                for kf_id in ("kf_sit_4", "kf_after_1"):
                    jon = solved.frame(kf_id).entities["ch_jon"]
                    if kf_id == "kf_after_1" or phase != "impact":
                        self.assertAlmostEqual(jon.position[1], want, places=9, msg=(seat, phase, kf_id))
                        self.assertEqual((jon.support, jon.support_id, jon.pose), ("on_entity", "pr_seat", "sit"))
                assert_continuous(self, plan, solved, {"sh_sit", "sh_after"}, "ch_jon")


class TestReachAndCustody(unittest.TestCase):
    def ball_plan(self) -> dict:
        data = fixture_data()
        data["bible"]["props"].append({"id": "pr_ball", "name": "ball", "shape": "sphere", "size_x_m": 0.2,
                                       "size_y_m": 0.2, "size_z_m": 0.2, "color_hex": "#ff0000", "movable": True})
        data["bible"]["phys_bodies"].append({"entity_id": "pr_ball", "regime": "rigid", "material": "rubber",
                                             "mass_class": "s", "friction": "med", "restitution": "high",
                                             "failure_mode": "none", "container_fill": "none"})
        return data

    def test_throw_needs_the_object_in_reach(self):
        # The ball rests on the table 2.5 m from Jon at the counter; nobody lifts it before the throw.
        data = self.ball_plan()
        data["bible"]["initial_states"].append(make_state(
            "init", "pr_ball", mark_id="mk_table", offset_x_m=-0.5, support="on_entity", support_id="pr_table",
            pose="none", expression="none"))
        add_shot(data, "sh_toss", 4, "ch_jon",
                 [make_keyframe("kf_toss_1", 1, "ev_throw", "anticipation"),
                  make_keyframe("kf_toss_2", 2, "ev_throw", "release"), make_keyframe("kf_toss_3", 3, "ev_throw", "rest")],
                 events=[make_event("ev_throw", "kf_toss_1", "throw", actor_id="ch_jon", target_id="pr_ball",
                                    outcome_rest="rest", outcome_mark_id="mk_floor_glass")],
                 states=[make_state("kf_toss_1", "ch_jon", mark_id="mk_counter", pose="throw", facing_id="ch_mara")],
                 secondary="ch_mara")
        plan, solved = solve(data)
        self.assertFalse(solved.events["ev_throw"].feasible)
        self.assertIn("beyond reach", solved.events["ev_throw"].note)
        from src.previs import validator
        found = [v for v in validator.validate(plan, solved) if v.code == "event_feasible" and v.severity == "error"]
        self.assertEqual([v.entity_id for v in found], ["pr_ball"])

    def test_catch_shown_at_impact_lands_in_the_drawn_hand(self):
        # Jon throws the ball to Mara (outcome held); the last keyframe shows the catch at impact with Mara reaching.
        data = self.ball_plan()
        next(s for s in data["bible"]["initial_states"] if s["entity_id"] == "ch_jon")["holding_id"] = "pr_ball"
        data["bible"]["initial_states"].append(make_state("init", "pr_ball", support="held", support_id="ch_jon",
                                                          pose="none", expression="none"))
        for st in timeline_of(data, "sh_jon")["states"]:
            st["holding_id"] = "pr_ball"
        add_shot(data, "sh_toss", 4, "ch_jon",
                 [make_keyframe("kf_toss_1", 1, "ev_throw", "anticipation"),
                  make_keyframe("kf_toss_2", 2, "ev_throw", "release"),
                  make_keyframe("kf_toss_3", 3, "ev_throw", "falling"), make_keyframe("kf_toss_4", 4, "ev_throw", "impact")],
                 events=[make_event("ev_throw", "kf_toss_1", "throw", actor_id="ch_jon", target_id="pr_ball",
                                    outcome_rest="held", outcome_support_id="ch_mara")],
                 states=[make_state("kf_toss_1", "ch_jon", mark_id="mk_counter", pose="throw", facing_id="ch_mara",
                                    holding_id="pr_ball"),
                         make_state("kf_toss_2", "ch_jon", mark_id="mk_counter", pose="throw", facing_id="ch_mara"),
                         make_state("kf_toss_4", "ch_mara", mark_id="mk_mara", pose="reach", facing_id="ch_jon",
                                    holding_id="pr_ball")],
                 secondary="ch_mara")
        plan, solved = solve(data)
        self.assertTrue(solved.events["ev_throw"].feasible, solved.events["ev_throw"].note)
        frame = solved.frame("kf_toss_4")
        ball, mara = frame.entities["pr_ball"], frame.entities["ch_mara"]
        self.assertEqual((ball.support, ball.support_id), ("held", "ch_mara"))      # caught, not airborne
        self.assertLess(math.dist(ball.position, physics.hand_position(mara)), 1e-9)
        from src.previs import validator
        self.assertEqual(validator.validate(plan, solved), [])

    def test_exit_takes_the_held_object_along(self):
        data = fixture_data()
        bible = data["bible"]
        bible["props"].append({"id": "pr_mug", "name": "mug", "shape": "cylinder", "size_x_m": 0.08, "size_y_m": 0.1,
                               "size_z_m": 0.08, "color_hex": "#ffffff", "movable": True})
        bible["marks"].append({"id": "mk_door", "location_id": "lo_kitchen", "name": "door", "x_m": 0.2, "z_m": 3.0})
        next(s for s in bible["initial_states"] if s["entity_id"] == "ch_jon")["holding_id"] = "pr_mug"
        bible["initial_states"].append(make_state("init", "pr_mug", support="held", support_id="ch_jon",
                                                  pose="none", expression="none"))
        for st in timeline_of(data, "sh_jon")["states"]:
            st["holding_id"] = "pr_mug"
        add_shot(data, "sh_leave", 4, "ch_jon",
                 [make_keyframe("kf_leave_1", 1, "ev_exit", "in_progress"), make_keyframe("kf_leave_2", 2)],
                 events=[make_event("ev_exit", "kf_leave_1", "exit", actor_id="ch_jon", outcome_mark_id="mk_door")],
                 states=[make_state("kf_leave_1", "ch_jon", mark_id="mk_counter", pose="walk", facing_id="mk_door",
                                    holding_id="pr_mug")])
        add_shot(data, "sh_after", 5, "ch_mara", [make_keyframe("kf_after_1", 1)])
        plan, solved = solve(data)
        self.assertTrue(solved.frame("kf_leave_1").entities["pr_mug"].present)
        for kf_id in ("kf_leave_2", "kf_after_1"):
            self.assertFalse(solved.frame(kf_id).entities["ch_jon"].present)
            self.assertFalse(solved.frame(kf_id).entities["pr_mug"].present, kf_id)
        # Between the moment Jon is out of the door and the next keyframe the mug is gone with him.
        a, b = solved.frame("kf_leave_1"), solved.frame("kf_leave_2")
        exit_done = solved.events["ev_exit"].start_t_physical + solved.events["ev_exit"].phase_times["impact"]
        self.assertLess(exit_done, b.t_physical_s)
        t_phys = (exit_done + b.t_physical_s) / 2
        t_edit = a.t_s + (t_phys - a.t_physical_s) / (b.t_physical_s - a.t_physical_s) * (b.t_s - a.t_s)
        mid = physics.sample_frame(plan, solved, t_edit)
        self.assertFalse(mid.entities["ch_jon"].present)
        self.assertFalse(mid.entities["pr_mug"].present)


CUP = {"id": "pr_cup", "name": "white cup", "shape": "cylinder", "size_x_m": 0.09, "size_y_m": 0.07,
       "size_z_m": 0.09, "color_hex": "#ffffff", "movable": True}
TRAY = {"id": "pr_tray", "name": "tray", "shape": "cylinder", "size_x_m": 0.4, "size_y_m": 0.02, "size_z_m": 0.3,
        "color_hex": "#cccccc", "movable": True}


def add_prop(data: dict, prop: dict, mass: str = "xs", friction: str = "med") -> None:
    data["bible"]["props"].append(dict(prop))
    data["bible"]["phys_bodies"].append({"entity_id": prop["id"], "regime": "rigid", "material": "ceramic",
                                         "mass_class": mass, "friction": friction, "restitution": "low",
                                         "failure_mode": "none", "container_fill": "none"})


def tray_plan() -> dict:
    """
    Fixture plus a tray Jon holds from the start, with a cup standing on it. Like the cafe plan, the cup is
    placed by a mark (Jon's mark plus an offset), not relative to the hand.
    """
    data = fixture_data()
    bible = data["bible"]
    add_prop(data, TRAY, mass="s")
    add_prop(data, CUP)
    bible["marks"] += [{"id": "mk_walk_end", "location_id": "lo_kitchen", "name": "walk end", "x_m": 1.0, "z_m": 3.0},
                       {"id": "mk_cup_floor", "location_id": "lo_kitchen", "name": "cup floor", "x_m": 3.3, "z_m": 3.5}]
    jon = next(s for s in bible["initial_states"] if s["entity_id"] == "ch_jon")
    jon.update(holding_id="pr_tray", pose="carry")
    bible["initial_states"] += [
        make_state("init", "pr_tray", mark_id="mk_counter", support="held", support_id="ch_jon", pose="none",
                   expression="none"),
        make_state("init", "pr_cup", mark_id="mk_counter", offset_x_m=0.1, offset_z_m=-0.04, support="on_entity",
                   support_id="pr_tray", pose="none", expression="none")]
    for st in timeline_of(data, "sh_jon")["states"]:
        st.update(holding_id="pr_tray", pose="carry")
    return data


def add_carry_walk(data: dict) -> None:
    """Jon walks from the counter to mk_walk_end carrying the tray."""
    add_shot(data, "sh_carry", 4, "ch_jon",
             [make_keyframe("kf_carry_1", 1, "ev_carry", "in_progress"), make_keyframe("kf_carry_2", 2, "ev_carry", "rest")],
             events=[make_event("ev_carry", "kf_carry_1", "walk_to", actor_id="ch_jon", outcome_mark_id="mk_walk_end",
                                outcome_rest="rest")],
             states=[make_state("kf_carry_1", "ch_jon", mark_id="mk_counter", pose="walk", holding_id="pr_tray",
                                facing_id="mk_walk_end"),
                     make_state("kf_carry_2", "ch_jon", mark_id="mk_walk_end", pose="carry", holding_id="pr_tray",
                                facing_id="mk_walk_end")])


def shelf_plan() -> dict:
    """Fixture plus a 1.6 m shelf beside Mara with a ball on its edge (0.65 m from her)."""
    data = fixture_data()
    bible = data["bible"]
    bible["props"] += [{"id": "pr_shelf", "name": "shelf", "shape": "box", "size_x_m": 0.4, "size_y_m": 1.6,
                        "size_z_m": 0.3, "color_hex": "#553311", "movable": False},
                       {"id": "pr_ball", "name": "ball", "shape": "sphere", "size_x_m": 0.1, "size_y_m": 0.1,
                        "size_z_m": 0.1, "color_hex": "#ff0000", "movable": True}]
    bible["phys_bodies"] += [
        {"entity_id": "pr_shelf", "regime": "static", "material": "wood", "mass_class": "fixed", "friction": "med",
         "restitution": "low", "failure_mode": "none", "container_fill": "none"},
        {"entity_id": "pr_ball", "regime": "rigid", "material": "rubber", "mass_class": "s", "friction": "med",
         "restitution": "dead", "failure_mode": "none", "container_fill": "none"}]
    bible["marks"].append({"id": "mk_shelf", "location_id": "lo_kitchen", "name": "shelf", "x_m": 1.5, "z_m": 1.3})
    bible["initial_states"] += [
        make_state("init", "pr_shelf", mark_id="mk_shelf", pose="none", expression="none"),
        make_state("init", "pr_ball", mark_id="mk_shelf", offset_x_m=0.15, support="on_entity", support_id="pr_shelf",
                   pose="none", expression="none")]
    return data


class TestCarriedCaughtAndContinued(unittest.TestCase):
    """Support chains move with their carrier; flights that end 'airborne' are finished by what follows them."""

    def assert_on_tray(self, frame, msg: str = "") -> None:
        cup, tray = frame.entities["pr_cup"], frame.entities["pr_tray"]
        self.assertLessEqual(abs(cup.position[0] - tray.position[0]), (0.4 - 0.09) / 2 + 1e-9, msg)
        self.assertLessEqual(abs(cup.position[2] - tray.position[2]), (0.3 - 0.09) / 2 + 1e-9, msg)
        self.assertAlmostEqual(cup.position[1], tray.position[1] + 0.02, places=9, msg=msg)

    def test_cup_on_a_carried_tray_moves_with_it(self):
        # The cafe bug: cups on the tray Leo carries stayed at their marks while the tray went with his hand.
        data = tray_plan()
        add_carry_walk(data)
        plan, solved = solve(data)
        from src.previs import validator
        self.assertEqual(validator.validate(plan, solved), [])
        for frame in solved.frames:
            self.assert_on_tray(frame, frame.kf_id)
            cup, tray = frame.entities["pr_cup"], frame.entities["pr_tray"]
            # Its stated place on the tray (mark offset relative to the tray's mark) is kept.
            self.assertAlmostEqual(cup.position[0] - tray.position[0], 0.1, places=9)
            self.assertAlmostEqual(cup.position[2] - tray.position[2], -0.04, places=9)
        walking = solved.frame("kf_carry_1")
        self.assertTrue(walking.entities["ch_jon"].in_motion)
        self.assertNotEqual(walking.entities["pr_cup"].velocity, (0.0, 0.0, 0.0))
        self.assertEqual(walking.entities["pr_cup"].velocity, walking.entities["pr_tray"].velocity)
        # Between keyframes too (the gray-box animatic).
        carry = next(s for s in solved.shots if s.shot_id == "sh_carry")
        for i in range(int((carry.end_t_s - carry.start_t_s) * 48)):
            self.assert_on_tray(physics.sample_frame(plan, solved, carry.start_t_s + i / 48.0), f"sample {i}")
        assert_continuous(self, plan, solved, {"sh_carry"}, "pr_cup")

    def test_cup_sliding_off_a_held_tray_lands_on_its_mark(self):
        # Outcome first: the slide off the tray heads for where the fall lands and leaves the edge fast enough.
        data = tray_plan()
        add_shot(data, "sh_spill", 4, "pr_cup",
                 [make_keyframe("kf_spill_1", 1, "ev_slide", "release"), make_keyframe("kf_spill_2", 2, "ev_drop", "falling"),
                  make_keyframe("kf_spill_3", 3, "ev_drop", "rest")],
                 events=[make_event("ev_tip", "kf_spill_1", "gesture", actor_id="ch_jon", target_id="pr_tray"),
                         make_event("ev_slide", "kf_spill_1", "slide", target_id="pr_cup", cause_event_id="ev_tip",
                                    strength="gentle", outcome_rest="airborne"),
                         make_event("ev_drop", "kf_spill_1", "fall", target_id="pr_cup", cause_event_id="ev_slide",
                                    outcome_mark_id="mk_cup_floor", outcome_rest="rest")],
                 states=[make_state("kf_spill_3", "pr_cup", mark_id="mk_cup_floor", pose="none", expression="none")])
        plan, solved = solve(data)
        self.assertTrue(all(e.feasible for e in solved.events.values()),
                        {k: e.note for k, e in solved.events.items() if not e.feasible})
        from src.previs import validator
        self.assertEqual([v for v in validator.validate(plan, solved) if v.severity == "error"], [])
        tray = solved.frame("kf_spill_1").entities["pr_tray"]
        start = solved.frame("kf_spill_1").entities["pr_cup"].position
        self.assertLessEqual(abs(start[0] - tray.position[0]), 0.2 + 1e-9)         # sliding on the tray
        landed = solved.frame("kf_spill_3").entities["pr_cup"].position
        for got, want in zip(landed, (3.3, 0.0, 3.5)):
            self.assertAlmostEqual(got, want, places=6)

    def test_fall_caught_by_a_separate_catch_ends_in_the_hand(self):
        # The cafe's kf_catch_2: the flight ends 'airborne' and a catch caused by it receives the object. The
        # catch's contact keyframe must show it at the catcher's hand, not on the floor.
        data = shelf_plan()
        add_shot(data, "sh_shelf", 4, "ch_mara",
                 [make_keyframe("kf_shelf_1", 1, "ev_drop", "release"), make_keyframe("kf_shelf_2", 2, "ev_drop", "falling"),
                  make_keyframe("kf_shelf_3", 3, "ev_grab", "contact"), make_keyframe("kf_shelf_4", 4, "ev_grab", "rest")],
                 events=[make_event("ev_drop", "kf_shelf_1", "fall", target_id="pr_ball", outcome_rest="airborne"),
                         make_event("ev_grab", "kf_shelf_3", "catch", actor_id="ch_mara", target_id="pr_ball",
                                    cause_event_id="ev_drop", outcome_support_id="ch_mara", outcome_rest="held")],
                 states=[make_state("kf_shelf_2", "pr_ball", mark_id="mk_shelf", offset_x_m=0.15, support="airborne",
                                    pose="none", expression="none"),
                         make_state("kf_shelf_3", "ch_mara", mark_id="mk_mara", pose="reach", holding_id="pr_ball"),
                         make_state("kf_shelf_4", "ch_mara", mark_id="mk_mara", pose="reach", holding_id="pr_ball")],
                 secondary="pr_ball")
        plan, solved = solve(data)
        self.assertTrue(all(e.feasible for e in solved.events.values()))
        from src.previs import validator
        self.assertEqual([v for v in validator.validate(plan, solved) if v.severity == "error"], [])
        contact = solved.frame("kf_shelf_3")
        ball, mara = contact.entities["pr_ball"], contact.entities["ch_mara"]
        self.assertEqual((ball.support, ball.support_id), ("held", "ch_mara"))
        self.assertLess(math.dist(ball.position, physics.hand_position(mara)), 1e-6)   # in the drawn hand
        self.assertAlmostEqual(ball.position[1], 0.55 * 1.68, places=6)               # caught at hand height
        falling = solved.frame("kf_shelf_2").entities["pr_ball"]
        self.assertEqual(falling.support, "airborne")
        self.assertTrue(0.55 * 1.68 < falling.position[1] < 1.6)
        # Mara turns to the catch while the ball falls, then faces her own way again; the ball stays in her hand.
        before, after = solved.frame("kf_jon_2"), solved.frame("kf_shelf_4")
        self.assertAlmostEqual(solved.frame("kf_shelf_1").entities["ch_mara"].yaw_deg, mara.yaw_deg, places=9)
        self.assertNotAlmostEqual(mara.yaw_deg, before.entities["ch_mara"].yaw_deg, places=3)
        self.assertAlmostEqual(after.entities["ch_mara"].yaw_deg, before.entities["ch_mara"].yaw_deg, places=9)
        self.assertLess(math.dist(after.entities["pr_ball"].position,
                                  physics.hand_position(after.entities["ch_mara"])), 1e-9)
        assert_continuous(self, plan, solved, {"sh_shelf"}, "pr_ball")

    def test_throw_caught_by_a_separate_catch_lands_in_the_hand(self):
        data = multi_event_plan()
        event_of(data, "ev_throw").update(outcome_rest="airborne", outcome_support_id="none")
        tl = timeline_of(data, "sh_play")
        tl["keyframes"][3].update(focus_event_id="ev_catch", depicted_phase="contact")
        plan, solved = solve(data)
        self.assertTrue(solved.events["ev_throw"].feasible, solved.events["ev_throw"].note)
        frame = solved.frame("kf_play_4")
        ball, mara = frame.entities["pr_ball"], frame.entities["ch_mara"]
        self.assertEqual((ball.support, ball.support_id), ("held", "ch_mara"))
        self.assertLess(math.dist(ball.position, physics.hand_position(mara)), 1e-9)
        from src.previs import validator
        self.assertEqual([v for v in validator.validate(plan, solved) if v.severity == "error"], [])

    def test_fall_continued_across_a_cut_is_one_flight(self):
        # sh_top shows the ball leaving the shelf (ends 'airborne'); sh_low continues the same fall to the floor.
        data = shelf_plan()
        add_shot(data, "sh_top", 4, "pr_ball",
                 [make_keyframe("kf_top_1", 1, "ev_drop", "release"), make_keyframe("kf_top_2", 2, "ev_drop", "falling")],
                 events=[make_event("ev_drop", "kf_top_1", "fall", target_id="pr_ball", outcome_rest="airborne")])
        add_shot(data, "sh_low", 5, "pr_ball",
                 [make_keyframe("kf_low_1", 1, "ev_drop2", "falling"), make_keyframe("kf_low_2", 2, "ev_drop2", "rest")],
                 events=[make_event("ev_drop2", "kf_low_1", "fall", target_id="pr_ball", cause_event_id="ev_drop",
                                    outcome_rest="rest")])
        plan, solved = solve(data)
        ev = solved.events
        self.assertTrue(ev["ev_drop"].feasible and ev["ev_drop2"].feasible)
        from src.previs import validator
        self.assertEqual([v for v in validator.validate(plan, solved) if v.severity == "error"], [])
        # One fall from 1.6 m: the second event starts where and when the first one stops.
        self.assertAlmostEqual(ev["ev_drop2"].start_t_physical,
                               ev["ev_drop"].start_t_physical + ev["ev_drop"].phase_times["impact"], places=9)
        total = ev["ev_drop2"].start_t_physical + ev["ev_drop2"].phase_times["impact"] - ev["ev_drop"].start_t_physical
        self.assertAlmostEqual(total, physics.fall_time(1.6), places=9)
        low = solved.frame("kf_low_1").entities["pr_ball"]
        self.assertEqual(low.support, "airborne")
        self.assertGreater(low.position[1], 0.05)              # still in the air after the cut
        self.assertLess(low.velocity[1], -2.0)                 # and already falling fast
        rest = solved.frame("kf_low_2").entities["pr_ball"].position
        for got, want in zip(rest, (1.65, 0.0, 1.3)):          # straight below the shelf edge
            self.assertAlmostEqual(got, want, places=6)

    def test_fall_with_no_landing_mark_rests_where_it_landed(self):
        # Planner request: with outcome_mark_id "none" the glass lands below the table edge; snapshots that only
        # restate its old mark must not pull it back there.
        data = fixture_data()
        for eid in ("ev_fall", "ev_shatter", "ev_spill"):
            event_of(data, eid)["outcome_mark_id"] = "none"
        for st in timeline_of(data, "sh_glass")["states"]:
            st.update(mark_id="mk_table", offset_x_m=0.62, offset_z_m=-0.2)
        plan, solved = solve(data)
        landing = solved.frame("kf_glass_3").entities["pr_glass"].position
        self.assertAlmostEqual(landing[0], 3.2, places=6)      # straight below where it left the table
        self.assertAlmostEqual(landing[1], 0.0, places=6)
        for kf_id in ("kf_jon_1", "kf_jon_2"):
            self.assertEqual(solved.frame(kf_id).entities["pr_glass"].position, landing)
        late = physics.sample_frame(plan, solved, solved.total_duration_s - 0.01).entities["pr_glass"].position
        self.assertEqual(late, landing)
        assert_continuous(self, plan, solved, {"sh_glass", "sh_jon"}, "pr_glass")
        # A snapshot that moves it somewhere new still places it.
        data2 = copy.deepcopy(data)
        timeline_of(data2, "sh_jon")["states"].append(make_state("kf_jon_2", "pr_glass", mark_id="mk_mara",
                                                                 pose="none", expression="none",
                                                                 flags=["shattered", "spilled", "fill_empty"]))
        _, solved2 = solve(data2)
        self.assertEqual(solved2.frame("kf_jon_2").entities["pr_glass"].position, (2.3, 0.0, 1.3))

    def test_animal_holds_things_in_its_mouth(self):
        data = fixture_data()
        bible = data["bible"]
        bible["characters"].append({"id": "ch_dog", "name": "Dog", "body": "quadruped", "height_m": 0.5,
                                    "appearance": "a dog", "wardrobe": "collar", "palette_hex": ["#aa7744", "#ffffff"]})
        add_prop(data, {"id": "pr_stick", "name": "stick", "shape": "box", "size_x_m": 0.05, "size_y_m": 0.05,
                        "size_z_m": 0.4, "color_hex": "#664400", "movable": True})
        bible["phys_bodies"].append({"entity_id": "ch_dog", "regime": "articulated", "material": "flesh",
                                     "mass_class": "m", "friction": "high", "restitution": "dead",
                                     "failure_mode": "none", "container_fill": "none"})
        bible["initial_states"] += [
            make_state("init", "ch_dog", mark_id="mk_mara", offset_x_m=1.0, holding_id="pr_stick", facing_id="pr_table"),
            make_state("init", "pr_stick", support="held", support_id="ch_dog", pose="none", expression="none")]
        _, solved = solve(data)
        frame = solved.frame("kf_wide_1")
        dog, stick = frame.entities["ch_dog"], frame.entities["pr_stick"]
        self.assertEqual(dog.shape, "quadruped")
        self.assertEqual(stick.position, physics.hand_position(dog))
        forward = math.dist((dog.position[0], 0, dog.position[2]), (stick.position[0], 0, stick.position[2]))
        self.assertAlmostEqual(forward, 0.82 * 0.5, places=9)                          # at the snout, no side offset
        self.assertAlmostEqual(stick.position[1], 0.87 * 0.5, places=9)


class TestSampleFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.solved = solve(fixture_data())

    def test_matches_keyframes_exactly(self):
        for frame in self.solved.frames:
            sampled = physics.sample_frame(self.plan, self.solved, frame.t_s)
            self.assertEqual(sampled.kf_id, frame.kf_id)
            for eid, ef in frame.entities.items():
                self.assertEqual(sampled.entities[eid].position, ef.position)

    def test_continuous_between_keyframes(self):
        dt = 1.0 / 480.0
        for shot in self.solved.shots:
            n = int((shot.end_t_s - shot.start_t_s) / dt)
            samples = [physics.sample_frame(self.plan, self.solved, shot.start_t_s + i * dt) for i in range(n)]
            for a, b in zip(samples, samples[1:]):
                for eid in a.entities:
                    step = math.dist(a.entities[eid].position, b.entities[eid].position)
                    self.assertLess(step, 0.05, f"{shot.shot_id} {eid} jumps {step:.3f} m at t={b.t_s:.4f}s")

    def test_mid_fall_interpolates_trajectory(self):
        f1, f2 = self.solved.frame("kf_glass_1"), self.solved.frame("kf_glass_2")
        mid = physics.sample_frame(self.plan, self.solved, (f1.t_s + f2.t_s) / 2)
        y = mid.entities["pr_glass"].position[1]
        self.assertLess(f2.entities["pr_glass"].position[1], y)
        self.assertLess(y, f1.entities["pr_glass"].position[1])
        self.assertTrue(mid.entities["pr_glass"].in_motion)
        self.assertAlmostEqual(mid.t_physical_s, (f1.t_physical_s + f2.t_physical_s) / 2, places=9)
        self.assertEqual(mid.kf_id, "kf_glass_1")

    def test_wide_hold_plays_the_fall_out(self):
        wide = self.solved.shots[0]
        late = physics.sample_frame(self.plan, self.solved, wide.end_t_s - 0.05)
        glass = late.entities["pr_glass"]
        self.assertAlmostEqual(glass.position[1], 0.0, places=6)
        self.assertEqual(glass.support, "ground")
        self.assertIn("shattered", glass.flags)
        self.assertIn("fill_empty", glass.flags)

    def test_clamps_out_of_range_times(self):
        first = physics.sample_frame(self.plan, self.solved, -1.0)
        self.assertEqual(first.kf_id, "kf_wide_1")
        last = physics.sample_frame(self.plan, self.solved, self.solved.total_duration_s + 5.0)
        self.assertEqual(last.kf_id, "kf_jon_2")


if __name__ == "__main__":
    unittest.main()
