import copy
import json
import unittest
from pathlib import Path

from src.previs import physics, validator
from src.previs.schema import ScenePlan
from src.previs.types import BlockingResult, Violation

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
FIXTURE_DATA = json.loads(FIXTURE.read_text())


def fixture_data() -> dict:
    return copy.deepcopy(FIXTURE_DATA)


def timeline_of(data: dict, shot_id: str) -> dict:
    return next(tl for tl in data["timeline"]["shots"] if tl["shot_id"] == shot_id)


def shot_of(data: dict, shot_id: str) -> dict:
    return next(s for s in data["bible"]["shots"] if s["id"] == shot_id)


def state_of(data: dict, kf_id: str, entity_id: str) -> dict:
    return next(s for tl in data["timeline"]["shots"] for s in tl["states"]
                if s["kf_id"] == kf_id and s["entity_id"] == entity_id)


def make_state(kf_id: str, entity_id: str, **overrides) -> dict:
    state = {
        "kf_id": kf_id, "entity_id": entity_id, "present": True, "mark_id": "none", "offset_x_m": 0.0,
        "offset_z_m": 0.0, "facing_id": "none", "pose": "none", "expression": "none", "holding_id": "none",
        "gaze_id": "none", "support": "ground", "support_id": "none", "flags": [], "weight_cue": "none",
        "secondary": [],
    }
    state.update(overrides)
    return state


def check(data: dict, blocking=None):
    plan = ScenePlan.model_validate(data)
    return validator.validate(plan, physics.solve(plan), blocking)


def errors(violations, code=None):
    return [v for v in violations if v.severity == "error" and (code is None or v.code == code)]


class TestFixture(unittest.TestCase):
    def test_fixture_has_no_violations(self):
        self.assertEqual(check(fixture_data()), [])

    def test_validate_solves_when_not_given(self):
        plan = ScenePlan.model_validate(fixture_data())
        self.assertEqual(validator.validate(plan), [])


class TestChecks(unittest.TestCase):
    def assert_error(self, data, code, **where):
        found = errors(check(data), code)
        self.assertTrue(found, f"expected a {code} error")
        for key, value in where.items():
            self.assertIn(value, [getattr(v, key) for v in found], f"{code}: no violation with {key}={value}: {found}")
        return found

    def test_custody_unexplained_holder_change(self):
        data = fixture_data()
        state_of(data, "kf_wide_2", "ch_mara")["holding_id"] = "pr_glass"
        found = self.assert_error(data, "custody", kf_id="kf_wide_2", entity_id="pr_glass")
        self.assertTrue(any("without a lift" in v.message for v in found))

    def test_custody_two_holders(self):
        data = fixture_data()
        state_of(data, "kf_jon_1", "ch_jon")["holding_id"] = "pr_glass"
        timeline_of(data, "sh_jon")["states"].append(make_state(
            "kf_jon_1", "ch_mara", mark_id="mk_mara", pose="stand", expression="neutral", holding_id="pr_glass"))
        found = self.assert_error(data, "custody", kf_id="kf_jon_1")
        self.assertTrue(any("at once" in v.message for v in found))

    def test_custody_held_without_holder(self):
        data = fixture_data()
        state_of(data, "kf_glass_2", "pr_glass").update(support="held", support_id="ch_jon")
        self.assert_error(data, "custody", kf_id="kf_glass_2", entity_id="pr_glass")

    def test_irreversible_flag_removal(self):
        data = fixture_data()
        timeline_of(data, "sh_jon")["states"].append(make_state(
            "kf_jon_2", "pr_glass", mark_id="mk_floor_glass", flags=["fill_empty"]))
        found = self.assert_error(data, "irreversible", kf_id="kf_jon_2", entity_id="pr_glass")
        self.assertIn("shattered", found[0].message)

    def test_fill_flags_exclusive(self):
        data = fixture_data()
        state_of(data, "kf_glass_2", "pr_glass")["flags"] = ["fill_full", "fill_half"]
        self.assert_error(data, "irreversible", kf_id="kf_glass_2")

    def test_ref_integrity(self):
        data = fixture_data()
        timeline_of(data, "sh_glass")["keyframes"][1]["focus_event_id"] = "ev_missing"
        timeline_of(data, "sh_jon")["cause_links"][0]["cause_event_id"] = "ev_nowhere"
        state_of(data, "kf_jon_2", "ch_jon")["mark_id"] = "mk_nowhere"
        found = self.assert_error(data, "ref_integrity", kf_id="kf_glass_2")
        messages = " ".join(v.message for v in found)
        self.assertIn("ev_missing", messages)
        self.assertIn("ev_nowhere", messages)
        self.assertIn("mk_nowhere", messages)

    def test_duplicate_ids(self):
        data = fixture_data()
        data["bible"]["marks"][1]["id"] = "mk_table"
        found = self.assert_error(data, "ref_integrity")
        self.assertTrue(any("more than once" in v.message for v in found))

    def test_keyframe_ids(self):
        data = fixture_data()
        timeline_of(data, "sh_glass")["keyframes"][2]["id"] = "kf_glass_9"
        for st in timeline_of(data, "sh_glass")["states"]:
            if st["kf_id"] == "kf_glass_3":
                st["kf_id"] = "kf_glass_9"
        for ev in timeline_of(data, "sh_glass")["events"]:
            if ev["start_kf_id"] == "kf_glass_3":
                ev["start_kf_id"] = "kf_glass_9"
        self.assert_error(data, "keyframe_ids", kf_id="kf_glass_9")

    def test_keyframe_count(self):
        data = fixture_data()
        shot_of(data, "sh_glass")["keyframe_count"] = 5
        self.assert_error(data, "keyframe_ids", shot_id="sh_glass")

    def test_duration(self):
        data = fixture_data()
        data["bible"]["scene"]["target_duration_s"] = 45.0
        self.assert_error(data, "duration")

    def test_ranges(self):
        data = fixture_data()
        shot_of(data, "sh_jon")["lens_mm"] = 5.0
        data["bible"]["characters"][0]["height_m"] = 3.1
        data["bible"]["marks"][2]["x_m"] = 7.0
        timeline_of(data, "sh_wide")["keyframes"][0]["move_progress"] = 1.5
        found = self.assert_error(data, "ranges", shot_id="sh_jon")
        self.assertEqual(len(found), 4)

    def test_count_needs_enter_or_exit(self):
        data = fixture_data()
        state_of(data, "kf_jon_2", "ch_jon")["present"] = False
        self.assert_error(data, "count", kf_id="kf_jon_2", entity_id="ch_jon")

    def exit_with_mug(self, mug_states=()) -> dict:
        """Jon holds a mug through the scene and exits through a door in an extra shot."""
        data = fixture_data()
        bible = data["bible"]
        bible["props"].append({"id": "pr_mug", "name": "mug", "shape": "cylinder", "size_x_m": 0.08, "size_y_m": 0.1,
                               "size_z_m": 0.08, "color_hex": "#ffffff", "movable": True})
        bible["phys_bodies"].append({"entity_id": "pr_mug", "regime": "rigid", "material": "ceramic", "mass_class": "xs",
                                     "friction": "med", "restitution": "low", "failure_mode": "none",
                                     "container_fill": "none"})
        bible["marks"].append({"id": "mk_door", "location_id": "lo_kitchen", "name": "door", "x_m": 0.2, "z_m": 3.0})
        next(s for s in bible["initial_states"] if s["entity_id"] == "ch_jon")["holding_id"] = "pr_mug"
        bible["initial_states"].append(make_state("init", "pr_mug", support="held", support_id="ch_jon"))
        for st in timeline_of(data, "sh_jon")["states"]:
            st["holding_id"] = "pr_mug"
        shot = copy.deepcopy(shot_of(data, "sh_jon"))
        shot.update(id="sh_leave", order=4, setup_id="su_leave", keyframe_count=2, move="static")
        bible["shots"].append(shot)
        kf = {"beat_id": "bt_react", "action": "Jon leaves.", "dialogue": "", "move_progress": 0.0}
        data["timeline"]["shots"].append({
            "shot_id": "sh_leave",
            "keyframes": [dict(kf, id="kf_leave_1", order=1, focus_event_id="ev_exit", depicted_phase="in_progress"),
                          dict(kf, id="kf_leave_2", order=2, focus_event_id="none", depicted_phase="none")],
            "events": [{"justification": "he walks out", "id": "ev_exit", "start_kf_id": "kf_leave_1", "kind": "exit",
                        "actor_id": "ch_jon", "target_id": "none", "cause_event_id": "none", "contact_part": "none",
                        "dir_ref": "none", "strength": "moderate", "outcome_support_id": "none",
                        "outcome_mark_id": "mk_door", "outcome_rest": "none", "outcome_flag": "none"}],
            "states": [make_state("kf_leave_1", "ch_jon", mark_id="mk_counter", pose="walk", expression="neutral",
                                  facing_id="mk_door", holding_id="pr_mug")] + list(mug_states),
            "cause_links": []})
        return data

    def test_carried_object_exits_with_its_holder(self):
        self.assertEqual(check(self.exit_with_mug()), [])
        # The planner may also state the mug's absence: covered by the holder's exit, not a count error.
        for support, support_id in (("held", "ch_jon"), ("none", "none")):
            gone = make_state("kf_leave_2", "pr_mug", present=False, support=support, support_id=support_id)
            self.assertEqual(check(self.exit_with_mug([gone])), [], support)

    def test_carried_object_enters_with_its_holder(self):
        data = fixture_data()
        bible = data["bible"]
        bible["props"].append({"id": "pr_tray", "name": "tray", "shape": "box", "size_x_m": 0.4, "size_y_m": 0.02,
                               "size_z_m": 0.3, "color_hex": "#bbbbbb", "movable": True})
        jon = next(s for s in bible["initial_states"] if s["entity_id"] == "ch_jon")
        jon.update(present=False, holding_id="pr_tray")
        bible["initial_states"].append(make_state("init", "pr_tray", present=False, support="held", support_id="ch_jon"))
        timeline_of(data, "sh_wide")["events"].append({
            "justification": "he comes in", "id": "ev_enter", "start_kf_id": "kf_wide_2", "kind": "enter",
            "actor_id": "ch_jon", "target_id": "none", "cause_event_id": "none", "contact_part": "none",
            "dir_ref": "none", "strength": "moderate", "outcome_support_id": "none", "outcome_mark_id": "mk_counter",
            "outcome_rest": "rest", "outcome_flag": "none"})
        for st in timeline_of(data, "sh_jon")["states"]:
            st["holding_id"] = "pr_tray"
        # The planner also states the tray's arrival; the holder's enter covers it.
        timeline_of(data, "sh_wide")["states"].append(make_state("kf_wide_2", "pr_tray", support="held",
                                                                 support_id="ch_jon"))
        self.assertEqual(errors(check(data), "count"), [])

    def test_held_by_absent_holder(self):
        stays = make_state("kf_leave_2", "pr_mug", present=True, support="held", support_id="ch_jon")
        found = self.assert_error(self.exit_with_mug([stays]), "supported", kf_id="kf_leave_2", entity_id="pr_mug")
        self.assertIn("not present", found[0].message)

    def test_custody_change_by_snapshot_without_event(self):
        # Mara carries a cup; a later snapshot puts it on the table with no place event and no Mara snapshot.
        data = fixture_data()
        data["bible"]["props"].append({"id": "pr_cup", "name": "cup", "shape": "cylinder", "size_x_m": 0.08,
                                       "size_y_m": 0.1, "size_z_m": 0.08, "color_hex": "#ffffff", "movable": True})
        next(s for s in data["bible"]["initial_states"] if s["entity_id"] == "ch_mara")["holding_id"] = "pr_cup"
        data["bible"]["initial_states"].append(make_state("init", "pr_cup", support="held", support_id="ch_mara"))
        for st in timeline_of(data, "sh_wide")["states"]:
            st["holding_id"] = "pr_cup"
        timeline_of(data, "sh_jon")["states"].append(make_state(
            "kf_jon_1", "pr_cup", mark_id="mk_table", support="on_entity", support_id="pr_table"))
        found = self.assert_error(data, "custody", kf_id="kf_jon_1", entity_id="pr_cup")
        self.assertIn("without a lift", found[0].message)

    def test_supported_airborne_at_rest(self):
        data = fixture_data()
        timeline_of(data, "sh_jon")["states"].append(make_state(
            "kf_jon_2", "pr_glass", mark_id="mk_floor_glass", support="airborne",
            flags=["shattered", "spilled", "fill_empty"]))
        self.assert_error(data, "supported", kf_id="kf_jon_2", entity_id="pr_glass")

    def test_fixed_entity_moves(self):
        data = fixture_data()
        timeline_of(data, "sh_jon")["states"].append(make_state("kf_jon_1", "pr_table", mark_id="mk_counter"))
        self.assert_error(data, "fixed", kf_id="kf_jon_1", entity_id="pr_table")

    def test_event_feasible(self):
        data = fixture_data()
        next(ev for ev in timeline_of(data, "sh_wide")["events"] if ev["id"] == "ev_knock")["target_id"] = "pr_table"
        self.assert_error(data, "event_feasible", shot_id="sh_wide", kf_id="kf_wide_2")

    def test_phase_order(self):
        data = fixture_data()
        kfs = timeline_of(data, "sh_glass")["keyframes"]
        kfs[0]["depicted_phase"], kfs[1]["depicted_phase"] = "falling", "release"
        self.assert_error(data, "phase_order", kf_id="kf_glass_2")

    def test_line_side(self):
        data = fixture_data()
        shot_of(data, "sh_jon")["line_side"] = "b"
        self.assert_error(data, "line_side", shot_id="sh_jon")
        shot_of(data, "sh_jon")["move"] = "orbit_left"
        self.assertEqual(errors(check(data), "line_side"), [])

    def test_line_side_resets_with_location(self):
        data = fixture_data()
        data["bible"]["locations"].append(copy.deepcopy(data["bible"]["locations"][0]) | {"id": "lo_hall"})
        data["bible"]["shots"][1].update(location_id="lo_hall", setup_id="su_hall")
        shot_of(data, "sh_jon")["line_side"] = "b"
        self.assertEqual(errors(check(data), "line_side"), [])

    def test_setup_location(self):
        data = fixture_data()
        data["bible"]["locations"].append(copy.deepcopy(data["bible"]["locations"][0]) | {"id": "lo_hall"})
        shot_of(data, "sh_jon").update(location_id="lo_hall", setup_id="su_wide")
        self.assert_error(data, "setup_location", shot_id="sh_jon")

    def test_subject_visible(self):
        def blocking(frac: float) -> BlockingResult:
            return BlockingResult(blocking_path="b.png", id_path="id.png", depth_path="d.npy", id_map={1: "pr_glass"},
                                  bboxes={}, visible_frac={"pr_glass": frac})
        found = errors(check(fixture_data(), {"kf_glass_2": blocking(0.01)}), "subject_visible")
        self.assertEqual([(v.kf_id, v.entity_id) for v in found], [("kf_glass_2", "pr_glass")])
        self.assertEqual(errors(check(fixture_data(), {"kf_glass_2": blocking(0.4)}), "subject_visible"), [])


def init_state_of(data: dict, entity_id: str) -> dict:
    return next(s for s in data["bible"]["initial_states"] if s["entity_id"] == entity_id)


class TestValidateBible(unittest.TestCase):
    """validate_bible(bible): the bible-only checks (planner request), the same facts validate() reports."""

    def bible_errors(self, data: dict, code=None):
        return errors(validator.validate_bible(ScenePlan.model_validate(data).bible), code)

    def test_sound_bibles_pass(self):
        self.assertEqual(validator.validate_bible(ScenePlan.model_validate(fixture_data()).bible), [])

    def test_each_bible_defect_is_reported(self):
        cases = []

        def case(code, edit):
            data = fixture_data()
            edit(data)
            cases.append((code, data))

        case("ranges", lambda d: shot_of(d, "sh_jon").update(lens_mm=5.0))
        case("ranges", lambda d: d["bible"]["marks"][2].update(x_m=7.0))
        case("ranges", lambda d: d["bible"]["characters"][0].update(height_m=3.1))
        case("duration", lambda d: d["bible"]["scene"].update(target_duration_s=45.0))
        case("keyframe_ids", lambda d: shot_of(d, "sh_glass").update(keyframe_count=0))
        case("ref_integrity", lambda d: shot_of(d, "sh_jon").update(subject_id="ch_nobody"))
        case("line_side", lambda d: shot_of(d, "sh_jon").update(line_side="b"))

        def two_locations(d):
            d["bible"]["locations"].append(copy.deepcopy(d["bible"]["locations"][0]) | {"id": "lo_hall"})
            shot_of(d, "sh_jon").update(location_id="lo_hall", setup_id="su_wide")
        case("setup_location", two_locations)
        # The "init" state: custody, support and fill rules no timeline repair can fix.
        case("custody", lambda d: init_state_of(d, "pr_glass").update(support="held", support_id="ch_jon"))
        case("custody", lambda d: (init_state_of(d, "ch_jon").update(holding_id="pr_glass"),
                                   init_state_of(d, "ch_mara").update(holding_id="pr_glass")))
        case("supported", lambda d: init_state_of(d, "pr_glass").update(support="airborne", support_id="none"))
        case("supported", lambda d: init_state_of(d, "pr_glass").update(support_id="ch_nobody"))
        case("irreversible", lambda d: init_state_of(d, "pr_glass").update(flags=["fill_full", "fill_empty"]))
        for code, data in cases:
            found = self.bible_errors(data, code)
            self.assertTrue(found, f"validate_bible misses the {code} defect")
            # validate() reports the same fact (code and message) for the full plan.
            full = {(v.code, v.message) for v in errors(check(data))}
            for v in found:
                self.assertIn((v.code, v.message), full)

    def test_timeline_defects_are_not_bible_defects(self):
        data = fixture_data()
        state_of(data, "kf_wide_2", "ch_mara")["holding_id"] = "pr_glass"       # a custody error at kf_wide_2
        timeline_of(data, "sh_wide")["keyframes"][0]["move_progress"] = 1.5     # a timeline range error
        self.assertTrue(errors(check(data)))
        self.assertEqual(self.bible_errors(data), [])

    def test_quadruped_shoulder_height_range(self):
        # A cat or a small dog stands 0.2-0.3 m at the shoulder; people stay within 0.3-2.5 m.
        data = fixture_data()
        data["bible"]["characters"][0].update(body="quadruped", height_m=0.25)
        self.assertEqual(self.bible_errors(data), [])
        self.assertEqual(errors(check(data), "ranges"), [])
        data["bible"]["characters"][0]["height_m"] = 0.15
        self.assertTrue(self.bible_errors(data, "ranges"))
        data["bible"]["characters"][0].update(body="human", height_m=0.25)
        self.assertTrue(self.bible_errors(data, "ranges"))
        self.assertEqual(validator.height_range_m("quadruped"), validator.QUADRUPED_HEIGHT_RANGE_M)
        self.assertEqual(validator.height_range_m("human"), validator.HEIGHT_RANGE_M)


class TestRepairSupport(unittest.TestCase):
    def setUp(self):
        self.data = fixture_data()

    def shots_for(self, violations, data=None):
        return validator.repair_shot_set(ScenePlan.model_validate(data or self.data), violations)

    def test_violating_shot_plus_later_references(self):
        v = Violation(code="custody", message="m", shot_id="sh_glass", kf_id="kf_glass_2", entity_id="pr_glass")
        self.assertEqual(self.shots_for([v]), ["sh_glass", "sh_jon"])

    def test_later_shots_without_shared_entities_are_skipped(self):
        v = Violation(code="custody", message="m", shot_id="sh_wide", kf_id="kf_wide_2", entity_id="ch_mara")
        self.assertEqual(self.shots_for([v]), ["sh_wide", "sh_jon"])     # sh_glass never references Mara

    def test_shot_from_keyframe_and_last_shot(self):
        v = Violation(code="phase_order", message="m", kf_id="kf_jon_2")
        self.assertEqual(self.shots_for([v]), ["sh_jon"])

    def test_dependency_closure_is_transitive(self):
        data = self.data
        shot_of(data, "sh_jon").update(secondary_id="none")
        for st in timeline_of(data, "sh_jon")["states"]:
            st["facing_id"] = "pr_glass"
        v = Violation(code="custody", message="m", shot_id="sh_wide", entity_id="ch_mara")
        self.assertEqual(self.shots_for([v], data), ["sh_wide"])
        # Once sh_glass also changes Mara, it is replanned, and so is sh_jon, which depends on its glass.
        timeline_of(data, "sh_glass")["states"].append(make_state(
            "kf_glass_1", "ch_mara", mark_id="mk_mara", pose="reach", expression="surprised"))
        self.assertEqual(self.shots_for([v], data), ["sh_wide", "sh_glass", "sh_jon"])

    def test_global_errors_select_all_shots_and_warnings_none(self):
        self.assertEqual(self.shots_for([Violation(code="duration", message="m")]), ["sh_wide", "sh_glass", "sh_jon"])
        warning = Violation(code="timing", message="m", shot_id="sh_wide", severity="warning")
        self.assertEqual(self.shots_for([warning]), [])

    def test_format_for_repair(self):
        violations = [
            Violation(code="custody", message="pr_glass is held by ch_mara and ch_jon at once", shot_id="sh_jon",
                      kf_id="kf_jon_1", entity_id="pr_glass"),
            Violation(code="timing", message="tight", severity="warning"),
        ]
        text = validator.format_for_repair(violations)
        self.assertIn("1 error(s)", text)
        self.assertIn("1. [custody] shot sh_jon, keyframe kf_jon_1, entity pr_glass: pr_glass is held", text)
        self.assertIn("- [timing] scene: tight", text)
        self.assertEqual(validator.format_for_repair([]), "No violations.")


if __name__ == "__main__":
    unittest.main()
