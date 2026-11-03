import copy
import json
import unittest
from pathlib import Path

from src.previs.schema import ScenePlan
from src.previs.state import INIT, build_index, event_mover, flags_with, merge_flags, reduce_states

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
FIXTURE_DATA = json.loads(FIXTURE.read_text())


def fixture_data() -> dict:
    return copy.deepcopy(FIXTURE_DATA)


def timeline_of(data: dict, shot_id: str) -> dict:
    return next(tl for tl in data["timeline"]["shots"] if tl["shot_id"] == shot_id)


def make_state(kf_id: str, entity_id: str, **overrides) -> dict:
    state = {
        "kf_id": kf_id, "entity_id": entity_id, "present": True, "mark_id": "none", "offset_x_m": 0.0,
        "offset_z_m": 0.0, "facing_id": "none", "pose": "none", "expression": "none", "holding_id": "none",
        "gaze_id": "none", "support": "ground", "support_id": "none", "flags": [], "weight_cue": "none",
        "secondary": [],
    }
    state.update(overrides)
    return state


def make_event(event_id: str, start_kf_id: str, kind: str, **overrides) -> dict:
    event = {
        "justification": "synthetic test event", "id": event_id, "start_kf_id": start_kf_id, "kind": kind,
        "actor_id": "none", "target_id": "none", "cause_event_id": "none", "contact_part": "none",
        "dir_ref": "none", "strength": "moderate", "outcome_support_id": "none", "outcome_mark_id": "none",
        "outcome_rest": "none", "outcome_flag": "none",
    }
    event.update(overrides)
    return event


def reduce(data: dict):
    plan = ScenePlan.model_validate(data)
    index = build_index(plan)
    return index, reduce_states(plan, index)


class TestSceneIndex(unittest.TestCase):
    def setUp(self):
        self.plan = ScenePlan.model_validate(fixture_data())
        self.index = build_index(self.plan)

    def test_tables(self):
        idx = self.index
        self.assertEqual(list(idx.characters), ["ch_mara", "ch_jon"])
        self.assertEqual(list(idx.props), ["pr_table", "pr_glass"])
        self.assertEqual(set(idx.entities), {"ch_mara", "ch_jon", "pr_table", "pr_glass"})
        self.assertEqual(list(idx.shots), ["sh_wide", "sh_glass", "sh_jon"])
        self.assertEqual(len(idx.marks), 5)
        self.assertEqual(idx.bodies["pr_glass"].mass_class, "xs")
        self.assertEqual(set(idx.events), {"ev_knock", "ev_fall", "ev_shatter", "ev_spill", "ev_turn"})
        self.assertEqual(idx.event_shot["ev_fall"], "sh_glass")
        self.assertEqual(len(idx.cause_links), 4)
        self.assertEqual(len(idx.invariants), 3)

    def test_keyframes_in_shot_then_keyframe_order(self):
        ids = [kf.id for _, kf in self.index.keyframes]
        self.assertEqual(ids, ["kf_wide_1", "kf_wide_2", "kf_glass_1", "kf_glass_2", "kf_glass_3", "kf_jon_1", "kf_jon_2"])
        self.assertEqual(self.index.shot_of_keyframe["kf_glass_2"], "sh_glass")
        self.assertEqual(self.index.keyframe_by_id["kf_glass_3"].focus_event_id, "ev_shatter")
        self.assertEqual(self.index.prev_keyframe_id("kf_wide_1"), INIT)
        self.assertEqual(self.index.prev_keyframe_id("kf_glass_1"), "kf_wide_2")
        self.assertEqual(self.index.event_span("ev_fall"), (2, 3))
        self.assertEqual(self.index.event_span("ev_spill"), (4, 4))

    def test_keyframes_sorted_by_order_field(self):
        data = fixture_data()
        timeline_of(data, "sh_glass")["keyframes"].reverse()
        index = build_index(ScenePlan.model_validate(data))
        self.assertEqual([kf.id for kf in index.shot_keyframes("sh_glass")], ["kf_glass_1", "kf_glass_2", "kf_glass_3"])

    def test_entity_helpers(self):
        idx = self.index
        self.assertEqual(idx.entity_kind("ch_mara"), "character")
        self.assertEqual(idx.entity_kind("pr_glass"), "prop")
        self.assertEqual(idx.entity_size("ch_jon"), (0.45, 1.8, 0.3))
        self.assertEqual(idx.entity_size("pr_table"), (1.4, 0.75, 0.8))
        self.assertEqual(idx.entity_colors("ch_mara"), ("#d4a017", "#1f3a5f"))
        self.assertEqual(idx.entity_colors("pr_glass"), ("#bfe6ff", "#bfe6ff"))
        self.assertEqual(idx.entity_shape("ch_jon"), "character")
        self.assertEqual(idx.entity_shape("pr_glass"), "cylinder")
        with self.assertRaises(KeyError):
            idx.entity_kind("pr_missing")

    def test_quadruped_shape(self):
        # Blocking draws Character.body == "quadruped" as a four-legged proxy; the solved frame says so too,
        # so the camera frames (and physics grips) the body that is drawn.
        data = fixture_data()
        next(c for c in data["bible"]["characters"] if c["id"] == "ch_jon")["body"] = "quadruped"
        idx = build_index(ScenePlan.model_validate(data))
        self.assertEqual(idx.entity_shape("ch_jon"), "quadruped")
        self.assertEqual(idx.entity_shape("ch_mara"), "character")
        self.assertEqual(idx.entity_kind("ch_jon"), "character")

    def test_event_mover(self):
        self.assertEqual(event_mover(self.index.events["ev_knock"]), "pr_glass")
        self.assertEqual(event_mover(self.index.events["ev_turn"]), "ch_jon")


class TestReduceStates(unittest.TestCase):
    def test_complete_snapshots_for_every_keyframe(self):
        index, states = reduce(fixture_data())
        self.assertEqual(list(states), [INIT] + [kf.id for _, kf in index.keyframes])
        for kf_id, snapshot in states.items():
            self.assertEqual(set(snapshot), set(index.entities))
            self.assertTrue(all(st.kf_id == kf_id for st in snapshot.values()))
        # Mara's kf_wide_2 snapshot carries forward through shots that never mention her.
        self.assertEqual(states["kf_glass_1"]["ch_mara"].pose, "reach")
        self.assertEqual(states["kf_jon_2"]["ch_mara"].weight_cue, "off_balance")

    def test_fixture_glass_outcomes(self):
        _, states = reduce(fixture_data())
        self.assertEqual(states["kf_wide_2"]["pr_glass"].support, "on_entity")
        self.assertEqual(states["kf_wide_2"]["pr_glass"].support_id, "pr_table")
        self.assertEqual(states["kf_glass_1"]["pr_glass"].support, "airborne")   # ev_knock outcome
        glass = states["kf_glass_3"]["pr_glass"]
        self.assertEqual((glass.mark_id, glass.support), ("mk_floor_glass", "ground"))
        self.assertEqual(set(glass.flags), {"shattered", "spilled", "fill_empty"})
        self.assertEqual(states["kf_jon_2"]["pr_glass"].flags, glass.flags)

    def test_outcomes_apply_without_snapshots(self):
        data = fixture_data()
        tl = timeline_of(data, "sh_glass")
        tl["states"] = [s for s in tl["states"] if s["kf_id"] != "kf_glass_3"]
        _, states = reduce(data)
        at_impact = states["kf_glass_3"]["pr_glass"]
        self.assertEqual((at_impact.mark_id, at_impact.support), ("mk_floor_glass", "ground"))   # ev_fall completed
        self.assertIn("shattered", at_impact.flags)        # break outcome visible at the depicted impact
        later = states["kf_jon_1"]["pr_glass"]
        self.assertIn("spilled", later.flags)
        self.assertEqual([f for f in later.flags if f.startswith("fill_")], ["fill_empty"])   # spill empties it

    def test_irreversible_flags_survive_snapshots(self):
        data = fixture_data()
        timeline_of(data, "sh_jon")["states"].append(
            make_state("kf_jon_1", "pr_glass", mark_id="mk_floor_glass", flags=["fill_full"]))
        _, states = reduce(data)
        flags = states["kf_jon_1"]["pr_glass"].flags
        self.assertIn("shattered", flags)
        self.assertIn("spilled", flags)

    def test_fill_flags_are_exclusive(self):
        self.assertEqual(merge_flags(["fill_full", "wet", "fill_half", "wet"]), ["wet", "fill_half"])
        self.assertEqual(flags_with(["fill_full"], "spilled", container=True), ["spilled", "fill_empty"])
        data = fixture_data()
        timeline_of(data, "sh_wide")["events"].append(make_event(
            "ev_sip", "kf_wide_1", "pour", actor_id="ch_mara", target_id="pr_glass", outcome_flag="fill_half"))
        _, states = reduce(data)
        self.assertEqual(states["kf_wide_2"]["pr_glass"].flags, ["fill_half"])

    def test_lift_sets_custody_and_release_clears_it(self):
        data = fixture_data()
        tl = timeline_of(data, "sh_wide")
        tl["states"] = [s for s in tl["states"] if s["kf_id"] != "kf_wide_2"]   # its holding_id would override
        tl["events"].append(make_event(
            "ev_lift", "kf_wide_1", "lift", actor_id="ch_mara", target_id="pr_glass", outcome_rest="held"))
        _, states = reduce(data)
        self.assertEqual(states["kf_wide_1"]["ch_mara"].holding_id, "none")
        self.assertEqual(states["kf_wide_2"]["ch_mara"].holding_id, "pr_glass")
        glass = states["kf_wide_2"]["pr_glass"]
        self.assertEqual((glass.support, glass.support_id), ("held", "ch_mara"))
        # ev_knock (outcome airborne) releases it.
        self.assertEqual(states["kf_glass_1"]["ch_mara"].holding_id, "none")
        self.assertEqual(states["kf_glass_1"]["pr_glass"].support, "airborne")

    def test_holding_snapshot_forces_held_support(self):
        data = fixture_data()
        timeline_of(data, "sh_wide")["states"][1]["holding_id"] = "pr_glass"
        _, states = reduce(data)
        glass = states["kf_wide_2"]["pr_glass"]
        self.assertEqual((glass.support, glass.support_id), ("held", "ch_mara"))

    def test_enter_and_exit_change_presence(self):
        data = fixture_data()
        jon_init = next(s for s in data["bible"]["initial_states"] if s["entity_id"] == "ch_jon")
        jon_init["present"] = False
        timeline_of(data, "sh_wide")["events"].append(make_event(
            "ev_enter", "kf_wide_2", "enter", actor_id="ch_jon", outcome_mark_id="mk_counter", outcome_rest="rest"))
        timeline_of(data, "sh_glass")["events"].append(make_event("ev_exit", "kf_glass_1", "exit", actor_id="ch_jon"))
        _, states = reduce(data)
        self.assertFalse(states["kf_wide_1"]["ch_jon"].present)
        self.assertTrue(states["kf_wide_2"]["ch_jon"].present)     # present from the enter's first keyframe
        self.assertTrue(states["kf_glass_1"]["ch_jon"].present)
        self.assertFalse(states["kf_glass_2"]["ch_jon"].present)   # gone once the exit completed

    def test_carried_objects_exit_and_enter_with_their_carrier(self):
        data = fixture_data()
        bible = data["bible"]
        bible["props"] += [
            {"id": "pr_mug", "name": "mug", "shape": "cylinder", "size_x_m": 0.08, "size_y_m": 0.1, "size_z_m": 0.08,
             "color_hex": "#ffffff", "movable": True},
            {"id": "pr_tray", "name": "tray", "shape": "box", "size_x_m": 0.4, "size_y_m": 0.02, "size_z_m": 0.3,
             "color_hex": "#bbbbbb", "movable": True},
            {"id": "pr_cup", "name": "cup", "shape": "cylinder", "size_x_m": 0.08, "size_y_m": 0.1, "size_z_m": 0.08,
             "color_hex": "#ffffff", "movable": True}]
        bible["characters"].append({"id": "ch_kid", "name": "Kid", "height_m": 1.2, "appearance": "a child",
                                    "wardrobe": "t-shirt", "palette_hex": ["#00ff00", "#0000ff"]})
        # Jon holds a mug and exits; the kid enters carrying a tray with a cup on it.
        next(s for s in bible["initial_states"] if s["entity_id"] == "ch_jon")["holding_id"] = "pr_mug"
        bible["initial_states"] += [
            make_state(INIT, "pr_mug", support="held", support_id="ch_jon"),
            make_state(INIT, "ch_kid", present=False, pose="stand", holding_id="pr_tray"),
            make_state(INIT, "pr_tray", present=False, support="held", support_id="ch_kid"),
            make_state(INIT, "pr_cup", present=False, support="on_entity", support_id="pr_tray")]
        for st in timeline_of(data, "sh_jon")["states"]:
            st["holding_id"] = "pr_mug"
        timeline_of(data, "sh_wide")["events"].append(make_event(
            "ev_enter", "kf_wide_2", "enter", actor_id="ch_kid", outcome_mark_id="mk_mara", outcome_rest="rest"))
        timeline_of(data, "sh_glass")["events"].append(make_event("ev_exit", "kf_glass_1", "exit", actor_id="ch_jon"))
        _, states = reduce(data)
        self.assertFalse(any(states["kf_wide_1"][e].present for e in ("ch_kid", "pr_tray", "pr_cup")))
        self.assertTrue(all(states["kf_wide_2"][e].present for e in ("ch_kid", "pr_tray", "pr_cup")))
        self.assertTrue(states["kf_glass_1"]["pr_mug"].present)
        for kf_id in ("kf_glass_2", "kf_glass_3"):
            self.assertFalse(states[kf_id]["ch_jon"].present)
            self.assertFalse(states[kf_id]["pr_mug"].present, kf_id)

    def test_snapshot_placing_a_held_object_releases_its_stale_holder(self):
        # Mara's holding_id is only carried forward; the cup's own snapshot puts it on the table.
        data = fixture_data()
        data["bible"]["props"].append({"id": "pr_cup", "name": "cup", "shape": "cylinder", "size_x_m": 0.08,
                                       "size_y_m": 0.1, "size_z_m": 0.08, "color_hex": "#ffffff", "movable": True})
        next(s for s in data["bible"]["initial_states"] if s["entity_id"] == "ch_mara")["holding_id"] = "pr_cup"
        data["bible"]["initial_states"].append(make_state(INIT, "pr_cup", support="held", support_id="ch_mara"))
        for st in timeline_of(data, "sh_wide")["states"]:
            st["holding_id"] = "pr_cup"
        timeline_of(data, "sh_jon")["states"].append(make_state(
            "kf_jon_1", "pr_cup", mark_id="mk_table", support="on_entity", support_id="pr_table"))
        _, states = reduce(data)
        self.assertEqual(states["kf_glass_3"]["ch_mara"].holding_id, "pr_cup")
        cup = states["kf_jon_1"]["pr_cup"]
        self.assertEqual((cup.support, cup.support_id, cup.mark_id), ("on_entity", "pr_table", "mk_table"))
        self.assertEqual(states["kf_jon_1"]["ch_mara"].holding_id, "none")
        # A holder that states its holding at the same keyframe still wins (explicit conflict).
        timeline_of(data, "sh_jon")["states"].append(make_state(
            "kf_jon_1", "ch_mara", mark_id="mk_mara", pose="carry", holding_id="pr_cup"))
        _, states = reduce(data)
        self.assertEqual((states["kf_jon_1"]["pr_cup"].support, states["kf_jon_1"]["pr_cup"].support_id),
                         ("held", "ch_mara"))

    def test_sit_down_and_stand_up_change_pose_and_seat(self):
        data = fixture_data()
        data["bible"]["props"].append({"id": "pr_chair", "name": "chair", "shape": "box", "size_x_m": 0.45,
                                       "size_y_m": 0.85, "size_z_m": 0.45, "color_hex": "#664422", "movable": False})
        data["bible"]["initial_states"].append(make_state(INIT, "pr_chair", mark_id="mk_counter"))
        timeline_of(data, "sh_wide")["events"].append(make_event(
            "ev_sit", "kf_wide_1", "sit_down", actor_id="ch_jon", outcome_support_id="pr_chair", outcome_rest="rest"))
        # Standing up with no stated outcome support still leaves the seat.
        timeline_of(data, "sh_glass")["events"].append(make_event("ev_up", "kf_glass_1", "stand_up", actor_id="ch_jon"))
        timeline_of(data, "sh_jon")["states"] = []
        _, states = reduce(data)
        jon = states["kf_wide_2"]["ch_jon"]
        self.assertEqual((jon.pose, jon.support, jon.support_id), ("sit", "on_entity", "pr_chair"))
        jon = states["kf_glass_2"]["ch_jon"]
        self.assertEqual((jon.pose, jon.support, jon.support_id), ("stand", "ground", "none"))

    def test_missing_initial_state_starts_absent(self):
        data = fixture_data()
        data["bible"]["initial_states"] = [s for s in data["bible"]["initial_states"] if s["entity_id"] != "ch_jon"]
        _, states = reduce(data)
        self.assertFalse(states[INIT]["ch_jon"].present)

    def test_dangling_references_do_not_raise(self):
        data = fixture_data()
        tl = timeline_of(data, "sh_glass")
        tl["events"].append(make_event("ev_ghost", "kf_glass_9", "push", target_id="pr_ghost"))
        tl["events"].append(make_event("ev_ghost2", "kf_glass_2", "push", target_id="pr_ghost", outcome_mark_id="mk_nowhere"))
        tl["keyframes"][1]["focus_event_id"] = "ev_missing"
        tl["states"].append(make_state("kf_glass_2", "pr_ghost"))
        tl["states"].append(make_state("kf_glass_7", "pr_glass"))
        index, states = reduce(data)
        self.assertEqual(set(states["kf_jon_2"]), set(index.entities))


if __name__ == "__main__":
    unittest.main()
