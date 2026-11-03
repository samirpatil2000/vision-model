"""
Planner tests: fixture planner, prompts, shot merging, and the Claude CLI transport with a mocked
subprocess (fake stream-json). No real `claude -p` call is ever made.
"""
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from src.previs.planner import FixturePlanner, make_planner, prompts
from src.previs.planner import claude_cli
from src.previs.planner.base import PlannerError, PlannerTimeoutError, PlannerTruncatedError, replace_shots
from src.previs.planner.claude_cli import (
    ClaudeCliJudge,
    ClaudeCliPlanner,
    ClaudeCliRunner,
    StreamRun,
    amend_shot_set,
    bible_violations,
    parse_answers,
)
from src.previs import physics, validator
from src.previs.planner.fixture import DEFAULT_FIXTURE
from src.previs.schema import SceneBible, ScenePlan, Timeline, bible_json_schema, timeline_json_schema
from src.previs.types import Violation

FIXTURE = ScenePlan.model_validate_json(DEFAULT_FIXTURE.read_text())
OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"
SETTINGS = {"switchModelsOnFlag": False, "permissions": {"defaultMode": "dontAsk"}}


def bible_dict() -> dict:
    return json.loads(FIXTURE.bible.model_dump_json())


def timeline_dict() -> dict:
    return json.loads(FIXTURE.timeline.model_dump_json())


def shot_dict(shot_id: str) -> dict:
    return next(s for s in timeline_dict()["shots"] if s["shot_id"] == shot_id)


def stream_lines(structured, session_id="sess-a", model=OPUS, *, is_error=False, subtype="success",
                 stop_reason="end_turn", result_text="", extra=(), with_result=True):
    """Fake stream-json: init, partial deltas, the main assistant message, then the result."""
    events = [
        {"type": "system", "subtype": "init", "session_id": session_id, "model": model, "tools": [],
         "mcp_servers": [], "permissionMode": "dontAsk", "apiKeySource": "none"},
        {"type": "stream_event", "session_id": session_id, "parent_tool_use_id": None,
         "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}}},
        {"type": "stream_event", "session_id": session_id, "parent_tool_use_id": None,
         "event": {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "{"}}},
        *extra,
        {"type": "assistant", "session_id": session_id, "parent_tool_use_id": None,
         "message": {"model": model, "role": "assistant", "content": [{"type": "text", "text": "done"}]}},
    ]
    if with_result:
        events.append({
            "type": "result", "subtype": subtype, "is_error": is_error, "result": result_text,
            "session_id": session_id, "total_cost_usd": 0.42, "stop_reason": stop_reason, "num_turns": 1,
            "usage": {"input_tokens": 100, "output_tokens": 200, "cache_read_input_tokens": 50},
            "modelUsage": {model: {"inputTokens": 100}}, "structured_output": structured,
        })
    return [json.dumps(e).encode() + b"\n" for e in events]


def ok(structured, session_id="sess", model=OPUS, **kw) -> dict:
    return {"lines": stream_lines(structured, session_id, model, **kw)}


class FakeStdout:
    def __init__(self, lines, hang=False, delay=0.0):
        self.lines = list(lines)
        self.hang = hang
        self.delay = delay
        self.released = threading.Event()   # set when the "process" is signalled

    def readline(self):
        if self.released.is_set():
            return b""
        if self.lines:
            if self.delay:
                time.sleep(self.delay)
            return self.lines.pop(0)
        if self.hang:
            self.released.wait(5.0)
        return b""

    def close(self):
        pass


class FakeStdin:
    def __init__(self):
        self.data = b""
        self.closed = False

    def write(self, data):
        self.data += data

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, response: dict):
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(response.get("lines", []), response.get("hang", False), response.get("delay", 0.0))
        self.stderr = io.BytesIO(response.get("stderr", b""))
        self._rc = response.get("returncode", 0)
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.stdout.hang and not self.stdout.released.is_set():
            if not self.stdout.released.wait(timeout):
                raise subprocess.TimeoutExpired("claude", timeout)
        self.returncode = -int(self.signals[0]) if self.signals else self._rc
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.stdout.released.set()

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)


class FakeClaude:
    """Patches subprocess.Popen/run inside claude_cli and replays scripted CLI responses."""

    def __init__(self, *responses, version="2.1.289 (Claude Code)"):
        self.responses = list(responses)
        self.calls = []
        self.version = version

    def popen(self, argv, **kw):
        if not self.responses:
            raise AssertionError(f"unexpected claude call: {argv[:6]}")
        proc = FakeProc(self.responses.pop(0))
        self.calls.append({"argv": list(argv), "env": dict(kw.get("env") or {}), "cwd": kw.get("cwd"), "proc": proc})
        return proc

    def run(self, argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=self.version + "\n", stderr="")

    def argv(self, i):
        return self.calls[i]["argv"]

    def env(self, i):
        return self.calls[i]["env"]

    def stdin(self, i):
        return self.calls[i]["proc"].stdin.data.decode("utf-8")

    def __enter__(self):
        self._patches = [mock.patch.object(claude_cli.subprocess, "Popen", self.popen),
                         mock.patch.object(claude_cli.subprocess, "run", self.run)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


def flag_value(argv, flag):
    return argv[argv.index(flag) + 1]


class PlannerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "run"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def runner_kw(self, **kw):
        kw.setdefault("replay", "off")
        kw.setdefault("cache_dir", self.tmp / "cache")
        kw.setdefault("cwd", self.tmp / "cwd")
        kw.setdefault("kill_grace_s", 1.0)
        return kw

    def planner(self, **kw) -> ClaudeCliPlanner:
        return ClaudeCliPlanner(claude_bin="claude-test", **self.runner_kw(**kw))

    def record(self, name) -> dict:
        return json.loads((self.work / "claude" / name).read_text())

    def assertSamePlan(self, a: ScenePlan, b: ScenePlan):
        self.assertEqual(a.model_dump(), b.model_dump())


# ---------------------------------------------------------------- fixture + prompts
class TestFixtureAndPrompts(unittest.TestCase):
    def test_fixture_planner_returns_fixture_and_never_edits(self):
        planner = FixturePlanner()
        plan = planner.plan("anything", Path("."))
        self.assertEqual(plan.model_dump(), FIXTURE.model_dump())
        self.assertIs(planner.repair(plan, "violation", ["sh_glass"], Path(".")), plan)
        self.assertIs(planner.amend(plan, "fact", "kf_glass_2", Path(".")), plan)

    def test_make_planner(self):
        self.assertIsInstance(make_planner("fixture"), FixturePlanner)
        self.assertIsInstance(make_planner("claude", replay="off"), ClaudeCliPlanner)
        with self.assertRaises(ValueError):
            make_planner("nope")

    def test_system_prompt_states_the_schema_rules(self):
        text = prompts.SYSTEM_PROMPT
        for phrase in (
            "Claude decides, code computes", "kf_<suffix>_1", "Never invent keyframe ids", '"none"',
            "focus_event_id and depicted_phase", "start_kf_id", "Outcome first", "cause_event_id",
            "complete state of one entity", "Write a snapshot only for an entity that changes",
            "Irreversibility", "Custody", "at most 30", "One keyframe is one instant, one beat",
            "mass_class", "friction", "action text contains no numbers", "keep reasoning out of the output",
            "at most 40 words",
            "Phase order: anticipation, contact, release, in_progress",
        ):
            self.assertIn(phrase, text)

    def test_system_prompt_embeds_the_fixture_as_few_shot(self):
        text = prompts.SYSTEM_PROMPT
        bible_block = text.split("Call A reply (bible):\n", 1)[1].split("\n\nKeyframe ids given", 1)[0]
        timeline_block = text.split("Call B reply (timeline):\n", 1)[1].split("\n\nWhy it is right", 1)[0]
        raw = json.loads(DEFAULT_FIXTURE.read_text())
        self.assertEqual(json.loads(bible_block), raw["bible"])
        self.assertEqual(json.loads(timeline_block), raw["timeline"])
        self.assertIn("kf_glass_4", text)   # the pre-created id list shown for call B

    def test_call_inputs(self):
        with self.assertRaises(ValueError):
            prompts.call_a_input("   ")
        a = prompts.call_a_input("A glass falls.")
        self.assertTrue(a.startswith("CALL A"))
        self.assertIn("<scene>\nA glass falls.\n</scene>", a)
        kf = {"sh_wide": ["kf_wide_1"], "sh_glass": ["kf_glass_1", "kf_glass_2"], "sh_jon": ["kf_jon_1"]}
        b = prompts.call_b_input(FIXTURE.bible.model_dump_json(), kf)
        self.assertTrue(b.startswith("CALL B"))
        self.assertIn("- sh_glass (order 2, keyframe_count 3, short, slow_2x, about 4 s): kf_glass_1, kf_glass_2", b)
        self.assertIn("Nominal runtime: 9.5 s (target 12 s, hard cap 30 s)", b)
        r = prompts.repair_input("bad thing", ["sh_glass"], kf)
        self.assertNotIn("<plan>", r)
        self.assertIn("sh_glass: kf_glass_1, kf_glass_2", r)
        self.assertIn("<plan>\n{}\n</plan>", prompts.repair_input("bad", ["sh_glass"], kf, plan_json="{}"))
        m = prompts.amend_input("{}", "Mara holds the glass", "kf_glass_2", ["sh_glass"], kf)
        self.assertTrue(m.startswith("AMEND"))
        self.assertIn("kf_glass_2", m)

    def test_bible_repair_input(self):
        body = prompts.bible_repair_input(" A glass falls. ", '{"scene":{}}', ["[ranges] lens", "[duration] long"])
        self.assertTrue(body.startswith("BIBLE REPAIR"))
        self.assertIn("<violations>\n1. [ranges] lens\n2. [duration] long\n</violations>", body)
        self.assertIn("<scene>\nA glass falls.\n</scene>", body)
        self.assertIn('<bible>\n{"scene":{}}\n</bible>', body)
        with self.assertRaises(ValueError):
            prompts.bible_repair_input("A glass falls.", "{}", [])

    def test_prompt_documents_bodies_seats_falls_and_remedies(self):
        text = prompts.SYSTEM_PROMPT
        for phrase in (
            'body is "human" for people and "quadruped" for four-legged animals', "the height at the shoulder",
            "support on_entity with support_id = the seat", "The feet stay on the floor",
            "within about 1 m/s x fall time of where it leaves its support",
            'use outcome_mark_id "none" so it lands straight below',
            'For a drop or fall with outcome_mark_id "none", keep the mark and offsets it had on its support',
            'When a violation proposes concrete remedies ("fix by one of: ...")', "Use only ids that already exist",
            "BIBLE REPAIR",
        ):
            self.assertIn(phrase, text)
        self.assertIn('"body":"human"', text)        # the few-shot bible carries Character.body
        self.assertIn("fix by one of", prompts.repair_input("v", ["sh_glass"]))
        self.assertIn('body "quadruped"', prompts.call_a_input("A dog barks."))

    def test_prompt_states_the_solver_seat_rule(self):
        """The seat rule in the prompt is the solver's: physics.seated_on (SEAT_MAX_M, ON_TOP_POSES)."""
        text = prompts.SYSTEM_PROMPT
        self.assertIn(f"A person whose support is an entity at most {physics.SEAT_MAX_M:g} m tall", text)
        clause = re.search(r"sits on it in every pose except ([a-z, ]+?):", text)
        self.assertIsNotNone(clause)
        self.assertEqual(set(re.split(r", | and ", clause.group(1))), set(physics.ON_TOP_POSES) - {"none"})
        for phrase in (
            "the hips rest on the seat and the feet stay on the floor",
            "stays seated: keep support on_entity with support_id = the seat",
            "To stand on a chair or crate use pose stand", "to lie on a bed or table use pose lie",
            f"A support taller than {physics.SEAT_MAX_M:g} m (counter, shelf, wall) is never a seat",
            "Animals always rest on top of their support",
            "on_entity = the top of the supporting entity, except a person seated on a seat",
            "keeps support on_entity = the seat (feet on the floor)",            # the final checklist
        ):
            self.assertIn(phrase, text)
        self.assertIn("sits with the feet on the floor: support on_entity = the seat", prompts.call_a_input("Ana sits."))

    def test_prompt_states_sit_down_stand_up_and_reach(self):
        text = prompts.SYSTEM_PROMPT
        for phrase in (
            "sit_down: the actor sits on the seat: target_id and outcome_support_id = the seat, outcome_rest rest",
            "from then on the actor's snapshots have pose sit",
            "stand_up: the actor gets up from a seat, or from lying or kneeling, to stand on the floor: "
            'outcome_support_id "none", outcome_rest rest',
            'from then on pose stand, support ground, support_id "none"',
            "A seated person leaves the seat only through stand_up",
            f"standing at most about {physics.REACH_M:g} m from it",
            "Throw only what the actor holds or can reach: otherwise add a lift first, or a walk_to",
            f"within about {physics.HAND_OFF_REACH_M:g} m of each other",
            f"{physics.REACH_M:g} m from where they stand with a lean",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn("reach about 0.6 m, and carry", text)      # the old arm-only reach contradicted the solver

    def test_prompt_states_quadruped_coat_palette(self):
        text = prompts.SYSTEM_PROMPT
        for phrase in (
            "for a quadruped its coat colours: the main coat colour first, then the legs or underside",
            "A collar, harness or leash is never in a quadruped's palette",
            "a quadruped's palette_hex is its coat (main coat, then legs or underside), its height_m the shoulder height",
            "for a quadruped the height at the shoulder, not the head",
        ):
            self.assertIn(phrase, text)
        self.assertIn("palette_hex = the coat colours (main coat first, then legs or underside; never the collar)",
                      prompts.call_a_input("A dog barks."))

    def test_prompt_asks_for_visually_distinct_prop_names(self):
        text = prompts.SYSTEM_PROMPT
        for phrase in (
            "A prop's name is what a viewer sees", 'Never a narrative or role name ("cup that falls"',
            'say the colour in the name ("white coffee cup" and "blue coffee cup"',
            "Prop names describe the look, and props of the same kind differ in colour",
        ):
            self.assertIn(phrase, text)
        self.assertIn("never the prop's role in the story", prompts.call_a_input("Two cups fall."))


# ---------------------------------------------------------------- shot merge rules
class TestReplaceShots(unittest.TestCase):
    def reply(self, shots, invariants=None) -> Timeline:
        return Timeline.model_validate({"invariants": invariants or [], "shots": shots})

    def test_replaces_only_listed_shots(self):
        glass = shot_dict("sh_glass")
        glass["keyframes"][0]["action"] = "The glass tips over the edge."
        merged = replace_shots(FIXTURE, self.reply([glass]), ["sh_glass"])
        by_id = {st.shot_id: st for st in merged.timeline.shots}
        self.assertEqual(by_id["sh_glass"].keyframes[0].action, "The glass tips over the edge.")
        for sid in ("sh_wide", "sh_jon"):
            orig = next(st for st in FIXTURE.timeline.shots if st.shot_id == sid)
            self.assertEqual(by_id[sid].model_dump(), orig.model_dump())
        self.assertEqual(merged.bible.model_dump(), FIXTURE.bible.model_dump())
        self.assertEqual([st.shot_id for st in merged.timeline.shots], ["sh_wide", "sh_glass", "sh_jon"])

    def test_rejects_other_missing_or_repeated_shots(self):
        glass, jon = shot_dict("sh_glass"), shot_dict("sh_jon")
        with self.assertRaisesRegex(PlannerError, "outside the requested set"):
            replace_shots(FIXTURE, self.reply([glass, jon]), ["sh_glass"])
        with self.assertRaisesRegex(PlannerError, "omits"):
            replace_shots(FIXTURE, self.reply([glass]), ["sh_glass", "sh_jon"])
        with self.assertRaisesRegex(PlannerError, "repeats"):
            replace_shots(FIXTURE, self.reply([glass, glass]), ["sh_glass"])

    def test_rejects_foreign_keyframe_ids(self):
        glass = shot_dict("sh_glass")
        glass["states"][0]["kf_id"] = "kf_jon_1"
        with self.assertRaisesRegex(PlannerError, "keyframe ids of other shots"):
            replace_shots(FIXTURE, self.reply([glass]), ["sh_glass"])

    def test_invariant_merge_rules(self):
        glass = shot_dict("sh_glass")
        ivs = {iv["id"]: iv for iv in timeline_dict()["invariants"]}
        broken = dict(ivs["iv_glass_broken"], from_kf_id="kf_glass_2")
        # Touching invariant replaced; untouched echo accepted; untouched originals kept.
        merged = replace_shots(FIXTURE, self.reply([glass], [broken, ivs["iv_table_fixed"]]), ["sh_glass"])
        got = {iv.id: iv for iv in merged.timeline.invariants}
        self.assertEqual(set(got), {"iv_glass_broken", "iv_table_fixed", "iv_glass_supported"})
        self.assertEqual(got["iv_glass_broken"].from_kf_id, "kf_glass_2")
        # Changing an invariant that lies outside the repaired shots is rejected.
        changed = dict(ivs["iv_glass_supported"], until_kf_id="kf_wide_2")
        with self.assertRaisesRegex(PlannerError, "changes invariant iv_glass_supported"):
            replace_shots(FIXTURE, self.reply([glass], [changed]), ["sh_glass"])

    def test_rejects_duplicate_event_ids(self):
        glass = shot_dict("sh_glass")
        glass["events"][2]["id"] = "ev_turn"      # already used in sh_jon
        with self.assertRaisesRegex(PlannerError, "duplicate event ids"):
            replace_shots(FIXTURE, self.reply([glass]), ["sh_glass"])

    def test_bad_shot_ids_are_caller_errors(self):
        with self.assertRaises(ValueError):
            replace_shots(FIXTURE, self.reply([]), [])
        with self.assertRaises(ValueError):
            replace_shots(FIXTURE, self.reply([]), ["sh_nope"])

    def test_amend_shot_set(self):
        self.assertEqual(amend_shot_set(FIXTURE, "kf_glass_2"), ["sh_glass", "sh_jon"])
        self.assertEqual(amend_shot_set(FIXTURE, "kf_wide_1"), ["sh_wide", "sh_glass", "sh_jon"])
        self.assertEqual(amend_shot_set(FIXTURE, "kf_jon_1"), ["sh_jon"])
        with self.assertRaises(ValueError):
            amend_shot_set(FIXTURE, "kf_nope_1")


# ---------------------------------------------------------------- bible checks + bible repair round
def mutated_bible(change) -> dict:
    bible = bible_dict()
    change(bible)
    return bible


def by_entity(items, entity_id) -> dict:
    return next(i for i in items if i["entity_id"] == entity_id)


def second_location(b) -> None:
    b["locations"].append(dict(b["locations"][0], id="lo_yard"))
    b["shots"][2].update(location_id="lo_yard", setup_id="su_wide")


BIBLE_DEFECTS = {
    "shot sh_wide lists unknown beat bt_nope": lambda b: b["shots"][0]["beat_ids"].append("bt_nope"),
    "shot sh_wide subject ch_nope does not exist": lambda b: b["shots"][0].update(subject_id="ch_nope"),
    "shot sh_jon secondary pr_nope does not exist": lambda b: b["shots"][2].update(secondary_id="pr_nope"),
    "shot sh_glass is in unknown location lo_nope": lambda b: b["shots"][1].update(location_id="lo_nope"),
    "mark mk_table is in unknown location lo_nope": lambda b: b["marks"][0].update(location_id="lo_nope"),
    "beat bt_knock has unknown cause_beat_id bt_nope": lambda b: b["beats"][1].update(cause_beat_id="bt_nope"),
    "initial state of ch_mara: mark_id 'mk_nope' does not exist":
        lambda b: by_entity(b["initial_states"], "ch_mara").update(mark_id="mk_nope"),
    "initial state of ch_jon: kf_id 'kf_wide_1' must be \"init\"":
        lambda b: by_entity(b["initial_states"], "ch_jon").update(kf_id="kf_wide_1"),
    "initial state of pr_glass: support_id 'pr_nope' does not exist":
        lambda b: by_entity(b["initial_states"], "pr_glass").update(support_id="pr_nope"),
    "pr_glass has no initial state": lambda b: b["initial_states"].remove(by_entity(b["initial_states"], "pr_glass")),
    "pr_table has no phys_body": lambda b: b["phys_bodies"].remove(by_entity(b["phys_bodies"], "pr_table")),
    "ch_jon has 2 initial states": lambda b: b["initial_states"].append(by_entity(b["initial_states"], "ch_jon")),
    "phys_body for unknown entity pr_nope": lambda b: b["phys_bodies"].append(dict(b["phys_bodies"][0], entity_id="pr_nope")),
    "mark id mk_table is used more than once": lambda b: b["marks"].append(b["marks"][0]),
    "[keyframe_ids] shot id sh_xxxxxxxxxxxxxxxxxxxxxxxxxxx is too long":
        lambda b: b["shots"][0].update(id="sh_" + "x" * 27),
    "[keyframe_ids] shot sh_wide keyframe_count is 0": lambda b: b["shots"][0].update(keyframe_count=0),
    "[keyframe_ids] shot sh_glass keyframe_count is 5": lambda b: b["shots"][1].update(keyframe_count=5),
    "[ranges] shot sh_wide lens 8 mm is outside 10-300 mm": lambda b: b["shots"][0].update(lens_mm=8),
    "[ranges] mark mk_table (999, 2) is outside lo_kitchen (0..5 x 0..4 m)": lambda b: b["marks"][0].update(x_m=999),
    "[ranges] ch_mara height 3.1 m is outside 0.3-2.5 m": lambda b: b["characters"][0].update(height_m=3.1),
    "[ranges] pr_glass has a non-positive size": lambda b: b["props"][1].update(size_y_m=0),
    "[ranges] location lo_kitchen has a non-positive extent": lambda b: b["locations"][0].update(depth_m=0),
    "[duration] target_duration_s is 40 s": lambda b: b["scene"].update(target_duration_s=40),
    "[line_side] sh_jon puts the camera on side 'b' of the ch_jon/ch_mara line but sh_wide used side 'a'":
        lambda b: b["shots"][2].update(line_side="b"),
    "[setup_location] setup su_wide is used in lo_kitchen and in lo_yard (sh_jon)": second_location,
}


def local_checks_only(test: unittest.TestCase) -> None:
    """Pin bible_violations to the planner's own checks and wording: hide validator.validate_bible."""
    patcher = mock.patch.object(validator, "validate_bible", None, create=True)
    patcher.start()
    test.addCleanup(patcher.stop)


def add_prop(b: dict, prop_id: str, **changes) -> None:
    """A copy of the fixture glass (prop, phys_body, init state on the table) under a new id."""
    b["props"].append(dict(next(p for p in b["props"] if p["id"] == "pr_glass"), id=prop_id, **changes))
    b["phys_bodies"].append(dict(by_entity(b["phys_bodies"], "pr_glass"), entity_id=prop_id))
    b["initial_states"].append(dict(by_entity(b["initial_states"], "pr_glass"), entity_id=prop_id, offset_x_m=-0.4))


class TestBibleChecks(PlannerTestCase):
    def setUp(self):
        super().setUp()
        local_checks_only(self)

    def test_sound_bibles_pass(self):
        self.assertEqual(bible_violations(FIXTURE.bible), [])
        crossed_by_orbit = mutated_bible(lambda b: b["shots"][2].update(line_side="b", move="orbit_left"))
        self.assertEqual(bible_violations(SceneBible.model_validate(crossed_by_orbit)), [])

    def test_bible_defects_no_timeline_repair_can_fix(self):
        for expected, change in BIBLE_DEFECTS.items():
            with self.subTest(expected):
                found = bible_violations(SceneBible.model_validate(mutated_bible(change)))
                self.assertTrue(any(expected in f for f in found), found)

    def test_look_alike_props_must_be_told_apart(self):
        """Two movable props of one shape, size and colour (the cafe's 'coffee cup that falls' and 'coffee cup
        that is caught', both #f2efe8): no frame, prompt or judge can tell them apart, so call A must fix it."""
        twin = SceneBible.model_validate(mutated_bible(lambda b: add_prop(b, "pr_glass_b", name="glass that is caught",
                                                                            color_hex="#c4e9ff")))
        found = bible_violations(twin)
        self.assertEqual(len(found), 1, found)
        self.assertTrue(found[0].startswith("[distinct_props] props pr_glass ('glass of water', #bfe6ff) and "
                                            "pr_glass_b ('glass that is caught', #c4e9ff) are the same shape, size and colour"))
        self.assertIn("'white coffee cup', 'blue coffee cup'", found[0])
        # Told apart by colour, shape or size, or one of them never moves: sound.
        for label, changes in (("colour", {"color_hex": "#2244aa"}), ("shape", {"shape": "box"}),
                               ("size", {"size_y_m": 0.3}), ("fixed", {"movable": False})):
            with self.subTest(label):
                other = mutated_bible(lambda b: add_prop(b, "pr_glass_b", **changes))
                self.assertEqual(bible_violations(SceneBible.model_validate(other)), [])

    def test_bible_defect_is_repaired_before_call_b(self):
        bad = mutated_bible(lambda b: b["shots"][0].update(lens_mm=8))
        with FakeClaude(ok(bad, "sess-a"), ok(bible_dict(), "sess-fix"), ok(timeline_dict(), "sess-b")) as fake:
            plan = self.planner().plan("A glass falls.", self.work)
        self.assertSamePlan(plan, FIXTURE)
        self.assertEqual(len(fake.calls), 3)
        fix = fake.argv(1)
        self.assertEqual(fix[-3:], ["--resume", "sess-a", "--fork-session"])
        self.assertEqual(json.loads(flag_value(fix, "--json-schema")), bible_json_schema())
        self.assertEqual(flag_value(fix, "--system-prompt-file"), flag_value(fake.argv(0), "--system-prompt-file"))
        self.assertEqual(fake.env(1)["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "64000")
        body = fake.stdin(1)
        self.assertTrue(body.startswith("BIBLE REPAIR"))
        self.assertIn("1. [ranges] shot sh_wide lens 8 mm is outside 10-300 mm", body)
        self.assertIn("<scene>\nA glass falls.\n</scene>", body)
        self.assertIn('"lens_mm":8.0', body)
        # B forks the repaired session and is built on the repaired bible.
        self.assertEqual(fake.argv(2)[-3:], ["--resume", "sess-fix", "--fork-session"])
        self.assertTrue(fake.stdin(2).startswith("CALL B"))
        self.assertNotIn('"lens_mm":8.0', fake.stdin(2))
        self.assertIsNone(self.record("bible_repair_1.json")["error"])

    def test_unrepairable_bible_stops_before_call_b(self):
        bad = mutated_bible(lambda b: b["shots"][0].update(lens_mm=8))
        with FakeClaude(ok(bad, "sess-a"), ok(bad, "sess-fix1"), ok(bad, "sess-fix2")) as fake:
            with self.assertRaisesRegex(PlannerError, "repaired bible still fails 1 check"):
                self.planner(replay="record").plan("A glass falls.", self.work)
        self.assertEqual(len(fake.calls), 3)                     # A, the fork, the fresh fallback; never B
        self.assertIn("--resume", fake.argv(1))
        self.assertNotIn("--resume", fake.argv(2))
        self.assertEqual(fake.stdin(2), fake.stdin(1))           # the body carries the scene and the bible
        # The unrepairable bible is not replayed into the next run.
        self.assertEqual(list((self.tmp / "cache").rglob("*.json")), [])

    def test_bible_repair_is_replayed_with_the_rest_of_the_plan(self):
        bad = mutated_bible(lambda b: b["shots"][0].update(lens_mm=8))
        with FakeClaude(ok(bad, "sess-a"), ok(bible_dict(), "sess-fix"), ok(timeline_dict(), "sess-b")):
            first = self.planner(replay="record").plan("A glass falls.", self.work)
        with FakeClaude() as fake:
            again = self.planner(replay="record").plan("A glass falls.", self.work)
        self.assertEqual(fake.calls, [])
        self.assertSamePlan(again, first)
        self.assertTrue(self.record("bible_repair_2.json")["replayed"])


def init_custody_defect(b: dict) -> None:
    """pr_glass starts in Mara's hand while Mara holds nothing: an 'init' custody defect only the validator sees."""
    by_entity(b["initial_states"], "pr_glass").update(support="held", support_id="ch_mara", mark_id="mk_mara",
                                                      offset_x_m=0.0, offset_z_m=0.0)


def fake_validate_bible(bible: SceneBible):
    """Stand-in for validator.validate_bible: one init custody error, plus a warning that must not block."""
    glass = next(s for s in bible.initial_states if s.entity_id == "pr_glass")
    mara = next(s for s in bible.initial_states if s.entity_id == "ch_mara")
    out = [Violation("ranges", "pr_table is wider than most kitchen tables", entity_id="pr_table", severity="warning")]
    if glass.support == "held" and mara.holding_id != "pr_glass":
        out.append(Violation("custody", "pr_glass is held by ch_mara at init but ch_mara holds none",
                             kf_id="init", entity_id="pr_glass"))
    return out


class TestSharedBibleChecks(PlannerTestCase):
    """bible_violations uses validator.validate_bible when it exists, and falls back to the local checks."""

    def test_validator_bible_errors_are_reported(self):
        bad = SceneBible.model_validate(mutated_bible(init_custody_defect))
        with mock.patch.object(validator, "validate_bible", side_effect=fake_validate_bible, create=True) as shared:
            found = bible_violations(bad)
            sound = bible_violations(FIXTURE.bible)
        self.assertEqual(shared.call_count, 2)
        self.assertEqual(found, ["[custody] keyframe init, entity pr_glass: pr_glass is held by ch_mara at init "
                                 "but ch_mara holds none"])
        self.assertEqual(sound, [])                                    # warnings never trigger a bible repair

    def test_a_defect_both_report_is_listed_once(self):
        bad = SceneBible.model_validate(mutated_bible(lambda b: (b["shots"][0].update(lens_mm=8),
                                                                 b["shots"][1].update(keyframe_count=5))))
        shared = [Violation("ranges", "lens_mm 8 is outside 10-300 mm", shot_id="sh_wide")]
        with mock.patch.object(validator, "validate_bible", return_value=shared, create=True):
            found = bible_violations(bad)
        self.assertEqual(found, ["[ranges] shot sh_wide: lens_mm 8 is outside 10-300 mm",
                                 "[keyframe_ids] shot sh_glass keyframe_count is 5; it must be 1 to 4"])
        # Another defect of the same shot under the same code is not swallowed.
        two = SceneBible.model_validate(mutated_bible(lambda b: (b["shots"][0].update(subject_id="ch_nope"),
                                                                 b["shots"][0]["beat_ids"].append("bt_nope"))))
        shared = [Violation("ref_integrity", "subject_id ch_nope does not exist", shot_id="sh_wide")]
        with mock.patch.object(validator, "validate_bible", return_value=shared, create=True):
            found = bible_violations(two)
        self.assertEqual(found, ["[ref_integrity] shot sh_wide: subject_id ch_nope does not exist",
                                 "[ref_integrity] shot sh_wide lists unknown beat bt_nope"])

    def test_local_checks_are_the_fallback(self):
        bad = SceneBible.model_validate(mutated_bible(lambda b: b["shots"][0].update(lens_mm=8)))
        expected = ["[ranges] shot sh_wide lens 8 mm is outside 10-300 mm"]
        with mock.patch.object(validator, "validate_bible", None, create=True):
            self.assertEqual(bible_violations(bad), expected)                     # not there yet
        with mock.patch.object(validator, "validate_bible", side_effect=KeyError("pr_nope"), create=True):
            self.assertEqual(bible_violations(bad), expected)                     # fails on this bible

    def test_validator_only_defect_is_repaired_before_call_b(self):
        bad = mutated_bible(init_custody_defect)
        with mock.patch.object(validator, "validate_bible", side_effect=fake_validate_bible, create=True), \
                FakeClaude(ok(bad, "sess-a"), ok(bible_dict(), "sess-fix"), ok(timeline_dict(), "sess-b")) as fake:
            plan = self.planner().plan("A glass falls.", self.work)
        self.assertSamePlan(plan, FIXTURE)
        self.assertEqual(len(fake.calls), 3)
        body = fake.stdin(1)
        self.assertTrue(body.startswith("BIBLE REPAIR"))
        self.assertIn("1. [custody] keyframe init, entity pr_glass: pr_glass is held by ch_mara at init", body)
        self.assertNotIn("wider than most kitchen tables", body)
        self.assertEqual(fake.argv(2)[-3:], ["--resume", "sess-fix", "--fork-session"])
        self.assertTrue(fake.stdin(2).startswith("CALL B"))

    @unittest.skipUnless(callable(getattr(validator, "validate_bible", None)), "validator.validate_bible not available")
    def test_real_validator_bible_checks(self):
        self.assertEqual([v for v in validator.validate_bible(FIXTURE.bible) if v.severity == "error"], [])
        self.assertEqual(bible_violations(FIXTURE.bible), [])
        found = bible_violations(SceneBible.model_validate(mutated_bible(init_custody_defect)))
        self.assertTrue(any(line.startswith("[custody]") and "pr_glass" in line for line in found), found)


# ---------------------------------------------------------------- argv, stdin, env
class TestInvocation(PlannerTestCase):
    def test_call_a_and_b_argv(self):
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")) as fake:
            plan = self.planner().plan("A glass falls off a table.", self.work)
        self.assertSamePlan(plan, FIXTURE)
        a, b = fake.argv(0), fake.argv(1)
        prefix = ["claude-test", "-p", "--model", OPUS, "--effort", "high", "--safe-mode", "--restricted",
                  "--strict-mcp-config", "--tools", "", "--disable-slash-commands", "--permission-prompts", "none",
                  "--prompt-suggestions", "false", "--no-chrome", "--settings"]
        for argv in (a, b):
            self.assertEqual(argv[:len(prefix)], prefix)
            self.assertEqual(flag_value(argv, "--max-turns"), "3")
            self.assertEqual(flag_value(argv, "--output-format"), "stream-json")
            self.assertIn("--verbose", argv)
            self.assertIn("--include-partial-messages", argv)
            for banned in ("--fallback-model", "--max-budget-usd", "--bare", "--system-prompt", "--append-system-prompt"):
                self.assertNotIn(banned, argv)
            self.assertEqual(json.loads(Path(flag_value(argv, "--settings")).read_text()), SETTINGS)
        # A: fresh session with the static bible schema.
        self.assertEqual(a[-2], "--session-id")
        uuid.UUID(a[-1])
        self.assertNotIn("--resume", a)
        self.assertEqual(json.loads(flag_value(a, "--json-schema")), bible_json_schema())
        # B: fork of A with the per-scene timeline schema and the byte-identical system prompt file.
        self.assertEqual(b[-3:], ["--resume", "sess-a", "--fork-session"])
        self.assertEqual(json.loads(flag_value(b, "--json-schema")), timeline_json_schema(FIXTURE.bible))
        prompt_file = Path(flag_value(a, "--system-prompt-file"))
        self.assertEqual(flag_value(b, "--system-prompt-file"), str(prompt_file))
        self.assertEqual(prompt_file, (self.work / "claude" / "system_prompt.md").resolve())
        self.assertEqual(prompt_file.read_text(encoding="utf-8"), prompts.SYSTEM_PROMPT)
        self.assertEqual(fake.calls[0]["cwd"], str(self.tmp / "cwd"))

    def test_input_goes_on_stdin_only(self):
        token = "UNIQUE-SCENE-TOKEN"
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")) as fake:
            self.planner().plan(f"{token}: a glass falls.", self.work)
        stdin_a, stdin_b = fake.stdin(0), fake.stdin(1)
        self.assertTrue(stdin_a.startswith("CALL A"))
        self.assertIn(token, stdin_a)
        self.assertTrue(stdin_b.startswith("CALL B"))
        self.assertIn('"id":"ch_mara"', stdin_b)
        self.assertIn("kf_glass_4", stdin_b)
        for i in range(2):
            argv = fake.argv(i)
            self.assertFalse(any(token in arg or "CALL " in arg for arg in argv))
            self.assertTrue(argv[2].startswith("--"))             # nothing positional after -p
            self.assertTrue(fake.calls[i]["proc"].stdin.closed)

    def test_env_allowlist_subscription(self):
        parent = {
            "PATH": "/usr/bin:/bin", "HOME": str(self.tmp), "HTTPS_PROXY": "http://proxy:3128",
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret-123",
            "CLAUDE_CODE_SIMPLE": "1", "CLAUDE_EFFORT": "low", "ANTHROPIC_BASE_URL": "http://elsewhere",
            "ANTHROPIC_API_KEY": "sk-must-not-pass", "ANTHROPIC_AUTH_TOKEN": "tok", "CLAUDECODE": "1",
            "CLAUDE_CODE_CHILD_SESSION": "1", "CLAUDE_CODE_EFFORT_LEVEL": "max", "API_TIMEOUT_MS": "5",
            "CLAUDE_CODE_MESSAGING_TOKEN": "secret",
        }
        with mock.patch.dict(os.environ, parent):
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
            with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")) as fake:
                self.planner().plan("A glass falls.", self.work)
        for i, max_out in ((0, "64000"), (1, "128000")):
            env = fake.env(i)
            for bad in ("CLAUDE_CODE_SIMPLE", "CLAUDE_EFFORT", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY",
                        "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_EFFORT_LEVEL",
                        "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CONFIG_DIR", "API_FORCE_IDLE_TIMEOUT",
                        "CLAUDE_CODE_RETRY_WATCHDOG"):
                self.assertNotIn(bad, env)
            self.assertEqual(env["PATH"], "/usr/bin:/bin")
            self.assertEqual(env["HOME"], str(self.tmp))
            self.assertEqual(env["HTTPS_PROXY"], "http://proxy:3128")
            self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "oauth-secret-123")
            self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], max_out)
            self.assertEqual(env["API_TIMEOUT_MS"], "2700000")
            self.assertEqual(env["CLAUDE_CODE_MAX_RETRIES"], "1")
            self.assertEqual(env["DISABLE_AUTOUPDATER"], "1")
        # Records list env names only, never values.
        for path in (self.work / "claude").iterdir():
            self.assertNotIn("oauth-secret-123", path.read_text())
        self.assertIn("CLAUDE_CODE_OAUTH_TOKEN", self.record("a_1.json")["env_keys"])

    def test_api_key_mode_config_dir_and_idle_override(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")) as fake:
                self.planner(auth="api_key", config_dir=str(self.tmp / "cfg"), force_idle_timeout_ms=900000,
                             restricted=False).plan("A glass falls.", self.work)
        env, argv = fake.env(0), fake.argv(0)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-test")
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], str(self.tmp / "cfg"))
        self.assertEqual(env["API_FORCE_IDLE_TIMEOUT"], "900000")
        self.assertIn("--bare", argv)
        self.assertNotIn("--restricted", argv)

    def test_api_key_mode_without_key_fails_before_spawning(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with FakeClaude() as fake:
                with self.assertRaises(PlannerError) as ctx:
                    self.planner(auth="api_key").plan("A glass falls.", self.work)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(fake.calls, [])

    def test_bad_configuration_values(self):
        with self.assertRaises(ValueError):
            self.planner(auth="oauth")
        with self.assertRaises(ValueError):
            self.planner(replay="sometimes")
        with mock.patch.dict(os.environ, {"PREVIS_CLAUDE_REPLAY": "replay"}):
            self.assertEqual(ClaudeCliRunner(cache_dir=self.tmp).replay, "replay")

    def test_records_are_written(self):
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")):
            self.planner().plan("A glass falls.", self.work)
        rec = self.record("a_1.json")
        self.assertEqual(rec["session_id"], "sess-a")
        self.assertEqual(rec["total_cost_usd"], 0.42)
        self.assertEqual(rec["usage"]["output_tokens"], 200)
        self.assertEqual(rec["cli_version"], "2.1.289")
        self.assertTrue(rec["stdin"].startswith("CALL A"))
        self.assertEqual(json.loads(flag_value(rec["argv"], "--json-schema")), bible_json_schema())
        self.assertEqual(rec["schema"], bible_json_schema())
        self.assertNotIn(prompts.SYSTEM_PROMPT[:200], json.dumps(rec["argv"]))
        self.assertEqual(self.record("b_1.json")["resumed_from"], "sess-a")
        events = (self.work / "claude" / "a_1.events.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(e)["type"] for e in events], ["system", "assistant", "result"])
        state = json.loads((self.work / "claude" / "session.json").read_text())
        self.assertEqual(state["session_id"], "sess-b")


# ---------------------------------------------------------------- sessions: B fallback, repair, amend
class TestSessions(PlannerTestCase):
    def test_b_falls_back_to_fresh_session(self):
        failed = {"lines": stream_lines(None, "sess-b1", is_error=True, subtype="error_during_execution",
                                        result_text="schema change on resume not supported")}
        with FakeClaude(ok(bible_dict(), "sess-a"), failed, ok(timeline_dict(), "sess-b2")) as fake:
            plan = self.planner().plan("A glass falls.", self.work)
        self.assertSamePlan(plan, FIXTURE)
        self.assertEqual(len(fake.calls), 3)
        fresh = fake.argv(2)
        self.assertNotIn("--resume", fresh)
        self.assertEqual(fresh[-2], "--session-id")
        self.assertEqual(fake.stdin(2), fake.stdin(1))          # B's body already carries the bible
        self.assertIn("error_during_execution", self.record("b_1.json")["error"])
        self.assertIsNone(self.record("b_2.json")["error"])
        self.assertEqual(json.loads((self.work / "claude" / "session.json").read_text())["session_id"], "sess-b2")

    def test_b_reply_missing_a_shot_also_falls_back(self):
        partial = timeline_dict()
        partial["shots"] = partial["shots"][:2]
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(partial, "sess-b1"), ok(timeline_dict(), "sess-b2")) as fake:
            plan = self.planner().plan("A glass falls.", self.work)
        self.assertSamePlan(plan, FIXTURE)
        self.assertEqual(len(fake.calls), 3)
        self.assertIn("do not match the bible shots", self.record("b_1.json")["error"])

    def _repair_reply(self):
        glass = shot_dict("sh_glass")
        glass["keyframes"][1]["action"] = "The glass falls, water streaming out."
        return {"invariants": [iv for iv in timeline_dict()["invariants"] if iv["id"] == "iv_glass_broken"],
                "shots": [glass]}

    def test_repair_forks_the_last_good_session(self):
        planner = self.planner()
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b"), ok(self._repair_reply(), "sess-r")) as fake:
            plan = planner.plan("A glass falls.", self.work)
            repaired = planner.repair(plan, "airborne glass outside a fall at kf_glass_2", ["sh_glass"], self.work)
        argv = fake.argv(2)
        self.assertEqual(argv[-3:], ["--resume", "sess-b", "--fork-session"])
        self.assertEqual(json.loads(flag_value(argv, "--json-schema")), timeline_json_schema(FIXTURE.bible))
        self.assertEqual(flag_value(argv, "--system-prompt-file"), flag_value(fake.argv(0), "--system-prompt-file"))
        body = fake.stdin(2)
        self.assertTrue(body.startswith("REPAIR"))
        self.assertIn("airborne glass outside a fall", body)
        self.assertNotIn("<plan>", body)
        by_id = {st.shot_id: st for st in repaired.timeline.shots}
        self.assertEqual(by_id["sh_glass"].keyframes[1].action, "The glass falls, water streaming out.")
        self.assertEqual(by_id["sh_jon"].model_dump(), FIXTURE.timeline.shots[2].model_dump())
        self.assertEqual(json.loads((self.work / "claude" / "session.json").read_text())["session_id"], "sess-r")

    def test_repair_resumes_session_from_disk_for_the_same_plan(self):
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")):
            plan = self.planner().plan("A glass falls.", self.work)
        with FakeClaude(ok(self._repair_reply(), "sess-r")) as fake:
            self.planner().repair(plan, "violation", ["sh_glass"], self.work)
        self.assertEqual(fake.argv(0)[-3:], ["--resume", "sess-b", "--fork-session"])

    def test_repair_goes_fresh_without_a_matching_session(self):
        with FakeClaude(ok(self._repair_reply(), "sess-r")) as fake:
            repaired = self.planner().repair(FIXTURE, "violation", ["sh_glass"], self.work)
        argv = fake.argv(0)
        self.assertNotIn("--resume", argv)
        self.assertEqual(argv[-2], "--session-id")
        self.assertIn("<plan>", fake.stdin(0))
        self.assertIn(FIXTURE.model_dump_json(), fake.stdin(0))
        self.assertEqual(repaired.timeline.shots[1].keyframes[1].action, "The glass falls, water streaming out.")

    def test_repair_falls_back_to_fresh_when_the_fork_fails(self):
        planner = self.planner()
        failed = {"lines": stream_lines(None, "sess-x", is_error=True, subtype="error_during_execution", result_text="boom")}
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b"), failed, ok(self._repair_reply(), "sess-r")) as fake:
            plan = planner.plan("A glass falls.", self.work)
            planner.repair(plan, "violation", ["sh_glass"], self.work)
        self.assertIn("--resume", fake.argv(2))
        self.assertNotIn("--resume", fake.argv(3))
        self.assertIn("<plan>", fake.stdin(3))

    def test_repair_rejects_replies_touching_other_shots(self):
        planner = self.planner()
        everything = timeline_dict()
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b"),
                        ok(everything, "sess-r1"), ok(everything, "sess-r2")) as fake:
            plan = planner.plan("A glass falls.", self.work)
            with self.assertRaisesRegex(PlannerError, "outside the requested set"):
                planner.repair(plan, "violation", ["sh_glass"], self.work)
        self.assertEqual(len(fake.calls), 4)
        # A rejected reply never becomes the last good session.
        self.assertEqual(json.loads((self.work / "claude" / "session.json").read_text())["session_id"], "sess-b")

    def test_amend_uses_a_fresh_session_and_replaces_dependent_shots(self):
        jon = shot_dict("sh_jon")
        jon["states"][1]["expression"] = "afraid"
        with FakeClaude(ok({"invariants": [], "shots": [jon]}, "sess-m")) as fake:
            amended = self.planner().amend(FIXTURE, "Jon looks afraid, not surprised.", "kf_jon_2", self.work)
        argv, body = fake.argv(0), fake.stdin(0)
        self.assertNotIn("--resume", argv)
        self.assertEqual(argv[-2], "--session-id")
        self.assertTrue(body.startswith("AMEND"))
        self.assertIn("kf_jon_2", body)
        self.assertIn("Jon looks afraid", body)
        self.assertIn("<plan>", body)
        self.assertEqual(amended.timeline.shots[2].states[1].expression, "afraid")
        self.assertEqual(amended.timeline.shots[0].model_dump(), FIXTURE.timeline.shots[0].model_dump())
        # Invariants that touch sh_jon but were not returned are dropped; the rest are kept.
        self.assertEqual({iv.id for iv in amended.timeline.invariants}, {"iv_glass_supported"})


# ---------------------------------------------------------------- result checks
class TestResultChecks(PlannerTestCase):
    def plan_a_only(self, response, **kw):
        with FakeClaude(response) as fake:
            with self.assertRaises(PlannerError) as ctx:
                self.planner(**kw).plan("A glass falls.", self.work)
        return ctx.exception, fake

    def test_model_rerouted(self):
        err, fake = self.plan_a_only(ok(bible_dict(), "sess-a", model="claude-opus-5"))
        self.assertIn("model rerouted", str(err))
        self.assertFalse(err.retryable)

    def test_model_rerouted_from_usage_when_no_assistant_message(self):
        lines = [line for line in stream_lines(bible_dict(), "sess-a", model="claude-sonnet-5-5")
                 if json.loads(line)["type"] != "assistant"]
        err, _ = self.plan_a_only({"lines": lines})
        self.assertIn("model rerouted", str(err))

    def test_auxiliary_models_are_ignored(self):
        extra = (
            {"type": "assistant", "parent_tool_use_id": "toolu_1", "message": {"model": HAIKU, "content": []}},
            {"type": "assistant", "parent_tool_use_id": None, "message": {"model": "<synthetic>", "content": []}},
        )
        a = {"lines": stream_lines(bible_dict(), "sess-a", extra=extra)}
        with FakeClaude(a, ok(timeline_dict(), "sess-b")):
            plan = self.planner().plan("A glass falls.", self.work)
        self.assertSamePlan(plan, FIXTURE)

    def test_truncation_is_detected(self):
        err, _ = self.plan_a_only(ok(bible_dict(), "sess-a", stop_reason="max_tokens"))
        self.assertIsInstance(err, PlannerTruncatedError)
        err, _ = self.plan_a_only(ok(None, "sess-a", is_error=True, subtype="error_during_execution",
                                     result_text="The response above may be incomplete"))
        self.assertIsInstance(err, PlannerTruncatedError)

    def test_error_results(self):
        err, _ = self.plan_a_only(ok(None, "sess-a", is_error=True, subtype="error_max_turns"))
        self.assertIn("error_max_turns", str(err))
        err, _ = self.plan_a_only(ok(None, "sess-a"))
        self.assertIn("no structured_output", str(err))
        err, _ = self.plan_a_only(ok(bible_dict(), "sess-a", stop_reason="refusal"))
        self.assertIn("refused", str(err))
        err, _ = self.plan_a_only({"lines": stream_lines(None, "sess-a", with_result=False), "returncode": 1})
        self.assertIn("no result message", str(err))
        err, _ = self.plan_a_only(ok({**bible_dict(), "shots": []}, "sess-a"))
        self.assertIn("at least one shot", str(err))

    def test_usage_limit_is_not_retried(self):
        limit = ok(None, "sess-b1", is_error=True, subtype="success", result_text="You've hit your session limit · resets 3pm")
        with FakeClaude(ok(bible_dict(), "sess-a"), limit) as fake:
            with self.assertRaisesRegex(PlannerError, "usage limit") as ctx:
                self.planner().plan("A glass falls.", self.work)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(fake.calls), 2)                 # no fresh fallback for B

    def test_unknown_flag_is_named(self):
        err, fake = self.plan_a_only({"lines": [], "stderr": b"error: unknown option '--restricted'\n", "returncode": 1})
        self.assertIn("--restricted", str(err))
        self.assertFalse(err.retryable)

    def test_compaction_is_fatal(self):
        compact = ({"type": "system", "subtype": "compact_boundary", "session_id": "sess-a"},)
        err, fake = self.plan_a_only({"lines": stream_lines(bible_dict(), "sess-a", extra=compact)})
        self.assertIn("compact", str(err))
        self.assertEqual(fake.calls[0]["proc"].signals[0], signal.SIGINT)


# ---------------------------------------------------------------- watchdog
class TestWatchdog(PlannerTestCase):
    def test_idle_timeout_terminates_the_process(self):
        init = stream_lines(None, "sess-a")[:1]
        started = time.monotonic()
        with FakeClaude({"lines": init, "hang": True}) as fake:
            with self.assertRaisesRegex(PlannerTimeoutError, "no stream event"):
                self.planner(idle_timeout_s=0.2).plan("A glass falls.", self.work)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(fake.calls[0]["proc"].signals[0], signal.SIGINT)
        self.assertIn("no stream event", self.record("a_1.json")["abort_reason"])

    def test_first_text_timeout(self):
        thinking = stream_lines(None, "sess-a")[:2] * 40     # init + thinking deltas, never any text
        with FakeClaude({"lines": thinking, "hang": True, "delay": 0.02}):
            with self.assertRaisesRegex(PlannerTimeoutError, "no output text"):
                self.planner(idle_timeout_s=5, first_text_timeout_s=0.25).plan("A glass falls.", self.work)

    def test_total_timeout(self):
        lines = stream_lines(None, "sess-a", with_result=False) * 20
        with FakeClaude({"lines": lines, "hang": True, "delay": 0.03}):
            with self.assertRaisesRegex(PlannerTimeoutError, "total time limit"):
                self.planner(idle_timeout_s=5, total_timeout_s=0.3).plan("A glass falls.", self.work)

    def test_real_process_is_stopped_by_the_watchdog(self):
        script = self.tmp / "fake-claude"
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, sys, time\n"
            "if '--version' in sys.argv:\n"
            "    print('2.1.289 (Claude Code)'); sys.exit(0)\n"
            "sys.stdin.read()\n"
            "print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': 'real-1'}), flush=True)\n"
            "time.sleep(30)\n"
        )
        script.chmod(0o755)
        planner = ClaudeCliPlanner(claude_bin=str(script), **self.runner_kw(idle_timeout_s=0.3))
        started = time.monotonic()
        with self.assertRaises(PlannerTimeoutError):
            planner.plan("A glass falls.", self.work)
        self.assertLess(time.monotonic() - started, 5.0)
        rec = self.record("a_1.json")
        self.assertEqual(rec["session_id"], "real-1")
        self.assertIsNotNone(rec["returncode"])

    def fake_cli(self, body: str) -> list:
        """argv of a fake CLI script, run through the interpreter (a freshly written executable can take
        a noticeable moment to exec the first time on macOS)."""
        script = self.tmp / "fake_cli.py"
        script.write_text(textwrap.dedent(body))
        return [sys.executable, str(script)]

    @staticmethod
    def gone(pid: int, within_s: float = 3.0) -> bool:
        deadline = time.monotonic() + within_s
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False

    def test_result_received_before_an_abort_is_judged_on_its_own(self):
        runner = ClaudeCliRunner(**self.runner_kw())
        done = StreamRun()
        for line in stream_lines(bible_dict(), "sess-a"):
            done.feed(line)
        done.abort_reason = "no stream event for 3 s"
        self.assertEqual(runner.check(done, OPUS), bible_dict())
        cut = StreamRun()
        cut.feed(stream_lines(None, "sess-a")[0])
        cut.abort_reason = "no stream event for 3 s"
        with self.assertRaisesRegex(PlannerTimeoutError, "no stream event"):
            runner.check(cut, OPUS)

    def test_a_descendant_holding_stdout_does_not_cost_the_reply(self):
        lines_file, pid_file = self.tmp / "lines.jsonl", self.tmp / "child.pid"
        lines_file.write_bytes(b"".join(stream_lines(bible_dict(), "real-a")))
        argv = self.fake_cli(f"""
            import subprocess, sys
            sys.stdin.read()
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(8)"])   # inherits stdout
            open({str(pid_file)!r}, "w").write(str(child.pid))
            sys.stdout.write(open({str(lines_file)!r}).read())
            sys.stdout.flush()
        """)
        runner = ClaudeCliRunner(**self.runner_kw(idle_timeout_s=2.0, kill_grace_s=0.5))
        started = time.monotonic()
        run = runner.stream(argv, "body", {"PATH": os.environ.get("PATH", "")})
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertIsNone(run.abort_reason)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(runner.check(run, OPUS), bible_dict())
        self.assertTrue(self.gone(int(pid_file.read_text())))

    def test_the_watchdog_stops_the_whole_process_group(self):
        pid_file = self.tmp / "child.pid"
        argv = self.fake_cli(f"""
            import json, signal, subprocess, sys
            sys.stdin.read()
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(8)"])   # inherits stdout
            open({str(pid_file)!r}, "w").write(str(child.pid))
            signal.signal(signal.SIGINT, signal.SIG_IGN)      # a wrapper that outlives SIGINT; its child does not
            print(json.dumps({{"type": "system", "subtype": "init", "session_id": "real-b"}}), flush=True)
            child.wait()
        """)
        runner = ClaudeCliRunner(**self.runner_kw(idle_timeout_s=0.5, kill_grace_s=0.5))
        started = time.monotonic()
        run = runner.stream(argv, "body", {"PATH": os.environ.get("PATH", "")})
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertIn("no stream event", run.abort_reason)
        with self.assertRaises(PlannerTimeoutError):
            runner.check(run, OPUS)
        self.assertTrue(self.gone(int(pid_file.read_text())))


# ---------------------------------------------------------------- replay cache
class TestReplayCache(PlannerTestCase):
    def test_record_then_replay(self):
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")) as fake:
            first = self.planner(replay="record").plan("A glass falls.", self.work)
        self.assertEqual(len(fake.calls), 2)
        for mode in ("record", "replay"):
            with FakeClaude() as fake:                      # any live call would fail the test
                again = self.planner(replay=mode).plan("A glass falls.", self.work)
            self.assertEqual(fake.calls, [])
            self.assertSamePlan(again, first)
        rec = self.record("b_2.json")
        self.assertTrue(rec["replayed"])
        self.assertEqual(rec["session_id"], "sess-b")

    def test_replay_miss_is_an_error(self):
        with FakeClaude() as fake:
            with self.assertRaisesRegex(PlannerError, "replay miss") as ctx:
                self.planner(replay="replay").plan("A glass falls.", self.work)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(fake.calls, [])

    def test_off_mode_never_touches_the_cache(self):
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")):
            self.planner(replay="off").plan("A glass falls.", self.work)
        self.assertFalse((self.tmp / "cache").exists())

    def test_only_accepted_replies_are_cached(self):
        failed = ok(None, "sess-b1", is_error=True, subtype="error_during_execution")
        with FakeClaude(ok(bible_dict(), "sess-a"), failed, ok(timeline_dict(), "sess-b2")):
            self.planner(replay="record").plan("A glass falls.", self.work)
        entries = [json.loads(p.read_text()) for p in (self.tmp / "cache").rglob("*.json")]
        self.assertEqual(sorted(e["session_id"] for e in entries), ["sess-a", "sess-b2"])

    def test_key_depends_on_input_and_cli_version(self):
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b")):
            self.planner(replay="record").plan("A glass falls.", self.work)
        # New scene text: A misses; it yields the same bible, so B (keyed by the bible) is reused.
        with FakeClaude(ok(bible_dict(), "sess-a2")) as fake:
            self.planner(replay="record").plan("A cup falls.", self.work)
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(self.record("b_2.json")["replayed"])
        with FakeClaude(ok(bible_dict(), "sess-a3"), ok(timeline_dict(), "sess-b3"), version="2.1.300 (Claude Code)") as fake:
            self.planner(replay="record").plan("A glass falls.", self.work)
        self.assertEqual(len(fake.calls), 2)

    def test_repair_key_is_bound_to_the_plan(self):
        reply = {"invariants": [], "shots": [shot_dict("sh_jon")]}
        with FakeClaude(ok(reply, "sess-r")):
            self.planner(replay="record").repair(FIXTURE, "violation", ["sh_jon"], self.work)
        with FakeClaude() as fake:
            self.planner(replay="record").repair(FIXTURE, "violation", ["sh_jon"], self.work)
        self.assertEqual(fake.calls, [])
        other = FIXTURE.model_copy(deep=True)
        other.timeline.shots[0].keyframes[0].dialogue = "Hm?"
        with FakeClaude(ok(reply, "sess-r2")) as fake:
            self.planner(replay="record").repair(other, "violation", ["sh_jon"], self.work)
        self.assertEqual(len(fake.calls), 1)

    def _no_op_repair_rounds(self, planner, rounds=3):
        plan = planner.plan("A glass falls.", self.work)
        for _ in range(rounds):
            plan = planner.repair(plan, "an error the model believes is already right", ["sh_glass"], self.work)
        return plan

    def test_repair_rounds_that_change_nothing_still_reach_the_model(self):
        touching = [iv for iv in timeline_dict()["invariants"]
                    if "kf_glass" in iv["from_kf_id"] or "kf_glass" in iv["until_kf_id"]]
        unchanged = {"invariants": touching, "shots": [shot_dict("sh_glass")]}
        replies = [ok(unchanged, f"sess-r{n}") for n in (1, 2, 3)]
        with FakeClaude(ok(bible_dict(), "sess-a"), ok(timeline_dict(), "sess-b"), *replies) as fake:
            first = self._no_op_repair_rounds(self.planner(replay="record"))
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual([self.record(f"repair_{n}.json")["replayed"] for n in (1, 2, 3)], [False] * 3)
        self.assertEqual(fake.argv(3)[-3:], ["--resume", "sess-r1", "--fork-session"])
        self.assertEqual(fake.argv(4)[-3:], ["--resume", "sess-r2", "--fork-session"])
        # A rerun of the same scene replays the whole chain, round by round.
        with FakeClaude() as fake:
            again = self._no_op_repair_rounds(self.planner(replay="record"))
        self.assertEqual(fake.calls, [])
        self.assertSamePlan(again, first)
        self.assertEqual([self.record(f"repair_{n}.json")["replayed"] for n in (4, 5, 6)], [True] * 3)


# ---------------------------------------------------------------- judge
class TestJudge(PlannerTestCase):
    def setUp(self):
        super().setUp()
        self.image = self.tmp / "frames" / "out.png"
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"\x89PNG fake image bytes")
        self.attempt = self.tmp / "attempt_0"

    def judge(self, **kw) -> ClaudeCliJudge:
        return ClaudeCliJudge(claude_bin="claude-test", **self.runner_kw(**kw))

    def test_facts_argv_and_answers(self):
        reply = {"answers": [{"q_index": 2, "answer": "no"}, {"q_index": 1, "answer": "yes"}]}
        with FakeClaude(ok(reply, "sess-f", model=HAIKU)) as fake:
            answers = self.judge().facts(str(self.image), ["Is Mara standing?", "Is the glass broken?", "Is Jon present?"],
                                         self.attempt)
        self.assertEqual(answers, ["yes", "no", "unsure"])
        argv = fake.argv(0)
        self.assertEqual(flag_value(argv, "--model"), HAIKU)
        self.assertEqual(flag_value(argv, "--effort"), "low")
        i = argv.index("--tools")
        self.assertEqual(argv[i:i + 6], ["--tools", "Read", "--allowedTools", "Read", "--add-dir", str(self.image.parent.resolve())])
        for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
            self.assertIn(flag, argv)
        self.assertNotIn("--session-id", argv)
        self.assertEqual(json.loads(flag_value(argv, "--json-schema")), claude_cli.FACTS_SCHEMA)
        self.assertEqual(fake.env(0)["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "8000")
        body = fake.stdin(0)
        self.assertIn(str(self.image.resolve()), body)
        self.assertIn("2. Is the glass broken?", body)
        self.assertTrue((self.attempt / "claude" / "facts_1.json").exists())

    def test_facts_edge_cases(self):
        with FakeClaude() as fake:
            self.assertEqual(self.judge().facts(str(self.image), [], self.attempt), [])
            with self.assertRaises(FileNotFoundError):
                self.judge().facts(str(self.tmp / "missing.png"), ["q?"], self.attempt)
        self.assertEqual(fake.calls, [])
        zero_based = {"answers": [{"q_index": 0, "answer": "YES"}, {"q_index": 1, "answer": "maybe"}]}
        self.assertEqual(parse_answers(zero_based, 2), ["yes", "unsure"])
        dupes = {"answers": [{"q_index": 1, "answer": "no"}, {"q_index": 1, "answer": "yes"}, {"q_index": 9, "answer": "yes"}]}
        self.assertEqual(parse_answers(dupes, 2), ["no", "unsure"])

    def test_facts_wrong_model_is_rejected(self):
        reply = {"answers": [{"q_index": 1, "answer": "yes"}]}
        with FakeClaude(ok(reply, "sess-f", model=OPUS)):
            with self.assertRaisesRegex(PlannerError, "model rerouted"):
                self.judge().facts(str(self.image), ["q?"], self.attempt)

    def test_facts_cache_is_keyed_by_image_bytes(self):
        reply = {"answers": [{"q_index": 1, "answer": "yes"}]}
        with FakeClaude(ok(reply, "s1", model=HAIKU)):
            self.judge(replay="record").facts(str(self.image), ["q?"], self.attempt)
        with FakeClaude() as fake:
            self.assertEqual(self.judge(replay="record").facts(str(self.image), ["q?"], self.attempt), ["yes"])
        self.assertEqual(fake.calls, [])
        self.image.write_bytes(b"\x89PNG different bytes")
        with FakeClaude(ok(reply, "s2", model=HAIKU)) as fake:
            self.judge(replay="record").facts(str(self.image), ["q?"], self.attempt)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
