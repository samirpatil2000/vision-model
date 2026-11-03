"""
Deterministic plan checks with stable violation codes.

validate() reports facts the planner can act on: broken references, keyframe bookkeeping, runtime,
value ranges, custody, irreversibility, presence, support, physical feasibility, phase order, the
180-degree rule, setup/location consistency and (after blocking) subject visibility. Errors block
rendering; warnings are reported only. validate_bible() runs the checks that need only the scene bible
(call A), which no timeline repair can fix. repair_shot_set() picks the shots a repair round must
replan and format_for_repair() turns violations into a fact list for the planner.
"""
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src import config
from src.previs import physics
from src.previs.schema import (
    FILL_FLAGS,
    IRREVERSIBLE_FLAGS,
    NONE,
    EntityState,
    SceneBible,
    ScenePlan,
    Timeline,
    keyframe_ids_for_shot,
)
from src.previs.state import (
    CUSTODY_KINDS,
    INIT,
    QUADRUPED,
    SceneIndex,
    build_index,
    event_mover,
    reduce_states,
    support_chain,
)
from src.previs.types import BlockingResult, SolvedPlan, Violation

MIN_VISIBLE_FRAC = 0.05
HEIGHT_RANGE_M = (0.3, 2.5)              # people: standing height
QUADRUPED_HEIGHT_RANGE_M = (0.2, 2.5)    # animals: shoulder height (a cat or a toy dog stands 0.2-0.3 m)
LENS_RANGE_MM = (10.0, 300.0)
PHASE_RANK = {phase: rank for rank, phase in enumerate(
    ("anticipation", "contact", "release", "rising", "apex", "falling", "impact", "rebound", "aftermath", "rest"))}
LINE_FREE_MOVE_PREFIXES = ("orbit_", "crane_")


def height_range_m(body: str) -> Tuple[float, float]:
    """Allowed Character.height_m for a body plan (a quadruped's is its shoulder height)."""
    return QUADRUPED_HEIGHT_RANGE_M if body == QUADRUPED else HEIGHT_RANGE_M


def _err(code: str, message: str, shot_id: str = NONE, kf_id: str = NONE, entity_id: str = NONE,
         severity: str = "error") -> Violation:
    return Violation(code=code, message=message, shot_id=shot_id, kf_id=kf_id, entity_id=entity_id, severity=severity)


def validate(plan: ScenePlan, solved: Optional[SolvedPlan] = None,
             blocking: Optional[Dict[str, BlockingResult]] = None) -> List[Violation]:
    """Run every check. Solves the plan first when `solved` is not given."""
    index = build_index(plan)
    states = reduce_states(plan, index)
    if solved is None:
        solved = physics.solve(plan)
    violations: List[Violation] = []
    violations += _check_refs(plan, index)
    violations += _check_keyframe_ids(plan, index)
    violations += _check_duration(plan, solved)
    violations += _check_ranges(plan, index)
    violations += _check_custody(index, states)
    violations += _check_irreversible(plan, index, states)
    violations += _check_count(index, states)
    violations += _check_fixed(index, states)
    violations += _check_supported(index, states, solved)
    violations += _check_feasible(index, solved)
    violations += _check_phase_order(index)
    violations += _check_line_side(index)
    violations += _check_setup_location(index)
    if blocking:
        violations += _check_visible(index, blocking)
    return violations


def validate_bible(bible: SceneBible) -> List[Violation]:
    """
    Every check that needs only the scene bible (call A): references and duplicate ids, keyframe_count,
    target duration, ranges (heights, sizes, extents, marks inside their location, lens), the 180-degree
    rule, one location per setup, and the "init" state's custody, support and fill rules. These are the
    same checks (codes and messages) validate() runs on the bible part of a full plan.
    """
    plan = ScenePlan(bible=bible, timeline=Timeline(invariants=[], shots=[]))
    index = build_index(plan)
    states = {INIT: reduce_states(plan, index)[INIT]}
    violations: List[Violation] = []
    violations += _check_refs(plan, index)
    violations += _check_keyframe_count(index)
    violations += _check_target_duration(plan)
    violations += _check_ranges(plan, index)
    violations += _check_custody(index, states)
    violations += _check_irreversible(plan, index, states)
    violations += _check_supported(index, states, None)
    violations += _check_line_side(index)
    violations += _check_setup_location(index)
    return violations


# ---------------------------------------------------------------- ref_integrity
def _check_state_refs(st: EntityState, index: SceneIndex, valid_kfs: Set[str], shot_id: str) -> List[Violation]:
    out: List[Violation] = []

    def bad(field: str, value: str) -> None:
        out.append(_err("ref_integrity", f"state of {st.entity_id} at {st.kf_id}: {field} '{value}' does not exist",
                        shot_id, st.kf_id, st.entity_id))

    if st.kf_id not in valid_kfs:
        bad("kf_id", st.kf_id)
    if st.entity_id not in index.entities:
        bad("entity_id", st.entity_id)
    if st.mark_id != NONE and st.mark_id not in index.marks:
        bad("mark_id", st.mark_id)
    if st.facing_id != NONE and st.facing_id not in index.entities and st.facing_id not in index.marks:
        bad("facing_id", st.facing_id)
    for name in ("holding_id", "gaze_id", "support_id"):
        value = getattr(st, name)
        if value != NONE and value not in index.entities:
            bad(name, value)
    return out


def _check_refs(plan: ScenePlan, index: SceneIndex) -> List[Violation]:
    bible, timeline = plan.bible, plan.timeline
    out: List[Violation] = []

    def err(message: str, **where: str) -> None:
        out.append(_err("ref_integrity", message, **where))

    # 1. Unique IDs across every namespace.
    seen: Set[str] = set()
    all_ids = ([c.id for c in bible.characters] + [p.id for p in bible.props] + [lo.id for lo in bible.locations]
               + [m.id for m in bible.marks] + [b.id for b in bible.beats] + [s.id for s in bible.shots]
               + [kf.id for tl in timeline.shots for kf in tl.keyframes] + [ev.id for tl in timeline.shots for ev in tl.events]
               + [iv.id for iv in timeline.invariants] + [cl.id for tl in timeline.shots for cl in tl.cause_links])
    for item_id in all_ids:
        if item_id in seen:
            err(f"id {item_id} is used more than once")
        seen.add(item_id)
    for label, ids in (("phys_bodies", [b.entity_id for b in bible.phys_bodies]),
                       ("initial_states", [s.entity_id for s in bible.initial_states]),
                       ("timeline shots", [tl.shot_id for tl in timeline.shots])):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        for dupe in dupes:
            err(f"{label} lists {dupe} more than once")

    # 2. Bible references.
    for mark in bible.marks:
        if mark.location_id not in index.locations:
            err(f"mark {mark.id} is in unknown location {mark.location_id}")
    for beat in bible.beats:
        if beat.cause_beat_id != NONE and beat.cause_beat_id not in index.beats:
            err(f"beat {beat.id} has unknown cause_beat_id {beat.cause_beat_id}")
    for shot in bible.shots:
        if shot.location_id not in index.locations:
            err(f"shot {shot.id} is in unknown location {shot.location_id}", shot_id=shot.id)
        for beat_id in shot.beat_ids:
            if beat_id not in index.beats:
                err(f"shot {shot.id} lists unknown beat {beat_id}", shot_id=shot.id)
        if shot.subject_id not in index.entities:
            err(f"shot {shot.id} subject {shot.subject_id} does not exist", shot_id=shot.id)
        if shot.secondary_id != NONE and shot.secondary_id not in index.entities:
            err(f"shot {shot.id} secondary {shot.secondary_id} does not exist", shot_id=shot.id)
    for body in bible.phys_bodies:
        if body.entity_id not in index.entities:
            err(f"phys_body for unknown entity {body.entity_id}")
    placed = {s.entity_id for s in bible.initial_states}
    for eid in index.entities:
        if eid not in index.bodies:
            out.append(_err("ref_integrity", f"{eid} has no phys_body; class defaults are used", entity_id=eid,
                            severity="warning"))
        if eid not in placed:
            out.append(_err("ref_integrity", f"{eid} has no initial state; it starts absent", entity_id=eid,
                            severity="warning"))
    for st in bible.initial_states:
        out += _check_state_refs(st, index, {INIT}, NONE)

    # 3. Timeline references.
    valid_kfs = set(index.keyframe_by_id)
    for tl in timeline.shots:
        sid = tl.shot_id
        if sid not in index.shots:
            err(f"timeline references unknown shot {sid}", shot_id=sid)
        for kf in tl.keyframes:
            if kf.beat_id not in index.beats:
                err(f"keyframe {kf.id} has unknown beat {kf.beat_id}", shot_id=sid, kf_id=kf.id)
            if kf.focus_event_id != NONE and kf.focus_event_id not in index.events:
                err(f"keyframe {kf.id} focuses unknown event {kf.focus_event_id}", shot_id=sid, kf_id=kf.id)
        for ev in tl.events:
            if ev.start_kf_id not in valid_kfs:
                err(f"event {ev.id} starts at unknown keyframe {ev.start_kf_id}", shot_id=sid)
            for name in ("actor_id", "target_id", "outcome_support_id"):
                value = getattr(ev, name)
                if value != NONE and value not in index.entities:
                    err(f"event {ev.id} {name} '{value}' does not exist", shot_id=sid)
            if ev.actor_id == NONE and ev.target_id == NONE and ev.kind != "none":
                err(f"event {ev.id} ({ev.kind}) has neither actor_id nor target_id", shot_id=sid)
            if ev.outcome_mark_id != NONE and ev.outcome_mark_id not in index.marks:
                err(f"event {ev.id} outcome_mark_id '{ev.outcome_mark_id}' does not exist", shot_id=sid)
            if ev.cause_event_id != NONE and (ev.cause_event_id not in index.events or ev.cause_event_id == ev.id):
                err(f"event {ev.id} cause_event_id '{ev.cause_event_id}' is not another existing event", shot_id=sid)
        for st in tl.states:
            out += _check_state_refs(st, index, valid_kfs, sid)
        for link in tl.cause_links:
            for name in ("cause_event_id", "effect_event_id"):
                if getattr(link, name) not in index.events:
                    err(f"cause link {link.id} {name} '{getattr(link, name)}' does not exist", shot_id=sid)
    for iv in timeline.invariants:
        if iv.subject_id not in index.entities or (iv.object_id != NONE and iv.object_id not in index.entities):
            err(f"invariant {iv.id} references an unknown entity")
        for name in ("from_kf_id", "until_kf_id"):
            if getattr(iv, name) not in valid_kfs:
                err(f"invariant {iv.id} {name} '{getattr(iv, name)}' does not exist")
    return out


# ---------------------------------------------------------------- keyframe_ids
def _check_keyframe_count(index: SceneIndex) -> List[Violation]:
    return [_err("keyframe_ids", f"shot {shot.id} keyframe_count must be at least 1", shot.id)
            for shot in index.shots.values() if shot.keyframe_count < 1]


def _check_keyframe_ids(plan: ScenePlan, index: SceneIndex) -> List[Violation]:
    out: List[Violation] = _check_keyframe_count(index)
    timelines = {tl.shot_id for tl in plan.timeline.shots}
    for shot in index.shots.values():
        if shot.id not in timelines:
            out.append(_err("keyframe_ids", f"shot {shot.id} has no timeline", shot.id))
    for tl in plan.timeline.shots:
        shot = index.shots.get(tl.shot_id)
        if shot is None:
            continue
        allowed = keyframe_ids_for_shot(shot)
        own = {kf.id for kf in tl.keyframes}
        for kf in tl.keyframes:
            if kf.id not in allowed:
                out.append(_err("keyframe_ids", f"keyframe {kf.id} is not one of {shot.id}'s ids {allowed}",
                                shot.id, kf.id))
        count = len(tl.keyframes)
        if count == 0 or abs(count - shot.keyframe_count) > 1:
            out.append(_err("keyframe_ids", f"shot {shot.id} has {count} keyframes; keyframe_count is "
                                            f"{shot.keyframe_count} (allowed {max(shot.keyframe_count - 1, 1)}-"
                                            f"{shot.keyframe_count + 1})", shot.id))
        orders = [kf.order for kf in tl.keyframes]
        if len(set(orders)) != len(orders):
            out.append(_err("keyframe_ids", f"shot {shot.id} repeats a keyframe order in {orders}", shot.id))
        for st in tl.states:
            if st.kf_id in index.keyframe_by_id and st.kf_id not in own:
                out.append(_err("keyframe_ids", f"{shot.id} lists a state for {st.kf_id}, a keyframe of another shot",
                                shot.id, st.kf_id, st.entity_id))
        for ev in tl.events:
            if ev.start_kf_id in index.keyframe_by_id and ev.start_kf_id not in own:
                out.append(_err("keyframe_ids", f"{shot.id} lists event {ev.id} starting at {ev.start_kf_id}, "
                                                f"a keyframe of another shot", shot.id, ev.start_kf_id))
    return out


# ---------------------------------------------------------------- duration / ranges
def _check_target_duration(plan: ScenePlan) -> List[Violation]:
    limit, target = config.MAX_VIDEO_SECONDS, plan.bible.scene.target_duration_s
    if 0 < target <= limit:
        return []
    return [_err("duration", f"target_duration_s is {target:g}s; it must be in (0, {limit:g}]")]


def _check_duration(plan: ScenePlan, solved: SolvedPlan) -> List[Violation]:
    out: List[Violation] = _check_target_duration(plan)
    limit = config.MAX_VIDEO_SECONDS
    if solved.total_duration_s > limit + 1e-6:
        detail = next((w.split(":", 1)[1].strip() for w in solved.warnings if w.startswith("duration:")), "")
        out.append(_err("duration", f"the solved cut runs {solved.total_duration_s:.2f}s, over the {limit:g}s limit"
                        + (f" ({detail})" if detail else "")))
    for warning in solved.warnings:
        if warning.startswith("timing:"):
            out.append(_err("timing", warning.split(":", 1)[1].strip(), severity="warning"))
    return out


def _check_ranges(plan: ScenePlan, index: SceneIndex) -> List[Violation]:
    out: List[Violation] = []
    for ch in plan.bible.characters:
        lo_h, hi_h = height_range_m(ch.body)
        if not lo_h <= ch.height_m <= hi_h:
            what = "shoulder height" if ch.body == QUADRUPED else "height"
            out.append(_err("ranges", f"{ch.id} {what} {ch.height_m:g} m is outside {lo_h}-{hi_h} m", entity_id=ch.id))
    for prop in plan.bible.props:
        if min(prop.size_x_m, prop.size_y_m, prop.size_z_m) <= 0:
            out.append(_err("ranges", f"{prop.id} has a non-positive size", entity_id=prop.id))
    for loc in plan.bible.locations:
        if loc.width_m <= 0 or loc.depth_m <= 0:
            out.append(_err("ranges", f"location {loc.id} has a non-positive extent"))
    for mark in plan.bible.marks:
        loc = index.locations.get(mark.location_id)
        if loc is not None and not (0 <= mark.x_m <= loc.width_m and 0 <= mark.z_m <= loc.depth_m):
            out.append(_err("ranges", f"mark {mark.id} ({mark.x_m:g}, {mark.z_m:g}) is outside {loc.id} "
                                      f"(0..{loc.width_m:g} x 0..{loc.depth_m:g} m)"))
    lo_l, hi_l = LENS_RANGE_MM
    for shot in plan.bible.shots:
        if not lo_l <= shot.lens_mm <= hi_l:
            out.append(_err("ranges", f"shot {shot.id} lens {shot.lens_mm:g} mm is outside {lo_l:g}-{hi_l:g} mm",
                            shot.id))
    for tl in plan.timeline.shots:
        for kf in tl.keyframes:
            if not 0 <= kf.move_progress <= 1:
                out.append(_err("ranges", f"keyframe {kf.id} move_progress {kf.move_progress:g} is outside 0-1",
                                tl.shot_id, kf.id))
    return out


# ---------------------------------------------------------------- state continuity
def _timeline_order(index: SceneIndex) -> List[str]:
    return [INIT] + [kf.id for _, kf in index.keyframes]


def _position(index: SceneIndex, kf_id: str) -> int:
    return -1 if kf_id == INIT else index.kf_position[kf_id]


def _covered(index: SceneIndex, kinds: Iterable[str], entity: str, prev_kf: str, kf_id: str) -> bool:
    """True when an event of `kinds` moving/changing `entity` spans the interval (prev_kf, kf_id]."""
    lo, hi = _position(index, prev_kf), _position(index, kf_id)
    kinds = set(kinds)
    for eid, ev in index.events.items():
        if ev.kind not in kinds or entity not in (event_mover(ev), ev.target_id):
            continue
        span = index.event_span(eid)
        if span is not None and span[0] <= hi and span[1] >= lo:
            return True
    return False


def _custody_in_progress(index: SceneIndex, entity: str, kf_id: str) -> bool:
    """A throw/drop/place/hand_off of `entity` spans this keyframe, so its holder may already have let go."""
    if kf_id == INIT:
        return False
    pos = index.kf_position[kf_id]
    for eid, ev in index.events.items():
        span = index.event_span(eid)
        if ev.kind in CUSTODY_KINDS and event_mover(ev) == entity and span is not None and span[0] <= pos <= span[1]:
            return True
    return False


def _check_custody(index: SceneIndex, states: Dict[str, Dict[str, EntityState]]) -> List[Violation]:
    out: List[Violation] = []
    prev_kf, prev_holder = None, {}
    for kf_id in _timeline_order(index):
        st = states[kf_id]
        shot_id = index.shot_of_keyframe.get(kf_id, NONE)
        holders: Dict[str, List[str]] = {}
        for eid, s in st.items():
            if s.holding_id == NONE:
                continue
            if s.holding_id == eid:
                out.append(_err("custody", f"{eid} holds itself", shot_id, kf_id, eid))
            else:
                holders.setdefault(s.holding_id, []).append(eid)
        for obj, hs in holders.items():
            if len(hs) > 1:
                out.append(_err("custody", f"{obj} is held by {' and '.join(hs)} at once", shot_id, kf_id, obj))
        for eid, s in st.items():
            if s.support == "held" and (s.support_id not in st or st[s.support_id].holding_id != eid) \
                    and not _custody_in_progress(index, eid, kf_id):
                out.append(_err("custody", f"{eid} has support 'held' by {s.support_id}, but {s.support_id} is not "
                                           f"holding it (holding_id must name {eid})", shot_id, kf_id, eid))
        holder = {obj: hs[0] for obj, hs in holders.items()}
        if prev_kf is not None:
            for obj in sorted(set(prev_holder) | set(holder)):
                before, after = prev_holder.get(obj, NONE), holder.get(obj, NONE)
                if before != after and not _covered(index, CUSTODY_KINDS, obj, prev_kf, kf_id):
                    out.append(_err("custody", f"{obj} passes from {before} to {after} between {prev_kf} and {kf_id} "
                                               f"without a lift/catch/hand_off/place/drop/throw event", shot_id, kf_id, obj))
        prev_kf, prev_holder = kf_id, holder
    return out


def _check_irreversible(plan: ScenePlan, index: SceneIndex, states: Dict[str, Dict[str, EntityState]]) -> List[Violation]:
    out: List[Violation] = []
    for st in plan.bible.initial_states:
        if len({f for f in st.flags if f in FILL_FLAGS}) > 1:
            out.append(_err("irreversible", f"{st.entity_id} starts with several fill flags {st.flags}; they are "
                                            f"mutually exclusive", kf_id=INIT, entity_id=st.entity_id))
    for tl in plan.timeline.shots:
        for st in tl.states:
            if st.kf_id not in index.kf_position or st.entity_id not in index.entities:
                continue
            prev_flags = states[index.prev_keyframe_id(st.kf_id)][st.entity_id].flags
            lost = [f for f in prev_flags if f in IRREVERSIBLE_FLAGS and f not in st.flags]
            if lost:
                out.append(_err("irreversible", f"{st.entity_id} loses irreversible flag(s) {lost} at {st.kf_id}; "
                                                f"once set they stay", tl.shot_id, st.kf_id, st.entity_id))
            fills = {f for f in st.flags if f in FILL_FLAGS}
            if len(fills) > 1:
                out.append(_err("irreversible", f"{st.entity_id} has several fill flags {sorted(fills)} at {st.kf_id}; "
                                                f"they are mutually exclusive", tl.shot_id, st.kf_id, st.entity_id))
    return out


def _check_count(index: SceneIndex, states: Dict[str, Dict[str, EntityState]]) -> List[Violation]:
    out: List[Violation] = []
    order = _timeline_order(index)
    for prev_kf, kf_id in zip(order, order[1:]):
        for eid, st in states[kf_id].items():
            before = states[prev_kf][eid].present
            if st.present == before:
                continue
            kind = "enter" if st.present else "exit"
            # What a character carries (held, attached, on a held tray) enters and exits with it.
            carriers = set(support_chain(states[prev_kf], eid)) | set(support_chain(states[kf_id], eid))
            if not any(_covered(index, (kind,), who, prev_kf, kf_id) for who in [eid, *sorted(carriers)]):
                verb = "appears" if st.present else "disappears"
                out.append(_err("count", f"{eid} {verb} between {prev_kf} and {kf_id} without an {kind} event",
                                index.shot_of_keyframe.get(kf_id, NONE), kf_id, eid))
    return out


def _placement(st: EntityState) -> tuple:
    return st.mark_id, round(st.offset_x_m, 4), round(st.offset_z_m, 4), st.support, st.support_id


def _check_fixed(index: SceneIndex, states: Dict[str, Dict[str, EntityState]]) -> List[Violation]:
    out: List[Violation] = []
    order = _timeline_order(index)
    for prev_kf, kf_id in zip(order, order[1:]):
        for eid, st in states[kf_id].items():
            if physics.is_immovable(index, eid) and _placement(st) != _placement(states[prev_kf][eid]):
                out.append(_err("fixed", f"{eid} is fixed (immovable) but moves between {prev_kf} and {kf_id}",
                                index.shot_of_keyframe.get(kf_id, NONE), kf_id, eid))
    return out


def _check_supported(index: SceneIndex, states: Dict[str, Dict[str, EntityState]],
                     solved: Optional[SolvedPlan]) -> List[Violation]:
    out: List[Violation] = []
    active = {f.kf_id: set(f.active_event_ids) for f in solved.frames} if solved is not None else {}
    for kf_id in _timeline_order(index):
        st = states[kf_id]
        shot_id = index.shot_of_keyframe.get(kf_id, NONE)
        for eid, s in st.items():
            if not s.present:
                continue
            if s.support == "airborne":
                flying = any(ev in index.events and index.events[ev].kind in physics.AIRBORNE_KINDS
                             and event_mover(index.events[ev]) == eid for ev in active.get(kf_id, ()))
                if not flying:
                    out.append(_err("supported", f"{eid} is airborne at {kf_id} but no fall/drop/throw/bounce moves it "
                                                 f"then; give it a support or an active fall", shot_id, kf_id, eid))
            elif s.support == "held" and s.support_id in st and not st[s.support_id].present:
                out.append(_err("supported", f"{eid} is held by {s.support_id}, which is not present; it leaves and "
                                             f"enters with its holder", shot_id, kf_id, eid))
            elif s.support in ("on_entity", "attached"):
                if s.support_id not in st or s.support_id == eid:
                    out.append(_err("supported", f"{eid} rests {s.support} but support_id '{s.support_id}' names no "
                                                 f"other entity", shot_id, kf_id, eid))
                elif not st[s.support_id].present:
                    out.append(_err("supported", f"{eid} rests on {s.support_id}, which is not present", shot_id, kf_id, eid))
                elif _support_cycle(eid, st):
                    out.append(_err("supported", f"{eid} is part of a support cycle", shot_id, kf_id, eid))
    return out


def _support_cycle(eid: str, st: Dict[str, EntityState]) -> bool:
    cur, seen = eid, set()
    while cur in st and st[cur].support in ("on_entity", "attached", "held") and st[cur].support_id in st:
        if cur in seen:
            return True
        seen.add(cur)
        cur = st[cur].support_id
        if cur == eid:
            return True
    return False


# ---------------------------------------------------------------- events and timing
def _check_feasible(index: SceneIndex, solved: SolvedPlan) -> List[Violation]:
    out: List[Violation] = []
    for eid, se in solved.events.items():
        ev = index.events.get(eid)
        kf_id = ev.start_kf_id if ev is not None else NONE
        mover = event_mover(ev) if ev is not None else NONE
        if not se.feasible:
            out.append(_err("event_feasible", f"{eid} ({se.kind}) is physically infeasible: {se.note}",
                            se.shot_id, kf_id, mover))
        elif se.note:
            out.append(_err("event_feasible", f"{eid} ({se.kind}): {se.note}", se.shot_id, kf_id, mover,
                            severity="warning"))
    return out


def _check_phase_order(index: SceneIndex) -> List[Violation]:
    out: List[Violation] = []
    for eid, kf_ids in index.focus_keyframes.items():
        ev = index.events.get(eid)
        if ev is None:
            continue
        start = index.kf_position.get(ev.start_kf_id)
        if start is not None and index.kf_position[kf_ids[0]] < start:
            out.append(_err("phase_order", f"{kf_ids[0]} shows {eid} before its start keyframe {ev.start_kf_id}",
                            index.shot_of_keyframe[kf_ids[0]], kf_ids[0], event_mover(ev)))
        ranked = [(k, index.keyframe_by_id[k].depicted_phase) for k in kf_ids
                  if index.keyframe_by_id[k].depicted_phase in PHASE_RANK]
        for (k1, p1), (k2, p2) in zip(ranked, ranked[1:]):
            if PHASE_RANK[p2] < PHASE_RANK[p1]:
                out.append(_err("phase_order", f"{k2} shows {eid} at '{p2}' after {k1} showed '{p1}'; phases of one "
                                               f"event must not go backward", index.shot_of_keyframe[k2], k2,
                                event_mover(ev)))
    return out


# ---------------------------------------------------------------- shots
def _check_line_side(index: SceneIndex) -> List[Violation]:
    """180-degree rule within a run of same-location shots: a subject/secondary pair keeps its line side."""
    out: List[Violation] = []
    last: Dict[frozenset, object] = {}
    location = None
    for shot in index.shots.values():
        if shot.location_id != location:
            last, location = {}, shot.location_id
        if shot.secondary_id == NONE or shot.secondary_id == shot.subject_id:
            continue
        pair = frozenset((shot.subject_id, shot.secondary_id))
        prev = last.get(pair)
        free = any(m.startswith(LINE_FREE_MOVE_PREFIXES) for m in (shot.move, getattr(prev, "move", "static")))
        if prev is not None and prev.line_side != shot.line_side and not free:
            out.append(_err("line_side", f"{shot.id} puts the camera on side '{shot.line_side}' of the "
                                         f"{shot.subject_id}/{shot.secondary_id} line but {prev.id} used side "
                                         f"'{prev.line_side}' (180-degree rule; cross only with an orbit or crane move)",
                            shot.id))
        last[pair] = shot
    return out


def _check_setup_location(index: SceneIndex) -> List[Violation]:
    out: List[Violation] = []
    setup_location: Dict[str, str] = {}
    for shot in index.shots.values():
        first = setup_location.setdefault(shot.setup_id, shot.location_id)
        if first != shot.location_id:
            out.append(_err("setup_location", f"setup {shot.setup_id} is used in {first} and in {shot.location_id} "
                                              f"({shot.id}); one setup belongs to one location", shot.id))
    return out


def _check_visible(index: SceneIndex, blocking: Dict[str, BlockingResult]) -> List[Violation]:
    out: List[Violation] = []
    for kf_id, result in blocking.items():
        shot = index.shots.get(index.shot_of_keyframe.get(kf_id, ""))
        if shot is None:
            continue
        frac = result.visible_frac.get(shot.subject_id, 0.0)
        if frac < MIN_VISIBLE_FRAC:
            out.append(_err("subject_visible", f"subject {shot.subject_id} is only {frac:.0%} visible in {kf_id} "
                                               f"(needs {MIN_VISIBLE_FRAC:.0%})", shot.id, kf_id, shot.subject_id))
    return out


# ---------------------------------------------------------------- repair support
def _shot_entities(plan: ScenePlan, index: SceneIndex) -> Dict[str, Dict[str, Set[str]]]:
    """Per shot: entities it references ("refs") and entities whose state its timeline sets ("changes")."""
    info: Dict[str, Dict[str, Set[str]]] = {}
    for sid, shot in index.shots.items():
        info[sid] = {"refs": {shot.subject_id, shot.secondary_id}, "changes": set()}
    for tl in plan.timeline.shots:
        entry = info.setdefault(tl.shot_id, {"refs": set(), "changes": set()})
        for st in tl.states:
            entry["changes"].add(st.entity_id)
            entry["refs"] |= {st.entity_id, st.holding_id, st.support_id, st.gaze_id, st.facing_id}
        for ev in tl.events:
            entry["changes"] |= {ev.actor_id, ev.target_id, ev.outcome_support_id}
            entry["refs"] |= {ev.actor_id, ev.target_id, ev.outcome_support_id}
    for entry in info.values():
        entry["refs"] = {e for e in entry["refs"] | entry["changes"] if e in index.entities}
        entry["changes"] = {e for e in entry["changes"] if e in index.entities}
    return info


def repair_shot_set(plan: ScenePlan, violations: List[Violation]) -> List[str]:
    """
    Shots a repair round must replan: every shot with an error, plus (transitively) every later shot that
    references an entity the replanned shots may change. Errors tied to no shot or entity select all shots.
    """
    index = build_index(plan)
    info = _shot_entities(plan, index)
    order = list(dict.fromkeys(list(index.shots) + [tl.shot_id for tl in plan.timeline.shots]))
    errors = [v for v in violations if v.severity == "error"]
    seeds: Dict[str, Set[str]] = {}
    for v in errors:
        sid = v.shot_id if v.shot_id in info else index.shot_of_keyframe.get(v.kf_id)
        if sid is None and v.entity_id in index.entities:
            sid = next((s for s in order if v.entity_id in info[s]["refs"]), None)
        if sid is None:
            return order
        seeds.setdefault(sid, set())
        seeds[sid] |= {v.entity_id} if v.entity_id in index.entities else info[sid]["changes"]
    selected: List[str] = []
    carried: Set[str] = set()
    for sid in order:
        if sid in seeds:
            selected.append(sid)
            carried |= seeds[sid]
        elif carried & info[sid]["refs"]:
            selected.append(sid)
            carried |= info[sid]["changes"]
    return selected


def _where(v: Violation) -> str:
    parts = [f"{label} {value}" for label, value in (("shot", v.shot_id), ("keyframe", v.kf_id), ("entity", v.entity_id))
             if value != NONE]
    return ", ".join(parts) if parts else "scene"


def format_for_repair(violations: List[Violation]) -> str:
    """Violations as a numbered fact list the planner can act on."""
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity != "error"]
    if not violations:
        return "No violations."
    lines = [f"The deterministic physics/continuity checker found {len(errors)} error(s). Fix every error; "
             f"keep all IDs and everything not mentioned unchanged."]
    lines += [f"{i}. [{v.code}] {_where(v)}: {v.message}" for i, v in enumerate(errors, 1)]
    if warnings:
        lines.append("Warnings (fix only if it does not disturb the rest):")
        lines += [f"- [{v.code}] {_where(v)}: {v.message}" for v in warnings]
    return "\n".join(lines)
