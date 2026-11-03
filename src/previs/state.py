"""
Scene index and state reducer.

build_index() turns a ScenePlan into lookup tables. reduce_states() carries complete entity
snapshots through the timeline: at each keyframe it first applies the outcomes of physics events
that have completed, then the planner's own snapshots for that keyframe.

Both tolerate dangling references (validator.py reports them), so they never raise on a plan
that passed pydantic validation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.previs.schema import (
    FILL_FLAGS,
    IRREVERSIBLE_FLAGS,
    NONE,
    Beat,
    CauseLink,
    Character,
    EntityState,
    Invariant,
    Keyframe,
    Location,
    Mark,
    PhysBody,
    PhysEvent,
    Prop,
    ScenePlan,
    Shot,
    ShotTimeline,
)
from src.previs.types import Vec3

Entity = Union[Character, Prop]

INIT = "init"
QUADRUPED = "quadruped"          # Character.body of animals, and the EntityFrame.shape blocking draws them with
CHARACTER_SIZE_XZ = (0.45, 0.3)
DEFAULT_COLOR = "#808080"

# Event kinds that move or change their actor; every other kind moves or changes its target.
SELF_KINDS = frozenset({"walk_to", "run_to", "enter", "exit", "sit_down", "stand_up", "gesture"})
# The only kinds allowed to change who holds an object.
CUSTODY_KINDS = frozenset({"lift", "catch", "hand_off", "place", "drop", "throw"})
# Kinds whose outcome is already visible at the depicted impact instant.
INSTANT_OUTCOME_KINDS = frozenset({"break", "tear", "spill", "impact", "collide"})
SETTLED_PHASES = frozenset({"aftermath", "rest"})
# Low postures: sit_down leaves a person in one of these, stand_up gets them out of it.
LOW_POSES = frozenset({"sit", "kneel", "crouch", "lie"})
# Supports that carry an entity along with whatever supports it (enter/exit take it in and out too).
CARRIED_SUPPORTS = ("held", "attached", "on_entity")


def event_mover(event: PhysEvent) -> str:
    """The entity an event moves or changes: the actor for self-moving kinds, else the target."""
    if event.kind in SELF_KINDS:
        return event.actor_id if event.actor_id != NONE else event.target_id
    return event.target_id if event.target_id != NONE else event.actor_id


@dataclass
class SceneIndex:
    """Lookup tables over one ScenePlan. Dicts preserve plan order (shots by `order`)."""
    plan: ScenePlan
    characters: Dict[str, Character]
    props: Dict[str, Prop]
    entities: Dict[str, Entity]
    marks: Dict[str, Mark]
    locations: Dict[str, Location]
    shots: Dict[str, Shot]
    beats: Dict[str, Beat]
    bodies: Dict[str, PhysBody]
    keyframes: List[Tuple[str, Keyframe]]           # (shot_id, keyframe), shot order then keyframe order
    keyframe_by_id: Dict[str, Keyframe]
    shot_of_keyframe: Dict[str, str]
    events: Dict[str, PhysEvent]
    event_shot: Dict[str, str]
    cause_links: Dict[str, CauseLink]
    invariants: Dict[str, Invariant]
    timelines: Dict[str, ShotTimeline] = field(default_factory=dict)
    kf_position: Dict[str, int] = field(default_factory=dict)          # kf_id -> index in `keyframes`
    focus_keyframes: Dict[str, List[str]] = field(default_factory=dict)  # event_id -> kf_ids it is the focus of

    # ------------------------------------------------------------ entity helpers
    def entity_kind(self, entity_id: str) -> str:
        if entity_id in self.characters:
            return "character"
        if entity_id in self.props:
            return "prop"
        raise KeyError(entity_id)

    def entity_size(self, entity_id: str) -> Vec3:
        if entity_id in self.characters:
            return (CHARACTER_SIZE_XZ[0], float(self.characters[entity_id].height_m), CHARACTER_SIZE_XZ[1])
        prop = self.props[entity_id]
        return (float(prop.size_x_m), float(prop.size_y_m), float(prop.size_z_m))

    def entity_colors(self, entity_id: str) -> Tuple[str, str]:
        if entity_id in self.characters:
            palette = self.characters[entity_id].palette_hex
            primary = palette[0] if palette else DEFAULT_COLOR
            return primary, palette[1] if len(palette) > 1 else primary
        color = self.props[entity_id].color_hex
        return color, color

    def entity_shape(self, entity_id: str) -> str:
        """Blocking shape: "character" for people, "quadruped" for animals (Character.body), else the prop's."""
        if entity_id in self.characters:
            return QUADRUPED if self.characters[entity_id].body == QUADRUPED else "character"
        return self.props[entity_id].shape

    def body(self, entity_id: str) -> PhysBody:
        """The entity's physics body, or a class default when the planner omitted it."""
        if entity_id in self.bodies:
            return self.bodies[entity_id]
        if entity_id in self.characters:
            return PhysBody(entity_id=entity_id, regime="articulated", material="flesh", mass_class="l",
                            friction="high", restitution="dead", failure_mode="none", container_fill="none")
        return PhysBody(entity_id=entity_id, regime="rigid", material="wood", mass_class="s",
                        friction="med", restitution="low", failure_mode="none", container_fill="none")

    # ------------------------------------------------------------ timeline helpers
    def prev_keyframe_id(self, kf_id: str) -> str:
        """The keyframe before `kf_id` in timeline order, or "init" for the first one."""
        pos = self.kf_position[kf_id]
        return self.keyframes[pos - 1][1].id if pos > 0 else INIT

    def event_span(self, event_id: str) -> Optional[Tuple[int, int]]:
        """(first, last) keyframe positions an event covers: its start keyframe and every keyframe focusing it."""
        event = self.events[event_id]
        positions = [self.kf_position[k] for k in self.focus_keyframes.get(event_id, [])]
        if event.start_kf_id in self.kf_position:
            positions.append(self.kf_position[event.start_kf_id])
        if not positions:
            return None
        start = self.kf_position.get(event.start_kf_id, min(positions))
        return start, max(positions)

    def shot_keyframes(self, shot_id: str) -> List[Keyframe]:
        return [kf for sid, kf in self.keyframes if sid == shot_id]


def build_index(plan: ScenePlan) -> SceneIndex:
    """Index a plan. First occurrence wins for duplicate IDs (the validator reports duplicates)."""
    bible, timeline = plan.bible, plan.timeline

    def first_by_id(items, key=lambda item: item.id):
        out: Dict[str, object] = {}
        for item in items:
            out.setdefault(key(item), item)
        return out

    characters = first_by_id(bible.characters)
    props = {pid: p for pid, p in first_by_id(bible.props).items() if pid not in characters}
    shot_list = sorted(enumerate(bible.shots), key=lambda pair: (pair[1].order, pair[0]))
    shots = first_by_id([s for _, s in shot_list])
    shot_rank = {sid: rank for rank, sid in enumerate(shots)}

    # Timeline shots in bible shot order (unknown shots last, in timeline order), keyframes by `order`.
    timelines = first_by_id(timeline.shots, key=lambda t: t.shot_id)
    ordered_tls = sorted(enumerate(timelines.values()),
                         key=lambda pair: (shot_rank.get(pair[1].shot_id, len(shot_rank)), pair[0]))
    keyframes: List[Tuple[str, Keyframe]] = []
    keyframe_by_id: Dict[str, Keyframe] = {}
    shot_of_keyframe: Dict[str, str] = {}
    for _, tl in ordered_tls:
        for _, kf in sorted(enumerate(tl.keyframes), key=lambda pair: (pair[1].order, pair[0])):
            if kf.id in keyframe_by_id:
                continue
            keyframes.append((tl.shot_id, kf))
            keyframe_by_id[kf.id] = kf
            shot_of_keyframe[kf.id] = tl.shot_id

    events: Dict[str, PhysEvent] = {}
    event_shot: Dict[str, str] = {}
    cause_links: Dict[str, CauseLink] = {}
    for _, tl in ordered_tls:
        for ev in tl.events:
            if ev.id not in events:
                events[ev.id] = ev
                event_shot[ev.id] = tl.shot_id
        for cl in tl.cause_links:
            cause_links.setdefault(cl.id, cl)

    focus_keyframes: Dict[str, List[str]] = {}
    for _, kf in keyframes:
        if kf.focus_event_id != NONE:
            focus_keyframes.setdefault(kf.focus_event_id, []).append(kf.id)

    return SceneIndex(
        plan=plan,
        characters=characters,
        props=props,
        entities={**characters, **props},
        marks=first_by_id(bible.marks),
        locations=first_by_id(bible.locations),
        shots=shots,
        beats=first_by_id(sorted(bible.beats, key=lambda b: b.order)),
        bodies=first_by_id(bible.phys_bodies, key=lambda b: b.entity_id),
        keyframes=keyframes,
        keyframe_by_id=keyframe_by_id,
        shot_of_keyframe=shot_of_keyframe,
        events=events,
        event_shot=event_shot,
        cause_links=cause_links,
        invariants=first_by_id(timeline.invariants),
        timelines=dict(timelines),
        kf_position={kf.id: pos for pos, (_, kf) in enumerate(keyframes)},
        focus_keyframes=focus_keyframes,
    )


# ---------------------------------------------------------------- flag rules
def merge_flags(flags: List[str]) -> List[str]:
    """Dedupe flags in order; fill flags are mutually exclusive (the last one stated wins)."""
    fills = [f for f in flags if f in FILL_FLAGS]
    out: List[str] = []
    for f in flags:
        if f in FILL_FLAGS or f in out:
            continue
        out.append(f)
    if fills:
        out.append(fills[-1])
    return out


def is_container(entity_id: str, flags: List[str], index: SceneIndex) -> bool:
    if any(f in FILL_FLAGS for f in flags):
        return True
    return entity_id in index.entities and index.body(entity_id).container_fill != NONE


def flags_with(flags: List[str], flag: str, container: bool) -> List[str]:
    """Flags after adding an outcome flag: fill flags replace each other; spilling empties a container."""
    out = [f for f in flags if f not in FILL_FLAGS] if flag in FILL_FLAGS else list(flags)
    if flag not in out:
        out.append(flag)
    if flag == "spilled" and container:
        out = [f for f in out if f not in FILL_FLAGS] + ["fill_empty"]
    return merge_flags(out)


def add_flag(state: EntityState, flag: str, index: SceneIndex) -> None:
    state.flags = flags_with(state.flags, flag, is_container(state.entity_id, state.flags, index))


# ---------------------------------------------------------------- reducer
def _copy(state: EntityState, kf_id: str) -> EntityState:
    return state.model_copy(update={"kf_id": kf_id}, deep=True)


def _default_state(entity_id: str, index: SceneIndex) -> EntityState:
    """Snapshot for an entity the planner never placed: absent until an event or snapshot says otherwise."""
    is_char = entity_id in index.characters
    return EntityState(
        kf_id=INIT, entity_id=entity_id, present=False, mark_id=NONE, offset_x_m=0.0, offset_z_m=0.0,
        facing_id=NONE, pose="stand" if is_char else "none", expression="neutral" if is_char else "none",
        holding_id=NONE, gaze_id=NONE, support="ground", support_id=NONE, flags=[], weight_cue="none",
        secondary=[],
    )


def _set_holder(states: Dict[str, EntityState], obj: str, holder: str) -> None:
    """Make `holder` the only holder of `obj` (NONE releases it)."""
    for st in states.values():
        if st.holding_id == obj and st.entity_id != holder:
            st.holding_id = NONE
    if holder != NONE and holder in states and holder != obj:
        states[holder].holding_id = obj
        states[obj].support, states[obj].support_id = "held", holder
    elif states[obj].support == "held":
        states[obj].support, states[obj].support_id = "ground", NONE


def _outcome_holder(event: PhysEvent, index: SceneIndex) -> Optional[str]:
    """New holder of the moved entity after the event: an id, NONE (released), or None (unchanged)."""
    receiver = event.outcome_support_id if event.outcome_support_id in index.characters else None
    actor = event.actor_id if event.actor_id in index.characters else None
    if event.kind == "hand_off":
        return receiver
    if event.outcome_rest == "held":
        return receiver or actor
    if event.kind in ("lift", "catch"):
        return actor
    if event.kind in ("place", "drop", "throw") or event.outcome_rest in ("rest", "airborne", "rolling"):
        return NONE
    return None


def outcome_support(event: PhysEvent, mover: str, current: str = "ground") -> Optional[Tuple[str, str]]:
    """
    (support, support_id) stated by a non-held outcome, or None when the outcome leaves support unchanged.
    `current` is the mover's support before the event: standing up always leaves a seat.
    """
    if event.outcome_rest in ("rest", "rolling"):
        sid = event.outcome_support_id
        return ("on_entity", sid) if sid != NONE and sid != mover else ("ground", NONE)
    if event.outcome_rest == "airborne":
        return "airborne", NONE
    if event.kind == "stand_up" and current == "on_entity":
        return "ground", NONE
    return None


def outcome_pose(event: PhysEvent, pose: str) -> str:
    """A person's pose once the event has completed: sit_down seats them, stand_up gets them up."""
    if event.kind == "sit_down" and pose not in LOW_POSES:
        return "sit"
    if event.kind == "stand_up" and (pose in LOW_POSES or pose == "fall"):
        return "stand"
    return pose


def carried_by(states: Mapping[str, Any], carrier: str) -> List[str]:
    """
    Entities that travel with `carrier`: held by, attached to or resting on it, directly or through
    each other (cups on a tray in a waiter's hand). Works on EntityState and EntityFrame maps alike.
    """
    out: List[str] = []
    frontier, seen = [carrier], {carrier}
    while frontier:
        cur = frontier.pop()
        for eid, st in states.items():
            if eid not in seen and st.support in CARRIED_SUPPORTS and st.support_id == cur:
                seen.add(eid)
                out.append(eid)
                frontier.append(eid)
    return out


def support_chain(states: Mapping[str, Any], entity_id: str) -> List[str]:
    """The entities carrying `entity_id`, nearest first (its holder or surface, that one's carrier, ...)."""
    out: List[str] = []
    cur = entity_id
    while cur in states and states[cur].support in CARRIED_SUPPORTS:
        nxt = states[cur].support_id
        if nxt not in states or nxt == entity_id or nxt in out:
            break
        out.append(nxt)
        cur = nxt
    return out


def set_present(states: Dict[str, EntityState], entity_id: str, present: bool) -> None:
    """Presence of an entity and of everything it carries (a character exits with what it holds)."""
    for eid in [entity_id] + carried_by(states, entity_id):
        states[eid].present = present


def apply_outcome(states: Dict[str, EntityState], event: PhysEvent, index: SceneIndex) -> None:
    """Apply a completed event's stated outcome (presence, custody, support, mark, pose, flag) in place."""
    mover = event_mover(event)
    if mover not in states:
        return
    st = states[mover]
    # 1. Presence, shared by everything the mover carries.
    if event.kind in ("enter", "exit"):
        set_present(states, mover, event.kind == "enter")
    # 2. Custody.
    holder = _outcome_holder(event, index)
    if holder is not None:
        _set_holder(states, mover, holder)
    # 3. Support, unless the entity is now held.
    support = outcome_support(event, mover, st.support)
    if (holder is None or holder == NONE) and support is not None:
        st.support, st.support_id = support
    if mover in index.characters:
        st.pose = outcome_pose(event, st.pose)
    # 4. Mark.
    if event.outcome_mark_id != NONE:
        st.mark_id, st.offset_x_m, st.offset_z_m = event.outcome_mark_id, 0.0, 0.0
    # 5. Flags.
    if event.outcome_flag != NONE:
        add_flag(st, event.outcome_flag, index)
    if event.kind == "spill" and is_container(mover, st.flags, index):
        add_flag(st, "fill_empty", index)


def _release_stale_holders(states: Dict[str, EntityState], stated: Mapping[str, EntityState]) -> None:
    """
    A planner snapshot that puts a present object anywhere but in X's hand releases X when X has no
    snapshot of its own at this keyframe: X's holding_id was only carried forward, and the object's stated
    placement wins (the custody check then asks for the event that took it out of X's hand). An absent
    object's placement is moot (it left with its holder), so it releases nobody.
    """
    for obj, snap in stated.items():
        if not snap.present:
            continue
        for holder, st in states.items():
            if st.holding_id == obj and holder not in stated \
                    and not (snap.support == "held" and snap.support_id == holder):
                st.holding_id = NONE


def _normalize(states: Dict[str, EntityState]) -> None:
    """Flag rules plus one-way custody consistency: when A holds B, B is held by A."""
    for st in states.values():
        st.flags = merge_flags(st.flags)
    for st in states.values():
        held = st.holding_id
        if held != NONE and held in states and held != st.entity_id:
            states[held].support, states[held].support_id = "held", st.entity_id


def event_completed_at(event: PhysEvent, span: Tuple[int, int], pos: int, keyframe: Keyframe) -> bool:
    """True when the event's outcome holds at the keyframe at timeline position `pos`."""
    if pos > span[1]:
        return True
    if keyframe.focus_event_id != event.id:
        return False
    if keyframe.depicted_phase in SETTLED_PHASES:
        return True
    return event.kind in INSTANT_OUTCOME_KINDS and keyframe.depicted_phase in ("impact", "rebound")


def reduce_states(plan: ScenePlan, index: SceneIndex) -> Dict[str, Dict[str, EntityState]]:
    """
    Complete snapshots of every entity, keyed by kf_id ("init" included). At each keyframe:
    (1) outcomes of events completed by then, (2) planner snapshots for that keyframe, which override
    every field they state except that irreversible flags are never removed.
    """
    # 1. Initial snapshot.
    current = {eid: _default_state(eid, index) for eid in index.entities}
    seen = set()
    for st in plan.bible.initial_states:
        if st.entity_id in current and st.entity_id not in seen:
            current[st.entity_id] = _copy(st, INIT)
            seen.add(st.entity_id)
    _normalize(current)
    reduced: Dict[str, Dict[str, EntityState]] = {INIT: current}

    snapshots: Dict[str, Dict[str, EntityState]] = {}
    for tl in plan.timeline.shots:
        for st in tl.states:
            if st.entity_id in index.entities:
                snapshots.setdefault(st.kf_id, {})[st.entity_id] = st

    # 2. Events in completion order.
    pending = []
    for order, (eid, ev) in enumerate(index.events.items()):
        span = index.event_span(eid)
        if span is not None:
            pending.append((span[1], span[0], order, ev, span))
    pending.sort(key=lambda item: item[:3])
    applied, entered = set(), set()

    # 3. Walk the keyframes.
    for pos, (_, kf) in enumerate(index.keyframes):
        states = {eid: _copy(st, kf.id) for eid, st in current.items()}
        for _, start, _, ev, span in pending:
            # An entering entity (and what it carries) is present from the event's first keyframe,
            # before it reaches its mark.
            if ev.kind == "enter" and start <= pos and ev.id not in entered:
                entered.add(ev.id)
                if event_mover(ev) in states:
                    set_present(states, event_mover(ev), True)
            if ev.id not in applied and event_completed_at(ev, span, pos, kf):
                apply_outcome(states, ev, index)
                applied.add(ev.id)
        stated = snapshots.get(kf.id, {})
        for eid, snap in stated.items():
            kept = [f for f in states[eid].flags if f in IRREVERSIBLE_FLAGS]
            new = _copy(snap, kf.id)
            new.flags = merge_flags(list(new.flags) + [f for f in kept if f not in new.flags])
            states[eid] = new
        _release_stale_holders(states, stated)
        _normalize(states)
        reduced[kf.id] = states
        current = states
    return reduced
