"""
Physics planner interface.

A planner is the "Claude decides" half of the previs pipeline: it turns scene text into a
ScenePlan (bible + timeline), repairs the shots the validator flags, and amends the plan when an
accepted render disagrees with it. Code (reducer, solver, camera) computes everything else.

`replace_shots` is the one merge rule every planner uses for repair and amend: only the listed
shots' timelines change, and a reply that touches any other shot is rejected.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence

from src.previs.schema import Invariant, ScenePlan, Timeline, keyframe_ids_for_shot


class PlannerError(RuntimeError):
    """A planner call failed or returned an unusable plan. `retryable` is False for configuration
    errors (unknown CLI flag, missing credentials, usage limit, model reroute) that a retry cannot fix."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class PlannerTimeoutError(PlannerError):
    """The streaming watchdog aborted the call (no events, no output text, or total time exceeded)."""


class PlannerTruncatedError(PlannerError):
    """The reply hit the output-token cap; partial JSON is never stitched."""


class PhysicsPlanner(ABC):
    """Produces and edits ScenePlans. `work_dir` is the run directory (raw calls go to work_dir/claude/)."""

    @abstractmethod
    def plan(self, scene_text: str, work_dir: Path) -> ScenePlan:
        raise NotImplementedError

    @abstractmethod
    def repair(self, plan: ScenePlan, violations_text: str, shot_ids: List[str], work_dir: Path) -> ScenePlan:
        """Return the full plan with ONLY the listed shots' timelines replaced."""
        raise NotImplementedError

    @abstractmethod
    def amend(self, plan: ScenePlan, facts_text: str, kf_id: str, work_dir: Path) -> ScenePlan:
        """Return the full plan updated so it agrees with facts observed at keyframe `kf_id`."""
        raise NotImplementedError


def shot_keyframe_ids(plan: ScenePlan) -> Dict[str, List[str]]:
    """Pre-created keyframe ids per shot, in shot order."""
    return {s.id: keyframe_ids_for_shot(s) for s in sorted(plan.bible.shots, key=lambda s: s.order)}


def ordered_shot_ids(plan: ScenePlan, shot_ids: Sequence[str]) -> List[str]:
    """Validate a requested shot set and return it in shot order. Raises ValueError on bad input."""
    order = {s.id: s.order for s in plan.bible.shots}
    requested = [s.strip().lower() for s in shot_ids]
    if not requested:
        raise ValueError("shot_ids is empty")
    unknown = [s for s in requested if s not in order]
    if unknown:
        raise ValueError(f"unknown shot ids: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate shot ids: {requested}")
    return sorted(requested, key=lambda s: order[s])


def replace_shots(plan: ScenePlan, reply: Timeline, shot_ids: Sequence[str]) -> ScenePlan:
    """
    Merge a repair/amend reply into `plan`, replacing exactly the shots in `shot_ids`.

    Rejected (PlannerError) when the reply:
      - contains a shot outside the set, omits a listed shot, or repeats one;
      - uses, inside a listed shot, a keyframe id that belongs to another shot (keyframes, snapshots,
        event start_kf_id) — that would silently edit the other shot's state;
      - changes an invariant that lies entirely outside the listed shots;
      - creates duplicate event / cause-link / invariant ids across the merged plan.
    Invariants: originals with an endpoint in the listed shots are replaced by the reply's
    invariants that have an endpoint there; all others are kept unchanged.
    """
    wanted = ordered_shot_ids(plan, shot_ids)
    got = [st.shot_id for st in reply.shots]
    extra = sorted(set(got) - set(wanted))
    missing = [s for s in wanted if s not in got]
    if extra:
        raise PlannerError(f"reply touches shots outside the requested set {wanted}: {extra}")
    if missing:
        raise PlannerError(f"reply omits requested shots: {missing}")
    if len(set(got)) != len(got):
        raise PlannerError(f"reply repeats shots: {got}")

    # 1. Each replaced shot may only use its own pre-created keyframe ids.
    kf_ids = shot_keyframe_ids(plan)
    for st in reply.shots:
        own = set(kf_ids[st.shot_id])
        used = ([k.id for k in st.keyframes] + [s.kf_id for s in st.states]
                + [e.start_kf_id for e in st.events])
        foreign = sorted({k for k in used if k not in own})
        if foreign:
            raise PlannerError(f"reply shot {st.shot_id} uses keyframe ids of other shots: {foreign}")

    # 2. Invariants: keep the ones outside the repaired shots, take the reply's inside them.
    touched = {k for s in wanted for k in kf_ids[s]}

    def touches(iv: Invariant) -> bool:
        return iv.from_kf_id in touched or iv.until_kf_id in touched

    kept = [iv for iv in plan.timeline.invariants if not touches(iv)]
    accepted: List[Invariant] = []
    for iv in reply.invariants:
        if touches(iv):
            accepted.append(iv)
        elif iv not in kept:
            raise PlannerError(f"reply changes invariant {iv.id} outside the requested shots {wanted}")
    invariants = kept + accepted

    # 3. Rebuild the timeline in shot order.
    by_id = {st.shot_id: st for st in reply.shots}
    shots = [by_id.get(st.shot_id, st) for st in plan.timeline.shots]
    present = {st.shot_id for st in shots}
    order = {s.id: s.order for s in plan.bible.shots}
    shots += [by_id[s] for s in wanted if s not in present]
    shots.sort(key=lambda st: order.get(st.shot_id, 10 ** 6))

    # 4. IDs must stay unique across the merged plan.
    for label, ids in (
        ("invariant", [iv.id for iv in invariants]),
        ("event", [e.id for st in shots for e in st.events]),
        ("cause link", [c.id for st in shots for c in st.cause_links]),
    ):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise PlannerError(f"merged plan has duplicate {label} ids: {dupes}")

    merged = ScenePlan(bible=plan.bible, timeline=Timeline(invariants=invariants, shots=shots))
    return ScenePlan.model_validate(merged.model_dump())
