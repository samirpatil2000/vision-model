"""
Deterministic physics solver: Claude decides, code computes.

The planner states intent: which event happens, its strength class, and where things end up.
solve() turns that into geometry and time. It is outcome-first: from the start pose (reduced state
before the event) and the stated outcome (mark / support / holder) it back-solves velocities with
class-level tables (mass, friction, restitution, strength impulse), checks feasibility, places every
keyframe on the physical and edited clocks, and builds an EntityFrame for every entity at every
keyframe. sample_frame() evaluates the same model at any edited time for the gray-box animatic.

Clocks: physical time is world time; edited time is the cut. Inside a shot the two are related
piecewise-linearly through the keyframes (slope = the shot's time_scale, ramped where keyframes
need a minimum on-screen gap). Shots are concatenated on the edited clock; on the physical clock a
shot may overlap the previous one (cutting back to show an action from another angle).
"""
import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src import config
from src.previs.schema import NONE, EntityState, Location, PhysEvent, ScenePlan
from src.previs.state import (
    CARRIED_SUPPORTS,
    INIT,
    QUADRUPED,
    SceneIndex,
    apply_outcome,
    build_index,
    carried_by,
    event_completed_at,
    event_mover,
    flags_with,
    is_container,
    outcome_pose,
    outcome_support,
    reduce_states,
    support_chain,
)
from src.previs.types import EntityFrame, FrameState, SolvedEvent, SolvedPlan, SolvedShot, Vec3

# ---------------------------------------------------------------- class tables
G = 9.81
MASS_KG: Dict[str, float] = {"xs": 0.05, "s": 0.5, "m": 4.0, "l": 40.0, "xl": 300.0, "fixed": math.inf}
RESTITUTION_E: Dict[str, float] = {"dead": 0.05, "low": 0.2, "med": 0.45, "high": 0.75}
FRICTION_MU: Dict[str, float] = {"low": 0.15, "med": 0.4, "high": 0.8}
# Impulse a body part delivers per strength class (N*s). For light objects the speed is further
# capped by how fast a hand moves at that effort (LIMB_SPEED_MAX_MS), which keeps "gentle" gentle.
STRENGTH_IMPULSE_NS: Dict[str, Tuple[float, float]] = {
    "gentle": (0.05, 2.0), "moderate": (0.5, 8.0), "hard": (4.0, 40.0), "violent": (20.0, 400.0),
}
LIMB_SPEED_MAX_MS: Dict[str, float] = {"gentle": 6.0, "moderate": 12.0, "hard": 22.0, "violent": 40.0}
MAX_LIFT_KG: Dict[str, float] = {"gentle": 5.0, "moderate": 20.0, "hard": 50.0, "violent": 100.0}
THROW_ANGLE_DEG: Dict[str, float] = {"gentle": 45.0, "moderate": 35.0, "hard": 20.0, "violent": 10.0}
THROW_TUMBLE_DPS: Dict[str, float] = {"gentle": 180.0, "moderate": 360.0, "hard": 540.0, "violent": 720.0}
FALL_TUMBLE_DPS = 270.0
WALK_SPEED = 1.4
RUN_SPEED = 4.0
ROLL_FRICTION_FACTOR = 0.2
DURATION_CLASS_S: Dict[str, float] = {"short": 2.0, "medium": 3.5, "long": 5.0}
TIME_SCALE: Dict[str, float] = {"real": 1.0, "slow_2x": 2.0, "slow_4x": 4.0, "ramp": 2.0}
FIXED_DURATION_S: Dict[str, float] = {
    "lift": 0.8, "place": 0.8, "catch": 0.8, "hand_off": 0.8, "sit_down": 1.2, "stand_up": 1.2, "gesture": 0.6,
    "break": 0.4, "impact": 0.4, "collide": 0.5, "bounce": 0.8, "tear": 0.8, "spill": 1.0, "settle": 1.0,
    "pour": 1.5, "fall_over": 0.8,
}

ANTICIPATION_S = 0.35        # wind-up before a volitional action's contact/release
REACTION_S = 0.25            # a person reacting to a cause
MIN_KF_GAP_S = 0.25          # minimum edited time between keyframes of one shot
END_HOLD_S = 0.5             # minimum edited hold after a shot's last keyframe
REACH_M = 1.2                # hand reach from a character's base (with a lean)
HAND_OFF_REACH_M = 1.6       # two people reaching toward each other
STOP_SHORT_M = 0.6           # walking "to" an entity stops this far from it
MAX_FALL_DRIFT_MS = 1.0      # horizontal speed a free fall may carry beyond its push (tipping, slop)
MIN_FLIGHT_S = 0.05
FLIGHT_END_TOL_S = 1e-6      # a ballistic body at its last impact instant has landed (not airborne)
SEAT_MAX_M = 1.0             # a person on a support at most this tall uses it as a seat (see seated_on)
MIN_LOCOMOTION_S = 0.5
MIN_BOUNCE_MS = 0.3
MAX_BOUNCES = 6
TRAJECTORY_SAMPLES = 12
EPS = 1e-9

BALLISTIC_KINDS = frozenset({"fall", "drop", "throw"})
AIRBORNE_KINDS = BALLISTIC_KINDS | {"bounce"}      # events that legitimately hold an entity in the air
SLIDE_KINDS = frozenset({"push", "pull", "slide", "roll"})
LOCOMOTION_KINDS = frozenset({"walk_to", "run_to", "enter", "exit"})
HANDLING_KINDS = frozenset({"lift", "place", "catch", "hand_off"})
BODY_KINDS = frozenset({"sit_down", "stand_up", "gesture"})
TRIGGER_AT_START_KINDS = frozenset({"break", "impact", "collide", "bounce"})   # effects begin at contact
PASSIVE_KINDS = frozenset({"fall", "collide", "impact", "bounce", "slide", "roll", "break", "spill", "settle"})
# Poses that put a person's base on top of its support (feet on a crate, body on a bed); in any other
# pose a person on a low support is seated on it with the feet on the support's floor.
ON_TOP_POSES = frozenset({"none", "stand", "walk", "run", "lie"})

# pose -> (forward reach in metres, hand height as a fraction of body height)
HAND_POSE: Dict[str, Tuple[float, float]] = {
    "reach": (0.55, 0.55), "push": (0.55, 0.55), "pull": (0.55, 0.55),
    "carry": (0.30, 0.45), "lift": (0.30, 0.45),
    "throw": (0.15, 0.85),
    "sit": (0.30, 0.30), "crouch": (0.30, 0.30), "kneel": (0.30, 0.30),
}
HAND_DEFAULT = (0.25, 0.45)
HAND_LATERAL_M = 0.18
# An animal holds things in its mouth, just under the snout tip of the blocking proxy
# (blocking.mannequin.quadruped_segments): pose -> (forward, height) as fractions of the shoulder height.
QUADRUPED_MOUTH: Dict[str, Tuple[float, float]] = {
    "stand": (0.82, 0.87), "walk": (0.84, 0.82), "run": (0.91, 0.74), "brace": (0.79, 0.79),
    "sit": (0.68, 1.03), "crouch": (0.90, 0.43), "lie": (0.81, 0.44), "fall": (0.82, 0.17),
}
QUADRUPED_POSE_ALIAS: Dict[str, str] = {"kneel": "lie", "pull": "brace"}   # every other pose draws as "stand"


# ---------------------------------------------------------------- vector helpers
def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _lerp(a: Vec3, b: Vec3, f: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f)


def _dist_xz(a: Vec3, b: Vec3) -> float:
    return math.hypot(b[0] - a[0], b[2] - a[2])


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))


def _unit_xz(dx: float, dz: float) -> Optional[Tuple[float, float]]:
    n = math.hypot(dx, dz)
    return (dx / n, dz / n) if n > 1e-9 else None


def _lerp_angle(a: float, b: float, f: float) -> float:
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + delta * f


def yaw_toward(a: Vec3, b: Vec3) -> float:
    """Yaw (degrees) that faces from a to b: 0 faces +z, positive turns toward +x."""
    return math.degrees(math.atan2(b[0] - a[0], b[2] - a[2]))


# ---------------------------------------------------------------- closed-form physics
def fall_time(height_m: float) -> float:
    """Free-fall time from rest through `height_m`: sqrt(2h/g)."""
    return math.sqrt(2.0 * height_m / G) if height_m > 0 else 0.0


def strength_speed_range(strength: str, mass_kg: float) -> Tuple[float, float]:
    """(min, max) launch speed a strength class gives a body of `mass_kg`."""
    j_min, j_max = STRENGTH_IMPULSE_NS[strength]
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        return 0.0, 0.0
    return j_min / mass_kg, min(j_max / mass_kg, LIMB_SPEED_MAX_MS[strength])


def slide_speed(distance_m: float, friction: str, roll: bool = False) -> float:
    """Initial speed that stops a sliding (or rolling) body after `distance_m`: v0 = sqrt(2*mu*g*d)."""
    mu = FRICTION_MU[friction] * (ROLL_FRICTION_FACTOR if roll else 1.0)
    return math.sqrt(2.0 * mu * G * max(distance_m, 0.0))


def slide_distance(speed_ms: float, friction: str, roll: bool = False) -> float:
    """Stopping distance of a body launched at `speed_ms`: d = v0^2 / (2*mu*g)."""
    mu = FRICTION_MU[friction] * (ROLL_FRICTION_FACTOR if roll else 1.0)
    return speed_ms ** 2 / (2.0 * mu * G)


@dataclass
class ThrowSolution:
    feasible: bool
    speed_ms: float
    angle_deg: float
    flight_time_s: float
    velocity: Vec3
    note: str = ""


def solve_throw(release: Vec3, landing: Vec3, strength: str, mass_kg: float) -> ThrowSolution:
    """
    Back-solve the launch velocity that carries a body from `release` to `landing`. The strength class
    picks the preferred launch angle; nearby angles (5 degree steps) are tried when it cannot reach.
    """
    dx, dz = landing[0] - release[0], landing[2] - release[2]
    rng, dy = math.hypot(dx, dz), landing[1] - release[1]
    v_min, v_max = strength_speed_range(strength, mass_kg)
    if rng < 0.05:
        return ThrowSolution(False, 0.0, 90.0, 0.0, (0.0, 0.0, 0.0),
                             "landing point is directly below or above the hand; use drop or lift instead of throw")
    base = THROW_ANGLE_DEG[strength]
    angles = sorted(range(5, 86, 5), key=lambda a: (abs(a - base), a))
    best_speed = math.inf
    for angle in [base] + [float(a) for a in angles if a != base]:
        theta = math.radians(angle)
        rise = rng * math.tan(theta) - dy
        if rise <= 0:
            continue
        speed = math.sqrt(G * rng ** 2 / (2.0 * math.cos(theta) ** 2 * rise))
        best_speed = min(best_speed, speed)
        if speed <= v_max + EPS:
            flight = rng / (speed * math.cos(theta))
            ux, uz = dx / rng, dz / rng
            horiz = speed * math.cos(theta)
            notes = []
            if angle != base:
                notes.append(f"launch angle {angle:.0f} deg instead of the {strength} default {base:.0f} deg")
            if speed < v_min - EPS:
                notes.append(f"needs only {speed:.1f} m/s, softer than a {strength} throw")
            return ThrowSolution(True, speed, angle, flight, (horiz * ux, speed * math.sin(theta), horiz * uz),
                                 "; ".join(notes))
    need = f"at least {best_speed:.1f} m/s" if math.isfinite(best_speed) else "an impossible launch"
    return ThrowSolution(False, best_speed if math.isfinite(best_speed) else 0.0, base, 0.0, (0.0, 0.0, 0.0),
                         f"reaching a point {rng:.2f} m away and {dy:+.2f} m up needs {need}, but a {strength} "
                         f"throw of a {mass_kg:g} kg body reaches at most {v_max:.1f} m/s")


def hand_geometry(pose: str, quadruped: bool = False) -> Tuple[float, float, float]:
    """(forward m, height fraction, lateral m) of the gripping point: a person's right hand, an animal's mouth."""
    if quadruped:
        if pose not in QUADRUPED_MOUTH:
            pose = QUADRUPED_POSE_ALIAS.get(pose, "stand")
        forward, up = QUADRUPED_MOUTH[pose]
        return forward, up, 0.0
    forward, up = HAND_POSE.get(pose, HAND_DEFAULT)
    return forward, up, HAND_LATERAL_M


def _hand_point(base: Vec3, yaw_deg: float, pose: str, height: float, side: str = "right",
                quadruped: bool = False) -> Vec3:
    forward, up, lateral = hand_geometry(pose, quadruped)
    if quadruped:
        forward *= height                # the mouth table is in shoulder heights
    yaw = math.radians(yaw_deg)
    fx, fz = math.sin(yaw), math.cos(yaw)
    rx, rz = -fz, fx                     # right = forward x up
    lateral = lateral if side == "right" else -lateral
    return (base[0] + fx * forward + rx * lateral, base[1] + up * height, base[2] + fz * forward + rz * lateral)


def hand_yaw_toward(base: Vec3, point: Vec3, pose: str, quadruped: bool = False) -> Optional[float]:
    """Yaw that turns a body at `base` so its gripping point (offset to the right of its facing) lies on the
    line toward `point`; None when the point is straight above the base."""
    if _dist_xz(base, point) <= 1e-9:
        return None
    forward, _, lateral = hand_geometry(pose, quadruped)
    return yaw_toward(base, point) + math.degrees(math.atan2(lateral, forward))


def is_immovable(index: SceneIndex, entity_id: str) -> bool:
    """Fixed-mass bodies and props marked movable=False never move."""
    if index.body(entity_id).mass_class == "fixed":
        return True
    return entity_id in index.props and not index.props[entity_id].movable


def seated_on(index: SceneIndex, entity_id: str, pose: str, support_id: str) -> bool:
    """
    True when a person uses its support as a seat: a human on a support at most SEAT_MAX_M tall, in any
    pose but standing or lying on it, keeps its base (feet) on the support's own floor level. A chair's
    size_y includes its backrest, so its top is no seat height. Animals and props rest on top.
    """
    character = index.characters.get(entity_id)
    if character is None or character.body != "human" or support_id == entity_id or support_id not in index.entities:
        return False
    return pose not in ON_TOP_POSES and index.entity_size(support_id)[1] <= SEAT_MAX_M + EPS


def resting_height(index: SceneIndex, entity_id: str, pose: str, support_id: str, support_base_y: float) -> float:
    """Base height of an entity resting on `support_id`, whose own base is at `support_base_y`."""
    if seated_on(index, entity_id, pose, support_id):
        return support_base_y
    return support_base_y + index.entity_size(support_id)[1]


def is_quadruped(index: SceneIndex, entity_id: str) -> bool:
    character = index.characters.get(entity_id)
    return character is not None and character.body == QUADRUPED


def hand_position(holder_frame: EntityFrame, side: str = "right") -> Vec3:
    """World position of a character's hand for its pose (an animal's mouth): where held objects attach."""
    return _hand_point(holder_frame.position, holder_frame.yaw_deg, holder_frame.pose, holder_frame.size[1], side,
                       quadruped=holder_frame.shape == QUADRUPED)


def grip_point(entities: Dict[str, EntityFrame], character_id: str) -> Optional[Vec3]:
    """
    Where a character's right hand (an animal's mouth) must be drawn: on the object it holds, which the solver
    places in the hand, except at the contact instant of a catch or lift (the hand meets the object where it
    is). None when the character holds nothing present.
    """
    holder = entities.get(character_id)
    held = entities.get(holder.holding_id) if holder is not None else None
    if held is None or not held.present or held.entity_id == character_id:
        return None
    return held.position


# ---------------------------------------------------------------- motion model
@dataclass
class _Motion:
    """Solved geometry of one event. Times are offsets (s) from the event start on the physical clock."""
    event: PhysEvent
    mover: str
    model: str                    # "ballistic" | "slide" | "linear" | "static"
    p0: Vec3
    p1: Vec3
    duration: float               # how long the mover moves (ballistic: flight + bounces)
    phases: Dict[str, float]
    trigger: float                # offset at which events it causes begin
    end: float                    # offset at which the event is over
    feasible: bool = True
    note: str = ""
    volitional: bool = False
    v0: Vec3 = (0.0, 0.0, 0.0)    # ballistic launch velocity
    speed0: float = 0.0           # slide launch speed
    decel: float = 0.0            # slide deceleration
    exit_speed: float = 0.0       # speed when leaving a support edge (slides)
    exit_velocity: Vec3 = (0.0, 0.0, 0.0)   # velocity when leaving a support edge (slides)
    t_impact: float = 0.0         # ballistic flight time
    bounces: List[Tuple[float, float]] = field(default_factory=list)   # (start offset, vertical speed)
    tumble_dps: float = 0.0
    face_travel: bool = False
    pose: str = ""                # pose override for a character while moving
    windup: float = 0.0           # seconds before the start during which the body moves windup_from -> p0
    windup_from: Vec3 = (0.0, 0.0, 0.0)
    turn: Optional[Tuple[str, float]] = None      # (catcher, yaw): a catcher faces the flight it catches
    full_landing: Optional[Vec3] = None           # set when this flight is cut mid-air for a continuation
    full_flight: float = 0.0                      # flight time from p0 to full_landing
    v_end: Vec3 = (0.0, 0.0, 0.0)                 # velocity where a cut flight ends
    start: Optional[float] = None
    window_end: Optional[float] = None

    def position(self, t: float) -> Vec3:
        if t < 0.0 and self.windup > 0.0:
            return _lerp(self.windup_from, self.p0, min(max((t + self.windup) / self.windup, 0.0), 1.0))
        t = max(t, 0.0)
        if self.model == "ballistic":
            if t <= self.t_impact:
                return (self.p0[0] + self.v0[0] * t, self.p0[1] + self.v0[1] * t - 0.5 * G * t * t,
                        self.p0[2] + self.v0[2] * t)
            for b_start, vb in self.bounces:
                tau = t - b_start
                if tau < 2.0 * vb / G:
                    return (self.p1[0], self.p1[1] + vb * tau - 0.5 * G * tau * tau, self.p1[2])
            return self.p1
        if self.model == "slide":
            total = self._slide_s(self.duration)
            if total <= EPS:
                return self.p1
            return _lerp(self.p0, self.p1, self._slide_s(min(t, self.duration)) / total)
        if self.model == "linear":
            return _lerp(self.p0, self.p1, min(t / self.duration, 1.0)) if self.duration > EPS else self.p1
        return self.p0

    def velocity(self, t: float) -> Vec3:
        if not self.moving(t):
            return (0.0, 0.0, 0.0)
        if t < 0.0:
            return _mul(_sub(self.p0, self.windup_from), 1.0 / self.windup)
        if self.model == "ballistic":
            if t < self.t_impact:
                return (self.v0[0], self.v0[1] - G * t, self.v0[2])
            for b_start, vb in self.bounces:
                tau = t - b_start
                if tau < 2.0 * vb / G:
                    return (0.0, vb - G * tau, 0.0)
            return (0.0, 0.0, 0.0)
        span = _dist(self.p0, self.p1)
        direction = _mul(_sub(self.p1, self.p0), 1.0 / span) if span > EPS else (0.0, 0.0, 0.0)
        if self.model == "slide":
            return _mul(direction, max(self.speed0 - self.decel * t, 0.0))
        return _mul(_sub(self.p1, self.p0), 1.0 / self.duration)

    def moving(self, t: float) -> bool:
        if -self.windup <= t < 0.0:
            return _dist(self.windup_from, self.p0) > 1e-6
        if self.model == "static" or not (0.0 <= t < self.duration):
            return False
        return self.model == "ballistic" or _dist(self.p0, self.p1) > 1e-6

    def in_flight(self, t: float) -> bool:
        # At the final landing instant the body is down (in a catcher's hand, on the floor), not airborne;
        # the tolerance absorbs the rounding of keyframe times placed exactly on that instant.
        return self.model == "ballistic" and 0.0 <= t < self.duration - FLIGHT_END_TOL_S

    def rotation(self, t: float) -> Vec3:
        if self.tumble_dps <= 0 or not (0.0 <= t < self.t_impact):
            return (0.0, 0.0, 0.0)
        ux, uz = _unit_xz(self.v0[0], self.v0[2]) or (0.0, 1.0)
        angle = self.tumble_dps * t
        return (angle * uz, 0.0, -angle * ux)

    def travel_yaw(self) -> Optional[float]:
        return yaw_toward(self.p0, self.p1) if _dist_xz(self.p0, self.p1) > 1e-6 else None

    def active(self, t_phys: float) -> bool:
        return (self.start is not None and self.window_end is not None
                and self.start - self.windup - EPS <= t_phys < self.window_end - EPS)

    def path(self) -> List[Vec3]:
        """TRAJECTORY_SAMPLES points along the part of the path the event actually plays."""
        length = self.duration
        if self.start is not None and self.window_end is not None:
            length = min(length, self.window_end - self.start)
        if self.model == "static" or length <= EPS:
            return []
        n = TRAJECTORY_SAMPLES
        return [self.position(length * i / (n - 1)) for i in range(n)]

    def _slide_s(self, t: float) -> float:
        return self.speed0 * t - 0.5 * self.decel * t * t


def _phase_table(duration: float, volitional: bool, **override: float) -> Dict[str, float]:
    """Physical offset of every depicted phase; kinds override the ones their mechanics define."""
    phases = {
        "anticipation": -ANTICIPATION_S if volitional else 0.0,
        "contact": 0.0, "release": 0.0,
        "rising": duration / 2, "apex": duration / 2, "falling": duration / 2,
        "in_progress": duration / 2, "none": duration / 2,
        "impact": duration, "rebound": duration,
        "aftermath": duration + 0.3, "rest": duration + 0.5,
    }
    phases.update(override)
    return phases


# ---------------------------------------------------------------- pose resolution
class _Resolver:
    """Positions and yaws of all entities at one instant from reduced states (+ motion overrides)."""

    def __init__(self, index: SceneIndex, states: Dict[str, EntityState], overrides: Dict[str, Vec3],
                 prev_pos: Dict[str, Vec3], prev_yaw: Dict[str, float], forced_yaw: Dict[str, float]):
        self.index, self.states, self.overrides = index, states, overrides
        self.prev_pos, self.prev_yaw, self.forced_yaw = prev_pos, prev_yaw, forced_yaw
        self._pos: Dict[str, Vec3] = {}
        self._yaw: Dict[str, float] = {}
        self._busy: set = set()
        self._xz_busy: set = set()

    def _support_of(self, eid: str) -> Optional[str]:
        sid = self.states[eid].support_id
        return sid if sid in self.states and sid != eid else None

    def stated_xz(self, eid: str, depth: int = 0) -> Tuple[float, float]:
        """Where the planner's state puts an entity (its mark plus offset, else offset from its support's
        stated place), ignoring motion and hands."""
        st = self.states.get(eid)
        if st is None:
            return (self.prev_pos[eid][0], self.prev_pos[eid][2]) if eid in self.prev_pos else (0.0, 0.0)
        mark = self.index.marks.get(st.mark_id)
        if mark is not None:
            return mark.x_m + st.offset_x_m, mark.z_m + st.offset_z_m
        sid = self._support_of(eid)
        if st.support in CARRIED_SUPPORTS and sid and depth < 8:
            sx, sz = self.stated_xz(sid, depth + 1)
            return sx + st.offset_x_m, sz + st.offset_z_m
        if eid in self.prev_pos:
            return self.prev_pos[eid][0], self.prev_pos[eid][2]
        return st.offset_x_m, st.offset_z_m

    def base_xz(self, eid: str) -> Tuple[float, float]:
        if eid in self.overrides:
            return self.overrides[eid][0], self.overrides[eid][2]
        st = self.states.get(eid)
        if st is not None and st.support in ("on_entity", "attached") and eid not in self._xz_busy:
            sid = self._support_of(eid)
            if sid:
                self._xz_busy.add(eid)
                try:
                    carried = self._carried_xz(eid, sid)
                finally:
                    self._xz_busy.discard(eid)
                if carried is not None:
                    return carried
        return self.stated_xz(eid)

    def _carried_xz(self, eid: str, sid: str) -> Optional[Tuple[float, float]]:
        """
        An entity resting on a support that is not where the plan states it (held in a hand, carried by
        one, moved by its own event) travels with it: it keeps its stated place relative to the support,
        inside the support's footprint. None when the support is where it is stated.
        """
        actual = self.position(sid)
        sx, sz = self.stated_xz(sid)
        if abs(actual[0] - sx) <= 1e-9 and abs(actual[2] - sz) <= 1e-9:
            return None
        ex, ez = self.stated_xz(eid)
        support_size, own_size = self.index.entity_size(sid), self.index.entity_size(eid)
        half_x = max(support_size[0] - own_size[0], 0.0) / 2.0
        half_z = max(support_size[2] - own_size[2], 0.0) / 2.0
        return (actual[0] + min(max(ex - sx, -half_x), half_x), actual[2] + min(max(ez - sz, -half_z), half_z))

    def position(self, eid: str) -> Vec3:
        if eid in self.overrides:
            return self.overrides[eid]
        if eid in self._pos:
            return self._pos[eid]
        x, z = self.base_xz(eid)
        st = self.states.get(eid)
        if st is None or eid in self._busy:
            return (x, 0.0, z)
        self._busy.add(eid)
        sid = self._support_of(eid)
        if st.support == "held" and sid:
            holder = self.states[sid]
            pos = _hand_point(self.position(sid), self.yaw(sid), holder.pose, self.index.entity_size(sid)[1],
                              quadruped=is_quadruped(self.index, sid))
        elif st.support == "on_entity" and sid:
            pos = (x, resting_height(self.index, eid, st.pose, sid, self.position(sid)[1]), z)
        elif st.support == "attached" and sid:
            pos = (x, self.position(sid)[1] + self.index.entity_size(sid)[1], z)
        elif st.support == "airborne":
            pos = (x, self.prev_pos[eid][1] if eid in self.prev_pos else 0.0, z)
        else:
            pos = (x, 0.0, z)
        self._busy.discard(eid)
        self._pos[eid] = pos
        return pos

    def yaw(self, eid: str) -> float:
        if eid in self.forced_yaw:
            return self.forced_yaw[eid]
        if eid in self._yaw:
            return self._yaw[eid]
        result = self.prev_yaw.get(eid, 0.0)
        st = self.states.get(eid)
        if st is not None and st.facing_id not in (NONE, eid):
            target = None
            if st.facing_id in self.states:
                target = self.base_xz(st.facing_id)
            elif st.facing_id in self.index.marks:
                mark = self.index.marks[st.facing_id]
                target = (mark.x_m, mark.z_m)
            if target is not None:
                x, z = self.base_xz(eid)
                if math.hypot(target[0] - x, target[1] - z) > 1e-6:
                    result = math.degrees(math.atan2(target[0] - x, target[1] - z))
        self._yaw[eid] = result
        return result


# ---------------------------------------------------------------- per-event geometry
@dataclass
class _Geo:
    """What the geometry builders know: the world just before the event (and the stated state after it)."""
    index: SceneIndex
    states: Dict[str, EntityState]
    poses: Dict[str, Tuple[Vec3, float]]
    location: Optional[Location]
    prev: Optional[_Motion]
    after_states: Optional[Dict[str, EntityState]] = None
    after_poses: Optional[Dict[str, Tuple[Vec3, float]]] = None
    ahead: Optional["_Ahead"] = None

    def held_after(self, eid: str, holder: str) -> Optional[Vec3]:
        """Where `eid` sits in `holder`'s hand once the event's outcome holds (pose and facing as stated then)."""
        if self.after_states is None or self.after_poses is None or eid not in self.after_states:
            return None
        st = self.after_states[eid]
        return self.after_poses[eid][0] if st.support == "held" and st.support_id == holder else None

    def pos(self, eid: str) -> Vec3:
        return self.poses[eid][0]

    def yaw(self, eid: str) -> float:
        return self.poses[eid][1]

    def mass(self, eid: str) -> float:
        return MASS_KG[self.index.body(eid).mass_class]

    def immovable(self, eid: str) -> bool:
        return is_immovable(self.index, eid)

    def hand(self, eid: str, pose: str, toward: Optional[Vec3] = None) -> Vec3:
        yaw = yaw_toward(self.pos(eid), toward) if toward is not None and _dist_xz(self.pos(eid), toward) > 1e-6 \
            else self.yaw(eid)
        return _hand_point(self.pos(eid), yaw, pose, self.index.entity_size(eid)[1],
                           quadruped=is_quadruped(self.index, eid))


@dataclass
class _Catch:
    """The catch that ends a flight: who catches, and their body (base, yaw, pose) when the hand meets it."""
    event: PhysEvent
    catcher: str
    base: Vec3
    yaw: float
    pose: str
    height: float
    quadruped: bool

    def hand(self, yaw: Optional[float] = None) -> Vec3:
        return _hand_point(self.base, self.yaw if yaw is None else yaw, self.pose, self.height,
                           quadruped=self.quadruped)


@dataclass
class _Ahead:
    """What comes after an event for the same mover: a flight that ends 'airborne' is continued (and only
    then lands or is caught) by the mover's next event."""
    index: SceneIndex
    states: Dict[str, Dict[str, EntityState]]
    static: Dict[str, Dict[str, Tuple[Vec3, float]]]
    next_of: Dict[str, str]

    def next(self, ev: PhysEvent) -> Optional[PhysEvent]:
        return self.index.events.get(self.next_of.get(ev.id, ""))

    def flight_end(self, ev: PhysEvent) -> PhysEvent:
        """The fall/drop that finishes the flight `ev` starts: `ev` itself unless it ends airborne and a
        fall or drop of the same mover continues it (one fall shown across a cut)."""
        seen = {ev.id}
        while ev.kind in BALLISTIC_KINDS and ev.outcome_rest == "airborne":
            nxt = self.next(ev)
            if nxt is None or nxt.kind not in ("fall", "drop") or nxt.id in seen:
                break
            seen.add(nxt.id)
            ev = nxt
        return ev

    def catch_of(self, ev: PhysEvent) -> Optional[_Catch]:
        """The catch that ends the flight `ev` starts, when the flight ends airborne and a character's catch
        of the same mover follows it."""
        last = self.flight_end(ev)
        if last.kind not in BALLISTIC_KINDS or last.outcome_rest != "airborne":
            return None
        catch = self.next(last)
        mover = event_mover(ev)
        if catch is None or catch.kind != "catch" or catch.actor_id not in self.index.characters \
                or catch.actor_id == mover or event_mover(catch) != mover:
            return None
        kfs = [k for k in [catch.start_kf_id] + self.index.focus_keyframes.get(catch.id, []) if k in self.index.kf_position]
        if not kfs:
            return None
        contact = min(kfs, key=lambda k: self.index.kf_position[k])
        who = catch.actor_id
        base, yaw = self.static[contact][who]
        return _Catch(catch, who, base, yaw, self.states[contact][who].pose, self.index.entity_size(who)[1],
                      is_quadruped(self.index, who))


def _is_volitional(ev: PhysEvent, index: SceneIndex) -> bool:
    return ev.actor_id in index.characters and ev.kind not in PASSIVE_KINDS


def _inside_footprint(p: Vec3, center: Vec3, size: Vec3) -> bool:
    return abs(p[0] - center[0]) <= size[0] / 2 + EPS and abs(p[2] - center[2]) <= size[2] / 2 + EPS


def _exit_distance(p: Vec3, ux: float, uz: float, center: Vec3, size: Vec3) -> float:
    """Distance along (ux, uz) from p to the edge of an axis-aligned footprint (0 if already outside)."""
    if not _inside_footprint(p, center, size):
        return 0.0
    exits = []
    for comp, u, c, half in ((p[0], ux, center[0], size[0] / 2), (p[2], uz, center[2], size[2] / 2)):
        if u > EPS:
            exits.append((c + half - comp) / u)
        elif u < -EPS:
            exits.append((c - half - comp) / u)
    return max(min(exits), 0.0) if exits else 0.0


def _nearest_edge_point(location: Optional[Location], p: Vec3) -> Vec3:
    if location is None:
        return p
    x, z = min(max(p[0], 0.0), location.width_m), min(max(p[2], 0.0), location.depth_m)
    options = [(x, (0.0, z)), (location.width_m - x, (location.width_m, z)), (z, (x, 0.0)),
               (location.depth_m - z, (x, location.depth_m))]
    _, (ex, ez) = min(options, key=lambda o: o[0])
    return (ex, 0.0, ez)


def _destination_xz(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> Optional[Tuple[float, float]]:
    mark = g.index.marks.get(ev.outcome_mark_id)
    if mark is not None:
        return mark.x_m, mark.z_m
    sid = ev.outcome_support_id
    if sid in g.index.entities and sid != mover:
        if _inside_footprint(p0, g.pos(sid), g.index.entity_size(sid)):
            return p0[0], p0[2]
        return g.pos(sid)[0], g.pos(sid)[2]
    return None


def _pose_after(mover: str, g: _Geo) -> str:
    """The mover's pose once the event's outcome holds (the stated after-state, else its pose before)."""
    for states in (g.after_states, g.states):
        if states is not None and mover in states:
            return states[mover].pose
    return NONE


def _rest_height(ev: PhysEvent, mover: str, g: _Geo) -> float:
    sid = ev.outcome_support_id
    if ev.outcome_rest != "held" and sid in g.index.entities and sid != mover:
        return resting_height(g.index, mover, _pose_after(mover, g), sid, g.pos(sid)[1])
    return 0.0


def _settled_height(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> float:
    """
    Base height of a person once a sit_down / stand_up has played out: the height the resolver gives the
    stated after-state, or (past the end of the timeline) the reducer's outcome under the same seat rule.
    """
    if g.after_poses is not None and mover in g.after_poses:
        return g.after_poses[mover][0][1]
    after = {eid: st.model_copy(deep=True) for eid, st in g.states.items()}
    apply_outcome(after, ev, g.index)
    st = after.get(mover)
    if st is None:
        return p0[1]
    if st.support == "on_entity" and st.support_id in g.poses and st.support_id != mover:
        return resting_height(g.index, mover, st.pose, st.support_id, g.pos(st.support_id)[1])
    return 0.0 if st.support == "ground" else p0[1]


def _catcher(ev: PhysEvent, g: _Geo) -> Optional[str]:
    if ev.outcome_rest == "held" and ev.outcome_support_id in g.index.characters:
        return ev.outcome_support_id
    return None


def _static_motion(ev: PhysEvent, mover: str, p0: Vec3, duration: float, volitional: bool,
                   note: str = "", feasible: bool = True, **phase_override: float) -> _Motion:
    phases = _phase_table(duration, volitional, **phase_override)
    end = max(phases["rest"], phases["aftermath"], duration)
    return _Motion(ev, mover, "static", p0, p0, duration, phases, duration, end, feasible, note, volitional)


def _linear_motion(ev: PhysEvent, mover: str, p0: Vec3, p1: Vec3, duration: float, volitional: bool,
                   note: str = "", feasible: bool = True, face_travel: bool = False, pose: str = "") -> _Motion:
    phases = _phase_table(duration, volitional)
    model = "linear" if _dist(p0, p1) > 1e-6 else "static"
    return _Motion(ev, mover, model, p0, p1, duration, phases, duration, phases["rest"], feasible, note,
                   volitional, face_travel=face_travel, pose=pose)


def _ballistic_motion(ev: PhysEvent, mover: str, p0: Vec3, p1: Vec3, v0: Vec3, flight: float, restitution: float,
                      tumble_dps: float, volitional: bool, note: str = "", feasible: bool = True) -> _Motion:
    v_impact = abs(v0[1] - G * flight)
    bounces: List[Tuple[float, float]] = []
    t, vb = flight, restitution * v_impact
    while vb >= MIN_BOUNCE_MS and len(bounces) < MAX_BOUNCES:
        bounces.append((t, vb))
        t += 2.0 * vb / G
        vb *= restitution
    first = bounces[0][1] if bounces else 0.0
    apex = min(max(v0[1] / G, 0.0), flight)
    aftermath = max(flight + 0.3, flight + 2.0 * first / G + 0.1)
    phases = _phase_table(flight, volitional,
                          rising=apex / 2, apex=apex, falling=(apex + flight) / 2, in_progress=(apex + flight) / 2,
                          none=(apex + flight) / 2, impact=flight, rebound=flight + first / G,
                          aftermath=aftermath, rest=max(aftermath, t) + 0.2)
    return _Motion(ev, mover, "ballistic", p0, p1, t, phases, flight, phases["rest"], feasible, note, volitional,
                   v0=v0, t_impact=flight, bounces=bounces, tumble_dps=tumble_dps)


def _interception(p0: Vec3, catch: _Catch, carried: Tuple[float, float]) -> Tuple[Vec3, float, float]:
    """
    (point, flight time, catcher yaw) where an object dropping from p0 (horizontal velocity `carried`)
    meets its catcher's hand. The hand takes it at the hand's height, or as soon as it drops off its
    support when it starts below the hand; meanwhile it drifts (a free fall's slop) toward the catcher's
    reach, and the catcher turns so the hand is on the line toward that point.
    """
    forward, up, lateral = hand_geometry(catch.pose, catch.quadruped)
    if catch.quadruped:
        forward *= catch.height
    hand_y = catch.base[1] + up * catch.height
    y = min(hand_y, p0[1] - 0.5 * G * MIN_FLIGHT_S ** 2)
    flight = fall_time(p0[1] - y)
    x0, z0 = p0[0] + carried[0] * flight, p0[2] + carried[1] * flight
    radius = math.hypot(forward, lateral)
    away = _unit_xz(x0 - catch.base[0], z0 - catch.base[2]) or \
        (math.sin(math.radians(catch.yaw)), math.cos(math.radians(catch.yaw)))
    rx, rz = catch.base[0] + away[0] * radius, catch.base[2] + away[1] * radius
    span = math.hypot(rx - x0, rz - z0)
    step = min(span, MAX_FALL_DRIFT_MS * flight)
    f = step / span if span > EPS else 0.0
    point = (x0 + (rx - x0) * f, y, z0 + (rz - z0) * f)
    yaw = hand_yaw_toward(catch.base, point, catch.pose, catch.quadruped)
    return point, flight, catch.yaw if yaw is None else yaw


def _split_flight(motion: _Motion, fraction: float = 0.5) -> None:
    """Cut a ballistic flight short mid-air: the event shows its first part, its continuation the rest."""
    t = max(motion.t_impact * fraction, MIN_FLIGHT_S / 2)
    motion.full_landing, motion.full_flight = motion.p1, motion.t_impact
    end = motion.position(t)
    motion.v_end = (motion.v0[0], motion.v0[1] - G * t, motion.v0[2])
    motion.p1, motion.t_impact, motion.duration, motion.trigger, motion.bounces = end, t, t, t, []
    apex = min(max(motion.v0[1] / G, 0.0), t)
    motion.phases = _phase_table(t, motion.volitional, rising=apex / 2, apex=apex, falling=(apex + t) / 2,
                                 in_progress=(apex + t) / 2, none=(apex + t) / 2, impact=t, rebound=t)
    motion.end = motion.phases["rest"]


def _continue_flight(ev: PhysEvent, mover: str, prev: _Motion, g: _Geo) -> _Motion:
    """The rest of a flight its previous event cut mid-air: same path, from where and how fast it was."""
    flight = max(prev.full_flight - prev.t_impact, MIN_FLIGHT_S / 2)
    caught = prev.turn is not None or _catcher(ev, g) is not None
    restitution = 0.0 if caught else RESTITUTION_E[g.index.body(mover).restitution]
    motion = _ballistic_motion(ev, mover, prev.p1, prev.full_landing, prev.v_end, flight, restitution,
                               prev.tumble_dps, _is_volitional(ev, g.index))
    motion.turn = prev.turn
    if g.ahead is not None and g.ahead.flight_end(ev) is not ev:
        _split_flight(motion)
    return motion


def _geo_fall(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    """
    drop / fall: free fall from rest (plus any edge speed) to the stated landing. A fall that ends
    'airborne' is finished by the mover's next event: a fall/drop continuing it (the flight is cut
    mid-air here, the landing is the continuation's) or a catch (the flight ends where the hand meets it).
    """
    volitional = _is_volitional(ev, g.index)
    dest = _destination_xz(ev, mover, p0, g) or (p0[0], p0[2])
    if mover in g.index.characters and ev.kind == "fall":
        # A person falling over: a body motion to the ground, not a free fall of the base.
        p1 = (dest[0], _rest_height(ev, mover, g), dest[1])
        return _linear_motion(ev, mover, p0, p1, FIXED_DURATION_S["fall_over"], False, pose="fall")
    if g.immovable(mover):
        return _static_motion(ev, mover, p0, 0.0, volitional, f"{mover} is fixed and cannot {ev.kind}", False)
    prev = g.prev
    if prev is not None and prev.model == "ballistic" and prev.full_landing is not None:
        return _continue_flight(ev, mover, prev, g)
    landing = g.ahead.flight_end(ev) if g.ahead is not None else ev
    catch = g.ahead.catch_of(ev) if g.ahead is not None else None
    # Horizontal speed it leaves its support with (a slide off an edge), plus a free fall's slop around it.
    carried = (prev.exit_velocity[0], prev.exit_velocity[2]) if prev is not None else (0.0, 0.0)
    drift_limit = MAX_FALL_DRIFT_MS + math.hypot(*carried)
    notes, feasible, turn = [], True, None
    if catch is not None:
        p1, flight, yaw = _interception(p0, catch, carried)
        turn = (catch.catcher, yaw)
        reach = _dist_xz(catch.base, p1)
        if reach > REACH_M + EPS:
            feasible = False
            notes.append(f"{catch.catcher} is {reach:.2f} m from where {mover} falls, beyond reach ({REACH_M} m); "
                         f"move {catch.catcher} closer or let {mover} land")
        restitution = 0.0
        vx, vz = (p1[0] - p0[0]) / flight, (p1[2] - p0[2]) / flight
    else:
        catcher = _catcher(landing, g)
        if catcher:
            p1 = g.held_after(mover, catcher) or g.hand(catcher, "carry", toward=p0)
        else:
            end = _destination_xz(landing, mover, p0, g) or (p0[0], p0[2])
            p1 = (end[0], _rest_height(landing, mover, g), end[1])
        height = p0[1] - p1[1]
        if height < -0.05:
            feasible = False
            notes.append(f"{mover} would have to fall upward from y={p0[1]:.2f} m to y={p1[1]:.2f} m")
        flight = max(fall_time(max(height, 0.0)), MIN_FLIGHT_S)
        vx, vz = (p1[0] - p0[0]) / flight, (p1[2] - p0[2]) / flight
        if math.hypot(vx - carried[0], vz - carried[1]) > MAX_FALL_DRIFT_MS + EPS:
            feasible = False
            notes.append(_fall_drift_note(landing, mover, p0, p1, height, flight, drift_limit * flight, g))
        restitution = 0.0 if catcher else RESTITUTION_E[g.index.body(mover).restitution]
    tumble = FALL_TUMBLE_DPS if mover in g.index.props else 0.0
    # From rest (vy = 0) unless the drop is shorter than the minimum flight (then it ends exactly at p1).
    vy = (p1[1] - p0[1] + 0.5 * G * flight * flight) / flight if flight > fall_time(max(p0[1] - p1[1], 0.0)) + EPS \
        else 0.0
    motion = _ballistic_motion(ev, mover, p0, p1, (vx, vy, vz), flight, restitution, tumble, volitional,
                               "; ".join(notes), feasible)
    motion.turn = turn
    if landing is not ev:
        _split_flight(motion)
    return motion


def _fall_drift_note(ev: PhysEvent, mover: str, p0: Vec3, p1: Vec3, height: float, flight: float,
                     reach: float, g: _Geo) -> str:
    """Repair facts for a free fall that would have to drift too far sideways: what would make it valid,
    stated with the plan's own ids (repairs cannot add marks, so only existing ones are offered)."""
    distance = _dist_xz(p0, p1)
    fixes = []
    shot = g.index.shots.get(g.index.event_shot.get(ev.id, ""))
    near = sorted(((math.hypot(m.x_m - p0[0], m.z_m - p0[2]), m.id) for m in g.index.marks.values()
                   if shot is not None and m.location_id == shot.location_id), key=lambda t: t[0])
    near = [(d, mid) for d, mid in near if d <= reach]
    if near:
        fixes.append(f"set {ev.id}.outcome_mark_id to a mark within reach: "
                     + ", ".join(f"{mid} ({d:.2f} m)" for d, mid in near[:4]))
    fixes.append(f"set {ev.id}.outcome_mark_id to \"none\" so {mover} lands straight below where it leaves its support")
    cause = g.index.events.get(ev.cause_event_id)
    if cause is not None and cause.kind in ("push", "slide", "roll", "collide", "impact"):
        need = max(distance / flight - MAX_FALL_DRIFT_MS, 0.0)
        fixes.append(f"make the cause {cause.id} ({cause.kind}) stronger so {mover} leaves its support at about "
                     f"{need:.1f} m/s")
    return (f"{mover} lands {distance:.2f} m from where it starts falling ({height:.2f} m drop), but a free fall "
            f"from there reaches at most {reach:.2f} m; fix by one of: " + "; or ".join(fixes))


def _geo_throw(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    actor = ev.actor_id if ev.actor_id in g.index.characters else None
    if actor is None:
        return _static_motion(ev, mover, p0, 0.0, False, "a throw needs a character actor_id", False)
    if g.immovable(mover):
        return _static_motion(ev, mover, p0, 0.0, True, f"{mover} is fixed and cannot be thrown", False)
    # The flight ends where the event (or, when it ends airborne, its continuation) says: a catcher's hand,
    # a mark or a support.
    end_ev = g.ahead.flight_end(ev) if g.ahead is not None else ev
    catch = g.ahead.catch_of(ev) if g.ahead is not None else None
    catcher = _catcher(end_ev, g)
    dest = _destination_xz(end_ev, mover, p0, g)
    turn = None
    if catch is not None:
        # A separate catch receives it: the catcher faces the thrower and takes it in the hand.
        facing = yaw_toward(catch.base, g.pos(actor)) if _dist_xz(catch.base, g.pos(actor)) > 1e-6 else catch.yaw
        landing, turn = catch.hand(facing), (catch.catcher, facing)
    elif catcher:
        landing = g.held_after(mover, catcher) or g.hand(catcher, "carry", toward=g.pos(actor))
    elif dest is not None:
        landing = (dest[0], _rest_height(end_ev, mover, g), dest[1])
    else:
        return _static_motion(ev, mover, p0, 0.0, True,
                              "a throw needs outcome_mark_id, outcome_support_id or a catcher (outcome_rest held)", False)
    release = g.hand(actor, "throw", toward=landing)
    sol = solve_throw(release, landing, ev.strength, g.mass(mover))
    if not sol.feasible:
        return _static_motion(ev, mover, release, 0.0, True, sol.note, False)
    restitution = 0.0 if catcher or catch is not None else RESTITUTION_E[g.index.body(mover).restitution]
    notes, feasible = [sol.note] if sol.note else [], True
    # The wind-up carries the object from where it is to the release point: it must be in hand or in reach.
    reach = _dist_xz(g.pos(actor), p0)
    if reach > REACH_M + EPS:
        feasible = False
        notes.append(f"{actor} is {reach:.2f} m from {mover}, beyond reach ({REACH_M} m); lift it first "
                     f"or walk there first")
    motion = _ballistic_motion(ev, mover, release, landing, sol.velocity, sol.flight_time_s, restitution,
                               THROW_TUMBLE_DPS[ev.strength], True, "; ".join(notes), feasible)
    motion.windup, motion.windup_from = ANTICIPATION_S, p0     # the hand carries it to the release point
    motion.turn = turn
    if end_ev is not ev:
        _split_flight(motion)
    return motion


def _onward(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> Optional[Tuple[Vec3, bool]]:
    """
    Where the fall that continues an edge-drop slide ends, as (point, lands): the landing of that flight
    (lands=True), or the hand that catches it (lands=False: the hand comes to the object).
    """
    nxt = g.ahead.next(ev) if g.ahead is not None else None
    if nxt is None or nxt.kind not in ("fall", "drop") or event_mover(nxt) != mover:
        return None
    catch = g.ahead.catch_of(nxt)
    if catch is not None:
        return catch.hand(), False
    end = g.ahead.flight_end(nxt)
    catcher = _catcher(end, g)
    if catcher:
        return g.hand(catcher, "carry", toward=p0), False
    dest = _destination_xz(end, mover, p0, g)
    if dest is None:
        return None
    return (dest[0], _rest_height(end, mover, g), dest[1]), True


def _geo_slide(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    """
    push / pull / slide / roll: friction-decelerated slide to a mark, to rest, or off a support edge. A
    slide off an edge with no stated direction heads for where the fall it starts ends (outcome first)
    and leaves the edge fast enough for that fall to get there.
    """
    volitional = _is_volitional(ev, g.index)
    if g.immovable(mover):
        return _static_motion(ev, mover, p0, 0.0, volitional, f"{mover} is fixed and cannot {ev.kind}", False)
    body = g.index.body(mover)
    decel = FRICTION_MU[body.friction] * (ROLL_FRICTION_FACTOR if ev.kind == "roll" else 1.0) * G
    mass = g.mass(mover)
    v_min, v_max = strength_speed_range(ev.strength, mass)
    actor = ev.actor_id if ev.actor_id in g.index.entities and ev.actor_id != mover else None
    notes, feasible = [], True
    onward = _onward(ev, mover, p0, g) if ev.outcome_rest == "airborne" else None

    # 1. Direction: toward the outcome mark, else away from (pull: toward) the actor, else toward where the
    #    continuing fall ends, else the incoming motion.
    mark = g.index.marks.get(ev.outcome_mark_id)
    direction = None
    if mark is not None:
        direction = _unit_xz(mark.x_m - p0[0], mark.z_m - p0[2])
    if direction is None and actor is not None:
        a = g.pos(actor)
        sign = -1.0 if ev.kind == "pull" else 1.0
        direction = _unit_xz(sign * (p0[0] - a[0]), sign * (p0[2] - a[2]))
    if direction is None and onward is not None:
        direction = _unit_xz(onward[0][0] - p0[0], onward[0][2] - p0[2])
    if direction is None and g.prev is not None:
        direction = _unit_xz(g.prev.p1[0] - g.prev.p0[0], g.prev.p1[2] - g.prev.p0[2])
    ux, uz = direction or (0.0, 1.0)

    # 2. Distance and end point.
    edge_drop = ev.outcome_rest == "airborne"
    if edge_drop:
        st = g.states.get(mover)
        sid = st.support_id if st is not None and st.support == "on_entity" else NONE
        if sid in g.index.entities and sid != mover:
            dist = _exit_distance(p0, ux, uz, g.pos(sid), g.index.entity_size(sid))
        else:
            dist, feasible = 0.0, False
            notes.append(f"{mover} is not standing on an entity, so a {ev.kind} cannot send it off an edge")
        p1 = (p0[0] + ux * dist, p0[1], p0[2] + uz * dist)
    elif mark is not None:
        dist = math.hypot(mark.x_m - p0[0], mark.z_m - p0[2])
        p1 = (mark.x_m, _rest_height(ev, mover, g), mark.z_m)
    else:
        nominal = math.sqrt(STRENGTH_IMPULSE_NS[ev.strength][0] * STRENGTH_IMPULSE_NS[ev.strength][1]) / mass
        dist = min(max(nominal, v_min), v_max) ** 2 / (2.0 * decel)
        p1 = (p0[0] + ux * dist, _rest_height(ev, mover, g), p0[2] + uz * dist)
    if not edge_drop and abs(p1[1] - p0[1]) > 0.05:
        feasible = False
        notes.append(f"a {ev.kind} cannot move {mover} from y={p0[1]:.2f} m to y={p1[1]:.2f} m; use a {ev.kind} "
                     f"with outcome_rest airborne followed by a fall")

    # 3. Speed from distance and friction; feasibility against the strength class.
    v_need = math.sqrt(2.0 * decel * dist)
    if v_need > v_max + EPS:
        feasible = False
        notes.append(f"sliding {mover} {dist:.2f} m needs {mass * v_need:.2f} N*s ({v_need:.1f} m/s) but a "
                     f"{ev.strength} {ev.kind} gives at most {mass * v_max:.2f} N*s")
    if edge_drop:
        # Leave the edge fast enough that the fall reaches its landing (the fall reports it if not even
        # the strongest slide of this class gets it there).
        v_exit = 0.0
        if onward is not None and onward[1]:
            t_fall = fall_time(p1[1] - onward[0][1])
            if t_fall > EPS:
                v_exit = _exit_speed_for(((onward[0][0] - p1[0]) / t_fall, (onward[0][2] - p1[2]) / t_fall), (ux, uz))
        speed0 = min(max(math.hypot(v_need, v_exit), v_min), v_max) if feasible else v_need
        disc = max(speed0 ** 2 - 2.0 * decel * dist, 0.0)
        duration = (speed0 - math.sqrt(disc)) / decel if decel > EPS and speed0 > EPS else 0.0
        exit_speed = math.sqrt(disc)
    else:
        speed0, exit_speed = v_need, 0.0
        duration = speed0 / decel if decel > EPS else 0.0
        if mark is not None and feasible and v_need < v_min - EPS:
            notes.append(f"moving {mover} {dist:.2f} m needs less than a {ev.strength} {ev.kind}")
    if ev.actor_id in g.index.characters and ev.kind in ("push", "pull") and actor is not None:
        reach = _dist_xz(g.pos(actor), p0)
        if reach > REACH_M + EPS:
            feasible = False
            notes.append(f"{actor} is {reach:.2f} m from {mover}, beyond reach ({REACH_M} m); walk there first")

    phases = _phase_table(duration, volitional, release=min(0.1, duration))
    return _Motion(ev, mover, "slide", p0, p1, duration, phases, duration, phases["rest"], feasible,
                   "; ".join(notes), volitional, speed0=speed0, decel=decel, exit_speed=exit_speed,
                   exit_velocity=(ux * exit_speed, 0.0, uz * exit_speed))


def _exit_speed_for(need: Tuple[float, float], direction: Tuple[float, float]) -> float:
    """
    Smallest speed along `direction` (unit xz) that leaves the horizontal velocity `need` within the free
    fall's slop (MAX_FALL_DRIFT_MS) of it; the closest one when none does.
    """
    along = need[0] * direction[0] + need[1] * direction[1]
    across2 = max(need[0] ** 2 + need[1] ** 2 - along ** 2, 0.0)
    if across2 > MAX_FALL_DRIFT_MS ** 2:
        return max(along, 0.0)
    speed = along - math.sqrt(MAX_FALL_DRIFT_MS ** 2 - across2)
    return speed + 1e-7 if speed > 0.0 else 0.0          # (margin against rounding in the fall's check)


def _geo_locomotion(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    if mover not in g.index.characters:
        return _static_motion(ev, mover, p0, 0.0, False, f"only characters can {ev.kind}", False)
    mark = g.index.marks.get(ev.outcome_mark_id)
    note = ""
    if ev.kind == "enter":
        p1 = (mark.x_m, 0.0, mark.z_m) if mark is not None else (p0[0], 0.0, p0[2])
        p0 = _nearest_edge_point(g.location, p1)
    elif ev.kind == "exit":
        p1 = (mark.x_m, 0.0, mark.z_m) if mark is not None else _nearest_edge_point(g.location, p0)
    elif mark is not None:
        p1 = (mark.x_m, 0.0, mark.z_m)
    elif ev.target_id in g.index.entities and ev.target_id != mover:
        target = g.pos(ev.target_id)
        dist = _dist_xz(p0, target)
        f = max(dist - STOP_SHORT_M, 0.0) / dist if dist > EPS else 0.0
        p1 = (p0[0] + (target[0] - p0[0]) * f, 0.0, p0[2] + (target[2] - p0[2]) * f)
    else:
        p1, note = p0, f"{ev.kind} has no outcome_mark_id or target; {mover} stays in place"
    speed = RUN_SPEED if ev.kind == "run_to" else WALK_SPEED
    duration = max(_dist_xz(p0, p1) / speed, MIN_LOCOMOTION_S)
    return _linear_motion(ev, mover, p0, p1, duration, True, note, face_travel=True,
                          pose="run" if ev.kind == "run_to" else "walk")


def _geo_handling(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    """lift / place / catch / hand_off: the object travels between a support and a hand."""
    duration = FIXED_DURATION_S[ev.kind]
    actor = ev.actor_id if ev.actor_id in g.index.characters else None
    if actor is None or actor == mover:
        return _static_motion(ev, mover, p0, duration, False, f"{ev.kind} needs a character actor_id", False)
    if g.immovable(mover):
        return _static_motion(ev, mover, p0, duration, True, f"{mover} is fixed and cannot be handled", False)
    notes, feasible = [], True
    mass = g.mass(mover)
    if mass > MAX_LIFT_KG[ev.strength] + EPS:
        feasible = False
        notes.append(f"{mover} ({mass:g} kg) is too heavy for a {ev.strength} {ev.kind} "
                     f"(at most {MAX_LIFT_KG[ev.strength]:g} kg)")
    st = g.states.get(mover)
    if st is not None and st.support == "held" and st.support_id == actor and ev.kind in ("place", "hand_off"):
        p0 = g.hand(actor, "carry")
    reach_from, reach_limit = p0, REACH_M
    if ev.kind in ("lift", "catch"):
        p1 = g.held_after(mover, actor) or g.hand(actor, "carry", toward=p0)
    elif ev.kind == "place":
        dest = _destination_xz(ev, mover, p0, g) or (p0[0], p0[2])
        p1 = (dest[0], _rest_height(ev, mover, g), dest[1])
        reach_from = p1
    else:
        receiver = ev.outcome_support_id if ev.outcome_support_id in g.index.characters else None
        if receiver is None or receiver == actor:
            return _static_motion(ev, mover, p0, duration, True,
                                  "hand_off needs outcome_support_id = the receiving character", False)
        p1 = g.held_after(mover, receiver) or g.hand(receiver, "carry", toward=g.pos(actor))
        reach_from, reach_limit = g.pos(receiver), HAND_OFF_REACH_M
    reach = _dist_xz(g.pos(actor), reach_from)
    if reach > reach_limit + EPS:
        feasible = False
        notes.append(f"{actor} is {reach:.2f} m away, beyond reach ({reach_limit} m); walk there first")
    motion = _linear_motion(ev, mover, p0, p1, duration, True, "; ".join(notes), feasible)
    if ev.kind == "catch" and g.prev is not None and g.prev.turn is not None and g.prev.turn[0] == actor:
        motion.turn = g.prev.turn       # still facing the flight it has just caught
    return motion


def _geo_instant(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    """break / spill / pour / tear / impact / collide / bounce / settle / sit_down / stand_up / gesture."""
    volitional = _is_volitional(ev, g.index)
    duration = FIXED_DURATION_S.get(ev.kind, 0.0)
    mark = g.index.marks.get(ev.outcome_mark_id)
    if ev.kind in ("sit_down", "stand_up"):
        # The body ends at the height its new support gives it: seated on a chair or standing on the
        # floor, the base is on the floor; only a tall support or a stand/lie on top raises it.
        x, z = (mark.x_m, mark.z_m) if mark is not None else (p0[0], p0[2])
        return _linear_motion(ev, mover, p0, (x, _settled_height(ev, mover, p0, g), z), duration, volitional)
    p1 = (mark.x_m, _rest_height(ev, mover, g) if ev.kind not in BODY_KINDS else p0[1], mark.z_m) \
        if mark is not None else p0
    if ev.kind in BODY_KINDS:
        return _linear_motion(ev, mover, p0, p1, duration, volitional)
    motion = _linear_motion(ev, mover, p0, p1, duration, volitional)
    motion.phases.update(impact=0.0, rebound=min(0.15, duration), aftermath=0.75 * duration, rest=duration)
    motion.end = duration
    motion.trigger = 0.0 if ev.kind in TRIGGER_AT_START_KINDS else duration
    return motion


def _geometry(ev: PhysEvent, mover: str, p0: Vec3, g: _Geo) -> _Motion:
    if ev.kind == "none":
        return _static_motion(ev, mover, p0, 0.0, False)
    if mover not in g.index.entities:
        return _static_motion(ev, mover, p0, 0.0, False, "event has no existing actor_id or target_id", False)
    if ev.kind in ("fall", "drop"):
        return _geo_fall(ev, mover, p0, g)
    if ev.kind == "throw":
        return _geo_throw(ev, mover, p0, g)
    if ev.kind in SLIDE_KINDS:
        return _geo_slide(ev, mover, p0, g)
    if ev.kind in LOCOMOTION_KINDS:
        return _geo_locomotion(ev, mover, p0, g)
    if ev.kind in HANDLING_KINDS:
        return _geo_handling(ev, mover, p0, g)
    return _geo_instant(ev, mover, p0, g)


# ---------------------------------------------------------------- context
@dataclass
class _Context:
    index: SceneIndex
    states: Dict[str, Dict[str, EntityState]]
    static: Dict[str, Dict[str, Tuple[Vec3, float]]]     # kf_id -> entity -> (position, yaw) without motion
    motions: Dict[str, _Motion]
    causes: Dict[str, List[Tuple[str, str]]]             # event -> [(cause event, "causes" | "enables")]
    before_kf: Dict[str, str]                            # event -> keyframe before its start keyframe
    starting_at: Dict[str, List[str]]                    # kf_id -> events whose span starts there
    prev_motion: Dict[str, str]                          # event -> previous event of the same mover
    snapshot_kfs: Dict[str, List[str]]                   # entity -> keyframes with a planner snapshot of it
    by_mover: Dict[str, List[_Motion]] = field(default_factory=dict)


def _static_poses(index: SceneIndex, states: Dict[str, Dict[str, EntityState]]) -> Dict[str, Dict[str, Tuple[Vec3, float]]]:
    poses: Dict[str, Dict[str, Tuple[Vec3, float]]] = {}
    prev_pos: Dict[str, Vec3] = {}
    prev_yaw: Dict[str, float] = {}
    for kf_id in [INIT] + [kf.id for _, kf in index.keyframes]:
        resolver = _Resolver(index, states[kf_id], {}, prev_pos, prev_yaw, {})
        poses[kf_id] = {eid: (resolver.position(eid), resolver.yaw(eid)) for eid in index.entities}
        prev_pos = {eid: p for eid, (p, _) in poses[kf_id].items()}
        prev_yaw = {eid: y for eid, (_, y) in poses[kf_id].items()}
    return poses


def _chained(history: List[Tuple[_Motion, int]], start_pos: int) -> Optional[_Motion]:
    """The mover's previous motion when no settled keyframe separates it from an event starting at start_pos."""
    if not history:
        return None
    last, last_end = history[-1]
    return last if last.event.outcome_rest in ("airborne", "rolling") or last_end >= start_pos - 1 else None


def _placed_by(history: List[Tuple[_Motion, int]], start_pos: int, snapshot_pos: List[int],
               state: Optional[EntityState]) -> Optional[_Motion]:
    """
    The motion whose end position an entity still occupies at keyframe position start_pos - 1: a chained
    motion, or its last moving event when no planner snapshot re-placed it afterwards (and it is not held).
    """
    chained = _chained(history, start_pos)
    if chained is not None:
        return chained
    if state is None or state.support == "held":
        return None
    moving = [(m, end) for m, end in history if m.model != "static"]
    if not moving:
        return None
    motion, end = moving[-1]
    last_snapshot = max((p for p in snapshot_pos if p <= start_pos - 1), default=-1)
    return motion if end >= last_snapshot else None


def _outcome_keyframe(index: SceneIndex, event: PhysEvent, span: Tuple[int, int]) -> Optional[str]:
    """First keyframe whose reduced state includes the event's outcome (None past the end of the timeline)."""
    for pos in (span[1], span[1] + 1):
        if pos < len(index.keyframes) and event_completed_at(event, span, pos, index.keyframes[pos][1]):
            return index.keyframes[pos][1].id
    return None


def _catch_keyframe(index: SceneIndex, states: Dict[str, Dict[str, EntityState]], event: PhysEvent) -> Optional[str]:
    """
    A keyframe that shows a thrown or dropped object arriving in its catcher's hand (the event focused at
    impact with the object held by the catcher there). The landing is that hand, in that keyframe's pose.
    """
    if event.kind not in BALLISTIC_KINDS or event.outcome_rest != "held" or event.outcome_support_id not in index.characters:
        return None
    mover = event_mover(event)
    for kf_id in index.focus_keyframes.get(event.id, []):
        st = states[kf_id].get(mover)
        if index.keyframe_by_id[kf_id].depicted_phase in ("impact", "rebound") and st is not None \
                and st.support == "held" and st.support_id == event.outcome_support_id:
            return kf_id
    return None


def _placement_key(st: EntityState) -> Optional[tuple]:
    """What a state says about where an entity is: its mark and offset, or its support and offset when it
    has no mark; None when it says nothing (no mark and a free support: the solver keeps it where it is)."""
    offsets = (round(st.offset_x_m, 4), round(st.offset_z_m, 4))
    if st.mark_id != NONE:
        return (st.mark_id,) + offsets
    if st.support in CARRIED_SUPPORTS:
        return (st.support, st.support_id) + offsets
    return None


def _places(snapshot: EntityState, before: EntityState) -> bool:
    """
    True when a planner snapshot puts an entity somewhere new. A snapshot that restates where the entity
    was (typically the mark it fell from, after a fall with outcome_mark_id "none") does not move it
    back: physics keeps it where its last motion left it.
    """
    key = _placement_key(snapshot)
    return key is not None and key != _placement_key(before)


def _prepare(plan: ScenePlan) -> _Context:
    """Everything that does not depend on timing: index, reduced states, static poses, event geometry."""
    index = build_index(plan)
    states = reduce_states(plan, index)
    static = _static_poses(index, states)
    snapshot_kfs: Dict[str, List[str]] = {}
    for tl in plan.timeline.shots:
        for st in tl.states:
            if st.kf_id in index.kf_position and st.entity_id in index.entities \
                    and _places(st, states[index.prev_keyframe_id(st.kf_id)][st.entity_id]):
                snapshot_kfs.setdefault(st.entity_id, []).append(st.kf_id)
    spans = {eid: index.event_span(eid) for eid in index.events}
    plan_order = {eid: i for i, eid in enumerate(index.events)}
    order = sorted((eid for eid in index.events if spans[eid] is not None), key=lambda e: (spans[e][0], plan_order[e]))
    next_of: Dict[str, str] = {}
    last_of: Dict[str, str] = {}
    for eid in order:
        mover = event_mover(index.events[eid])
        if mover in last_of:
            next_of[last_of[mover]] = eid
        last_of[mover] = eid
    ahead = _Ahead(index, states, static, next_of)
    motions: Dict[str, _Motion] = {}
    before_kf: Dict[str, str] = {}
    starting_at: Dict[str, List[str]] = {}
    prev_motion: Dict[str, str] = {}
    history: Dict[str, List[Tuple[_Motion, int]]] = {}
    for eid in order:
        ev = index.events[eid]
        start_pos, end_pos = spans[eid]
        start_kf = index.keyframes[start_pos][1].id
        before = index.prev_keyframe_id(start_kf)
        before_kf[eid] = before
        starting_at.setdefault(start_kf, []).append(eid)
        # 1. Where everything is when the event starts: earlier motions win over stale reduced positions.
        poses = dict(static[before])
        for other, hist in history.items():
            placed = _placed_by(hist, start_pos, [index.kf_position[k] for k in snapshot_kfs.get(other, [])],
                                states[before].get(other))
            if placed is not None and other in poses:
                poses[other] = (placed.p1, poses[other][1])
        # 2. Geometry of this event.
        mover = event_mover(ev)
        prev = _chained(history.get(mover, []), start_pos)
        if history.get(mover):
            prev_motion[eid] = history[mover][-1][0].event.id
        p0 = poses[mover][0] if mover in poses else (0.0, 0.0, 0.0)
        shot = index.shots.get(index.event_shot.get(eid, ""))
        location = index.locations.get(shot.location_id) if shot is not None else None
        after = _catch_keyframe(index, states, ev) or _outcome_keyframe(index, ev, spans[eid])
        geo = _Geo(index, states[before], poses, location, prev,
                   states[after] if after else None, static[after] if after else None, ahead)
        motion = _geometry(ev, mover, p0, geo)
        motions[eid] = motion
        if mover in index.entities:
            history.setdefault(mover, []).append((motion, end_pos))

    causes: Dict[str, List[Tuple[str, str]]] = {eid: [] for eid in motions}
    for eid in motions:
        cause = index.events[eid].cause_event_id
        if cause in motions and cause != eid:
            causes[eid].append((cause, "causes"))
    for link in index.cause_links.values():
        if link.kind in ("causes", "enables") and link.effect_event_id in motions and link.cause_event_id in motions \
                and link.cause_event_id != link.effect_event_id \
                and all(c != link.cause_event_id for c, _ in causes[link.effect_event_id]):
            causes[link.effect_event_id].append((link.cause_event_id, link.kind))
    return _Context(index, states, static, motions, causes, before_kf, starting_at, prev_motion, snapshot_kfs)


# ---------------------------------------------------------------- timing
@dataclass
class _Schedule:
    kf_times: Dict[str, Tuple[float, float]]    # kf_id -> (edited t_s, physical t)
    starts: Dict[str, float]                    # event -> physical start
    shots: List[SolvedShot]
    total: float
    warnings: List[str]


def _shot_groups(index: SceneIndex) -> List[Tuple[str, list]]:
    groups = [(sid, index.shot_keyframes(sid)) for sid in index.shots]
    known = set(index.shots)
    for sid in dict.fromkeys(sid for sid, _ in index.keyframes):
        if sid not in known:
            groups.append((sid, index.shot_keyframes(sid)))
    return groups


def _schedule(ctx: _Context, hold_scale: float) -> _Schedule:
    """
    Place keyframes and events on both clocks. Non-physics holds (keyframe slots, shot nominal
    durations) are multiplied by `hold_scale`; physics-determined gaps are never compressed.
    """
    index, motions = ctx.index, ctx.motions
    starts: Dict[str, float] = {}
    kf_times: Dict[str, Tuple[float, float]] = {}
    shots: List[SolvedShot] = []
    warnings: List[str] = []

    def start_for(eid: str, land_t: float, offset: float) -> float:
        """Start an event so `offset` lands at `land_t`, unless its causes dictate the start."""
        m = motions[eid]
        react = REACTION_S if m.volitional and m.event.kind != "catch" else 0.0
        triggers = [(starts[c] + motions[c].trigger, kind) for c, kind in ctx.causes[eid] if c in starts]
        if triggers and (not m.volitional or m.event.kind == "catch") and any(k == "causes" for _, k in triggers):
            return max(t for t, _ in triggers)    # physics cannot wait: it starts at its trigger
        start = land_t - offset
        if triggers:
            start = max(start, max(t for t, _ in triggers) + react)
        before = ctx.before_kf.get(eid)
        if before in kf_times:
            start = max(start, kf_times[before][1])
        prev = ctx.prev_motion.get(eid)
        if prev in starts:
            # One thing at a time: finish the previous motion (and wind up) before starting this one.
            start = max(start, starts[prev] + motions[prev].duration - min(m.phases["anticipation"], 0.0))
        return start

    edit_cursor = phys_cursor = 0.0
    for shot_id, kfs in _shot_groups(index):
        shot = index.shots.get(shot_id)
        scale = TIME_SCALE.get(shot.time_scale, 1.0) if shot is not None else 1.0
        nominal = DURATION_CLASS_S.get(shot.duration_class, 2.0) * hold_scale if shot is not None else 0.0
        spacing = nominal / max(len(kfs), 1)
        e0, prev = edit_cursor, None
        for i, kf in enumerate(kfs):
            eid = kf.focus_event_id if kf.focus_event_id in motions else None
            offset = motions[eid].phases.get(kf.depicted_phase, 0.0) if eid else 0.0
            if prev is None:
                land = phys_cursor
            else:
                slot = max(e0 + i * spacing, prev[0] + MIN_KF_GAP_S)
                land = prev[1] + (slot - prev[0]) / scale
            if eid is None:
                t = land
            else:
                if eid not in starts:
                    starts[eid] = start_for(eid, land, offset)
                t = starts[eid] + offset
            if prev is None:
                e = e0
            else:
                if t < prev[1] - EPS:
                    warnings.append(f"timing: {kf.id} shows {eid} '{kf.depicted_phase}' at t={t:.2f}s, before the "
                                    f"previous keyframe of {shot_id} (t={prev[1]:.2f}s); clamped")
                    t = prev[1]
                e = max(prev[0] + (t - prev[1]) * scale, prev[0] + MIN_KF_GAP_S)
            kf_times[kf.id] = (e, t)
            for other in ctx.starting_at.get(kf.id, []):
                if other not in starts:
                    starts[other] = start_for(other, t, 0.0)
            prev = (e, t)
        if prev is None:
            end_e, end_t = e0 + nominal, phys_cursor + nominal / scale
        else:
            end_e = max(e0 + nominal, prev[0] + END_HOLD_S)
            end_t = prev[1] + (end_e - prev[0]) / scale
        shots.append(SolvedShot(shot_id=shot_id, start_t_s=e0, end_t_s=end_e, keyframe_ids=[kf.id for kf in kfs]))
        edit_cursor, phys_cursor = end_e, end_t

    for eid, cause_list in ctx.causes.items():
        for cause, _ in cause_list:
            if eid in starts and cause in starts and starts[eid] < starts[cause] - 1e-6:
                warnings.append(f"timing: {eid} starts at t={starts[eid]:.2f}s, before its cause {cause} "
                                f"(t={starts[cause]:.2f}s)")
    return _Schedule(kf_times, starts, shots, edit_cursor, warnings)


def _fit_duration(ctx: _Context, max_duration_s: float) -> _Schedule:
    """Full holds when they fit; otherwise the largest uniform hold scale that fits (binary search)."""
    sched = _schedule(ctx, 1.0)
    if sched.total <= max_duration_s + 1e-6:
        return sched
    tightest = _schedule(ctx, 0.0)
    if tightest.total > max_duration_s + 1e-6:
        tightest.warnings.append(
            f"duration: the cut runs {tightest.total:.2f}s even with every non-physics hold removed "
            f"(limit {max_duration_s:g}s); use fewer shots/keyframes or shorter actions")
        return tightest
    lo, hi, best = 0.0, 1.0, tightest
    for _ in range(40):
        mid = (lo + hi) / 2
        candidate = _schedule(ctx, mid)
        if candidate.total <= max_duration_s + 1e-6:
            lo, best = mid, candidate
        else:
            hi = mid
    return best


def _apply_schedule(ctx: _Context, sched: _Schedule) -> None:
    """Set event starts and the window each event drives its mover (cut short by the mover's next event)."""
    by_mover: Dict[str, List[_Motion]] = {}
    for eid, motion in ctx.motions.items():
        motion.start = sched.starts.get(eid)
        motion.window_end = None
        if motion.start is not None:
            by_mover.setdefault(motion.mover, []).append(motion)
    for motions in by_mover.values():
        motions.sort(key=lambda m: m.start)
        for m in motions:
            later = [o.start - o.windup for o in motions if o.start > m.start + EPS]
            m.window_end = max(min([m.start + m.end] + later), m.start)
    ctx.by_mover = by_mover


# ---------------------------------------------------------------- frames
def _leading_motions(ctx: _Context, t_phys: float) -> Dict[str, _Motion]:
    """The motion driving each entity at t: the most recently started active one (moving ones win ties)."""
    lead: Dict[str, _Motion] = {}
    for mover, motions in ctx.by_mover.items():
        if mover not in ctx.index.entities:
            continue
        for m in motions:
            if m.active(t_phys):
                cur = lead.get(mover)
                if cur is None or (m.start, m.model != "static") >= (cur.start, cur.model != "static"):
                    lead[mover] = m
    return lead


def _entity_frame(index: SceneIndex, eid: str, st: EntityState, position: Vec3, yaw: float,
                  motion: Optional[_Motion], t_phys: float) -> EntityFrame:
    primary, secondary = index.entity_colors(eid)
    frame = EntityFrame(
        entity_id=eid, kind=index.entity_kind(eid), present=st.present, position=position, yaw_deg=yaw,
        rotation_deg=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), pose=st.pose, expression=st.expression,
        holding_id=st.holding_id, gaze_id=st.gaze_id, support=st.support, support_id=st.support_id,
        flags=list(st.flags), weight_cue=st.weight_cue, secondary=list(st.secondary),
        size=index.entity_size(eid), color_hex=primary, secondary_color_hex=secondary, shape=index.entity_shape(eid),
    )
    if motion is not None:
        _drive(frame, motion, t_phys - motion.start, index)
    return frame


def _drive(frame: EntityFrame, motion: _Motion, rel: float, index: SceneIndex) -> None:
    """Overlay an active motion on an entity frame (position is set by the caller)."""
    if motion.event.kind == "enter":
        frame.present = True
    if motion.model == "static":
        return
    frame.trajectory = motion.path()
    if motion.moving(rel):
        frame.in_motion = True
        frame.velocity = motion.velocity(rel)
        if motion.pose and frame.kind == "character" and (motion.pose == "fall" or frame.pose in ("stand", "none", "walk", "run")):
            frame.pose = motion.pose
    if motion.in_flight(rel):
        frame.support, frame.support_id = "airborne", NONE
    if frame.entity_id in index.props:
        frame.rotation_deg = motion.rotation(rel)


def _carry_order(entities: Dict[str, EntityFrame]) -> List[str]:
    """Entity ids with every carrier before what it carries (holder, then tray, then the cups on it)."""
    return sorted(entities, key=lambda eid: len(support_chain(entities, eid)))


def _carry_holder_velocity(entities: Dict[str, EntityFrame]) -> None:
    """Held entities move with their holder; entities resting on a moving support move with it."""
    for eid in _carry_order(entities):
        ef = entities[eid]
        carrier = entities.get(ef.support_id)
        if carrier is None or carrier is ef or ef.in_motion:
            continue
        if ef.support == "held" or (ef.support in ("on_entity", "attached") and any(carrier.velocity)):
            ef.velocity = carrier.velocity


def _placements(ctx: _Context, sched: _Schedule, kf_id: str, t_phys: float) -> Dict[str, Vec3]:
    """
    Entities resting where their last moving event left them: physics owns the position until a planner
    snapshot at or after the motion's end re-places the entity (held entities follow their holder).
    """
    out: Dict[str, Vec3] = {}
    pos = ctx.index.kf_position[kf_id]
    for eid, motions in ctx.by_mover.items():
        if eid not in ctx.index.entities or ctx.states[kf_id][eid].support == "held":
            continue
        ended = [m for m in motions if m.model != "static" and m.window_end <= t_phys + EPS]
        if not ended:
            continue
        last = ended[-1]
        replaced = any(ctx.index.kf_position[k] <= pos and sched.kf_times[k][1] >= last.window_end - EPS
                       for k in ctx.snapshot_kfs.get(eid, []))
        if not replaced:
            out[eid] = last.position(last.window_end - last.start)
    return out


def _build_frames(ctx: _Context, sched: _Schedule) -> List[FrameState]:
    index = ctx.index
    frames: List[FrameState] = []
    prev_pos = {eid: p for eid, (p, _) in ctx.static[INIT].items()}
    prev_yaw = {eid: y for eid, (_, y) in ctx.static[INIT].items()}
    for shot_id, kf in index.keyframes:
        t_edit, t_phys = sched.kf_times[kf.id]
        states = ctx.states[kf.id]
        lead = _leading_motions(ctx, t_phys)
        overrides = _placements(ctx, sched, kf.id, t_phys)
        overrides.update({mv: m.position(t_phys - m.start) for mv, m in lead.items() if m.model != "static"})
        forced_yaw = {}
        for mv, m in lead.items():
            if m.face_travel and m.moving(t_phys - m.start) and m.travel_yaw() is not None:
                forced_yaw[mv] = m.travel_yaw()
        travelling = dict(forced_yaw)
        for m in sorted((m for ms in ctx.by_mover.values() for m in ms if m.turn and m.active(t_phys)),
                        key=lambda m: m.start):
            who, yaw = m.turn            # a catcher faces the flight it is catching
            if who in index.entities and who not in travelling:
                forced_yaw[who] = yaw
        turned = [eid for eid in forced_yaw if eid not in travelling]
        resolver = _Resolver(index, states, overrides, prev_pos, prev_yaw, forced_yaw)
        entities = {eid: _entity_frame(index, eid, states[eid], resolver.position(eid), resolver.yaw(eid),
                                       lead.get(eid), t_phys)
                    for eid in index.entities}
        _carry_holder_velocity(entities)
        shot = index.shots.get(shot_id)
        active = sorted((m for ms in ctx.by_mover.values() for m in ms if m.active(t_phys)),
                        key=lambda m: (m.start, m.event.id))
        frames.append(FrameState(
            kf_id=kf.id, shot_id=shot_id, order_in_shot=kf.order, t_s=t_edit, t_physical_s=t_phys,
            location_id=shot.location_id if shot is not None else NONE, entities=entities,
            focus_event_id=kf.focus_event_id, depicted_phase=kf.depicted_phase,
            active_event_ids=[m.event.id for m in active],
        ))
        # A catcher turned toward a catch faces its own way again afterwards (as the static poses the event
        # geometry was built from assume), unless a snapshot states where it faces.
        unturned = _Resolver(index, states, overrides, prev_pos, prev_yaw, travelling) if turned else None
        prev_pos = {eid: ef.position for eid, ef in entities.items()}
        prev_yaw = {eid: ef.yaw_deg for eid, ef in entities.items()}
        for eid in turned:
            prev_yaw[eid] = unturned.yaw(eid)
    return frames


def solve(plan: ScenePlan, max_duration_s: float = config.MAX_VIDEO_SECONDS) -> SolvedPlan:
    """Solve event geometry, keyframe timing, and per-keyframe entity frames for a plan."""
    # 1. Timing-independent model: reduced states, static poses, event geometry and feasibility.
    ctx = _prepare(plan)
    # 2. Timing, compressing non-physics holds when the cut is too long.
    sched = _fit_duration(ctx, max_duration_s)
    _apply_schedule(ctx, sched)
    # 3. Entity frames at every keyframe.
    frames = _build_frames(ctx, sched)
    events = {
        eid: SolvedEvent(event_id=eid, kind=m.event.kind, shot_id=ctx.index.event_shot.get(eid, NONE),
                         start_t_physical=m.start, end_t_physical=m.window_end, phase_times=dict(m.phases),
                         feasible=m.feasible, note=m.note)
        for eid, m in ctx.motions.items() if m.start is not None
    }
    return SolvedPlan(frames=frames, events=events, shots=sched.shots, total_duration_s=sched.total,
                      warnings=list(sched.warnings))


# ---------------------------------------------------------------- sampling at arbitrary edited times
_SAMPLE_CACHE: List[Tuple[ScenePlan, SolvedPlan, _Context]] = []


def _sample_context(plan: ScenePlan, solved: SolvedPlan) -> _Context:
    for cached_plan, cached_solved, ctx in _SAMPLE_CACHE:
        if cached_plan is plan and cached_solved is solved:
            return ctx
    ctx = _prepare(plan)
    for eid, motion in ctx.motions.items():
        if eid in solved.events:
            motion.start = solved.events[eid].start_t_physical
    by_mover: Dict[str, List[_Motion]] = {}
    for eid, motion in ctx.motions.items():
        if eid in solved.events:
            motion.window_end = solved.events[eid].end_t_physical
            by_mover.setdefault(motion.mover, []).append(motion)
        else:
            motion.start = None
    for motions in by_mover.values():
        motions.sort(key=lambda m: m.start)
    ctx.by_mover = by_mover
    _SAMPLE_CACHE.insert(0, (plan, solved, ctx))
    del _SAMPLE_CACHE[2:]
    return ctx


def _apply_outcome_to_frame(frame: EntityFrame, event: PhysEvent, index: SceneIndex) -> None:
    """Outcome of an event that finished between two keyframes (flags, support, pose, presence)."""
    if event.kind == "exit":
        frame.present = False
    elif event.kind == "enter":
        frame.present = True
    if event.outcome_rest == "held" and event.outcome_support_id in index.characters:
        frame.support, frame.support_id = "held", event.outcome_support_id
    else:
        support = outcome_support(event, frame.entity_id, frame.support)
        if support is not None:
            frame.support, frame.support_id = support
    if frame.kind == "character":
        frame.pose = outcome_pose(event, frame.pose)
    container = is_container(frame.entity_id, frame.flags, index)
    if event.outcome_flag != NONE:
        frame.flags = flags_with(frame.flags, event.outcome_flag, container)
    if event.kind == "spill" and container:
        frame.flags = flags_with(frame.flags, "fill_empty", container)


def sample_frame(plan: ScenePlan, solved: SolvedPlan, t_s: float) -> FrameState:
    """
    State at an arbitrary edited time: active events follow their solved trajectories, entities that
    finished an event rest where it left them, everything else interpolates between the surrounding
    keyframes of the shot. Discrete fields come from the previous keyframe plus finished outcomes.
    """
    if not solved.frames:
        raise ValueError("solved plan has no keyframes to sample")
    ctx = _sample_context(plan, solved)
    index = ctx.index
    t = min(max(t_s, 0.0), solved.total_duration_s)

    # 1. Surrounding keyframes A (at or before t) and B (after t) within the shot playing at t.
    shot_id = solved.frames[0].shot_id
    for shot in solved.shots:
        if shot.start_t_s <= t + EPS and shot.keyframe_ids:
            shot_id = shot.shot_id
    in_shot = sorted((f for f in solved.frames if f.shot_id == shot_id), key=lambda f: f.t_s)
    before = [f for f in in_shot if f.t_s <= t + EPS]
    a = before[-1] if before else in_shot[0]
    after = [f for f in in_shot if f.t_s > a.t_s + EPS]
    b = after[0] if after else None
    if abs(t - a.t_s) <= EPS:
        out = copy.deepcopy(a)
        out.t_s, out.camera = t, None
        return out

    # 2. Physical time: linear between keyframes, the shot's time scale after the last one.
    shot_def = index.shots.get(shot_id)
    scale = TIME_SCALE.get(shot_def.time_scale, 1.0) if shot_def is not None else 1.0
    if b is not None:
        frac = (t - a.t_s) / (b.t_s - a.t_s)
        t_phys = a.t_physical_s + frac * (b.t_physical_s - a.t_physical_s)
    else:
        frac, t_phys = 0.0, a.t_physical_s + (t - a.t_s) / scale

    # 3. Entities.
    lead = _leading_motions(ctx, t_phys)
    entities: Dict[str, EntityFrame] = {}
    ended_here, starts_soon = set(), set()      # own motion ended in (a, t] / starts before b
    for eid, ef_a in a.entities.items():
        ef = copy.deepcopy(ef_a)
        ef.velocity, ef.rotation_deg, ef.in_motion, ef.trajectory = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, []
        ef_b = b.entities.get(eid) if b is not None else None
        own = ctx.by_mover.get(eid, [])
        for m in own:
            if a.t_physical_s + EPS < m.start + m.trigger <= t_phys + EPS:
                _apply_outcome_to_frame(ef, m.event, index)
        yaw = _lerp_angle(ef_a.yaw_deg, ef_b.yaw_deg, frac) if ef_b is not None else ef_a.yaw_deg
        m = lead.get(eid)
        if m is not None and m.model != "static":
            rel = t_phys - m.start
            ef.position = m.position(rel)
            _drive(ef, m, rel, index)
            if m.face_travel and m.moving(rel) and m.travel_yaw() is not None:
                yaw = m.travel_yaw()
        else:
            moving = [o for o in own if o.model != "static"]
            ended = [o for o in moving if a.t_physical_s + EPS < o.window_end <= t_phys + EPS]
            upcoming = [o for o in moving if b is not None and t_phys < o.start - o.windup < b.t_physical_s - EPS]
            if ended:
                last = max(ended, key=lambda o: o.window_end)
                ef.position = last.position(last.window_end - last.start)
                ended_here.add(eid)
            elif upcoming:
                nxt = min(upcoming, key=lambda o: o.start)
                ef.position = nxt.position(-nxt.windup)
                starts_soon.add(eid)
            elif ef_b is not None:
                ef.position = _lerp(ef_a.position, ef_b.position, frac)
                dt = b.t_physical_s - a.t_physical_s
                if dt > EPS:
                    ef.velocity = _mul(_sub(ef_b.position, ef_a.position), 1.0 / dt)
        ef.yaw_deg = yaw
        entities[eid] = ef

    # 4. Entities carried across the whole interval ride along with their carrier, carriers first: held
    #    ones with the hand (offset interpolated between the two keyframes), ones resting on a moving
    #    support (a tray in a walking waiter's hand) with it. Ones caught or set down mid-interval stay
    #    where their motion delivered them.
    for eid in _carry_order(entities):
        ef = entities[eid]
        hid = ef.support_id
        if ef.support not in CARRIED_SUPPORTS or hid not in entities or hid == eid \
                or (eid in lead and lead[eid].model != "static"):
            continue
        ef_a = a.entities[eid]
        if ef_a.support != ef.support or ef_a.support_id != hid:
            continue
        held = ef.support == "held"
        if not held:
            carrier_moves = _dist(entities[hid].position, a.entities[hid].position) > 1e-9 or (
                b is not None and _dist(b.entities[hid].position, a.entities[hid].position) > 1e-9)
            if not carrier_moves or eid in ended_here:
                continue
        offset_a = _sub(ef_a.position, a.entities[hid].position)
        offset = offset_a
        ef_b = b.entities[eid] if b is not None else None
        if ef_b is not None and ef_b.support == ef.support and ef_b.support_id == hid \
                and (held or eid not in starts_soon):
            offset = _lerp(offset_a, _sub(ef_b.position, b.entities[hid].position), frac)
        ef.position = _add(entities[hid].position, offset)
    _carry_holder_velocity(entities)

    # 5. What an entering or exiting entity carries comes and goes with it.
    for eid, ef in list(entities.items()):
        if ef.present != a.entities[eid].present:
            for cid in carried_by(entities, eid):
                entities[cid].present = ef.present

    active = sorted((m for ms in ctx.by_mover.values() for m in ms if m.active(t_phys)),
                    key=lambda m: (m.start, m.event.id))
    return FrameState(
        kf_id=a.kf_id, shot_id=a.shot_id, order_in_shot=a.order_in_shot, t_s=t, t_physical_s=t_phys,
        location_id=a.location_id, entities=entities, focus_event_id=a.focus_event_id,
        depicted_phase=a.depicted_phase, active_event_ids=[m.event.id for m in active],
    )
