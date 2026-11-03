"""
Mannequin geometry: a character's pose as world-space capsules.

segments(frame) builds a 15-capsule body (torso, shoulders, hips, head, nose, arms, legs, feet) plus a hair cap and eyes for every
schema pose, adds the weight-cue lean, rotates by yaw and moves it to the character's base position.
The right hand of a character holding something lands exactly on physics.hand_position(), so held
props (placed there by the solver) sit in the hand. Seated, crouching and kneeling hands rest on the
thighs; relaxed hands never reach past the arm's length or below the floor.

A character whose frame shape is "quadruped" (Character.body == "quadruped"; height = shoulder height)
gets a four-legged proxy instead: horizontal spine, four two-bone legs with paws, neck, head with
snout, ears and eyes, and a tail, posed for stand / walk / run / sit / lie / crouch / brace / fall.

Local body frame: x = the character's right, y = up, z = forward (yaw 0 faces +z). Lengths scale with
height; widths scale with height relative to a 1.75 m adult.
"""
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.previs import physics
from src.previs.schema import NONE
from src.previs.types import EntityFrame, Vec3

Segment = Tuple[Vec3, Vec3, float, str]

REFERENCE_HEIGHT_M = 1.75
UPPER_ARM = 0.19            # x height
FOREARM = 0.25              # elbow to palm centre, x height
SHOULDER_Y = 0.81
HIP_Y = 0.50
PELVIS_Y = 0.53
CHEST_Y = 0.76
HEAD_Y = 0.935
HEAD_R = 0.048              # x height
SHOULDER_HALF = 0.18        # metres at reference height
HIP_HALF = 0.09
ANKLE_HALF = 0.10

# Capsule radii in metres at the reference height.
RADII = {
    "torso": 0.13, "shoulders": 0.06, "hips": 0.11, "upper_arm": 0.045, "lower_arm": 0.038,
    "upper_leg": 0.07, "lower_leg": 0.055, "foot": 0.035,
}

# Upper-body lean (forward pitch, sideways roll) in degrees for each weight cue.
WEIGHT_CUE_LEAN = {
    "none": (0.0, 0.0), "leaning": (15.0, 0.0), "bracing": (-8.0, 0.0), "straining": (20.0, 0.0),
    "off_balance": (10.0, -12.0), "recoil": (-15.0, 0.0),
}
POSE_LEAN = {"run": 12.0, "crouch": 30.0, "reach": 20.0, "lift": 15.0, "push": 20.0, "pull": -12.0,
             "throw": -5.0, "brace": -8.0}
# Whole-body pitch about the ankles: lying and falling bodies are horizontal / tipping.
BODY_PITCH = {"lie": 90.0, "fall": 60.0}
# Poses whose right hand goes where physics.hand_position() says (reaching, handling, throwing).
PHYSICS_HAND_POSES = frozenset({"reach", "push", "pull", "lift", "carry", "throw"})
BOTH_HANDS_POSES = frozenset({"push", "pull", "lift", "carry"})
# Poses whose relaxed hands rest on the thighs (placed from the leg joints, not the standing table).
THIGH_HANDS_POSES = frozenset({"sit", "crouch", "kneel"})
SEAT_MIN_M = 0.2            # a seat this far above the feet is sat on with the shins down to the floor
SEAT_MAX = 0.30             # x height: the highest seat drawn with the feet on the floor (about knee height)


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    # Positive angle tips +y toward +z (a forward lean in the body frame).
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_axes(yaw_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) unit vectors of a body with this yaw."""
    yaw = math.radians(yaw_deg)
    forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    right = np.array([-forward[2], 0.0, forward[0]])          # same convention as physics.hand_position
    return right, np.array([0.0, 1.0, 0.0]), forward


def _seat_levels(base_y: float, seat_y: Optional[float], h: float) -> Tuple[float, float]:
    """(seat, floor) heights in the body frame of a sitting character.

    With `seat_y` (world height of the seat surface) well above the base, the base is the feet on the
    floor and the seat is drawn at that height (capped at about knee height). Otherwise a base well above
    the floor is the seat itself and the feet reach down to the floor at world y = 0; else the character
    sits on the floor."""
    if seat_y is not None and seat_y - base_y > SEAT_MIN_M:
        return min(seat_y - base_y, SEAT_MAX * h), 0.0
    if base_y > SEAT_MIN_M:
        return 0.0, -base_y
    return 0.0, 0.0


def _legs(pose: str, h: float, s: float, seat: float = 0.0, floor: float = 0.0) -> Dict[str, np.ndarray]:
    """Hip, knee, ankle and toe joints plus the pelvis centre for a pose (local frame).

    `seat` and `floor` (from _seat_levels) only matter for "sit"."""
    hip, ankle = HIP_HALF * s, ANKLE_HALF * s
    toe = 0.12 * s
    j = {
        "pelvis": (0.0, PELVIS_Y * h, 0.0),
        "hip_l": (-hip, HIP_Y * h, 0.0), "hip_r": (hip, HIP_Y * h, 0.0),
        "knee_l": (-hip, 0.28 * h, 0.01), "knee_r": (hip, 0.28 * h, 0.01),
        "ankle_l": (-ankle, 0.045 * h, 0.0), "ankle_r": (ankle, 0.045 * h, 0.0),
    }
    if pose == "walk":
        j.update({"knee_l": (-hip, 0.28 * h, 0.08 * h), "ankle_l": (-ankle, 0.045 * h, 0.13 * h),
                  "knee_r": (hip, 0.29 * h, -0.03 * h), "ankle_r": (ankle, 0.07 * h, -0.13 * h),
                  "pelvis": (0.0, 0.52 * h, 0.0), "hip_l": (-hip, 0.49 * h, 0.0), "hip_r": (hip, 0.49 * h, 0.0)})
    elif pose == "run":
        j.update({"knee_l": (-hip, 0.34 * h, 0.17 * h), "ankle_l": (-ankle, 0.14 * h, 0.12 * h),
                  "knee_r": (hip, 0.24 * h, -0.05 * h), "ankle_r": (ankle, 0.05 * h, -0.22 * h),
                  "pelvis": (0.0, 0.51 * h, 0.0), "hip_l": (-hip, 0.48 * h, 0.0), "hip_r": (hip, 0.48 * h, 0.0)})
    elif pose == "sit":
        if seat - floor > SEAT_MIN_M:     # on a seat: thighs on the seat, shins down to the floor
            j.update({"pelvis": (0.0, seat + 0.06 * h, -0.05 * h), "hip_l": (-hip, seat + 0.04 * h, -0.02 * h),
                      "hip_r": (hip, seat + 0.04 * h, -0.02 * h), "knee_l": (-hip, seat + 0.04 * h, 0.22 * h),
                      "knee_r": (hip, seat + 0.04 * h, 0.22 * h), "ankle_l": (-ankle, floor + 0.045 * h, 0.24 * h),
                      "ankle_r": (ankle, floor + 0.045 * h, 0.24 * h)})
        else:                # on the floor: knees up, feet flat
            j.update({"pelvis": (0.0, 0.08 * h, -0.05 * h), "hip_l": (-hip, 0.065 * h, -0.02 * h),
                      "hip_r": (hip, 0.065 * h, -0.02 * h), "knee_l": (-hip, 0.20 * h, 0.18 * h),
                      "knee_r": (hip, 0.20 * h, 0.18 * h), "ankle_l": (-ankle, 0.04 * h, 0.34 * h),
                      "ankle_r": (ankle, 0.04 * h, 0.34 * h)})
    elif pose == "crouch":
        j.update({"pelvis": (0.0, 0.32 * h, -0.08 * h), "hip_l": (-hip, 0.30 * h, -0.08 * h),
                  "hip_r": (hip, 0.30 * h, -0.08 * h), "knee_l": (-hip * 1.3, 0.30 * h, 0.12 * h),
                  "knee_r": (hip * 1.3, 0.30 * h, 0.12 * h), "ankle_l": (-ankle, 0.045 * h, 0.02 * h),
                  "ankle_r": (ankle, 0.045 * h, 0.02 * h)})
    elif pose == "kneel":
        j.update({"pelvis": (0.0, 0.28 * h, 0.0), "hip_l": (-hip, 0.25 * h, 0.0), "hip_r": (hip, 0.25 * h, 0.0),
                  "knee_l": (-hip, 0.04 * h, 0.04 * h), "knee_r": (hip, 0.04 * h, 0.04 * h),
                  "ankle_l": (-ankle, 0.03 * h, -0.2 * h), "ankle_r": (ankle, 0.03 * h, -0.2 * h)})
    elif pose in ("lift", "brace"):
        wide = 1.8 if pose == "brace" else 1.3
        j.update({"pelvis": (0.0, 0.49 * h, -0.03 * h), "hip_l": (-hip, 0.46 * h, -0.03 * h),
                  "hip_r": (hip, 0.46 * h, -0.03 * h), "knee_l": (-hip * wide, 0.27 * h, 0.06 * h),
                  "knee_r": (hip * wide, 0.27 * h, 0.06 * h), "ankle_l": (-ankle * wide, 0.045 * h, 0.0),
                  "ankle_r": (ankle * wide, 0.045 * h, 0.0)})
    elif pose in ("push", "throw"):       # right leg back
        j.update({"knee_r": (hip, 0.27 * h, -0.08 * h), "ankle_r": (ankle, 0.05 * h, -0.2 * h),
                  "knee_l": (-hip, 0.28 * h, 0.06 * h), "ankle_l": (-ankle, 0.045 * h, 0.08 * h)})
    elif pose == "pull":                  # left leg forward, weight back
        j.update({"knee_l": (-hip, 0.28 * h, 0.12 * h), "ankle_l": (-ankle, 0.045 * h, 0.2 * h),
                  "knee_r": (hip, 0.27 * h, -0.04 * h), "ankle_r": (ankle, 0.045 * h, -0.06 * h)})
    out = {k: np.array(v, dtype=float) for k, v in j.items()}
    out["toe_l"] = out["ankle_l"] + np.array([0.0, -0.02 * h, toe])
    out["toe_r"] = out["ankle_r"] + np.array([0.0, -0.02 * h, toe])
    if pose in ("kneel",):                # toes point back and down when kneeling
        out["toe_l"] = out["ankle_l"] + np.array([0.0, 0.0, -toe * 0.6])
        out["toe_r"] = out["ankle_r"] + np.array([0.0, 0.0, -toe * 0.6])
    return out


def _relaxed_hands(pose: str, h: float, s: float) -> Dict[str, np.ndarray]:
    """Hand targets that move with the upper body, in standing-body coordinates (the caller adds the
    pelvis drop, then the lean). A lying body keeps its arms along its sides, so they stay off the floor
    once the body is pitched flat."""
    side = 0.21 * s
    hands = {"hand_l": (-side, 0.40 * h, 0.02 * h), "hand_r": (side, 0.40 * h, 0.02 * h)}
    if pose == "walk":
        hands = {"hand_l": (-side, 0.41 * h, -0.08 * h), "hand_r": (side, 0.41 * h, 0.10 * h)}
    elif pose == "run":
        hands = {"hand_l": (-0.18 * s, 0.62 * h, 0.14 * h), "hand_r": (0.18 * s, 0.55 * h, -0.12 * h)}
    elif pose == "brace":
        hands = {"hand_l": (-0.12 * s, 0.84 * h, 0.22 * h), "hand_r": (0.12 * s, 0.84 * h, 0.22 * h)}
    elif pose == "fall":
        hands = {"hand_l": (-0.2 * s, 0.86 * h, 0.2 * h), "hand_r": (0.2 * s, 0.86 * h, 0.2 * h)}
    elif pose == "throw":
        hands["hand_l"] = (-0.2 * s, 0.72 * h, 0.22 * h)
    return {k: np.array(v, dtype=float) for k, v in hands.items()}


def _thigh_hands(j: Dict[str, np.ndarray], s: float) -> Dict[str, np.ndarray]:
    """Seated / crouching / kneeling hands resting on the thighs, 65% of the way to the knees (local frame).

    Placed from the leg joints, so they already sit at the dropped pelvis height (no drop, no lean)."""
    up_front = np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0)
    lift = (RADII["upper_leg"] + RADII["lower_arm"]) * s
    hands = {}
    for side in ("l", "r"):
        hip, knee = j[f"hip_{side}"], j[f"knee_{side}"]
        thigh = knee - hip
        axis = thigh / max(float(np.linalg.norm(thigh)), 1e-9)
        normal = up_front - axis * float(up_front @ axis)       # the thigh's top / front surface
        normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
        point = hip + 0.65 * thigh + normal * lift
        point[0] = hip[0] * 1.3
        hands[f"hand_{side}"] = point
    return hands


def _within_reach(shoulder: np.ndarray, target: np.ndarray, reach: float) -> np.ndarray:
    """`target`, pulled toward the shoulder when the arm cannot reach it."""
    span = target - shoulder
    dist = float(np.linalg.norm(span))
    return target if dist <= reach else shoulder + span * (reach / dist)


def _two_bone(shoulder: np.ndarray, target: np.ndarray, l1: float, l2: float, pole: np.ndarray) -> np.ndarray:
    """Elbow position reaching `target`; stretches straight when the target is out of reach."""
    span = target - shoulder
    dist = float(np.linalg.norm(span))
    if dist < 1e-6:
        return shoulder + pole / (np.linalg.norm(pole) + 1e-9) * l1
    direction = span / dist
    if dist >= l1 + l2 - 1e-6:
        return shoulder + span * (l1 / (l1 + l2))
    along = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
    height = math.sqrt(max(l1 * l1 - along * along, 0.0))
    perp = pole - direction * float(pole @ direction)
    norm = float(np.linalg.norm(perp))
    perp = perp / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])
    return shoulder + direction * along + perp * height


def _to_local(point: Vec3, base: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    rel = np.asarray(point, dtype=float) - base
    return np.array([rel @ axes[0], rel @ axes[1], rel @ axes[2]])


def is_quadruped(frame: EntityFrame) -> bool:
    """True for a character drawn with the four-legged proxy (blocking.build_scene sets the shape)."""
    return frame.shape == QUADRUPED


def segments(frame: EntityFrame, seat_y: Optional[float] = None) -> List[Segment]:
    """World-space capsules (p0, p1, radius_m, part) of a character's mannequin.

    `seat_y`: world height of the seat under a sitting character whose base stays on the floor (see
    blocking.seat_height); without it a base well above the floor is taken to be the seat."""
    if is_quadruped(frame):
        return quadruped_segments(frame)
    h = float(frame.size[1]) if frame.size[1] > 0 else REFERENCE_HEIGHT_M
    s = h / REFERENCE_HEIGHT_M
    pose = frame.pose if frame.pose != NONE else "stand"
    base = np.asarray(frame.position, dtype=float)
    axes = body_axes(frame.yaw_deg)

    # 1. Legs and pelvis, then the upper body pivoting at the pelvis.
    seat, floor = _seat_levels(float(base[1]), seat_y, h) if pose == "sit" else (0.0, 0.0)
    j = _legs(pose, h, s, seat, floor)
    pelvis = j["pelvis"]
    drop = pelvis[1] - PELVIS_Y * h                       # the upper body follows the pelvis down
    upper = {
        "chest": np.array([0.0, CHEST_Y * h + drop, pelvis[2]]),
        "torso_base": np.array([0.0, PELVIS_Y * h + 0.03 * h + drop, pelvis[2]]),
        "shoulder_l": np.array([-SHOULDER_HALF * s, SHOULDER_Y * h + drop, pelvis[2]]),
        "shoulder_r": np.array([SHOULDER_HALF * s, SHOULDER_Y * h + drop, pelvis[2]]),
        "head": np.array([0.0, HEAD_Y * h + drop, pelvis[2]]),
    }
    if pose not in THIGH_HANDS_POSES:
        upper.update({k: v + np.array([0.0, drop, pelvis[2]]) for k, v in _relaxed_hands(pose, h, s).items()})
    pitch, roll = WEIGHT_CUE_LEAN.get(frame.weight_cue, (0.0, 0.0))
    pitch += POSE_LEAN.get(pose, 0.0)
    lean = _rot_z(roll) @ _rot_x(pitch)
    for k in upper:
        upper[k] = pelvis + lean @ (upper[k] - pelvis)
    head_up = lean @ np.array([0.0, 1.0, 0.0])
    head_fwd = lean @ np.array([0.0, 0.0, 1.0])

    # 2. Hands: relaxed hands within the arm's reach and above the floor; physics-driven targets for
    #    handling poses and for anything held in the right hand (held props sit exactly there).
    l1, l2 = UPPER_ARM * h, FOREARM * h
    if pose in THIGH_HANDS_POSES:
        hands = _thigh_hands(j, s)
        hand_r, hand_l = hands["hand_r"], hands["hand_l"]
    else:
        hand_r, hand_l = upper.pop("hand_r"), upper.pop("hand_l")
    floor_y = floor + RADII["lower_arm"] * s
    relaxed = {}
    for side, hand in (("l", hand_l), ("r", hand_r)):
        hand = np.array(hand, dtype=float)
        if pose not in BODY_PITCH:                       # lying / falling bodies are pitched afterwards
            hand[1] = max(hand[1], floor_y)
        relaxed[side] = _within_reach(upper[f"shoulder_{side}"], hand, l1 + l2)
    hand_l, hand_r = relaxed["l"], relaxed["r"]
    holding = frame.holding_id not in (NONE, "")
    if pose in PHYSICS_HAND_POSES or holding:
        hand_r = _to_local(physics.hand_position(frame, "right"), base, axes)
        if pose in BOTH_HANDS_POSES:
            hand_l = _to_local(physics.hand_position(frame, "left"), base, axes)
    elbow_l = _two_bone(upper["shoulder_l"], hand_l, l1, l2, lean @ np.array([-0.35, -0.3, -1.0]))
    elbow_r = _two_bone(upper["shoulder_r"], hand_r, l1, l2, lean @ np.array([0.35, -0.3, -1.0]))

    # 3. Whole-body pitch about the ankles for falling / lying bodies.
    joints = {**j, **upper, "hand_l": hand_l, "hand_r": hand_r, "elbow_l": elbow_l, "elbow_r": elbow_r}
    body_pitch = BODY_PITCH.get(pose, 0.0)
    if body_pitch:
        pivot = (joints["ankle_l"] + joints["ankle_r"]) / 2.0
        tip = _rot_x(body_pitch)
        lift = np.array([0.0, 0.1 * s if pose == "lie" else 0.0, 0.0])
        joints = {k: pivot + tip @ (v - pivot) + lift for k, v in joints.items()}
        head_up, head_fwd = tip @ head_up, tip @ head_fwd

    # 4. Local -> world.
    rot = np.stack(axes, axis=1)                          # columns: right, up, forward

    def w(p: np.ndarray) -> Vec3:
        q = base + rot @ p
        return (float(q[0]), float(q[1]), float(q[2]))

    head = joints["head"]
    head_r = HEAD_R * h
    out: List[Segment] = [
        (w(joints["torso_base"]), w(joints["chest"]), RADII["torso"] * s, "torso"),
        (w(joints["shoulder_l"]), w(joints["shoulder_r"]), RADII["shoulders"] * s, "shoulders"),
        (w(joints["hip_l"]), w(joints["hip_r"]), RADII["hips"] * s, "hips"),
        (w(head - head_up * 0.02 * h), w(head + head_up * 0.02 * h), head_r, "head"),
        (w(head + head_fwd * head_r * 0.7), w(head + head_fwd * head_r * 1.08 - head_up * 0.01 * h), 0.009 * h, "nose"),
    ]
    # Orientation cues for the image model: a hair cap over the back/top of the skull and two eyes in
    # front. A bare sphere reads as either the face or the back of the head.
    head_right = np.cross(head_up, head_fwd)
    hair = head - head_fwd * head_r * 0.22 + head_up * head_r * 0.18
    out.append((w(hair - head_up * 0.01 * h), w(hair + head_up * 0.01 * h), head_r * 0.93, "hair"))
    for side, sign in (("l", -1.0), ("r", 1.0)):
        eye = head + head_fwd * head_r * 0.86 + head_right * sign * head_r * 0.36 + head_up * head_r * 0.18
        out.append((w(eye), w(eye), head_r * 0.13, f"eye_{side}"))
    for side in ("l", "r"):
        out += [
            (w(joints[f"shoulder_{side}"]), w(joints[f"elbow_{side}"]), RADII["upper_arm"] * s, f"upper_arm_{side}"),
            (w(joints[f"elbow_{side}"]), w(joints[f"hand_{side}"]), RADII["lower_arm"] * s, f"lower_arm_{side}"),
            (w(joints[f"hip_{side}"]), w(joints[f"knee_{side}"]), RADII["upper_leg"] * s, f"upper_leg_{side}"),
            (w(joints[f"knee_{side}"]), w(joints[f"ankle_{side}"]), RADII["lower_leg"] * s, f"lower_leg_{side}"),
            (w(joints[f"ankle_{side}"]), w(joints[f"toe_{side}"]), RADII["foot"] * s, f"foot_{side}"),
        ]
    return out


# ---------------------------------------------------------------- quadrupeds
QUADRUPED = "quadruped"          # EntityFrame.shape of a Character with body == "quadruped"
# Proportions as fractions of the SHOULDER height (Character.height_m for a quadruped).
Q_SPINE = 0.78                   # rump to chest along the spine (torso capsule axis)
Q_FORE = (0.31, 0.31)            # leg top to elbow, elbow to wrist
Q_HIND = (0.30, 0.31)            # hip to stifle, stifle to hock
Q_RADII = {"torso": 0.19, "chest": 0.215, "neck": 0.10, "head": 0.13, "snout": 0.065, "nose": 0.035, "ear": 0.04, "eye": 0.024,
           "tail": 0.04, "upper_leg": 0.07, "lower_leg": 0.05, "paw": 0.045}
Q_LEGS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class _QuadPose:
    """A quadruped pose in the body frame, in units of shoulder height. Legs are solved with two-bone IK
    toward their wrist (front) / hock (hind) targets, so limbs keep their lengths (a joint the IK would
    put below the ground is lifted onto it)."""
    rump: Tuple[float, float, float]            # rear end of the spine
    pitch: float                                # spine pitch, degrees (positive: chest up)
    head: Tuple[float, float, float]            # head centre
    head_pitch: float                           # snout pitch, degrees (positive: up)
    wrist: Dict[str, Tuple[float, float, float]]      # "fl" / "fr"
    hock: Dict[str, Tuple[float, float, float]]       # "rl" / "rr"
    hind_paw: Dict[str, Tuple[float, float, float]]   # "rl" / "rr" toe tips
    tail_tip: Tuple[float, float, float]
    fore_pole: Tuple[float, float, float] = (0.0, -0.2, -1.0)   # elbows point back (right legs; mirrored)
    hind_pole: Tuple[float, float, float] = (0.0, -0.2, 1.0)    # stifles point forward
    roll: float = 0.0                           # whole body rolled onto its side (degrees)


def _mirror(right: Tuple[float, float, float], front: bool) -> Dict[str, Tuple[float, float, float]]:
    x, y, z = right
    return {("fl" if front else "rl"): (-x, y, z), ("fr" if front else "rr"): (x, y, z)}


_Q_STAND = _QuadPose(
    rump=(0.0, 0.68, -0.40), pitch=3.0, head=(0.0, 1.02, 0.60), head_pitch=-15.0,
    wrist=_mirror((0.12, 0.06, 0.38), True), hock=_mirror((0.12, 0.20, -0.47), False),
    hind_paw=_mirror((0.12, 0.03, -0.38), False), tail_tip=(0.0, 0.98, -0.78))
QUADRUPED_POSES: Dict[str, _QuadPose] = {
    "stand": _Q_STAND,
    "walk": _QuadPose(            # diagonal pair: front-left and hind-right swing forward
        rump=(0.0, 0.68, -0.40), pitch=3.0, head=(0.0, 0.98, 0.62), head_pitch=-18.0,
        wrist={"fl": (-0.12, 0.12, 0.52), "fr": (0.12, 0.06, 0.28)},
        hock={"rl": (-0.12, 0.20, -0.58), "rr": (0.12, 0.24, -0.36)},
        hind_paw={"rl": (-0.12, 0.03, -0.50), "rr": (0.12, 0.08, -0.26)}, tail_tip=(0.0, 0.95, -0.80)),
    "run": _QuadPose(             # gallop: forelegs reaching, hind legs pushing off, spine low and long
        rump=(0.0, 0.62, -0.42), pitch=-2.0, head=(0.0, 0.86, 0.68), head_pitch=-8.0,
        wrist={"fl": (-0.12, 0.30, 0.78), "fr": (0.12, 0.06, 0.52)},
        hock={"rl": (-0.12, 0.30, -0.84), "rr": (0.12, 0.18, -0.62)},
        hind_paw={"rl": (-0.12, 0.26, -1.00), "rr": (0.12, 0.03, -0.56)}, tail_tip=(0.0, 0.70, -0.98)),
    "sit": _QuadPose(             # haunches on the ground, chest up on straight forelegs
        rump=(0.0, 0.22, -0.30), pitch=38.0, head=(0.0, 1.14, 0.44), head_pitch=-5.0,
        wrist=_mirror((0.12, 0.06, 0.40), True), hock=_mirror((0.15, 0.04, -0.40), False),
        hind_paw=_mirror((0.15, 0.03, -0.16), False), tail_tip=(0.10, 0.04, -0.62),
        hind_pole=(0.5, 0.3, 1.0)),
    "lie": _QuadPose(             # lying down on the belly, forelegs forward, head up
        rump=(0.0, 0.22, -0.40), pitch=2.0, head=(0.0, 0.56, 0.58), head_pitch=-8.0,
        wrist=_mirror((0.13, 0.04, 0.50), True), hock=_mirror((0.24, 0.04, -0.56), False),
        hind_paw=_mirror((0.26, 0.03, -0.34), False), tail_tip=(0.25, 0.04, -0.80),
        fore_pole=(0.6, -0.3, -1.0), hind_pole=(1.0, 0.1, 0.6)),
    "crouch": _QuadPose(          # front end down (stalking / play bow), hind legs standing
        rump=(0.0, 0.62, -0.40), pitch=-20.0, head=(0.0, 0.52, 0.66), head_pitch=0.0,
        wrist=_mirror((0.12, 0.04, 0.62), True), hock=_mirror((0.12, 0.20, -0.47), False),
        hind_paw=_mirror((0.12, 0.03, -0.38), False), tail_tip=(0.0, 0.95, -0.72)),
    "brace": _QuadPose(           # legs planted wide, weight back (pulling, bracing)
        rump=(0.0, 0.58, -0.44), pitch=8.0, head=(0.0, 0.92, 0.56), head_pitch=-10.0,
        wrist=_mirror((0.15, 0.06, 0.52), True), hock=_mirror((0.15, 0.18, -0.56), False),
        hind_paw=_mirror((0.15, 0.03, -0.48), False), tail_tip=(0.0, 0.80, -0.80)),
    "fall": replace(_Q_STAND, roll=90.0),     # knocked over: lying on its side, legs out
}
# Schema poses without a four-legged counterpart map to the nearest one.
QUADRUPED_POSE_ALIAS = {"none": "stand", "reach": "stand", "lift": "stand", "carry": "stand", "throw": "stand",
                        "push": "stand", "pull": "brace", "kneel": "lie"}


def quadruped_pose(pose: str) -> str:
    """The quadruped pose drawn for a schema pose."""
    return pose if pose in QUADRUPED_POSES else QUADRUPED_POSE_ALIAS.get(pose, "stand")


def _clamped_two_bone(top: np.ndarray, target: np.ndarray, l1: float, l2: float,
                      pole: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(middle joint, end joint) of a two-bone limb reaching toward `target` without stretching."""
    end = _within_reach(top, target, (l1 + l2) * 0.999)
    return _two_bone(top, end, l1, l2, pole), end


def quadruped_segments(frame: EntityFrame) -> List[Segment]:
    """World-space capsules (p0, p1, radius_m, part) of a four-legged proxy; frame.size[1] = shoulder height.

    Weight cues and held objects do not change the proxy (it has no hands)."""
    h = float(frame.size[1]) if frame.size[1] > 0 else 0.5
    qp = QUADRUPED_POSES[quadruped_pose(frame.pose)]
    r = {k: q * h for k, q in Q_RADII.items()}

    def v(p: Tuple[float, float, float]) -> np.ndarray:            # pose units -> metres
        return np.array(p, dtype=float) * h

    # 1. Spine, neck and head.
    spine = _rot_x(-qp.pitch)                                          # maps +z onto the spine direction
    rump = v(qp.rump)
    chest = rump + spine @ np.array([0.0, 0.0, Q_SPINE * h])
    head = v(qp.head)
    hp = math.radians(qp.head_pitch)
    head_fwd = np.array([0.0, math.sin(hp), math.cos(hp)])
    head_up = np.array([0.0, math.cos(hp), -math.sin(hp)])
    right = np.array([1.0, 0.0, 0.0])
    joints: Dict[str, np.ndarray] = {
        "rump": rump, "chest": chest, "head": head,
        "ribs": chest + spine @ v((0.0, -0.02, -0.24)), "brisket": chest + spine @ v((0.0, -0.02, 0.0)),
        "neck_base": chest + spine @ v((0.0, 0.10, 0.04)), "neck_top": head - head_fwd * 0.10 * h,
        "snout_0": head + head_fwd * 0.10 * h - head_up * 0.03 * h,
        "snout_1": head + head_fwd * 0.24 * h - head_up * 0.04 * h,
        "nose": head + head_fwd * 0.30 * h - head_up * 0.03 * h,
        "tail_0": rump + spine @ v((0.0, 0.06, -0.18)), "tail_1": v(qp.tail_tip),
    }
    for side, sign in (("l", -1.0), ("r", 1.0)):
        joints[f"eye_{side}"] = head + head_fwd * 0.105 * h + right * sign * 0.065 * h + head_up * 0.045 * h
        joints[f"ear_{side}_0"] = head + head_up * 0.10 * h + right * sign * 0.08 * h - head_fwd * 0.03 * h
        joints[f"ear_{side}_1"] = head - head_up * 0.02 * h + right * sign * 0.14 * h - head_fwd * 0.02 * h

    # 2. Legs: two-bone IK from the shoulder / hip toward the wrist / hock target, then the paw.
    for leg in Q_LEGS:
        sign = -1.0 if leg.endswith("l") else 1.0
        mirror = np.array([sign, 1.0, 1.0])
        if leg.startswith("f"):
            top = chest + spine @ v((sign * 0.12, -0.06, -0.02))
            mid, low = _clamped_two_bone(top, v(qp.wrist[leg]), Q_FORE[0] * h, Q_FORE[1] * h,
                                         np.array(qp.fore_pole) * mirror)
            paw = low + v((0.0, -0.03, 0.08))
        else:
            top = rump + spine @ v((sign * 0.12, -0.06, 0.02))
            mid, low = _clamped_two_bone(top, v(qp.hock[leg]), Q_HIND[0] * h, Q_HIND[1] * h,
                                         np.array(qp.hind_pole) * mirror)
            paw = v(qp.hind_paw[leg])
        joints.update({f"top_{leg}": top, f"mid_{leg}": mid, f"low_{leg}": low, f"paw_{leg}": paw})

    # 3. Knocked over: roll onto the left side about the spine, resting the torso on the ground.
    if qp.roll:
        roll = _rot_z(qp.roll)
        joints = {k: roll @ p for k, p in joints.items()}
        head_fwd, head_up = roll @ head_fwd, roll @ head_up
        spine_axis = (joints["rump"] + joints["chest"]) / 2.0
        shift = np.array([-spine_axis[0], r["chest"] - min(joints["rump"][1], joints["chest"][1]), 0.0])
        joints = {k: p + shift for k, p in joints.items()}
    for leg in Q_LEGS:                                                 # joints never sink into the ground
        for name, part in ((f"mid_{leg}", "upper_leg"), (f"low_{leg}", "lower_leg"), (f"paw_{leg}", "paw")):
            joints[name][1] = max(joints[name][1], r[part])

    # 4. Local -> world.
    base = np.asarray(frame.position, dtype=float)
    rot = np.stack(body_axes(frame.yaw_deg), axis=1)                    # columns: right, up, forward

    def w(p: np.ndarray) -> Vec3:
        q = base + rot @ p
        return (float(q[0]), float(q[1]), float(q[2]))

    j = joints
    out: List[Segment] = [
        (w(j["rump"]), w(j["chest"]), r["torso"], "torso"),
        (w(j["ribs"]), w(j["brisket"]), r["chest"], "chest"),
        (w(j["neck_base"]), w(j["neck_top"]), r["neck"], "neck"),
        (w(j["head"] - head_fwd * 0.03 * h), w(j["head"] + head_fwd * 0.03 * h), r["head"], "head"),
        (w(j["snout_0"]), w(j["snout_1"]), r["snout"], "snout"),
        (w(j["nose"]), w(j["nose"]), r["nose"], "nose"),
        (w(j["tail_0"]), w(j["tail_1"]), r["tail"], "tail"),
    ]
    for side in ("l", "r"):
        out.append((w(j[f"ear_{side}_0"]), w(j[f"ear_{side}_1"]), r["ear"], f"ear_{side}"))
        out.append((w(j[f"eye_{side}"]), w(j[f"eye_{side}"]), r["eye"], f"eye_{side}"))
    for leg in Q_LEGS:
        out += [
            (w(j[f"top_{leg}"]), w(j[f"mid_{leg}"]), r["upper_leg"], f"upper_leg_{leg}"),
            (w(j[f"mid_{leg}"]), w(j[f"low_{leg}"]), r["lower_leg"], f"lower_leg_{leg}"),
            (w(j[f"low_{leg}"]), w(j[f"paw_{leg}"]), r["paw"], f"paw_{leg}"),
        ]
    return out
