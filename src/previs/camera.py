"""
Camera solver: the planner's shot grammar (size, angle, lens, line/frame side, move) -> CameraState,
plus the pinhole projection every renderer shares.

Conventions (types.py): metres, y up, yaw 0 faces +z, positive yaw turns toward +x, pitch > 0 looks up.
Camera axes: forward f = (sin yaw cos pitch, sin pitch, cos yaw cos pitch), image-right r = f x up
(normalised), image-up u = r x f, then rolled about f. Pixels: +x right, +y down, principal point at
the image centre, square pixels, focal length from the vertical field of view of a 36x24 mm sensor.

Placement: the framed height of the shot size sets the distance (d = framed/2 / tan(vfov/2)), the
angle sets the pitch, and the camera sits on the ray of length d through the aim point, so the framed
height is exact at the subject (unless that would put the camera inside or too near the subject's own
body: then it backs out along the ray). Character shots keep headroom above the head, also when a
dolly tightens the frame. The horizontal side comes from the line side (pairs) or the subject's
facing; it is chosen once per shot over all its keyframes by a deterministic search (nearest to the
nominal side first) that avoids cameras inside props/people, blocked sight lines and, indoors, walls.
That choice is then checked against what the camera actually sees: a low-resolution id render of each
keyframe measures how much of the subject (head and torso for characters) is visible and whether a
character's face is toward the camera; when it is hidden, other sides (and a slightly higher camera)
are scanned and the least-hidden one wins, and a moving camera may still adjust single keyframes a
little. Distance, height and aim then follow the subject in each frame, and the move is applied by
the keyframe's move_progress; moves that displace the camera (orbit, crane, truck, handheld) stop
short of solids and of losing the subject. A static shot is locked off: one camera for all its
frames, framing the union of the subject over its keyframes (so a static lock can reuse pixels).

Framing measures what blocking draws (blocking.build_scene): a quadruped's four-legged body and a
seated character on its seat (blocking.seat_height), not the bare solved EntityFrame.
"""
import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.previs.blocking import (ROOM_HEIGHT_M, BlockingScene, Primitive, build_scene, entity_aabb, mannequin,
                                 seat_height, surface_height_below)
from src.previs.schema import NONE, ScenePlan, Shot
from src.previs.state import SceneIndex
from src.previs.types import CameraState, EntityFrame, FrameState, SolvedPlan

SENSOR_HALF_HEIGHT_MM = 12.0
FRAMED_HEIGHT_M = {"ews": 20.0, "ws": 5.0, "ms": 0.95, "mcu": 0.6, "cu": 0.38, "ecu": 0.18}
FS_FACTOR = 1.15
INSERT_FACTOR = 2.5
LOOK_AT_FRACTION = {"ms": 0.72, "mcu": 0.9, "cu": 0.9, "ecu": 0.93}       # of subject height; else mid-height
ANGLE_PITCH_DEG = {"eye": 0.0, "high": -25.0, "low": 15.0, "overhead": -80.0, "dutch": 0.0}
DUTCH_ROLL_DEG = 12.0
GROUP_SIZES = frozenset({"ews", "ws"})                # two-shots aim between subject and secondary
TIGHT_SIZES = frozenset({"ms", "mcu", "cu", "ecu", "insert"})
HEAD_AIM_SIZES = frozenset({"mcu", "cu", "ecu"})      # aim at the head, wherever the pose put it
MIN_CAMERA_HEIGHT_M = 0.15
WALL_MARGIN_M = 0.2
CEILING_MARGIN_M = 0.2
SECONDARY_OFFSET_DEG = 30.0
FACING_OFFSET_DEG = 25.0
FACE_THREE_QUARTER_DEG = 35.0  # character shots: camera this far off the face direction (3/4 front)
LINE_SIDE_LIMIT_DEG = 80.0     # nominal azimuths stay this close to the line-side normal (cost forbids > 85)
PAN_DEG, TILT_DEG = 20.0, 12.0
DOLLY_IN, DOLLY_OUT = 0.35, 0.5
TRUCK_M, CRANE_M, ORBIT_DEG = 1.5, 1.2, 40.0
HANDHELD_POS_M, HANDHELD_ANGLE_DEG = 0.03, 0.8
NEAR_M = 0.05
SEARCH_STEP_DEG = 5.0
CLEARANCE_M = 0.25             # minimum gap between the camera and any solid
CLEARANCE_COST = 1.0           # rendered placements: cost of a camera just clear of a solid (falls to 0 at CLEARANCE_M)
SHOT_DEVIATION_WEIGHT = 1.0    # cost of turning away from the nominal side when choosing a shot's side
FRAME_DEVIATION_WEIGHT = 3.0   # cost of leaving the shot's side in one frame (only when blocked there)
AIRBORNE_MARGIN = 1.25         # framed height over (drawn top - surface) for an airborne prop framed tight
FOLLOW_THRESHOLD = 0.1         # subject moved > this x framed height since the first keyframe: pan/tilt follow it
SAFE_FRAME = 0.45              # moves never push a must-see point beyond this x frame size from centre
HEADROOM = 0.08                # character shots: frame space above the head top, x framed height
HEADROOM_SIZES = frozenset({"fs", "ms", "mcu", "cu"})
SUBJECT_CLEARANCE_M = 0.08     # minimum gap between the camera and the subject (or what touches it): > NEAR_M
SUBJECT_STEP_M = 0.01          # resolution of backing the camera out of the subject's body
PUSH_OUT_WEIGHT = 0.5          # placement cost of framing wider than asked (camera backed out of the subject)
MOVE_STEPS = 8                 # displacing moves are checked at this many points along their path
# Visibility in the loop: a low-res id render of the actual scene measures what the camera sees.
VIS_WIDTH_PX = 128
MIN_VIS_PX = 4                 # fewer pixels than this when drawn alone: too small to measure
MIN_VISIBLE = 0.7              # subject (key parts) visible fraction below which the shot side is re-searched
VISIBILITY_WEIGHT = 10.0       # a fully hidden subject costs this much (dominates deviating from the nominal side)
SECONDARY_MIN_VISIBLE = 0.5    # the secondary only counts when hidden more than this (furniture may hide some)
SECONDARY_WEIGHT = 0.5         # ... and then this much relative to the subject
FACE_WEIGHT = 0.5              # a character subject seen from behind counts this much
FACE_SIDE_DEG = 75.0           # camera within this angle of the face direction: the face side reads
FACE_BACK_DEG = 120.0          # beyond this angle only the back of the head is seen
GOOD_VISIBLE = 0.9             # a moving camera adjusts a keyframe whose subject is less visible than this
KEYFRAME_SPREAD_DEG = 45.0     # ... by at most this much azimuth from the shot's side
VIS_STEP_DEG = 15.0            # azimuth step of the visibility scan
LIFT_DEG = 15.0                # the "slightly higher" camera tried when no azimuth shows the subject ...
MAX_LIFT_STEPS = 2             # ... up to this many steps higher
LIFT_COST = 0.5                # cost of each lift step (same units as 90 deg of azimuth deviation)
LOWER_BODY_PARTS = ("leg", "foot", "hips", "tail")   # character parts that may hide behind furniture
HEAD_TOP_PARTS = frozenset({"head", "hair", "ear_l", "ear_r"})   # the top of these is the top of the head
GRIP_PARTS = ("lower_arm", "forearm", "hand", "snout", "muzzle", "jaw")   # holder parts touching what is held
_LOWER_BODY_ID = "__lower_body__"

UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------- optics + projection
def vertical_fov_deg(lens_mm: float) -> float:
    return math.degrees(2.0 * math.atan(SENSOR_HALF_HEIGHT_MM / max(float(lens_mm), 1e-3)))


def horizontal_fov_deg(vfov_deg: float, width: int, height: int) -> float:
    return math.degrees(2.0 * math.atan(math.tan(math.radians(vfov_deg) / 2.0) * width / height))


def _forward(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])


def _basis(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _forward(yaw_deg, pitch_deg)
    yaw = math.radians(yaw_deg)
    r = np.array([-math.cos(yaw), 0.0, math.sin(yaw)])      # = normalise(f x up), also defined looking straight up/down
    u = np.cross(r, f)
    roll = math.radians(roll_deg)
    c, s = math.cos(roll), math.sin(roll)
    return r * c + u * s, u * c - r * s, f


def camera_basis(cam: CameraState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image-right, image-up, forward) unit vectors in world space."""
    return _basis(cam.yaw_deg, cam.pitch_deg, cam.roll_deg)


def focal_px(cam: CameraState) -> float:
    return (cam.height / 2.0) / math.tan(math.radians(cam.vfov_deg) / 2.0)


def world_to_camera(cam: CameraState, points: np.ndarray) -> np.ndarray:
    """Camera-space coordinates (x right, y up, z forward = depth) of world points [N,3]."""
    r, u, f = camera_basis(cam)
    rel = np.atleast_2d(np.asarray(points, dtype=float)) - np.asarray(cam.position, dtype=float)
    return np.stack([rel @ r, rel @ u, rel @ f], axis=1)


def project(cam: CameraState, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole projection: (uv [N,2] pixel coordinates, depth [N] along the optical axis).
    Points at or behind the camera plane get uv = nan and depth <= 0."""
    pc = world_to_camera(cam, points)
    depth = pc[:, 2]
    fpx = focal_px(cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.width / 2.0 + fpx * pc[:, 0] / depth
        v = cam.height / 2.0 - fpx * pc[:, 1] / depth
    uv = np.stack([u, v], axis=1)
    uv[depth <= 1e-9] = np.nan
    return uv, depth


# ---------------------------------------------------------------- framing
@dataclass
class _Target:
    look_at: np.ndarray                       # aim point
    framed_h: float                           # world height the frame spans at the aim point
    subject_id: str
    sight_points: List[np.ndarray]            # points that must not be hidden behind other solids
    facing_yaw: Optional[float] = None        # subject's facing (characters only)
    group: bool = False                       # two-shot aimed between subject and secondary
    fit_points: List[np.ndarray] = field(default_factory=list)   # aim at their angular centre when set
    min_height: float = MIN_CAMERA_HEIGHT_M


def _move_progress(frame: FrameState, index: SceneIndex) -> float:
    kf = index.keyframe_by_id.get(frame.kf_id)
    return min(max(float(kf.move_progress), 0.0), 1.0) if kf is not None else 0.0


def _present(frame: FrameState, entity_id: str) -> Optional[EntityFrame]:
    ef = frame.entities.get(entity_id)
    return ef if ef is not None and ef.present else None


def _drawn_bounds(prims: Sequence[Primitive], entity_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Bounds of everything drawn for an entity (body plus proxies such as a liquid arc or a puddle)."""
    mine = [p.bounds() for p in prims if p.entity_id == entity_id and p.alpha >= 1.0]
    if not mine:
        return None
    return np.min([b[0] for b in mine], axis=0), np.max([b[1] for b in mine], axis=0)


def _head_top(segments: Sequence[Tuple]) -> Optional[float]:
    """Top of a mannequin's head (skull, hair cap or an animal's ears), or None when it has no head."""
    tops = [max(p0[1], p1[1]) + r for p0, p1, r, part in segments if part in HEAD_TOP_PARTS]
    return float(max(tops)) if tops else None


def _drawn(scene: BlockingScene, frame: FrameState, entity_id: str) -> Optional[EntityFrame]:
    """An entity as blocking draws it: build_scene gives a Character.body == "quadruped" the four-legged
    shape (the solved frame keeps shape "character"). Pair with `scene.entities` in entity_aabb / _segments."""
    ef = scene.entities.get(entity_id) if scene.entities else None
    return ef if ef is not None else frame.entities.get(entity_id)


def _segments(ef: EntityFrame, entities: Dict[str, EntityFrame]) -> List[Tuple]:
    """A character's capsules exactly as drawn: on its seat (blocking.seat_height), or a quadruped's body."""
    return mannequin.segments(ef, seat_y=seat_height(ef, entities))


def _framing(shot: Shot, frame: FrameState, index: SceneIndex, aspect: float, scene: BlockingScene,
             distance_scale: float = 1.0) -> _Target:
    """Aim point, framed height and must-see points for this shot size and subject, from the geometry
    blocking draws (`scene`). `distance_scale` (a dolly) scales the height the frame spans at the subject."""
    prims, entities = scene.primitives, (scene.entities or frame.entities)
    subject = _drawn(scene, frame, shot.subject_id)
    if subject is None:
        loc = index.locations.get(frame.location_id)
        centre = np.array([loc.width_m / 2.0, 1.0, loc.depth_m / 2.0]) if loc else np.array([0.0, 1.0, 0.0])
        return _Target(centre, FRAMED_HEIGHT_M.get(shot.size, FS_FACTOR * 1.7), shot.subject_id, [centre])

    lo, hi = entity_aabb(subject, entities)
    base = np.array(subject.position, dtype=float)
    centre = (lo + hi) / 2.0
    is_char = subject.kind == "character"
    height = max(float(hi[1] - base[1]), 0.05) if is_char else max(float(hi[1] - lo[1]), 0.02)
    size = shot.size

    # 1. Framed height and aim point by shot size.
    if size == "insert":
        framed = INSERT_FACTOR * max(float(v) for v in subject.size)
    elif size == "fs":
        framed = FS_FACTOR * max(height, float(max(hi[0] - lo[0], hi[2] - lo[2])) / aspect)
    else:
        framed = FRAMED_HEIGHT_M.get(size, FRAMED_HEIGHT_M["ms"])
    if is_char:
        aim_y = base[1] + LOOK_AT_FRACTION.get(size, 0.5) * height
        segments = _segments(subject, entities)
        head = next(((np.array(p0) + np.array(p1)) / 2.0 for p0, p1, _, part in segments if part == "head"), base)
        weight = 1.0 if size in HEAD_AIM_SIZES else (0.5 if size == "ms" else 0.0)
        head_top = _head_top(segments)
        if size in HEADROOM_SIZES and head_top is not None:
            # Keep the whole head plus headroom in frame, also when a dolly tightens the framing.
            aim_y = max(aim_y, head_top - (0.5 - HEADROOM) * framed * distance_scale)
        look_at = np.array([base[0] + (head[0] - base[0]) * weight, aim_y, base[2] + (head[2] - base[2]) * weight])
    else:
        look_at = centre.copy()
    target = _Target(look_at, float(framed), subject.entity_id, [look_at.copy(), np.array([look_at[0], hi[1] - 0.02, look_at[2]])],
                     facing_yaw=float(subject.yaw_deg) if is_char else None)

    # 2. Tight shots of a prop keep its physics readable: an airborne prop is framed together with the
    #    surface below it (its detached shadow) and its spray, and debris / puddles fit the frame.
    drawn = _drawn_bounds(prims, subject.entity_id)
    if not is_char and size in TIGHT_SIZES and drawn is not None:
        d_lo, d_hi = drawn
        floor_y = surface_height_below(frame.entities, subject.entity_id, float(centre[0]), float(centre[2]), float(lo[1]))
        gap = float(lo[1]) - floor_y
        if subject.support == "airborne" and gap > 0.03:
            span = float(d_hi[1]) - floor_y
            target.framed_h = max(target.framed_h, span * AIRBORNE_MARGIN)
            target.look_at = np.array([centre[0], floor_y + span / 2.0, centre[2]])
            target.fit_points = [np.array([centre[0], floor_y, centre[2]]), np.array([centre[0], d_hi[1], centre[2]]),
                                 np.array([d_lo[0], centre[1], d_lo[2]]), np.array([d_hi[0], centre[1], d_hi[2]])]
            target.sight_points.append(np.array([centre[0], floor_y + 0.01, centre[2]]))
            target.min_height = max(MIN_CAMERA_HEIGHT_M, floor_y + 0.5 * gap)
        spread = float(max(d_hi[0] - d_lo[0], d_hi[2] - d_lo[2]))
        if spread > float(max(hi[0] - lo[0], hi[2] - lo[2])) + 0.05:
            target.framed_h = max(target.framed_h, 1.1 * spread / aspect)
            target.look_at[1] = min(target.look_at[1], floor_y + max(0.2 * target.framed_h, 0.03))

    # 3. Two-shots aim between subject and secondary.
    secondary = _drawn(scene, frame, shot.secondary_id) if shot.secondary_id != NONE else None
    if size in GROUP_SIZES and secondary is not None and secondary.present:
        s_lo, s_hi = entity_aabb(secondary, entities)
        s_mid = (s_lo + s_hi) / 2.0
        target.look_at = np.array([(look_at[0] + s_mid[0]) / 2.0, look_at[1], (look_at[2] + s_mid[2]) / 2.0])
        target.sight_points = [look_at.copy(), s_mid]
        target.group = True
    return target


# ---------------------------------------------------------------- horizontal placement
def _dir(azimuth_deg: float) -> np.ndarray:
    a = math.radians(azimuth_deg)
    return np.array([math.sin(a), 0.0, math.cos(a)])


def _azimuth_of(vec: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vec[0]), float(vec[2])))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _line_normal(shot: Shot, frame: FrameState) -> Optional[np.ndarray]:
    """Unit normal (xz) of the subject/secondary line on the shot's line side, or None.

    The line is oriented canonically (lexicographically smaller id -> larger id), so a pair keeps the
    same physical side whichever of the two is the subject (matches the validator's unordered pair)."""
    if shot.secondary_id in (NONE, shot.subject_id):
        return None
    a, b = _present(frame, shot.subject_id), _present(frame, shot.secondary_id)
    if a is None or b is None:
        return None
    first, second = (a, b) if a.entity_id < b.entity_id else (b, a)
    axis = np.array(second.position, dtype=float) - np.array(first.position, dtype=float)
    axis[1] = 0.0
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return None
    axis /= norm
    left = np.array([axis[2], 0.0, -axis[0]])        # left of the axis (yaw convention: right = (-fz, fx))
    return left if shot.line_side == "a" else -left


def _holder(frame: FrameState, entity_id: str) -> Optional[EntityFrame]:
    """The character holding a prop, directly or through what it rests on (a cup on a held tray)."""
    ef = _present(frame, entity_id)
    for _ in range(4):
        if ef is None or ef.kind == "character":
            return None
        if ef.support == "held":
            holder = _present(frame, ef.support_id)
            return holder if holder is not None and holder.kind == "character" else None
        if ef.support != "on_entity":
            return None
        ef = _present(frame, ef.support_id)
    return None


def _nominal_azimuth(shot: Shot, frame: FrameState, index: SceneIndex, target: _Target) -> float:
    """Azimuth (from the aim point toward the camera) the shot grammar asks for."""
    normal = _line_normal(shot, frame)
    subject = frame.entities.get(shot.subject_id)
    if normal is not None and subject is not None:
        toward = np.array(frame.entities[shot.secondary_id].position, dtype=float) - np.array(subject.position, dtype=float)
        toward[1] = 0.0
        toward /= max(np.linalg.norm(toward), 1e-9)
        base = _azimuth_of(normal)
        options = [base + SECONDARY_OFFSET_DEG, base - SECONDARY_OFFSET_DEG]
        if target.facing_yaw is not None:
            # Characters: a 3/4-front view of the face when that stays on the shot's side of the line,
            # otherwise the azimuth on the shot's side that sees most of the face (the side's edge).
            three_quarter = [target.facing_yaw + FACE_THREE_QUARTER_DEG, target.facing_yaw - FACE_THREE_QUARTER_DEG]
            on_side = [az for az in three_quarter if abs(_wrap(az - base)) <= LINE_SIDE_LIMIT_DEG]
            if on_side:
                return _wrap(min(on_side, key=lambda az: abs(_wrap(az - base))))
            face = _dir(target.facing_yaw)
            edges = [base + LINE_SIDE_LIMIT_DEG, base - LINE_SIDE_LIMIT_DEG]
            return _wrap(max(edges, key=lambda az: float(_dir(az) @ face)))
        return _wrap(min(options, key=lambda az: float(_dir(az) @ toward)))   # props: both read in frame
    sign = 1.0 if shot.line_side == "a" else -1.0
    if target.facing_yaw is not None:
        return _wrap(target.facing_yaw + sign * FACING_OFFSET_DEG)
    # A prop in someone's hands (or on something they hold, like a cup on a tray): from the holder's
    # front, never from behind the holder.
    holder = _holder(frame, shot.subject_id)
    if holder is not None:
        return _wrap(float(holder.yaw_deg) + sign * FACING_OFFSET_DEG)
    # Other props without a secondary: seen from the side of the nearest person (or the room centre).
    origin = np.array(subject.position if subject is not None else target.look_at, dtype=float)
    people = [np.array(ef.position, dtype=float) for ef in frame.entities.values() if ef.kind == "character" and ef.present]
    if people:
        toward = min(people, key=lambda q: float(np.linalg.norm((q - origin)[[0, 2]])))
    else:
        loc = index.locations.get(frame.location_id)
        toward = np.array([loc.width_m / 2.0, 0.0, loc.depth_m / 2.0]) if loc else origin + np.array([0.0, 0.0, 1.0])
    vec = toward - origin
    vec[1] = 0.0
    if np.linalg.norm(vec) < 1e-6:
        vec = np.array([0.0, 0.0, 1.0])
    return _wrap(_azimuth_of(vec) + sign * FACING_OFFSET_DEG)


def _clamp(position: np.ndarray, frame: FrameState, index: SceneIndex, min_height: float = MIN_CAMERA_HEIGHT_M) -> np.ndarray:
    """Keep the camera above the floor and, in interiors, inside the walls and under the ceiling."""
    out = position.copy()
    out[1] = max(out[1], min_height)
    loc = index.locations.get(frame.location_id)
    if loc is not None and loc.kind == "interior":
        out[0] = min(max(out[0], WALL_MARGIN_M), max(loc.width_m - WALL_MARGIN_M, WALL_MARGIN_M))
        out[2] = min(max(out[2], WALL_MARGIN_M), max(loc.depth_m - WALL_MARGIN_M, WALL_MARGIN_M))
        out[1] = min(out[1], ROOM_HEIGHT_M - CEILING_MARGIN_M)
    return out


def _ray(azimuth: float, pitch: float) -> np.ndarray:
    """Unit vector from the aim point toward a camera at `azimuth` looking with `pitch` degrees."""
    p = math.radians(pitch)
    d = _dir(azimuth)
    return np.array([d[0] * math.cos(p), -math.sin(p), d[2] * math.cos(p)])



def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(max(-c / a, 0.0), 1.0), 0.0
            elif t > 1.0:
                s, t = min(max((b - c) / a, 0.0), 1.0), 1.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def _segment_hits(a: np.ndarray, b: np.ndarray, prim: Primitive) -> bool:
    """Does the segment a->b pass through a primitive (capsules exact, other solids as oriented boxes)?"""
    if prim.kind == "capsule":
        return _segment_distance(a, b, np.array(prim.p0), np.array(prim.p1)) < prim.radius
    rot = np.array(prim.axes, dtype=float).T
    o = (a - np.array(prim.center)) @ rot
    d = (b - a) @ rot
    half = np.array(prim.half)
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(o[k]) > half[k]:
                return False
            continue
        ta, tb = (-half[k] - o[k]) / d[k], (half[k] - o[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return t1 > 1e-6 and t0 < 1.0 - 1e-6


def _inside(point: np.ndarray, prim: Primitive, margin: float) -> bool:
    if prim.kind == "capsule":
        return _segment_distance(point, point, np.array(prim.p0), np.array(prim.p1)) < prim.radius + margin
    local = (point - np.array(prim.center)) @ np.array(prim.axes, dtype=float).T
    return bool(np.all(np.abs(local) <= np.array(prim.half) + margin))


@dataclass
class _Placement:
    """Everything the side search needs to score one frame."""
    frame: FrameState
    target: _Target
    distance: float
    pitch: float
    occluders: List[Primitive]                # may hide the must-see points
    normal: Optional[np.ndarray]
    hfov: float
    vfov: float
    solids: List[Tuple[Primitive, float]] = field(default_factory=list)   # (solid, clearance) the camera keeps out of
    body: List[Primitive] = field(default_factory=list)                   # the subject's own solids
    body_radius: float = 0.0                  # farthest point of `body` from the aim point


def _grips(prim: Primitive, lo: np.ndarray, hi: np.ndarray, gap: float = 0.03) -> bool:
    """Is this holder primitive the hand/forearm (or muzzle) holding the prop with bounds lo..hi?"""
    if not any(k in prim.part for k in GRIP_PARTS):
        return False
    if prim.kind == "capsule":
        a, b = np.array(prim.p0, dtype=float), np.array(prim.p1, dtype=float)
        ab = b - a
        s = min(max(float(((lo + hi) / 2.0 - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0), 1.0)
        point, margin = a + ab * s, prim.radius + gap
    else:
        point, margin = np.array(prim.center, dtype=float), max(prim.half) + gap
    return bool(np.all(point >= lo - margin) and np.all(point <= hi + margin))


def _placement(shot: Shot, frame: FrameState, index: SceneIndex, width: int, height: int,
               distance_scale: float = 1.0, lift: float = 0.0, scene: Optional[BlockingScene] = None) -> _Placement:
    """Framing, distance and obstacles of one frame. `lift` raises the camera by that many degrees."""
    vfov = vertical_fov_deg(shot.lens_mm)
    scene = scene if scene is not None else build_scene(frame, index.plan, index)
    prims = scene.primitives
    target = _framing(shot, frame, index, width / height, scene, distance_scale)
    distance = (target.framed_h / 2.0) / math.tan(math.radians(vfov) / 2.0) * distance_scale
    # 1. What never hides the subject: itself, what it holds, the other half of a two-shot and the parts
    #    of its holder that touch it (hand, forearm). The holder's body still hides it.
    unhidden = {target.subject_id}
    close = {target.subject_id}
    subject = _drawn(scene, frame, target.subject_id)
    holder_id, held_box = None, None
    if subject is not None and subject.present:
        if subject.holding_id != NONE:
            unhidden.add(subject.holding_id)
            close.add(subject.holding_id)
        if subject.support == "held":
            holder_id, held_box = subject.support_id, entity_aabb(subject, scene.entities or frame.entities)
    if target.group:
        unhidden.add(shot.secondary_id)
    # 2. Solids keep the camera out (a smaller gap to the subject and what touches it).
    occluders: List[Primitive] = []
    solids: List[Tuple[Primitive, float]] = []
    for p in prims:
        if not p.occluder or p.alpha < 1.0:
            continue
        touching = p.entity_id == holder_id and _grips(p, *held_box)
        solids.append((p, SUBJECT_CLEARANCE_M if p.entity_id in close or touching else CLEARANCE_M))
        if p.entity_id not in unhidden and not touching:
            occluders.append(p)
    body = [p for p in prims if p.entity_id == target.subject_id and p.occluder and p.alpha >= 1.0]
    radius = max((float(np.linalg.norm(c - target.look_at)) for p in body for c in p.corners()), default=0.0)
    pitch = ANGLE_PITCH_DEG.get(shot.angle, 0.0)
    if lift:
        pitch = max(pitch - lift, ANGLE_PITCH_DEG["overhead"])
    return _Placement(frame, target, distance, pitch, occluders, _line_normal(shot, frame),
                      horizontal_fov_deg(vfov, width, height), vfov, solids, body, radius)


# ---------------------------------------------------------------- locked-off (static) shots
def _union_target(targets: Sequence[_Target], azimuth: float, pitch: float, aspect: float) -> _Target:
    """One framing for a locked-off camera: the smallest frame, seen from `azimuth` looking with `pitch`,
    that holds every keyframe's framing window (orthographic approximation at their mean depth). A
    subject that does not move gives back its own framing."""
    if len(targets) == 1:
        return targets[0]
    right, up, forward = _basis(_wrap(azimuth + 180.0), pitch, 0.0)     # the camera looks back at the aim
    lo, hi, depth = np.full(2, math.inf), np.full(2, -math.inf), []
    for t in targets:
        centre = np.array([float(t.look_at @ right), float(t.look_at @ up)])
        half = np.array([t.framed_h * aspect / 2.0, t.framed_h / 2.0])
        lo, hi = np.minimum(lo, centre - half), np.maximum(hi, centre + half)
        depth.append(float(t.look_at @ forward))
    mid = (lo + hi) / 2.0
    look_at = right * mid[0] + up * mid[1] + forward * (sum(depth) / len(depth))
    framed = max(float(hi[1] - lo[1]), float(hi[0] - lo[0]) / aspect)
    first = targets[0]
    # Fit points belong to one keyframe's window; the union window already holds them.
    return _Target(look_at, framed, first.subject_id, [p.copy() for t in targets for p in t.sight_points],
                   facing_yaw=first.facing_yaw, group=any(t.group for t in targets),
                   min_height=max(t.min_height for t in targets))


def _pin(shot: Shot, frames: Sequence[FrameState], placements: List[_Placement], azimuth: float,
         aspect: float) -> List[_Placement]:
    """The placements of a static shot's keyframes sharing one framing (the union of the subject over
    the keyframes where it is present) and one subject body to keep clear of, so every keyframe gets the
    same camera. Each keyframe keeps its own solids and must-see points (checked against its own
    occluders), so the camera must be clear in all of them."""
    if len(placements) == 1:
        return list(placements)
    present = [_present(f, shot.subject_id) is not None for f in frames]
    if not any(present):
        present = [True] * len(placements)
    keep = [pl for pl, here in zip(placements, present) if here]
    target = _union_target([pl.target for pl in keep], azimuth, placements[0].pitch, aspect)
    distance = (target.framed_h / 2.0) / math.tan(math.radians(placements[0].vfov) / 2.0)
    body = [p for pl in keep for p in pl.body]
    radius = max((float(np.linalg.norm(c - target.look_at)) for p in body for c in p.corners()), default=0.0)
    return [replace(pl, target=replace(target, sight_points=[p.copy() for p in pl.target.sight_points] if here else []),
                    distance=distance, body=body, body_radius=radius)
            for pl, here in zip(placements, present)]


def _clear_distance(pl: _Placement, ray: np.ndarray) -> float:
    """Camera distance along `ray`: the framing distance, or farther when that would put the camera
    inside (or nearer than SUBJECT_CLEARANCE_M to) the subject's own body — a close shot of a big prop
    or an insert of a tiny one frames wider instead of losing the subject."""
    limit = pl.body_radius + 2.0 * SUBJECT_CLEARANCE_M
    t = pl.distance
    while t < limit and any(_inside(pl.target.look_at + ray * t, p, SUBJECT_CLEARANCE_M) for p in pl.body):
        t += SUBJECT_STEP_M
    return t


def _position(pl: _Placement, index: SceneIndex, azimuth: float) -> np.ndarray:
    ray = _ray(azimuth, pl.pitch)
    return _clamp(pl.target.look_at + ray * _clear_distance(pl, ray), pl.frame, index, pl.target.min_height)


def _gap(point: np.ndarray, prim: Primitive) -> float:
    """Distance from a point to a primitive's surface (0 inside; other solids as oriented boxes)."""
    if prim.kind == "capsule":
        return max(_segment_distance(point, point, np.array(prim.p0), np.array(prim.p1)) - prim.radius, 0.0)
    local = (point - np.array(prim.center)) @ np.array(prim.axes, dtype=float).T
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.array(prim.half), 0.0)))


def _placement_cost(pl: _Placement, index: SceneIndex, azimuth: float, rendered: bool = False) -> float:
    """Distance lost to walls, framing widened by the subject's bulk, camera inside or near a solid,
    hidden must-see points, two-shot members out of frame.

    With `rendered`, what the camera sees is measured by rendering instead: hidden must-see points are
    not counted, and only a camera clipping a solid (nearer than SUBJECT_CLEARANCE_M) is ruled out; a
    camera within CLEARANCE_M of one (above a table top, beside a shoulder) costs a little."""
    if pl.normal is not None and abs(_wrap(azimuth - _azimuth_of(pl.normal))) > 85.0:
        return math.inf                                   # never cross the line
    ray = _ray(azimuth, pl.pitch)
    clear = _clear_distance(pl, ray)
    pos = _clamp(pl.target.look_at + ray * clear, pl.frame, index, pl.target.min_height)
    reach = float(np.linalg.norm(pos - pl.target.look_at))
    cost = 4.0 * max(0.0, 1.0 - reach / pl.distance)
    cost += PUSH_OUT_WEIGHT * min((clear - pl.distance) / pl.distance, 2.0)
    if not rendered:
        if any(_inside(pos, p, margin) for p, margin in pl.solids):
            cost += 100.0
        cost += sum(25.0 for pt in pl.target.sight_points if any(_segment_hits(pos, pt, p) for p in pl.occluders))
    else:
        near = 0.0
        for p, margin in pl.solids:
            gap = _gap(pos, p)
            if gap < SUBJECT_CLEARANCE_M:
                near = 100.0
                break
            if gap < margin:
                near = max(near, CLEARANCE_COST * (margin - gap) / (margin - SUBJECT_CLEARANCE_M))
        cost += near
    if pl.target.fit_points:
        # Fit points that cannot all be framed from here (the frame cannot widen).
        angles = np.array([_aim(pos, q) for q in pl.target.fit_points])
        yaw0 = _aim(pos, pl.target.look_at)[0]
        rel = np.array([_wrap(a - yaw0) for a in angles[:, 0]])
        cost += 4.0 * max(0.0, (angles[:, 1].max() - angles[:, 1].min()) / pl.vfov - 0.9)
        cost += 4.0 * max(0.0, (rel.max() - rel.min()) / pl.hfov - 0.9)
    if pl.target.group:
        forward = pl.target.look_at - pos
        forward[1] = 0.0
        forward /= max(np.linalg.norm(forward), 1e-9)
        for pt in pl.target.sight_points:
            to = pt - pos
            to[1] = 0.0
            cos = float(to @ forward) / max(np.linalg.norm(to), 1e-9)
            if math.degrees(math.acos(min(max(cos, -1.0), 1.0))) > 0.9 * pl.hfov / 2.0:
                cost += 20.0
    return cost


def _search_azimuth(nominal: float, placements: Sequence[_Placement], index: SceneIndex, weight: float,
                    rendered: bool = False) -> float:
    """Azimuth nearest `nominal` with the lowest mean placement cost over `placements` (see
    _placement_cost for `rendered`). Fixed scan order (0, +5, -5, +10, ...) and strict improvement make
    it deterministic."""
    best, best_cost = nominal, math.inf
    steps = [0.0] + [s * k * SEARCH_STEP_DEG for k in range(1, int(180.0 / SEARCH_STEP_DEG) + 1) for s in (1.0, -1.0)]
    for offset in steps:
        penalty = weight * abs(offset) / 90.0
        if penalty >= best_cost:
            break                                         # nothing further out can win
        az = _wrap(nominal + offset)
        cost = penalty + sum(_placement_cost(pl, index, az, rendered) for pl in placements) / len(placements)
        if cost < best_cost - 1e-9:
            best, best_cost = az, cost
    return best


def _aim(position: np.ndarray, point: np.ndarray) -> Tuple[float, float]:
    v = point - position
    return math.degrees(math.atan2(float(v[0]), float(v[2]))), math.degrees(math.atan2(float(v[1]), math.hypot(float(v[0]), float(v[2]))))


def _aim_fit(position: np.ndarray, target: _Target) -> Tuple[float, float]:
    """Aim at the look-at point, or at the angular centre of the fit points when the target has them."""
    yaw, pitch = _aim(position, target.look_at)
    if not target.fit_points:
        return yaw, pitch
    angles = np.array([_aim(position, p) for p in target.fit_points])
    rel_yaw = np.array([_wrap(a - yaw) for a in angles[:, 0]])
    return _wrap(yaw + (rel_yaw.min() + rel_yaw.max()) / 2.0), float((angles[:, 1].min() + angles[:, 1].max()) / 2.0)


def _offset_scale(position: np.ndarray, yaw: float, pitch: float, roll: float, d_yaw: float, d_pitch: float,
                  vfov: float, width: int, height: int, points: List[np.ndarray]) -> float:
    """Largest fraction (in 1/8 steps) of a pan/tilt offset that keeps the must-see points that start
    inside the safe frame inside it."""
    def inside(f: float) -> np.ndarray:
        cam = CameraState(position=tuple(position), yaw_deg=yaw + d_yaw * f, pitch_deg=pitch + d_pitch * f,
                          roll_deg=roll, lens_mm=0.0, vfov_deg=vfov, width=width, height=height, look_at=(0.0, 0.0, 0.0))
        uv, depth = project(cam, np.array(points))
        with np.errstate(invalid="ignore"):
            return (depth > NEAR_M) & (np.abs(uv[:, 0] - width / 2.0) <= SAFE_FRAME * width) & (
                np.abs(uv[:, 1] - height / 2.0) <= SAFE_FRAME * height)

    protect = inside(0.0)
    for k in range(8, 0, -1):
        if inside(k / 8.0)[protect].all():
            return k / 8.0
    return 0.0


CameraPath = Callable[[float], Tuple[np.ndarray, float, float]]   # move fraction -> (position, yaw, pitch)


def _move_fraction(pl: _Placement, path: CameraPath, roll: float, width: int, height: int) -> float:
    """Largest fraction (in 1/MOVE_STEPS steps along the path) of a move that displaces the camera
    (orbit, crane, truck, handheld) before it would come near (or clip) a solid it started clear of,
    hide a must-see point that was in view, or push one that started inside the safe frame out of it."""
    points = np.array(pl.target.sight_points + pl.target.fit_points)

    def state(f: float) -> Tuple[np.ndarray, ...]:
        pos, yaw, pitch = path(f)
        cam = CameraState(position=tuple(pos), yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, lens_mm=0.0,
                          vfov_deg=pl.vfov, width=width, height=height, look_at=(0.0, 0.0, 0.0))
        uv, depth = project(cam, points)
        with np.errstate(invalid="ignore"):
            framed = (depth > NEAR_M) & (np.abs(uv[:, 0] - width / 2.0) <= SAFE_FRAME * width) & (
                np.abs(uv[:, 1] - height / 2.0) <= SAFE_FRAME * height)
        seen = np.array([not any(_segment_hits(pos, q, o) for o in pl.occluders) for q in points], dtype=bool)
        gaps = np.array([_gap(pos, p) for p, _ in pl.solids])
        near = gaps < np.array([margin for _, margin in pl.solids])
        return framed, seen, near, gaps < SUBJECT_CLEARANCE_M

    framed0, seen0, near0, clip0 = state(0.0)
    for k in range(1, MOVE_STEPS + 1):
        framed, seen, near, clip = state(k / MOVE_STEPS)
        if (not framed[framed0].all() or not seen[seen0].all() or (near & ~near0).any()
                or (clip & ~clip0).any()):
            return (k - 1) / MOVE_STEPS
    return 1.0


def _jitter(kf_id: str) -> Tuple[np.ndarray, float, float, float]:
    rng = random.Random(int(hashlib.sha256(f"handheld|{kf_id}".encode()).hexdigest()[:16], 16))
    pos = np.array([rng.uniform(-1, 1) for _ in range(3)]) * HANDHELD_POS_M
    return pos, *(rng.uniform(-1, 1) * HANDHELD_ANGLE_DEG for _ in range(3))


# ---------------------------------------------------------------- visibility in the loop
def _vis_size(cam: CameraState) -> Tuple[int, int]:
    return VIS_WIDTH_PX, max(8, int(round(VIS_WIDTH_PX * cam.height / max(cam.width, 1))))


def visible_fractions(scene: BlockingScene, cam: CameraState, entity_ids: Sequence[str],
                      key_parts_only: bool = True) -> Dict[str, Optional[float]]:
    """Fraction of each entity's in-frame pixels that is not hidden, from a low-res id render of the
    scene (None when the entity covers too few pixels to measure). With `key_parts_only`, characters'
    legs, feet and hips do not count (a body behind a table or a counter still reads), but they still
    hide what is behind them."""
    from src.previs.blocking.box_renderer import BoxRenderer   # lazy: box_renderer imports this module

    width, height = _vis_size(cam)
    chars = {e for e in entity_ids if e in scene.entities and scene.entities[e].kind == "character"} if key_parts_only else set()
    # Only the id buffer matters: no environment (the camera stays inside the room), shadows or overlays.
    prims = [replace(p, casts_shadow=False,
                     entity_id=_LOWER_BODY_ID if p.entity_id in chars and any(k in p.part for k in LOWER_BODY_PARTS)
                     else p.entity_id)
             for p in scene.primitives if p.alpha >= 1.0]
    scene = replace(scene, primitives=prims, environment=None)
    rast = BoxRenderer(width, height).rasterize(scene, replace(cam, width=width, height=height), grid=False)
    out: Dict[str, Optional[float]] = {}
    for eid in entity_ids:
        alone = rast.alone.get(eid)
        count = int(alone.sum()) if alone is not None else 0
        visible = int((rast.ids == scene.id_of(eid)).sum()) if eid in scene.entity_order else 0
        out[eid] = visible / count if count >= MIN_VIS_PX else None
    return out


def _face_angle(ef: EntityFrame, cam_position: np.ndarray, entities: Optional[Dict[str, EntityFrame]] = None) -> Optional[float]:
    """Horizontal angle (deg) between a character's face direction (as drawn: seated, quadruped ...) and
    the direction to the camera; None when the face points up or down (lying) or it has no head."""
    parts = {part: (np.array(p0, dtype=float), np.array(p1, dtype=float))
             for p0, p1, _, part in _segments(ef, entities or {})}
    if "head" not in parts:
        return None
    head = (parts["head"][0] + parts["head"][1]) / 2.0
    face = parts["nose"][1] - head if "nose" in parts else _dir(ef.yaw_deg)
    if np.linalg.norm(face[[0, 2]]) < 0.3 * np.linalg.norm(face):
        return None
    to_cam = np.asarray(cam_position, dtype=float) - head
    a, b = face[[0, 2]], to_cam[[0, 2]]
    if np.linalg.norm(b) < 1e-6:
        return None
    cos = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(min(max(cos, -1.0), 1.0)))


@dataclass
class _View:
    """How well one camera shows a keyframe's subject (from the low-res id render)."""
    deficit: float                    # 0 = subject fully visible (and secondary seen, face toward camera)
    hidden: bool                      # subject under MIN_VISIBLE, secondary mostly hidden, or a face turned away
    clean: bool                       # subject at least GOOD_VISIBLE and nothing hidden


def _view(shot: Shot, frame: FrameState, scene: BlockingScene, cam: CameraState) -> _View:
    subject = _drawn(scene, frame, shot.subject_id)
    if subject is None or not subject.present:
        return _View(0.0, False, True)
    entities = scene.entities or frame.entities
    secondary = _present(frame, shot.secondary_id) if shot.secondary_id not in (NONE, shot.subject_id) else None
    if subject.support == "held":                 # the hand holding the subject does not hide it
        lo, hi = entity_aabb(subject, entities)
        scene = replace(scene, primitives=[p for p in scene.primitives
                                           if not (p.entity_id == subject.support_id and _grips(p, lo, hi))])
    vis = visible_fractions(scene, cam, [shot.subject_id] + ([secondary.entity_id] if secondary is not None else []))
    seen = vis[shot.subject_id]
    deficit = 0.0 if seen is None else 1.0 - seen
    hidden = seen is not None and seen < MIN_VISIBLE
    clean = seen is None or seen >= GOOD_VISIBLE
    other = vis[secondary.entity_id] if secondary is not None else None
    if other is not None and other < SECONDARY_MIN_VISIBLE:
        deficit += SECONDARY_WEIGHT * (SECONDARY_MIN_VISIBLE - other) / SECONDARY_MIN_VISIBLE
        hidden = True
    if subject.kind == "character" and shot.size not in GROUP_SIZES:
        angle = _face_angle(subject, np.array(cam.position), entities)
        away = 0.0 if angle is None else min(max((angle - FACE_SIDE_DEG) / (FACE_BACK_DEG - FACE_SIDE_DEG), 0.0), 1.0)
        if away > 0.0:
            deficit += FACE_WEIGHT * away
            hidden = True
    return _View(deficit, hidden, clean and not hidden)


@dataclass(frozen=True)
class _ShotSide:
    """Where a shot is filmed from: (azimuth, lift) for the whole shot, plus per-keyframe adjustments
    (within KEYFRAME_SPREAD_DEG, never across the line) where a keyframe's own render showed its subject
    partly hidden. A static shot has one `pinned` camera for all its frames instead."""
    azimuth: float
    lift: float = 0.0
    keyframes: Tuple[Tuple[str, float, float], ...] = ()
    pinned: Optional[CameraState] = field(default=None, compare=False)

    def for_keyframe(self, kf_id: str) -> Tuple[float, float]:
        return next(((az, lift) for k, az, lift in self.keyframes if k == kf_id), (self.azimuth, self.lift))


def _is_locked_off(shot: Shot) -> bool:
    """A static shot: one camera for all its keyframes (and every animatic frame between them)."""
    return shot.move == "static"


def _static_camera(shot: Shot, frames: Sequence[FrameState], index: SceneIndex, width: int, height: int,
                   side: Tuple[float, float], scenes: Optional[Sequence[BlockingScene]] = None
                   ) -> Tuple[CameraState, List[_Placement], float]:
    """(camera, the keyframes' pinned placements, azimuth used) of a static shot filmed from `side`:
    one camera framing the union of the subject over the keyframes, clear of solids in all of them."""
    scenes = list(scenes) if scenes is not None else [build_scene(f, index.plan, index) for f in frames]
    placements = [_placement(shot, f, index, width, height, lift=side[1], scene=s) for f, s in zip(frames, scenes)]
    placements = _pin(shot, frames, placements, side[0], width / height)
    cam, _, azimuth = _camera_for_side(shot, frames[0], index, width, height, frames, 0.0, side,
                                       placements=placements)
    return cam, placements, azimuth


def _lifts(shot: Shot, base: float, steps: int) -> List[float]:
    """Camera lifts to try: the current one, then up to `steps` steps higher (not past an overhead view)."""
    pitch = ANGLE_PITCH_DEG.get(shot.angle, 0.0)
    return [base + k * LIFT_DEG for k in range(steps + 1)
            if k == 0 or pitch - (base + k * LIFT_DEG) >= ANGLE_PITCH_DEG["overhead"]]


def _scan_steps(limit: float) -> List[float]:
    """Azimuth offsets 0, +step, -step, +2 step, ... up to +-limit (+180 and -180 counted once)."""
    steps = [0.0] + [s * k * VIS_STEP_DEG for k in range(1, int(limit / VIS_STEP_DEG) + 1) for s in (1.0, -1.0)]
    return [o for o in steps if o > -180.0]


def _on_side(shot: Shot, frame: FrameState, azimuth: float) -> bool:
    normal = _line_normal(shot, frame)
    return normal is None or abs(_wrap(azimuth - _azimuth_of(normal))) <= 85.0


def _visible_side(shot: Shot, frames: Sequence[FrameState], index: SceneIndex, width: int, height: int,
                  nominal: float, side: float, scenes: Optional[Sequence[BlockingScene]] = None) -> _ShotSide:
    """Check the clearance-searched `side` against low-res renders of the shot's keyframes.

    1. If a keyframe shows its subject (or secondary) hidden or a character facing away (for a moving
       camera: any keyframe whose subject is not clearly visible), candidates on the shot's side of the
       line, nearest the nominal first and then higher, are scored by visibility deficit (dominant; the
       worst keyframe counts as much as the average), clearance cost and deviation from the nominal.
    2. A moving camera may then adjust a keyframe whose subject is still partly hidden, within
       KEYFRAME_SPREAD_DEG of the shot's side. A static shot keeps one camera (`pinned`), scored as such.
    Deterministic: fixed candidate order, strict improvement."""
    scenes = list(scenes) if scenes is not None else [build_scene(f, index.plan, index) for f in frames]
    progress = [_move_progress(f, index) for f in frames]
    pinned: Dict[Tuple[float, float], CameraState] = {}

    def views(candidate: Tuple[float, float]) -> Tuple[List[_View], float]:
        if _is_locked_off(shot):
            cam, pls, azimuth = _static_camera(shot, frames, index, width, height, candidate, scenes)
            pinned[candidate] = cam
            return ([_view(shot, f, scene, cam) for f, scene in zip(frames, scenes)],
                    sum(_placement_cost(pl, index, azimuth, rendered=True) for pl in pls) / len(pls))
        out, clearance = [], 0.0
        for f, scene, p in zip(frames, scenes, progress):
            cam, pl, azimuth = _camera_for_side(shot, f, index, width, height, frames, p, candidate, scene)
            out.append(_view(shot, f, scene, cam))
            clearance += _placement_cost(pl, index, azimuth, rendered=True)
        return out, clearance / len(frames)

    def score(candidate: Tuple[float, float]) -> Tuple[float, List[_View]]:
        seen, clearance = views(candidate)
        deficits = [v.deficit for v in seen]
        deficit = (sum(deficits) / len(deficits) + max(deficits)) / 2.0
        deviation = SHOT_DEVIATION_WEIGHT * abs(_wrap(candidate[0] - nominal)) / 90.0
        return VISIBILITY_WEIGHT * deficit + clearance + deviation + LIFT_COST * candidate[1] / LIFT_DEG, seen

    # 1. The shot's side. A static camera settles for a subject that is not hidden; a moving one looks
    #    for a side that shows every keyframe well before adjusting single keyframes.
    def satisfied(seen: List[_View]) -> bool:
        return not any(v.hidden for v in seen) and (shot.move == "static" or all(v.clean for v in seen))

    best = (side, 0.0)
    best_cost, best_views = score(best)
    if not satisfied(best_views):
        for lift in _lifts(shot, 0.0, MAX_LIFT_STEPS):
            for offset in _scan_steps(180.0):
                if SHOT_DEVIATION_WEIGHT * abs(offset) / 90.0 + LIFT_COST * lift / LIFT_DEG >= best_cost:
                    break                                 # nothing further out can win
                azimuth = _wrap(nominal + offset)
                if not _on_side(shot, frames[0], azimuth):
                    continue                              # never cross the line
                cost, seen = score((azimuth, lift))
                if cost < best_cost - 1e-9:
                    best, best_cost, best_views = (azimuth, lift), cost, seen
            if satisfied(best_views):
                break                                     # seen without raising the camera (further)

    # 2. Per-keyframe adjustments of a moving camera.
    adjusted = []
    for f, scene, p, view in zip(frames, scenes, progress, best_views):
        if shot.move == "static" or view.clean:
            continue

        def frame_cost(candidate: Tuple[float, float]) -> float:
            cam, pl, azimuth = _camera_for_side(shot, f, index, width, height, frames, p, candidate, scene)
            return (VISIBILITY_WEIGHT * _view(shot, f, scene, cam).deficit + _placement_cost(pl, index, azimuth, rendered=True)
                    + FRAME_DEVIATION_WEIGHT * abs(_wrap(candidate[0] - best[0])) / 90.0
                    + LIFT_COST * (candidate[1] - best[1]) / LIFT_DEG)

        own, own_cost = best, frame_cost(best)
        for lift in _lifts(shot, best[1], 1):
            for offset in _scan_steps(KEYFRAME_SPREAD_DEG):
                if FRAME_DEVIATION_WEIGHT * abs(offset) / 90.0 + LIFT_COST * (lift - best[1]) / LIFT_DEG >= own_cost:
                    break
                candidate = (_wrap(best[0] + offset), lift)
                if candidate == best or not _on_side(shot, f, candidate[0]):
                    continue
                cost = frame_cost(candidate)
                if cost < own_cost - 1e-9:
                    own, own_cost = candidate, cost
        if own != best:
            adjusted.append((f.kf_id, *own))
    return _ShotSide(best[0], best[1], tuple(adjusted), pinned.get(best))


# ---------------------------------------------------------------- shot side (cached)
_SOLVE_CACHE: List[Tuple[ScenePlan, Dict[str, List[FrameState]]]] = []
_SIDE_CACHE: Dict[Tuple, _ShotSide] = {}


def shot_frames(solved: SolvedPlan) -> Dict[str, List[FrameState]]:
    """Keyframes of each shot in edited order."""
    out: Dict[str, List[FrameState]] = {}
    for f in sorted(solved.frames, key=lambda f: (f.t_s, f.order_in_shot)):
        out.setdefault(f.shot_id, []).append(f)
    return out


def _plan_shot_frames(shot: Shot, index: SceneIndex) -> List[FrameState]:
    """The shot's keyframes as the physics solver places them (solved once per plan and cached)."""
    plan = index.plan
    for cached_plan, frames in _SOLVE_CACHE:
        if cached_plan is plan:
            return frames.get(shot.id, [])
    from src.previs import physics

    try:
        frames = shot_frames(physics.solve(plan))
    except (ValueError, RuntimeError, KeyError, ZeroDivisionError):
        frames = {}
    _SOLVE_CACHE.insert(0, (plan, frames))
    del _SOLVE_CACHE[4:]
    return frames.get(shot.id, [])


def _frame_key(f: FrameState) -> Tuple:
    """Everything about a keyframe that changes what a camera sees (cache key)."""
    return (f.kf_id, f.t_s, tuple(sorted(
        (e, ef.present, ef.position, ef.yaw_deg, ef.rotation_deg, ef.pose, ef.weight_cue, ef.support, ef.support_id,
         ef.holding_id, tuple(ef.flags), tuple(ef.secondary), ef.size) for e, ef in f.entities.items())))


def _shot_side(shot: Shot, frames: Sequence[FrameState], index: SceneIndex, width: int, height: int) -> _ShotSide:
    """The shot's camera side: nominal on its first keyframe, searched for clearance over all its
    keyframes, then checked (and if need be re-chosen) against the keyframes' low-res renders."""
    key = (id(index.plan), shot.model_dump_json(), width, height, tuple(_frame_key(f) for f in frames))
    if key not in _SIDE_CACHE:
        if len(_SIDE_CACHE) > 256:
            _SIDE_CACHE.clear()
        scenes = [build_scene(f, index.plan, index) for f in frames]
        placements = [_placement(shot, f, index, width, height, scene=s) for f, s in zip(frames, scenes)]
        nominal = _nominal_azimuth(shot, frames[0], index, placements[0].target)
        if _is_locked_off(shot):
            placements = _pin(shot, frames, placements, nominal, width / height)
        side = _search_azimuth(nominal, placements, index, SHOT_DEVIATION_WEIGHT)
        _SIDE_CACHE[key] = _visible_side(shot, frames, index, width, height, nominal, side, scenes)
    return _SIDE_CACHE[key]


# ---------------------------------------------------------------- solve
def _dolly_scale(move: str, p: float) -> float:
    return {"dolly_in": 1.0 - DOLLY_IN * p, "dolly_out": 1.0 + DOLLY_OUT * p}.get(move, 1.0)


def solve_camera(shot: Shot, frame: FrameState, index: SceneIndex, width: int, height: int,
                 frames: Optional[Sequence[FrameState]] = None, move_progress: Optional[float] = None) -> CameraState:
    """
    Camera for one frame of a shot (a keyframe or any sampled animatic frame).

    `frames` are the shot's keyframes in order (solve_cameras passes them; otherwise they are derived
    from the plan). They fix the side of the scene the shot is filmed from, so every frame of a shot
    shares it (a moving camera may adjust it per keyframe, by `frame.kf_id`, where that keyframe's
    render showed its subject partly hidden); distance, height and aim follow the subject in `frame`.
    A static shot is locked off: every frame gets the same camera, framing the union of the subject
    over the keyframes. `move_progress` defaults to the keyframe's own value.
    """
    p = _move_progress(frame, index) if move_progress is None else min(max(float(move_progress), 0.0), 1.0)
    frames = [f for f in (frames if frames is not None else _plan_shot_frames(shot, index)) if f.shot_id == shot.id]
    shot_side = _shot_side(shot, frames or [frame], index, width, height)
    if shot_side.pinned is not None:
        return replace(shot_side.pinned)
    side = shot_side.for_keyframe(frame.kf_id)
    return _camera_for_side(shot, frame, index, width, height, frames, p, side)[0]


def _camera_for_side(shot: Shot, frame: FrameState, index: SceneIndex, width: int, height: int,
                     frames: Sequence[FrameState], p: float, side: Tuple[float, float],
                     scene: Optional[BlockingScene] = None, placements: Optional[Sequence[_Placement]] = None
                     ) -> Tuple[CameraState, _Placement, float]:
    """(camera, its placement, the azimuth used) for one frame filmed from the shot's `side`.
    `placements` (a static shot's pinned keyframes, see _static_camera) replace the frame's own: the
    camera then frames their shared target and keeps clear of the solids of all of them."""
    # 1. Optics and framing for this frame (a dolly scales the distance, the side may lift the camera).
    vfov = vertical_fov_deg(shot.lens_mm)
    if not placements:
        placements = [_placement(shot, frame, index, width, height, distance_scale=_dolly_scale(shot.move, p),
                                 lift=side[1], scene=scene)]
    pl = placements[0]
    target = pl.target
    roll = DUTCH_ROLL_DEG if shot.angle == "dutch" else 0.0

    # 2. Side of the scene: chosen for the whole shot (where the keyframes' renders showed the subject),
    #    re-checked for clearance (walls, solids) on this frame (on every keyframe of a static shot).
    azimuth = _search_azimuth(side[0], placements, index, FRAME_DEVIATION_WEIGHT, rendered=True)

    # 3. Position (clamped inside interiors), orbit / crane (re-aimed at the subject), aim. Both stop
    #    short of solids and of hiding the subject.
    position = _position(pl, index, azimuth)
    orbit = {"orbit_left": -ORBIT_DEG, "orbit_right": ORBIT_DEG}.get(shot.move, 0.0) * p
    crane = {"crane_up": CRANE_M, "crane_down": -CRANE_M}.get(shot.move, 0.0) * p
    if orbit or crane:
        def displaced(f: float) -> Tuple[np.ndarray, float, float]:
            pos = _position(pl, index, azimuth + orbit * f) if orbit else position
            pos = _clamp(pos + UP * crane * f, frame, index, target.min_height) if crane else pos
            return (pos, *_aim_fit(pos, target))

        position = displaced(_move_fraction(pl, displaced, roll, width, height))[0]
    yaw, pitch = _aim_fit(position, target)

    # 4. Composition: subject on a third, then pans / tilts. A subject that moves during the shot is
    #    followed (the follow is the pan/tilt); on a still subject the move sweeps by its nominal angle,
    #    limited so the subject's must-see points stay inside the frame.
    third = math.degrees(math.atan(math.tan(math.radians(horizontal_fov_deg(vfov, width, height)) / 2.0) / 3.0))
    yaw += {"left": -third, "right": third}.get(shot.frame_side, 0.0)
    d_yaw = {"pan_left": PAN_DEG * p, "pan_right": -PAN_DEG * p}.get(shot.move, 0.0)
    d_pitch = {"tilt_up": TILT_DEG * p, "tilt_down": -TILT_DEG * p}.get(shot.move, 0.0)
    if d_yaw or d_pitch:
        first = frames[0] if frames else frame
        start = _placement(shot, first, index, width, height).target if first is not frame else target
        moved = float(np.linalg.norm(target.look_at - start.look_at)) > FOLLOW_THRESHOLD * target.framed_h
        scale = 0.0 if moved else _offset_scale(position, yaw, pitch, roll, d_yaw, d_pitch, vfov, width, height,
                                                target.sight_points + target.fit_points)
        yaw, pitch = yaw + d_yaw * scale, pitch + d_pitch * scale

    # 5. Trucking slides the camera sideways without re-aiming (only as far as the subject stays in the
    #    safe frame and nothing solid is entered); handheld adds a small seeded shake.
    if shot.move in ("truck_left", "truck_right"):
        right, _, _ = _basis(yaw, pitch, 0.0)
        slide = right * TRUCK_M * p * (1.0 if shot.move == "truck_right" else -1.0)
        start_pos, aim = position, (yaw, pitch)
        trucked = lambda f: (_clamp(start_pos + slide * f, frame, index, target.min_height), *aim)
        position = trucked(_move_fraction(pl, trucked, roll, width, height))[0]
    if shot.move == "handheld":
        d_pos, j_yaw, j_pitch, j_roll = _jitter(frame.kf_id)
        start_pos, aim = position, (yaw, pitch)
        shaken = lambda f: (_clamp(start_pos + d_pos * f, frame, index, target.min_height), *aim)
        position = shaken(_move_fraction(pl, shaken, roll, width, height))[0]
        yaw, pitch, roll = yaw + j_yaw, pitch + j_pitch, roll + j_roll

    yaw, pitch = _wrap(yaw), min(max(pitch, -89.0), 89.0)
    look_at = position + _forward(yaw, pitch) * float(np.linalg.norm(target.look_at - position))
    cam = CameraState(
        position=tuple(round(float(v), 6) for v in position), yaw_deg=round(yaw, 6), pitch_deg=round(pitch, 6),
        roll_deg=round(roll, 6), lens_mm=float(shot.lens_mm), vfov_deg=round(vfov, 6), width=int(width),
        height=int(height), look_at=tuple(round(float(v), 6) for v in look_at),
    )
    return cam, pl, azimuth


def solve_cameras(plan: ScenePlan, solved: SolvedPlan, index: SceneIndex, width: int, height: int) -> None:
    """Fill `frame.camera` for every solved keyframe."""
    by_shot = shot_frames(solved)
    for frame in solved.frames:
        frame.camera = solve_camera(index.shots[frame.shot_id], frame, index, width, height,
                                    frames=by_shot.get(frame.shot_id))
