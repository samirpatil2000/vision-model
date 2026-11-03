"""
Gray-box blocking: the scene description shared by every blocking backend.

build_scene(frame, plan, index) turns one solved FrameState into primitives: mannequin capsules for
characters (a four-legged proxy when Character.body == "quadruped"; a sitting character on the top of
its supporting entity), solids for props (shards instead of the intact shape once a prop is shattered,
a liquid level inside containers), secondary-motion proxies (liquid arc, droplets, splash crown, puddle, shards,
debris, dust) and the location environment (floor, walls, ceiling, window on the key-light wall).
The box ray-caster (box_renderer.py) and the Blender backend (blender_builder.py) draw exactly these
primitives, so their id maps agree; write_result() turns either backend's buffers into the files and
the BlockingResult every later stage reads.

Pixel id convention: entity at position k of `index.entities` (characters, then props, in plan order)
has id k + 1 in id.png; 0 is environment/background. The mapping is the same for every keyframe.
"""
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from src.previs.blocking import mannequin
from src.previs.schema import ScenePlan
from src.previs.state import SceneIndex, event_mover
from src.previs.types import BlockingResult, CameraState, EntityFrame, FrameState, Vec3

RGB = Tuple[float, float, float]
Axes = Tuple[Vec3, Vec3, Vec3]
IDENTITY: Axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

ROOM_HEIGHT_M = 2.7
GRID_M = 0.5                      # floor / wall grid spacing
FILL_LEVEL = {"fill_full": 0.85, "fill_half": 0.5, "fill_empty": 0.0}
VISIBLE_CAP = 0.5                 # visible_frac denominator never exceeds this fraction of the frame
TRANSIENT_SECONDARY = frozenset({"liquid_arc", "droplets", "splash", "dust"})
SKIN = (0.74, 0.63, 0.55)
WATER = (0.29, 0.56, 0.78)
DUST = (0.85, 0.82, 0.77)
TIME_OF_DAY = {               # (brightness, tint)
    "day": (1.0, (1.0, 1.0, 1.0)), "dawn": (0.88, (1.0, 0.95, 0.9)),
    "dusk": (0.82, (1.0, 0.9, 0.82)), "night": (0.55, (0.8, 0.86, 1.0)),
}
# Direction TOWARD the key light (world, y up). "window_left" is a window in the x = 0 wall.
KEY_LIGHT_DIR = {
    "window_left": (-0.75, 0.6, -0.15), "window_right": (0.75, 0.6, -0.15), "overhead": (0.1, 1.0, 0.1),
    "practical": (0.3, 0.8, -0.4), "sun_high": (0.3, 0.9, -0.3), "sun_low": (-0.75, 0.35, -0.4),
    "moon": (0.4, 0.7, -0.5),
}


@dataclass
class Primitive:
    """One solid (or translucent overlay) in world space. Cylinders run along their local y axis."""
    kind: str                         # "capsule" | "box" | "cylinder" | "ellipsoid"
    entity_id: str
    color: RGB                        # sRGB in 0..1
    center: Vec3 = (0.0, 0.0, 0.0)
    axes: Axes = IDENTITY             # local x, y, z axes in world coordinates (orthonormal)
    half: Vec3 = (0.0, 0.0, 0.0)      # half extents along the local axes
    p0: Vec3 = (0.0, 0.0, 0.0)        # capsule ends and radius
    p1: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 0.0
    part: str = ""
    alpha: float = 1.0                # < 1: translucent overlay (dust), never in the id or depth map
    casts_shadow: bool = True
    occluder: bool = True             # considered by the camera's line-of-sight checks
    smooth: bool = False              # curved surface (Blender shades it smooth)

    def corners(self) -> np.ndarray:
        """8 corners of a box bounding the primitive (world)."""
        if self.kind == "capsule":
            p0, p1, r = np.array(self.p0), np.array(self.p1), self.radius
            lo, hi = np.minimum(p0, p1) - r, np.maximum(p0, p1) + r
            return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        rot = np.array(self.axes, dtype=float).T          # columns = local axes
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        return np.array(self.center) + (signs * np.array(self.half)) @ rot.T

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.corners()
        return c.min(axis=0), c.max(axis=0)


@dataclass
class Environment:
    kind: str                         # "interior" | "exterior"
    width: float
    depth: float
    height: float
    floor_color: RGB
    wall_color: RGB
    ceiling_color: RGB
    sky_color: RGB
    window: Optional[Tuple[str, float, float, float, float]] = None   # (wall "x0"|"x1", z0, z1, y0, y1)


@dataclass
class Lighting:
    direction: Vec3                   # unit vector toward the key light
    brightness: float
    tint: RGB
    ambient: float = 0.42


@dataclass
class BlockingScene:
    kf_id: str
    shot_id: str
    entity_order: List[str]           # id value = position + 1
    primitives: List[Primitive]
    environment: Optional[Environment]
    lighting: Lighting
    entities: Dict[str, EntityFrame] = field(default_factory=dict)

    def id_map(self) -> Dict[int, str]:
        return {k + 1: eid for k, eid in enumerate(self.entity_order)}

    def id_of(self, entity_id: str) -> int:
        return self.entity_order.index(entity_id) + 1


# ---------------------------------------------------------------- colour helpers
def hex_to_rgb(value: str, default: RGB = (0.5, 0.5, 0.5)) -> RGB:
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return default
    try:
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return default


def _mix(a: RGB, b: RGB, f: float) -> RGB:
    return tuple(float(x + (y - x) * f) for x, y in zip(a, b))


def _scale(a: RGB, f: float) -> RGB:
    return tuple(float(min(max(x * f, 0.0), 1.0)) for x in a)


def liquid_color(container: RGB) -> RGB:
    return _mix(container, WATER, 0.6)


# ---------------------------------------------------------------- geometry helpers
def _rot_euler(deg: Vec3) -> np.ndarray:
    """Extrinsic X, then Y, then Z rotation (degrees)."""
    rx, ry, rz = (math.radians(d) for d in deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _yaw_matrix(yaw_deg: float) -> np.ndarray:
    """Rotation taking local +z to the yaw direction (positive yaw turns +z toward +x)."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _axes_from(rot: np.ndarray) -> Axes:
    return tuple(tuple(float(v) for v in rot[:, k]) for k in range(3))


def prop_rotation(ef: EntityFrame) -> np.ndarray:
    """World rotation of a prop: yaw, then the airborne tumble (columns = local axes)."""
    return _rot_euler(ef.rotation_deg) @ _yaw_matrix(ef.yaw_deg)


def prop_center(ef: EntityFrame) -> np.ndarray:
    """Centre of a prop's body: half its height above the base point (the tumble pivots here)."""
    return np.array(ef.position, dtype=float) + np.array([0.0, ef.size[1] / 2.0, 0.0])


def entity_aabb(ef: EntityFrame, entities: Optional[Dict[str, EntityFrame]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """World axis-aligned bounds of an entity's body (mannequin for characters, rotated solid for props).

    `entities` (the frame's entities) lets a sitting character be posed on its seat (see seat_height)."""
    if ef.kind == "character":
        pts = []
        for p0, p1, r, _ in mannequin.segments(ef, seat_y=seat_height(ef, entities) if entities else None):
            pts += [np.array(p0) - r, np.array(p0) + r, np.array(p1) - r, np.array(p1) + r]
        pts = np.array(pts)
        return pts.min(axis=0), pts.max(axis=0)
    half = np.array(ef.size, dtype=float) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = prop_center(ef) + (signs * half) @ prop_rotation(ef).T
    return corners.min(axis=0), corners.max(axis=0)


def seat_height(ef: EntityFrame, entities: Dict[str, EntityFrame]) -> Optional[float]:
    """World height of the seat under a sitting character: the top of the entity supporting it, else None."""
    if ef.kind != "character" or ef.pose != "sit" or ef.support != "on_entity":
        return None
    seat = entities.get(ef.support_id)
    if seat is None or not seat.present or seat.kind == "character":
        return None
    return float(entity_aabb(seat)[1][1])


def surface_height_below(entities: Dict[str, EntityFrame], entity_id: str, x: float, z: float, below_y: float) -> float:
    """Height of the highest prop top under (x, z) at or below `below_y` (0 = floor)."""
    best = 0.0
    for eid, ef in entities.items():
        if eid == entity_id or not ef.present or ef.kind == "character" or "shattered" in ef.flags:
            continue
        lo, hi = entity_aabb(ef)
        if lo[0] <= x <= hi[0] and lo[2] <= z <= hi[2] and hi[1] <= below_y + 0.02:
            best = max(best, float(hi[1]))
    return best


def _rng(*key: str) -> random.Random:
    digest = hashlib.sha256("|".join(key).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


# ---------------------------------------------------------------- entity primitives
CHARACTER_PART_COLOR = {"head": "skin", "nose": "skin", "hips": "bottom", "foot_l": "feet", "foot_r": "feet",
                        "hair": "hair", "eye_l": "eye", "eye_r": "eye"}
# Quadrupeds: coat in the palette's first colour, lower legs and paws in the second, dark nose and eyes.
QUADRUPED_PART_COLOR = {"nose": "eye", "eye_l": "eye", "eye_r": "eye", "ear_l": "ear", "ear_r": "ear"}
HAIR = (0.24, 0.18, 0.14)
EYE = (0.08, 0.07, 0.07)
FACE_DETAIL_PARTS = frozenset({"nose", "hair", "eye_l", "eye_r", "ear_l", "ear_r"})   # drawn, but not occluders


def _part_role(part: str, quadruped: bool) -> str:
    if quadruped:
        return QUADRUPED_PART_COLOR.get(part) or ("bottom" if part.startswith(("lower_leg", "paw")) else "top")
    return CHARACTER_PART_COLOR.get(part) or ("bottom" if "leg" in part else "top")


def _character_primitives(ef: EntityFrame, seat_y: Optional[float] = None) -> List[Primitive]:
    top, bottom = hex_to_rgb(ef.color_hex), hex_to_rgb(ef.secondary_color_hex)
    palette = {"top": top, "bottom": bottom, "skin": SKIN, "feet": _scale(bottom, 0.55), "hair": HAIR, "eye": EYE,
               "ear": _scale(top, 0.8)}
    quadruped = mannequin.is_quadruped(ef)
    out = []
    for p0, p1, r, part in mannequin.segments(ef, seat_y=seat_y):
        role = _part_role(part, quadruped)
        out.append(Primitive("capsule", ef.entity_id, palette[role], p0=p0, p1=p1, radius=r, part=part,
                             smooth=True, occluder=part not in FACE_DETAIL_PARTS))
    return out


def _prop_primitives(ef: EntityFrame) -> List[Primitive]:
    color = hex_to_rgb(ef.color_hex)
    rot = prop_rotation(ef)
    axes = _axes_from(rot)
    center = prop_center(ef)
    sx, sy, sz = (max(float(v), 0.004) for v in ef.size)
    kind = {"box": "box", "plane": "box", "cylinder": "cylinder", "sphere": "ellipsoid"}.get(ef.shape, "box")
    fill = next((FILL_LEVEL[f] for f in reversed(ef.flags) if f in FILL_LEVEL), 0.0)
    is_container = fill > 0.0 and kind in ("box", "cylinder")
    if not is_container:
        return [Primitive(kind, ef.entity_id, color, center=tuple(center), axes=axes, half=(sx / 2, sy / 2, sz / 2),
                          part="body", smooth=kind != "box")]
    # Container: the liquid band at the bottom, the empty part of the vessel above it.
    up = rot[:, 1]
    liquid_h, rest_h = sy * fill, sy * (1.0 - fill)
    bottom = center - up * sy / 2.0
    out = [Primitive(kind, ef.entity_id, liquid_color(color), center=tuple(bottom + up * liquid_h / 2.0), axes=axes,
                     half=(sx / 2, liquid_h / 2, sz / 2), part="liquid", smooth=kind != "box")]
    if rest_h > 1e-4:
        out.append(Primitive(kind, ef.entity_id, color, center=tuple(bottom + up * (liquid_h + rest_h / 2.0)), axes=axes,
                             half=(sx / 2, rest_h / 2, sz / 2), part="body", smooth=kind != "box"))
    return out


def _scale_of(ef: EntityFrame) -> float:
    """Proxy scale relative to a 7 cm drinking glass, clamped to a sane range."""
    return min(max(max(ef.size[0], ef.size[2]) / 0.07, 0.6), 4.0)


def _mouth(ef: EntityFrame) -> np.ndarray:
    return prop_center(ef) + prop_rotation(ef)[:, 1] * ef.size[1] / 2.0


def _liquid_arc(ef: EntityFrame, color: RGB) -> List[Primitive]:
    """Tapered stream out of the vessel's mouth.

    A vessel in free fall carries its liquid with it, so what has already left the mouth drifts away
    along the mouth axis and lags behind the vessel's horizontal motion (the solved trajectory
    direction); a vessel at rest or in a hand pours a ballistic stream that falls from the spout."""
    k = _scale_of(ef)
    mouth = _mouth(ef)
    axis = prop_rotation(ef)[:, 1]
    up = np.array([0.0, 1.0, 0.0])
    travel = np.array(ef.velocity, dtype=float)
    if len(ef.trajectory) >= 2:
        travel = np.array(ef.trajectory[-1], dtype=float) - np.array(ef.trajectory[0], dtype=float)
    travel[1] = 0.0
    travel = travel / np.linalg.norm(travel) if np.linalg.norm(travel) > 1e-6 else np.zeros(3)
    if ef.support == "airborne" or ef.in_motion:
        ctrl = mouth + axis * 0.12 * k
        tail = mouth + axis * 0.15 * k - travel * 0.1 * k + up * 0.07 * k
    else:
        spout = axis - up * float(axis @ up)
        spout = spout / np.linalg.norm(spout) if np.linalg.norm(spout) > 1e-6 else travel
        ctrl = mouth + spout * 0.1 * k + up * 0.03 * k
        tail = mouth + spout * 0.14 * k - up * 0.25 * k
    pts = [(1 - t) ** 2 * mouth + 2 * (1 - t) * t * ctrl + t * t * tail for t in np.linspace(0.0, 1.0, 9)]
    out = []
    n = len(pts) - 1
    for i in range(n):
        r = 0.018 * (1.0 - 0.65 * i / n) * k ** 0.5
        out.append(Primitive("capsule", ef.entity_id, color, p0=tuple(pts[i]), p1=tuple(pts[i + 1]), radius=r,
                             part="liquid_arc", casts_shadow=False, occluder=False, smooth=True))
    return out


def _droplets(ef: EntityFrame, color: RGB, kf_id: str) -> List[Primitive]:
    rng = _rng(kf_id, ef.entity_id, "droplets")
    mouth = _mouth(ef) + prop_rotation(ef)[:, 1] * 0.08 * _scale_of(ef)
    k = _scale_of(ef)
    out = []
    for _ in range(10):
        offset = np.array([rng.uniform(-1, 1), rng.uniform(-0.6, 1.0), rng.uniform(-1, 1)]) * 0.1 * k
        r = rng.uniform(0.006, 0.012) * k ** 0.5
        out.append(Primitive("ellipsoid", ef.entity_id, color, center=tuple(mouth + offset), half=(r, r, r),
                             part="droplets", casts_shadow=False, occluder=False, smooth=True))
    return out


def _ground_point(ef: EntityFrame, entities: Dict[str, EntityFrame]) -> np.ndarray:
    x, y, z = ef.position
    return np.array([x, surface_height_below(entities, ef.entity_id, x, z, y + 0.05), z], dtype=float)


def _puddle(ef: EntityFrame, color: RGB, ground: np.ndarray) -> List[Primitive]:
    rng = _rng(ef.entity_id, "puddle")
    radius = min(max(0.2 * _scale_of(ef), 0.12), 0.6)
    out = [Primitive("cylinder", ef.entity_id, color, center=tuple(ground + [0, 0.0015, 0]),
                     half=(radius, 0.0015, radius * 0.85), part="puddle", casts_shadow=False, occluder=False)]
    for _ in range(2):
        ang = rng.uniform(0, 2 * math.pi)
        r = radius * rng.uniform(0.45, 0.65)
        c = ground + np.array([math.cos(ang) * radius * 0.55, 0.0016, math.sin(ang) * radius * 0.55])
        out.append(Primitive("cylinder", ef.entity_id, color, center=tuple(c), half=(r, 0.0015, r * 0.8),
                             part="puddle", casts_shadow=False, occluder=False))
    return out


def _splash(ef: EntityFrame, color: RGB, ground: np.ndarray, kf_id: str) -> List[Primitive]:
    rng = _rng(kf_id, ef.entity_id, "splash")
    k = _scale_of(ef)
    out = []
    for i in range(16):
        ang = 2 * math.pi * i / 16 + rng.uniform(-0.12, 0.12)
        d = np.array([math.cos(ang), 0.0, math.sin(ang)])
        p0 = ground + d * 0.06 * k + np.array([0, 0.004, 0])
        p1 = ground + d * rng.uniform(0.09, 0.12) * k + np.array([0, rng.uniform(0.035, 0.06) * k, 0])
        out.append(Primitive("capsule", ef.entity_id, color, p0=tuple(p0), p1=tuple(p1), radius=0.005 * k ** 0.5,
                             part="splash", casts_shadow=False, occluder=False, smooth=True))
        tip = p1 + d * 0.015 * k + np.array([0, 0.02 * k, 0])
        r = 0.006 * k ** 0.5
        out.append(Primitive("ellipsoid", ef.entity_id, color, center=tuple(tip), half=(r, r, r), part="splash",
                             casts_shadow=False, occluder=False, smooth=True))
    return out


def _fragments(ef: EntityFrame, color: RGB, ground: np.ndarray, part: str, count: int, spread: float,
               size: Tuple[float, float], thickness: float) -> List[Primitive]:
    """Scattered small boxes (shards / debris). Persistent: seeded by entity only, so they never move."""
    rng = _rng(ef.entity_id, part)
    k = _scale_of(ef)
    out = []
    for _ in range(count):
        dist = spread * k * math.sqrt(rng.uniform(0.02, 1.0))
        ang = rng.uniform(0, 2 * math.pi)
        half = (rng.uniform(*size) * k, thickness, rng.uniform(*size) * k * 0.7)
        rot = _rot_euler((rng.uniform(-35, 35), rng.uniform(0, 360), rng.uniform(-35, 35)))
        lift = max(abs(rot[1, 0]) * half[0], abs(rot[1, 2]) * half[2]) + thickness
        c = ground + np.array([math.cos(ang) * dist, lift, math.sin(ang) * dist])
        out.append(Primitive("box", ef.entity_id, color, center=tuple(c), axes=_axes_from(rot), half=half,
                             part=part, casts_shadow=False, occluder=False))
    return out


def _dust(ef: EntityFrame, ground: np.ndarray, kf_id: str) -> List[Primitive]:
    rng = _rng(kf_id, ef.entity_id, "dust")
    k = _scale_of(ef)
    out = []
    for _ in range(5):
        r = rng.uniform(0.12, 0.25) * k ** 0.5
        c = ground + np.array([rng.uniform(-0.25, 0.25) * k, rng.uniform(0.05, 0.3), rng.uniform(-0.25, 0.25) * k])
        out.append(Primitive("ellipsoid", ef.entity_id, DUST, center=tuple(c), half=(r, r * 0.7, r), part="dust",
                             alpha=0.25, casts_shadow=False, occluder=False, smooth=True))
    return out


def _transient_active(ef: EntityFrame, frame: FrameState, index: SceneIndex) -> bool:
    """Splashes, arcs, droplets and dust only exist while the entity is moving or an event acts on it."""
    if ef.in_motion:
        return True
    return any(eid in index.events and event_mover(index.events[eid]) == ef.entity_id
               for eid in frame.active_event_ids)


def _secondary_primitives(ef: EntityFrame, frame: FrameState, index: SceneIndex) -> List[Primitive]:
    color = hex_to_rgb(ef.color_hex)
    liquid = liquid_color(color)
    wanted = set(ef.secondary)
    if not _transient_active(ef, frame, index):
        wanted -= TRANSIENT_SECONDARY
    if "shattered" in ef.flags:
        wanted.add("shards")
    if "spilled" in ef.flags:
        wanted.add("puddle")
    ground = _ground_point(ef, frame.entities)
    out: List[Primitive] = []
    if "puddle" in wanted:
        out += _puddle(ef, liquid, ground)
    if "shards" in wanted:
        out += _fragments(ef, _mix(color, (1.0, 1.0, 1.0), 0.25), ground, "shards", 16, 0.25, (0.012, 0.03), 0.002)
    if "debris" in wanted:
        out += _fragments(ef, _scale(color, 0.8), ground, "debris", 10, 0.3, (0.01, 0.035), 0.008)
    if "splash" in wanted:
        out += _splash(ef, _mix(liquid, (1.0, 1.0, 1.0), 0.3), ground, frame.kf_id)
    if "liquid_arc" in wanted:
        out += _liquid_arc(ef, _mix(liquid, (1.0, 1.0, 1.0), 0.2))
    if "droplets" in wanted:
        out += _droplets(ef, _mix(liquid, (1.0, 1.0, 1.0), 0.3), frame.kf_id)
    if "dust" in wanted:
        out += _dust(ef, ground, frame.kf_id)
    return out


# ---------------------------------------------------------------- scene
def _environment(frame: FrameState, index: SceneIndex) -> Optional[Environment]:
    loc = index.locations.get(frame.location_id)
    if loc is None:
        return None
    floor = hex_to_rgb(loc.palette_hex[0] if loc.palette_hex else "", (0.6, 0.58, 0.55))
    wall = hex_to_rgb(loc.palette_hex[1] if len(loc.palette_hex) > 1 else "", (0.85, 0.84, 0.8))
    width, depth = max(float(loc.width_m), 0.5), max(float(loc.depth_m), 0.5)
    window = None
    if loc.kind == "interior" and loc.key_light in ("window_left", "window_right"):
        zc, half = depth / 2.0, min(0.6, depth / 3.0)
        window = ("x0" if loc.key_light == "window_left" else "x1", zc - half, zc + half, 0.9, 2.1)
    return Environment(kind=loc.kind, width=width, depth=depth, height=ROOM_HEIGHT_M, floor_color=floor,
                       wall_color=wall, ceiling_color=_mix(wall, (1.0, 1.0, 1.0), 0.35),
                       sky_color=_mix(wall, (0.62, 0.74, 0.88), 0.5), window=window)


def _lighting(frame: FrameState, index: SceneIndex) -> Lighting:
    loc = index.locations.get(frame.location_id)
    key = loc.key_light if loc is not None else "overhead"
    d = np.array(KEY_LIGHT_DIR.get(key, KEY_LIGHT_DIR["overhead"]), dtype=float)
    brightness, tint = TIME_OF_DAY.get(index.plan.bible.scene.time_of_day, TIME_OF_DAY["day"])
    if key == "moon":
        brightness *= 0.8
    return Lighting(direction=tuple(float(v) for v in d / np.linalg.norm(d)), brightness=brightness, tint=tint)


def _with_body(ef: EntityFrame, index: SceneIndex) -> EntityFrame:
    """The entity frame with shape "quadruped" when the bible's Character.body says so."""
    character = index.characters.get(ef.entity_id)
    if character is not None and character.body == mannequin.QUADRUPED and not mannequin.is_quadruped(ef):
        return replace(ef, shape=mannequin.QUADRUPED)
    return ef


def build_scene(frame: FrameState, plan: ScenePlan, index: SceneIndex) -> BlockingScene:
    """All blocking primitives for one keyframe (only present entities are drawn)."""
    primitives: List[Primitive] = []
    entities = {eid: _with_body(ef, index) for eid, ef in frame.entities.items()}
    for eid in index.entities:
        ef = entities.get(eid)
        if ef is None or not ef.present:
            continue
        if ef.kind == "character":
            primitives += _character_primitives(ef, seat_height(ef, entities))
        else:
            if "shattered" not in ef.flags:
                primitives += _prop_primitives(ef)
            primitives += _secondary_primitives(ef, frame, index)
    return BlockingScene(kf_id=frame.kf_id, shot_id=frame.shot_id, entity_order=list(index.entities),
                         primitives=primitives, environment=_environment(frame, index),
                         lighting=_lighting(frame, index), entities=entities)


# ---------------------------------------------------------------- outputs
def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def in_frame_fraction(cam: CameraState, corners: np.ndarray, width: int, height: int) -> float:
    """Fraction of an entity's projected bounding box inside the image (1 = fully in frame)."""
    from src.previs.camera import NEAR_M, project

    uv, depth = project(cam, corners)
    front = depth > NEAR_M
    if not front.any():
        return 0.0
    uv = uv[front]
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    area = max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6)
    ix = max(min(x1, width) - max(x0, 0.0), 0.0)
    iy = max(min(y1, height) - max(y0, 0.0), 0.0)
    return float(min(ix * iy / area, 1.0) * front.mean())


def screen_velocity(cam: CameraState, ef: EntityFrame, dt: float = 1.0 / 24.0) -> Tuple[float, float]:
    """Projected velocity (px/s) of the entity's body centre."""
    from src.previs.camera import NEAR_M, project

    lo, hi = entity_aabb(ef)
    c = (lo + hi) / 2.0
    pts = np.array([c, c + np.array(ef.velocity, dtype=float) * dt])
    uv, depth = project(cam, pts)
    if (depth <= NEAR_M).any():
        return (0.0, 0.0)
    v = (uv[1] - uv[0]) / dt
    return (round(float(v[0]), 3), round(float(v[1]), 3))


def write_result(out_dir: Path, scene: BlockingScene, cam: CameraState, sharp: np.ndarray, ids: np.ndarray,
                 depth: np.ndarray, alone: Dict[str, int], blur_px: float, renderer: str,
                 init: Optional[np.ndarray] = None) -> BlockingResult:
    """Write blocking.png / blocking_sharp.png / id.png / depth.npy / meta.json and build the result.

    blocking_sharp.png is the human-facing view; blocking.png is the init image for the image model
    (`init` when given, e.g. without the reference grid, else the sharp view), blurred by blur_px."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = ids.shape
    sharp_img = Image.fromarray(sharp, mode="RGB")
    sharp_img.save(out_dir / "blocking_sharp.png")
    init_img = Image.fromarray(init, mode="RGB") if init is not None else sharp_img
    blurred = init_img.filter(ImageFilter.GaussianBlur(radius=blur_px)) if blur_px > 0 else init_img
    blurred.save(out_dir / "blocking.png")
    Image.fromarray(ids.astype(np.uint8), mode="L").save(out_dir / "id.png")
    np.save(out_dir / "depth.npy", depth.astype(np.float32))

    id_map = scene.id_map()
    bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    visible: Dict[str, float] = {}
    velocity: Dict[str, Tuple[float, float]] = {}
    for value, eid in id_map.items():
        mask = ids == value
        count = int(mask.sum())
        box = _bbox(mask)
        if box is not None:
            bboxes[eid] = box
        ef = scene.entities.get(eid)
        drawn = [p for p in scene.primitives if p.entity_id == eid and p.alpha >= 1.0]
        if ef is None or not ef.present or not drawn or alone.get(eid, 0) == 0:
            visible[eid] = 0.0
            continue
        # Pixels when drawn alone on an unbounded image plane (off-frame parts count as not visible),
        # capped at half the frame: a subject that fills the frame is fully visible.
        corners = np.concatenate([p.corners() for p in drawn])
        frac = in_frame_fraction(cam, corners, width, height)
        unbounded = alone[eid] / max(frac, 1e-6)
        visible[eid] = round(min(count / min(unbounded, VISIBLE_CAP * width * height), 1.0), 4)
        velocity[eid] = screen_velocity(cam, ef)

    meta = {
        "kf_id": scene.kf_id, "shot_id": scene.shot_id, "renderer": renderer, "width": width, "height": height,
        "blur_px": blur_px, "camera": asdict(cam), "id_map": {str(k): v for k, v in id_map.items()},
        "bboxes": bboxes, "visible_frac": visible, "screen_velocity": velocity,
        "depth_range_m": [float(np.min(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else None,
                          float(np.max(depth[np.isfinite(depth)])) if np.isfinite(depth).any() else None],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return BlockingResult(
        blocking_path=str(out_dir / "blocking.png"), id_path=str(out_dir / "id.png"),
        depth_path=str(out_dir / "depth.npy"), id_map=id_map, bboxes=bboxes, visible_frac=visible,
        screen_velocity=velocity,
    )

