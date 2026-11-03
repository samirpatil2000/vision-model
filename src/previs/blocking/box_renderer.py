"""
Gray-box blocking renderer in pure numpy.

Every pixel's ray is intersected analytically with the scene primitives (capsules, oriented boxes,
elliptic cylinders, ellipsoids) and the location environment, keeping the nearest hit in a depth
buffer. That is the far-to-near painter's result without its failure cases (props resting on props,
interpenetrating mannequin limbs) and it yields exact id and depth maps for free.

Shading: Lambert from the location's key light plus ambient; floor/walls get a faint 0.5 m grid and a
bright window panel on the key-light wall. Shadows are soft vertical contact shadows on the highest
upward-facing surface under each caster: tight and dark for grounded entities, wider and fainter
with height, so an airborne prop shows a DETACHED shadow on the surface below it.
"""
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.previs import camera as camera_mod
from src.previs.blocking import GRID_M, BlockingScene, Environment, Lighting, Primitive, build_scene, write_result
from src.previs.schema import ScenePlan
from src.previs.state import SceneIndex
from src.previs.types import BlockingResult, CameraState, FrameState

NEAR = camera_mod.NEAR_M
BACKGROUND = (0.18, 0.18, 0.2)
EXTERIOR_MARGIN_M = 40.0           # ground plane margin around an exterior location
GRID_DARKEN = 0.9
WINDOW_COLOR = (0.94, 0.96, 1.0)
UP_NORMAL = 0.7                     # receivers: surfaces whose normal is within ~45 deg of up
SHADOW_STRENGTH = {"character": 0.42, "prop": 0.55}


def _smoothstep(edge0: float, edge1: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- ray / primitive intersections
# Rays are O + t*D with D = x*r + y*u + f (unnormalised), so t is the camera-space depth directly.
def _local(o: np.ndarray, d: np.ndarray, prim: Primitive) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = np.array(prim.axes, dtype=float).T           # columns = local axes
    return (o - np.array(prim.center)) @ rot, d @ rot, rot


def _hit_box(o: np.ndarray, d: np.ndarray, prim: Primitive) -> Tuple[np.ndarray, np.ndarray]:
    lo, ld, rot = _local(o, d, prim)
    half = np.array(prim.half)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / ld
        t1, t2 = (-half - lo) * inv, (half - lo) * inv
    tmin, tmax = np.fmin(t1, t2), np.fmax(t1, t2)
    t_near, t_far = np.nanmax(tmin, axis=1), np.nanmin(tmax, axis=1)
    hit = (t_far >= t_near) & (t_near > NEAR)
    axis = np.nanargmax(np.where(np.isnan(tmin), -np.inf, tmin), axis=1)
    rows = np.arange(len(ld))
    n_local = np.zeros_like(ld)
    n_local[rows, axis] = -np.sign(ld[rows, axis])
    return np.where(hit, t_near, np.inf), n_local @ rot.T


def _hit_cylinder(o: np.ndarray, d: np.ndarray, prim: Primitive) -> Tuple[np.ndarray, np.ndarray]:
    lo, ld, rot = _local(o, d, prim)
    hx, hy, hz = prim.half
    ox, oy, oz = lo[0] / hx, lo[1], lo[2] / hz
    dx, dy, dz = ld[:, 0] / hx, ld[:, 1], ld[:, 2] / hz
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - 1.0
    disc = b * b - 4.0 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t_side = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
        side_ok = (disc >= 0) & (a > 1e-12) & (t_side > NEAR) & (np.abs(oy + t_side * dy) <= hy)
        t_side = np.where(side_ok, t_side, np.inf)
        best_cap = np.full(len(ld), np.inf)
        for cap_y in (hy, -hy):
            t_cap = (cap_y - oy) / dy
            px, pz = ox + t_cap * dx, oz + t_cap * dz
            ok = (np.abs(dy) > 1e-12) & (t_cap > NEAR) & (px * px + pz * pz <= 1.0)
            best_cap = np.where(ok & (t_cap < best_cap), t_cap, best_cap)
    t = np.minimum(t_side, best_cap)
    p = lo + ld * np.where(np.isfinite(t), t, 0.0)[:, None]
    n_side = np.stack([p[:, 0] / (hx * hx), np.zeros(len(p)), p[:, 2] / (hz * hz)], axis=1)
    n_cap = np.stack([np.zeros(len(p)), np.sign(p[:, 1]), np.zeros(len(p))], axis=1)
    n_local = np.where((t_side <= best_cap)[:, None], n_side, n_cap)
    return t, n_local @ rot.T


def _hit_ellipsoid(o: np.ndarray, d: np.ndarray, prim: Primitive) -> Tuple[np.ndarray, np.ndarray]:
    lo, ld, rot = _local(o, d, prim)
    half = np.array(prim.half)
    so, sd = lo / half, ld / half
    a = np.einsum("ij,ij->i", sd, sd)
    b = 2.0 * (sd @ so)
    c = float(so @ so) - 1.0
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
    t = np.where((disc >= 0) & (t > NEAR), t, np.inf)
    p = so + sd * np.where(np.isfinite(t), t, 0.0)[:, None]
    return t, (p / half) @ rot.T


def _hit_capsule(o: np.ndarray, d: np.ndarray, prim: Primitive) -> Tuple[np.ndarray, np.ndarray]:
    pa, pb, r = np.array(prim.p0), np.array(prim.p1), prim.radius
    ba, oa = pb - pa, o - pa
    baba, baoa, oaoa = float(ba @ ba), float(ba @ oa), float(oa @ oa)
    bard, rdoa, rdrd = d @ ba, d @ oa, np.einsum("ij,ij->i", d, d)
    a = baba * rdrd - bard * bard
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa * baoa - r * r * baba
    h = b * b - a * c
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_body = (-b - np.sqrt(np.maximum(h, 0.0))) / a
        y = baoa + t_body * bard
        body = (h >= 0) & (a > 1e-12) & (y > 0) & (y < baba) & (t_body > NEAR)
        t = np.where(body, t_body, t)
        # Caps: the sphere at whichever end the cylinder test fell beyond.
        for end, sel in ((pa, ~body), (pb, ~body)):
            oc = o - end
            bb = d @ oc
            cc = float(oc @ oc) - r * r
            hh = bb * bb - rdrd * cc
            t_cap = (-bb - np.sqrt(np.maximum(hh, 0.0))) / rdrd
            ok = sel & (hh >= 0) & (t_cap > NEAR) & (t_cap < t)
            t = np.where(ok, t_cap, t)
    p = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
    s = np.clip(((p - pa) @ ba) / max(baba, 1e-12), 0.0, 1.0)
    n = p - (pa + s[:, None] * ba)
    return t, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


HITTERS = {"box": _hit_box, "cylinder": _hit_cylinder, "ellipsoid": _hit_ellipsoid, "capsule": _hit_capsule}


# ---------------------------------------------------------------- raster
class _Raster:
    """Per-pixel buffers for one camera: colour, depth, entity id, surface normal y."""

    def __init__(self, cam: CameraState, width: int, height: int):
        self.cam, self.width, self.height = cam, width, height
        r, u, f = camera_mod.camera_basis(cam)
        fpx = (height / 2.0) / np.tan(np.radians(cam.vfov_deg) / 2.0)
        xs = (np.arange(width) + 0.5 - width / 2.0) / fpx
        ys = -(np.arange(height) + 0.5 - height / 2.0) / fpx
        self.rays = xs[None, :, None] * r + ys[:, None, None] * u + f          # [H, W, 3]
        self.origin = np.asarray(cam.position, dtype=float)
        self.fpx = fpx
        self.depth = np.full((height, width), np.inf)
        self.color = np.empty((height, width, 3))
        self.color[:] = BACKGROUND
        self.ids = np.zeros((height, width), dtype=np.uint8)
        self.normal_y = np.zeros((height, width))
        self.alone: Dict[str, np.ndarray] = {}

    def screen_box(self, corners: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        uv, depth = camera_mod.project(replace(self.cam, width=self.width, height=self.height), corners)
        if (depth <= NEAR).all():
            return None
        if (depth <= NEAR).any():
            return 0, 0, self.width, self.height
        x0, y0 = np.floor(uv.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.width), min(y1, self.height)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _shade(color, normal: np.ndarray, rays: np.ndarray, light: Lighting) -> Tuple[np.ndarray, np.ndarray]:
    """(colours, normal y) with Lambert + ambient + a little sky fill; normals flipped to face the camera."""
    flip = np.einsum("ij,ij->i", normal, rays) > 0
    n = np.where(flip[:, None], -normal, normal)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    lambert = np.clip(n @ np.array(light.direction), 0.0, 1.0)
    shade = light.ambient + (1.0 - light.ambient - 0.08) * lambert + 0.08 * (0.5 + 0.5 * n[:, 1])
    tint = np.array(light.tint) * light.brightness
    return np.clip(np.asarray(color)[None, :] * shade[:, None] * tint[None, :], 0.0, 1.0), n[:, 1]


def _draw_primitive(rast: _Raster, prim: Primitive, entity_value: int, light: Lighting) -> None:
    box = rast.screen_box(prim.corners())
    if box is None:
        return
    x0, y0, x1, y1 = box
    rays = rast.rays[y0:y1, x0:x1].reshape(-1, 3)
    t, normal = HITTERS[prim.kind](rast.origin, rays, prim)
    hit = np.isfinite(t)
    if not hit.any():
        return
    t2d, hit2d = t.reshape(y1 - y0, x1 - x0), hit.reshape(y1 - y0, x1 - x0)
    depth_view = rast.depth[y0:y1, x0:x1]
    if prim.alpha < 1.0:                                    # translucent overlay: blend, no id/depth
        front = hit2d & (t2d < depth_view)
        if front.any():
            col, _ = _shade(prim.color, normal[front.ravel()], rays[front.ravel()], light)
            view = rast.color[y0:y1, x0:x1]
            view[front] = view[front] * (1.0 - prim.alpha) + col * prim.alpha
        return
    if prim.entity_id:
        mask = rast.alone.setdefault(prim.entity_id, np.zeros((rast.height, rast.width), dtype=bool))
        mask[y0:y1, x0:x1] |= hit2d
    front = hit2d & (t2d < depth_view)
    if not front.any():
        return
    sel = front.ravel()
    col, ny = _shade(prim.color, normal[sel], rays[sel], light)
    rast.color[y0:y1, x0:x1][front] = col
    rast.depth[y0:y1, x0:x1][front] = t2d[front]
    rast.ids[y0:y1, x0:x1][front] = entity_value
    rast.normal_y[y0:y1, x0:x1][front] = ny


# ---------------------------------------------------------------- environment
def _planes(env: Environment) -> List[Tuple[np.ndarray, float, str]]:
    """(inward normal, offset, name) for the room's planes: n . p = offset."""
    planes = [(np.array([0.0, 1.0, 0.0]), 0.0, "floor")]
    if env.kind == "interior":
        planes += [
            (np.array([0.0, -1.0, 0.0]), -env.height, "ceiling"),
            (np.array([1.0, 0.0, 0.0]), 0.0, "x0"), (np.array([-1.0, 0.0, 0.0]), -env.width, "x1"),
            (np.array([0.0, 0.0, 1.0]), 0.0, "z0"), (np.array([0.0, 0.0, -1.0]), -env.depth, "z1"),
        ]
    return planes


def _grid_lines(a: np.ndarray, b: np.ndarray, width: np.ndarray) -> np.ndarray:
    da = np.abs(a - np.round(a / GRID_M) * GRID_M)
    db = np.abs(b - np.round(b / GRID_M) * GRID_M)
    return (da < width) | (db < width)


def _draw_environment(rast: _Raster, env: Optional[Environment], light: Lighting, grid: bool = True) -> None:
    """Floor (and for interiors walls + ceiling) with the optional grid and the key-light window; sky outdoors."""
    if env is None:
        return
    rays = rast.rays.reshape(-1, 3)
    o = rast.origin
    if env.kind == "exterior":
        sky_t = np.clip(rays[:, 1] / np.linalg.norm(rays, axis=1) * 2.5, 0.0, 1.0)
        horizon = np.array(env.sky_color) * 0.85 + 0.15
        sky = horizon[None, :] * (1.0 - sky_t[:, None]) + np.array(env.sky_color)[None, :] * sky_t[:, None]
        rast.color[:] = (sky * light.brightness * np.array(light.tint)).reshape(rast.height, rast.width, 3)
    for normal, offset, name in _planes(env):
        denom = rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (offset - float(o @ normal)) / denom
        p = o + rays * np.where(np.isfinite(t), t, 0.0)[:, None]
        ok = (denom < -1e-9) & (t > NEAR)
        margin = EXTERIOR_MARGIN_M if env.kind == "exterior" else 1e-6
        if name in ("floor", "ceiling"):
            ok &= (p[:, 0] >= -margin) & (p[:, 0] <= env.width + margin) & (p[:, 2] >= -margin) & (p[:, 2] <= env.depth + margin)
        elif name in ("x0", "x1"):
            ok &= (p[:, 2] >= 0) & (p[:, 2] <= env.depth) & (p[:, 1] >= 0) & (p[:, 1] <= env.height)
        else:
            ok &= (p[:, 0] >= 0) & (p[:, 0] <= env.width) & (p[:, 1] >= 0) & (p[:, 1] <= env.height)
        depth = rast.depth.reshape(-1)
        ok &= t < depth
        if not ok.any():
            continue
        base = {"floor": env.floor_color, "ceiling": env.ceiling_color}.get(name, env.wall_color)
        col, ny = _shade(base, np.repeat(normal[None, :], int(ok.sum()), axis=0), rays[ok], light)
        q = p[ok]
        line_w = np.maximum(0.006, t[ok] / rast.fpx * 0.7)
        inside = (q[:, 0] >= 0) & (q[:, 0] <= env.width) & (q[:, 2] >= 0) & (q[:, 2] <= env.depth)
        if name in ("floor", "ceiling"):
            lines = _grid_lines(q[:, 0], q[:, 2], line_w) & inside
        elif name in ("x0", "x1"):
            lines = _grid_lines(q[:, 2], q[:, 1], line_w)
        else:
            lines = _grid_lines(q[:, 0], q[:, 1], line_w)
        if grid:
            col[lines] *= GRID_DARKEN
        if env.window is not None and name == env.window[0]:
            _, z0, z1, wy0, wy1 = env.window
            pane = (q[:, 2] >= z0) & (q[:, 2] <= z1) & (q[:, 1] >= wy0) & (q[:, 1] <= wy1)
            col[pane] = np.array(WINDOW_COLOR) * min(1.0, light.brightness * 1.1)
        rast.color.reshape(-1, 3)[ok] = col
        depth[ok] = t[ok]
        rast.ids.reshape(-1)[ok] = 0
        rast.normal_y.reshape(-1)[ok] = ny


# ---------------------------------------------------------------- shadows
def _caster_footprint(prim: Primitive, px: np.ndarray, pz: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """Shadow coverage 0..1 of one caster at receiver points (soft edges widen with the height gap)."""
    if prim.kind == "capsule":
        a, b = np.array(prim.p0)[[0, 2]], np.array(prim.p1)[[0, 2]]
        ab = b - a
        denom = float(ab @ ab)
        s = np.clip(((px - a[0]) * ab[0] + (pz - a[1]) * ab[1]) / denom, 0.0, 1.0) if denom > 1e-12 else np.zeros_like(px)
        dist = np.hypot(px - (a[0] + s * ab[0]), pz - (a[1] + s * ab[1]))
        radius = prim.radius + 0.03 + 0.12 * gap
        return 1.0 - _smoothstep(0.0, radius, dist - 0.5 * prim.radius)
    lo, hi = prim.bounds()
    cx, cz = (lo[0] + hi[0]) / 2.0, (lo[2] + hi[2]) / 2.0
    rx, rz = max((hi[0] - lo[0]) / 2.0, 0.01), max((hi[2] - lo[2]) / 2.0, 0.01)
    soft = 0.03 + 0.12 * gap
    ex, ez = np.abs(px - cx) / (rx + soft), np.abs(pz - cz) / (rz + soft)
    power = 2.0 if prim.kind in ("cylinder", "ellipsoid") else 6.0
    d = (ex ** power + ez ** power) ** (1.0 / power)
    inner = np.minimum(rx, rz) / (np.minimum(rx, rz) + soft)
    return 1.0 - _smoothstep(0.0, np.ones_like(d), (d - inner * 0.7) / np.maximum(1.0 - inner * 0.7, 1e-6))


def _apply_shadows(rast: _Raster, scene: BlockingScene) -> None:
    """Darken upward-facing pixels under each caster (strongest at contact, fading with height)."""
    receivers = (rast.normal_y > UP_NORMAL) & np.isfinite(rast.depth)
    if not receivers.any():
        return
    idx = np.nonzero(receivers.reshape(-1))[0]
    pts = rast.origin + rast.rays.reshape(-1, 3)[idx] * rast.depth.reshape(-1)[idx][:, None]
    owner = rast.ids.reshape(-1)[idx]
    light = np.ones(len(idx))
    id_of = {eid: k + 1 for k, eid in enumerate(scene.entity_order)}
    casters = [p for p in scene.primitives if p.casts_shadow and p.alpha >= 1.0 and p.entity_id]
    # Horizontal surfaces of props that can catch a shadow before it reaches what is below them.
    blockers = [p for p in scene.primitives if p.kind != "capsule" and p.alpha >= 1.0 and p.entity_id
                and p.part in ("body", "liquid")]
    blocker_bounds = [(p.entity_id, *p.bounds()) for p in blockers]
    for eid in dict.fromkeys(p.entity_id for p in casters):
        mine = [p for p in casters if p.entity_id == eid]
        ef = scene.entities.get(eid)
        strength = SHADOW_STRENGTH["character" if ef is not None and ef.kind == "character" else "prop"]
        cover = np.zeros(len(idx))
        for prim in mine:
            lo, hi = prim.bounds()
            reach = 0.05 + 0.12 * max(float(lo[1]), 0.0) + 0.1
            near = ((pts[:, 0] > lo[0] - reach) & (pts[:, 0] < hi[0] + reach) & (pts[:, 2] > lo[2] - reach)
                    & (pts[:, 2] < hi[2] + reach) & (pts[:, 1] <= lo[1] + 0.03) & (owner != id_of.get(eid, -1)))
            if not near.any():
                continue
            sel = np.nonzero(near)[0]
            gap = np.maximum(float(lo[1]) - pts[sel, 1], 0.0)
            cov = _caster_footprint(prim, pts[sel, 0], pts[sel, 2], gap) * strength / (1.0 + 1.2 * gap)
            # Only the highest surface under the caster receives its shadow.
            for owner_id, b_lo, b_hi in blocker_bounds:
                if owner_id == eid:
                    continue
                blocked = ((pts[sel, 0] >= b_lo[0]) & (pts[sel, 0] <= b_hi[0]) & (pts[sel, 2] >= b_lo[2])
                           & (pts[sel, 2] <= b_hi[2]) & (b_hi[1] > pts[sel, 1] + 0.03) & (b_hi[1] <= lo[1] + 0.03))
                cov = np.where(blocked, 0.0, cov)
            cover[sel] = np.maximum(cover[sel], cov)
        light *= 1.0 - np.clip(cover, 0.0, 0.85)
    rast.color.reshape(-1, 3)[idx] *= light[:, None]


# ---------------------------------------------------------------- renderer
class BoxRenderer:
    """Deterministic numpy blocking renderer (no external dependencies)."""

    name = "box"

    def __init__(self, width: int, height: int, blur_px: float = 0):
        self.width, self.height, self.blur_px = int(width), int(height), float(blur_px)

    def _camera(self, frame: FrameState, plan: ScenePlan, index: SceneIndex) -> CameraState:
        cam = frame.camera
        if cam is None:
            cam = camera_mod.solve_camera(index.shots[frame.shot_id], frame, index, self.width, self.height)
        return replace(cam, width=self.width, height=self.height)

    def rasterize(self, scene: BlockingScene, cam: CameraState, grid: bool = True) -> _Raster:
        """Draw environment, solids, shadows, then translucent overlays. The 0.5 m grid helps people read
        space in the gray-box views; the init image for Qwen omits it (Qwen renders grid lines as tiles)."""
        rast = _Raster(cam, self.width, self.height)
        _draw_environment(rast, scene.environment, scene.lighting, grid=grid)
        id_of = {eid: k + 1 for k, eid in enumerate(scene.entity_order)}
        for prim in scene.primitives:
            if prim.alpha >= 1.0:
                _draw_primitive(rast, prim, id_of.get(prim.entity_id, 0), scene.lighting)
        _apply_shadows(rast, scene)
        for prim in scene.primitives:
            if prim.alpha < 1.0:
                _draw_primitive(rast, prim, 0, scene.lighting)
        return rast

    def alone_counts(self, scene: BlockingScene, cam: CameraState) -> Dict[str, int]:
        """Pixels each entity covers when drawn without occluders (visible_frac denominators)."""
        rast = _Raster(replace(cam, width=self.width, height=self.height), self.width, self.height)
        for prim in scene.primitives:
            if prim.alpha >= 1.0:
                _draw_primitive(rast, prim, 0, scene.lighting)
        return {eid: int(mask.sum()) for eid, mask in rast.alone.items()}

    def render_image(self, frame: FrameState, plan: ScenePlan, index: SceneIndex) -> Image.Image:
        """The sharp blocking image only (gray-box animatic frames)."""
        scene = build_scene(frame, plan, index)
        rast = self.rasterize(scene, self._camera(frame, plan, index))
        return Image.fromarray(_to_uint8(rast.color), mode="RGB")

    def render(self, frame: FrameState, plan: ScenePlan, index: SceneIndex, out_dir) -> BlockingResult:
        cam = self._camera(frame, plan, index)
        scene = build_scene(frame, plan, index)
        rast = self.rasterize(scene, cam)
        init = self.rasterize(scene, cam, grid=False)
        alone = {eid: int(mask.sum()) for eid, mask in rast.alone.items()}
        depth = rast.depth.astype(np.float32)
        return write_result(Path(out_dir), scene, cam, _to_uint8(rast.color), rast.ids, depth, alone,
                            self.blur_px, self.name, init=_to_uint8(init.color))


def _to_uint8(color: np.ndarray) -> np.ndarray:
    return np.clip(np.round(color * 255.0), 0, 255).astype(np.uint8)
