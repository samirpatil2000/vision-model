"""
Gray-box physics animatic: the solved scene sampled at every output frame and rendered as blocking.

For each frame time: physics.sample_frame -> camera.solve_camera -> BoxRenderer.render_image, with
the shot / keyframe / depicted phase / active events and edited + physical time burned in. This is
the cheap animatic a human approves before any keyframe is generated.

Camera moves are keyed to keyframes (move_progress), so the camera at an in-between time is solved
for both surrounding keyframes on the same sampled state and interpolated: moves play smoothly while
the camera still tracks the moving subject.
"""
import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from src import config
from src.previs import physics
from src.previs.assembly import FrameSequence, PathLike, burn_label, even_size, format_timecode, frame_count, require_ffmpeg
from src.previs.schema import ScenePlan, Shot
from src.previs.state import SceneIndex
from src.previs.types import CameraState, FrameState, SolvedPlan

CameraSolver = Callable[[Shot, FrameState, SceneIndex, int, int], CameraState]


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def _lerp_angle(a: float, b: float, f: float) -> float:
    return a + ((b - a + 180.0) % 360.0 - 180.0) * f


def interpolate_camera(a: CameraState, b: CameraState, f: float) -> CameraState:
    """Camera between two solved states (positions/look-at linear, angles along the short arc)."""
    if f <= 0.0:
        return a
    if f >= 1.0:
        return b
    return CameraState(
        position=tuple(_lerp(p, q, f) for p, q in zip(a.position, b.position)),
        yaw_deg=_lerp_angle(a.yaw_deg, b.yaw_deg, f),
        pitch_deg=_lerp(a.pitch_deg, b.pitch_deg, f),
        roll_deg=_lerp(a.roll_deg, b.roll_deg, f),
        lens_mm=_lerp(a.lens_mm, b.lens_mm, f),
        vfov_deg=_lerp(a.vfov_deg, b.vfov_deg, f),
        width=a.width,
        height=a.height,
        look_at=tuple(_lerp(p, q, f) for p, q in zip(a.look_at, b.look_at)),
    )


def _neighbours(solved: SolvedPlan) -> Dict[str, Tuple[FrameState, Optional[FrameState]]]:
    """kf_id -> (that keyframe, the next keyframe in the same shot or None)."""
    out: Dict[str, Tuple[FrameState, Optional[FrameState]]] = {}
    by_shot: Dict[str, List[FrameState]] = {}
    for f in solved.frames:
        by_shot.setdefault(f.shot_id, []).append(f)
    for frames in by_shot.values():
        frames.sort(key=lambda f: f.t_s)
        for k, f in enumerate(frames):
            out[f.kf_id] = (f, frames[k + 1] if k + 1 < len(frames) else None)
    return out


def camera_at(frame: FrameState, shot: Shot, index: SceneIndex, neighbours: Dict[str, Tuple[FrameState, Optional[FrameState]]],
              solve_camera: CameraSolver, width: int, height: int) -> CameraState:
    """Camera for a sampled frame, interpolated between the surrounding keyframes' move progress."""
    cam_a = solve_camera(shot, frame, index, width, height)
    a, b = neighbours.get(frame.kf_id, (None, None))
    if a is None or b is None or b.t_s <= a.t_s or frame.t_s <= a.t_s:
        return cam_a
    as_b = dataclasses.replace(frame, kf_id=b.kf_id, order_in_shot=b.order_in_shot,
                               depicted_phase=b.depicted_phase, focus_event_id=b.focus_event_id)
    cam_b = solve_camera(shot, as_b, index, width, height)
    return interpolate_camera(cam_a, cam_b, (frame.t_s - a.t_s) / (b.t_s - a.t_s))


def build_graybox_animatic(run_dir: PathLike, plan: ScenePlan, solved: SolvedPlan, index: SceneIndex,
                           width: int, height: int, fps: int = config.FPS, renderer=None,
                           solve_camera: Optional[CameraSolver] = None, keep_frames: bool = False) -> str:
    """
    Encode graybox/graybox.mp4 covering the solved timeline at `fps`.
    `renderer` needs `render_image(frame, plan, index) -> PIL.Image` (default BoxRenderer);
    `solve_camera` defaults to camera.solve_camera.
    """
    if not solved.frames:
        raise ValueError("Solved plan has no keyframes to sample")
    size = even_size(int(width), int(height))
    require_ffmpeg()
    if renderer is None:
        from src.previs.blocking.box_renderer import BoxRenderer
        renderer = BoxRenderer(size[0], size[1])
    if solve_camera is None:
        from src.previs.camera import solve_camera as default_solver
        solve_camera = default_solver

    out_dir = Path(run_dir) / "graybox"
    sequence = FrameSequence(out_dir / "_frames")
    neighbours = _neighbours(solved)
    for i in range(frame_count(solved.total_duration_s, fps)):
        # 1. Physics state at the edited time, 2. camera, 3. blocking render, 4. burn-in.
        t = i / fps
        frame = physics.sample_frame(plan, solved, t)
        shot = index.shots[frame.shot_id]
        frame.camera = camera_at(frame, shot, index, neighbours, solve_camera, size[0], size[1])
        img = renderer.render_image(frame, plan, index)
        if not isinstance(img, Image.Image):
            raise TypeError(f"renderer.render_image returned {type(img).__name__}, expected PIL.Image")
        if img.size != size:
            img = img.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        phase = f" · {frame.depicted_phase}" if frame.depicted_phase != "none" else ""
        img = burn_label(img, [f"GRAYBOX  {frame.shot_id}  {frame.kf_id}{phase}"], corner="tl")
        events = ", ".join(frame.active_event_ids)
        img = burn_label(img, [f"{format_timecode(t, fps)}   phys {frame.t_physical_s:5.2f}s",
                               events and f"active: {events}"], corner="bl")
        sequence.add(img)
    return sequence.encode(out_dir / "graybox.mp4", fps, keep_frames=keep_frames)
