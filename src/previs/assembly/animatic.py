"""
Hold animatic: accepted keyframes laid on the solved timeline and encoded to H.264.

Each keyframe holds from its edited time t_s until the next keyframe of the same shot (the last one
holds to the shot end). Keyframes within a shot crossfade over CROSSFADE_FRAMES; shots hard-cut
unless the incoming shot's `transition` is "dissolve" (Shot.transition is read as the transition
INTO that shot). A Ken-Burns crop follows the shot's camera move, and the shot id + timecode are
burned into the bottom-left corner. Missing keyframes show a grey "pending" placeholder.
"""
import json
import math
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src import config
from src.previs.assembly import (
    FrameSequence, PathLike, burn_label, even_size, format_timecode, frame_count, load_frame, ordered_frames,
    placeholder, require_ffmpeg,
)
from src.previs.schema import ScenePlan
from src.previs.types import SolvedPlan

CROSSFADE_FRAMES = 5
DISSOLVE_FRAMES = 12
KB_ZOOM = 1.10               # crop zoom for pans/tilts/trucks/cranes/orbits and the dolly end point
HANDHELD_ZOOM = 1.05


@dataclass
class Segment:
    kf_id: str
    shot_id: str
    start_frame: int
    end_frame: int           # exclusive
    source: Optional[str]    # accepted frame path, None when pending

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class ShotSpan:
    shot_id: str
    start_frame: int
    end_frame: int           # exclusive
    move: str
    transition_in: str
    segments: List[Segment] = field(default_factory=list)


def build_timeline(plan: ScenePlan, solved: SolvedPlan, frames: Dict[str, PathLike], fps: int) -> List[ShotSpan]:
    """Frame-accurate shot spans and keyframe hold segments for the edited timeline."""
    shot_defs = {s.id: s for s in plan.bible.shots}
    total = frame_count(solved.total_duration_s, fps)
    keyframes = ordered_frames(solved)
    shots = [s for s in sorted(solved.shots, key=lambda s: s.start_t_s) if any(f.shot_id == s.shot_id for f in keyframes)]
    spans: List[ShotSpan] = []
    for i, shot in enumerate(shots):
        start = min(int(round(shot.start_t_s * fps)), total)
        end = int(round(shots[i + 1].start_t_s * fps)) if i + 1 < len(shots) else total
        end = min(max(end, start), total)
        if end <= start:
            continue
        sdef = shot_defs.get(shot.shot_id)
        span = ShotSpan(shot.shot_id, start, end, sdef.move if sdef else "static",
                        sdef.transition if sdef else "cut")
        kfs = [f for f in keyframes if f.shot_id == shot.shot_id]
        starts = [start] + [min(max(int(round(f.t_s * fps)), start), end - 1) for f in kfs[1:]]
        for k, f in enumerate(kfs):
            seg_end = starts[k + 1] if k + 1 < len(kfs) else end
            source = frames.get(f.kf_id)
            span.segments.append(Segment(f.kf_id, shot.shot_id, starts[k], max(seg_end, starts[k]),
                                         str(source) if source and Path(source).is_file() else None))
        spans.append(span)
    return spans


def ken_burns_box(move: str, progress: float, t_s: float, width: int, height: int,
                  seed: int = 0) -> Optional[Tuple[float, float, float, float]]:
    """Crop box (x0, y0, x1, y1) for a shot move at progress 0..1, or None for no crop (static)."""
    p = min(max(progress, 0.0), 1.0)
    zoom, cx, cy = KB_ZOOM, 0.5, 0.5          # cx, cy: crop centre as a fraction of the free travel
    if move == "dolly_in":
        zoom = 1.0 + (KB_ZOOM - 1.0) * p
    elif move == "dolly_out":
        zoom = KB_ZOOM - (KB_ZOOM - 1.0) * p
    elif move in ("pan_left", "truck_left", "orbit_left"):
        cx = 1.0 - p
    elif move in ("pan_right", "truck_right", "orbit_right"):
        cx = p
    elif move in ("tilt_up", "crane_up"):
        cy = 1.0 - p
    elif move in ("tilt_down", "crane_down"):
        cy = p
    elif move == "handheld":
        phase = (seed % 1000) / 1000.0 * 2 * math.pi
        zoom = HANDHELD_ZOOM
        cx = 0.5 + 0.30 * math.sin(2 * math.pi * 0.7 * t_s + phase) + 0.12 * math.sin(2 * math.pi * 1.9 * t_s)
        cy = 0.5 + 0.25 * math.sin(2 * math.pi * 0.5 * t_s + 2 * phase) + 0.10 * math.sin(2 * math.pi * 2.3 * t_s)
    else:
        return None
    if zoom <= 1.0 + 1e-6:
        return None
    cw, ch = width / zoom, height / zoom
    x0 = (width - cw) * min(max(cx, 0.0), 1.0)
    y0 = (height - ch) * min(max(cy, 0.0), 1.0)
    return (x0, y0, x0 + cw, y0 + ch)


class _Renderer:
    """Composes one output frame at a time from the cached keyframe images."""

    def __init__(self, spans: List[ShotSpan], size: Tuple[int, int], fps: int):
        self.spans, self.size, self.fps = spans, size, fps
        self.images: Dict[str, Image.Image] = {}
        for span in spans:
            for seg in span.segments:
                img = load_frame(seg.source, size)
                self.images[seg.kf_id] = img if img is not None else placeholder(size, seg.kf_id)

    def span_at(self, i: int) -> int:
        idx = 0
        for k, span in enumerate(self.spans):
            if span.start_frame <= i:
                idx = k
        return idx

    def shot_frame(self, span: ShotSpan, i: int) -> Tuple[Image.Image, Segment]:
        """The shot's own picture at output frame i (clamped into the shot), crossfades + Ken Burns applied."""
        i = min(max(i, span.start_frame), span.end_frame - 1)
        segs = [s for s in span.segments if s.length > 0]
        k = max((j for j, s in enumerate(segs) if s.start_frame <= i), default=0)
        seg = segs[k]
        img = self.images[seg.kf_id]
        if k + 1 < len(segs):
            nxt = segs[k + 1]
            fade = min(CROSSFADE_FRAMES, seg.length)
            if fade > 1 and i > nxt.start_frame - fade:
                img = Image.blend(img, self.images[nxt.kf_id], (i - (nxt.start_frame - fade)) / fade)
        progress = (i - span.start_frame) / max(1, span.end_frame - span.start_frame - 1)
        box = ken_burns_box(span.move, progress, (i - span.start_frame) / self.fps, self.size[0], self.size[1],
                            seed=zlib.crc32(span.shot_id.encode()))
        if box is not None:
            img = img.resize(self.size, Image.Resampling.BILINEAR, box=box)
        return img, seg

    def frame(self, i: int) -> Image.Image:
        k = self.span_at(i)
        span = self.spans[k]
        img, seg = self.shot_frame(span, i)

        # Dissolve windows straddle a cut; either side of the cut can own the frame.
        for j in (k, k + 1):
            if 0 < j < len(self.spans) and self.spans[j].transition_in == "dissolve":
                prev, cur = self.spans[j - 1], self.spans[j]
                length = min(DISSOLVE_FRAMES, prev.end_frame - prev.start_frame, cur.end_frame - cur.start_frame)
                w0 = cur.start_frame - length // 2
                if length > 1 and w0 <= i < w0 + length:
                    alpha = (i - w0 + 1) / (length + 1)
                    a_img, a_seg = self.shot_frame(prev, i)
                    b_img, b_seg = self.shot_frame(cur, i)
                    img = Image.blend(a_img, b_img, alpha)
                    seg = b_seg if alpha >= 0.5 else a_seg
        label = [f"{seg.shot_id}  {seg.kf_id}", format_timecode(i / self.fps, self.fps)]
        return burn_label(img, label, corner="bl")


def build_animatic(run_dir: PathLike, plan: ScenePlan, solved: SolvedPlan, frames: Dict[str, PathLike],
                   fps: int = config.FPS, width: Optional[int] = None, height: Optional[int] = None,
                   keep_frames: bool = False) -> str:
    """
    Render the hold animatic to animatic/animatic.mp4 and write animatic/timeline.json.
    width/height default to the first accepted frame's size (else the "final" tier).
    """
    if not solved.frames:
        raise ValueError("Solved plan has no keyframes to assemble")
    out_dir = Path(run_dir) / "animatic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Output size.
    if width is None or height is None:
        first = next((p for p in frames.values() if p and Path(p).is_file()), None)
        if first is not None:
            with Image.open(first) as img:
                width, height = img.size
        else:
            width, height = config.TIERS["final"].width, config.TIERS["final"].height
    size = even_size(int(width), int(height))

    # 2. Frame-accurate timeline.
    spans = build_timeline(plan, solved, frames, fps)
    if not spans:
        raise ValueError("No shot in the solved plan has keyframes")
    total = spans[-1].end_frame
    timeline = {
        "fps": fps, "width": size[0], "height": size[1], "frame_count": total, "duration_s": total / fps,
        "crossfade_frames": CROSSFADE_FRAMES, "dissolve_frames": DISSOLVE_FRAMES,
        "shots": [{k: v for k, v in asdict(s).items() if k != "segments"} | {
            "start_t_s": s.start_frame / fps, "end_t_s": s.end_frame / fps} for s in spans],
        "segments": [asdict(seg) | {
            "start_t_s": seg.start_frame / fps, "end_t_s": seg.end_frame / fps,
            "timecode": format_timecode(seg.start_frame / fps, fps),
            "status": "frame" if seg.source else "pending"} for s in spans for seg in s.segments],
    }
    (out_dir / "timeline.json").write_text(json.dumps(timeline, indent=2))

    # 3. PNG sequence -> H.264.
    require_ffmpeg()
    renderer = _Renderer(spans, size, fps)
    sequence = FrameSequence(out_dir / "_frames")
    for i in range(total):
        sequence.add(renderer.frame(i))
    return sequence.encode(out_dir / "animatic.mp4", fps, keep_frames=keep_frames)
