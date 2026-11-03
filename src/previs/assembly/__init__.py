"""
Assembly: storyboard pages + PDF, the hold animatic, and the 24 fps gray-box physics animatic.

This module holds the helpers the three builders share: fonts, timecodes, image fitting and
placeholders, burned-in labels, the PNG-sequence -> H.264 encoder, ffprobe, and reading
solved.json back into a SolvedPlan. Builders live in storyboard.py, animatic.py and graybox.py.
"""
import dataclasses
import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src import config
from src.previs.types import CameraState, EntityFrame, FrameState, SolvedEvent, SolvedPlan, SolvedShot

PLACEHOLDER_RGB = (128, 128, 128)
ENCODE_TIMEOUT_S = 900
PROBE_TIMEOUT_S = 60

PathLike = Union[str, Path]


# ---------------------------------------------------------------- text + images
@lru_cache(maxsize=16)
def font(size: int) -> ImageFont.FreeTypeFont:
    """Pillow's bundled scalable font: identical on every machine, no system font lookup."""
    return ImageFont.load_default(size=size)


def format_timecode(t_s: float, fps: int = config.FPS) -> str:
    """Edited time as mm:ss.ff, where ff is the frame within the second at `fps`."""
    total = max(0, int(round(t_s * fps)))
    seconds, frames = divmod(total, fps)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{frames:02d}"


def even_size(width: int, height: int) -> Tuple[int, int]:
    """H.264 with yuv420p needs even dimensions."""
    if width < 2 or height < 2:
        raise ValueError(f"Video size must be at least 2x2, got {width}x{height}")
    return width - width % 2, height - height % 2


def fit_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Scale and centre-crop to exactly `size` (cover), RGB."""
    img = img.convert("RGB")
    if img.size == tuple(size):
        return img
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)


def load_frame(path: Optional[PathLike], size: Tuple[int, int]) -> Optional[Image.Image]:
    """The image at `path` fitted to `size`, or None when the path is empty or missing."""
    if not path or not Path(path).is_file():
        return None
    with Image.open(path) as img:
        return fit_image(img, size)


def placeholder(size: Tuple[int, int], kf_id: str, label: str = "pending") -> Image.Image:
    """Grey panel standing in for a keyframe that has not been rendered yet."""
    w, h = size
    img = Image.new("RGB", size, PLACEHOLDER_RGB)
    draw = ImageDraw.Draw(img)
    big, small = font(max(12, h // 7)), font(max(10, h // 16))
    draw.line([(0, 0), (w - 1, h - 1)], fill=(140, 140, 140), width=2)
    draw.line([(0, h - 1), (w - 1, 0)], fill=(140, 140, 140), width=2)
    draw.text((w / 2, h / 2), label, font=big, fill=(235, 235, 235), anchor="mm",
              stroke_width=2, stroke_fill=(90, 90, 90))
    draw.text((w / 2, h / 2 + h // 7), kf_id, font=small, fill=(225, 225, 225), anchor="mt")
    return img


def burn_label(img: Image.Image, lines: Sequence[str], corner: str = "bl", size: Optional[int] = None) -> Image.Image:
    """Draw text lines on a translucent dark box in a corner ("tl", "tr", "bl", "br"). Returns a new RGB image."""
    lines = [line for line in lines if line]
    base = img.convert("RGB")
    if not lines:
        return base
    w, h = base.size
    fnt = font(size or max(10, h // 28))
    pad, gap = max(4, h // 90), max(2, h // 200)
    boxes = [fnt.getbbox(line) for line in lines]
    line_h = max(b[3] for b in boxes) + gap
    box_w = max(b[2] for b in boxes) + 2 * pad
    box_h = line_h * len(lines) + 2 * pad - gap
    x0 = pad if corner in ("tl", "bl") else w - box_w - pad
    y0 = pad if corner in ("tl", "tr") else h - box_h - pad
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((x0 + pad, y0 + pad + i * line_h), line, font=fnt, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def frame_count(duration_s: float, fps: int) -> int:
    """Number of output frames covering `duration_s` at `fps` (at least one)."""
    return max(1, int(round(duration_s * fps)))


# ---------------------------------------------------------------- ffmpeg
def require_ffmpeg() -> str:
    """Absolute path of the ffmpeg binary, or RuntimeError when it is not installed."""
    found = shutil.which(config.FFMPEG)
    if found is None:
        raise RuntimeError(f"ffmpeg not found ('{config.FFMPEG}'); install it or set FFMPEG")
    return found


class FrameSequence:
    """A PNG sequence written frame by frame, then encoded to H.264 (yuv420p) with ffmpeg."""

    def __init__(self, frame_dir: PathLike):
        self.frame_dir = Path(frame_dir)
        shutil.rmtree(self.frame_dir, ignore_errors=True)
        self.frame_dir.mkdir(parents=True)
        self.count = 0
        self.size: Optional[Tuple[int, int]] = None

    def add(self, img: Image.Image) -> None:
        img = img.convert("RGB")
        if self.size is None:
            self.size = img.size
        elif img.size != self.size:
            raise ValueError(f"Frame {self.count} is {img.size}, sequence is {self.size}")
        img.save(self.frame_dir / f"frame_{self.count:05d}.png", compress_level=1)
        self.count += 1

    def encode(self, out_path: PathLike, fps: int, keep_frames: bool = False) -> str:
        if self.count == 0:
            raise ValueError("No frames to encode")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # -nostdin + DEVNULL: ffmpeg must never touch the terminal, or a backgrounded run stops on SIGTTOU/SIGTTIN.
        cmd = [
            require_ffmpeg(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(self.frame_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-movflags", "+faststart", str(out_path),
        ]
        try:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=ENCODE_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffmpeg timed out after {ENCODE_TIMEOUT_S}s encoding {out_path}") from exc
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}) encoding {out_path}: {proc.stderr.strip()[-800:]}")
        if not keep_frames:
            shutil.rmtree(self.frame_dir, ignore_errors=True)
        return str(out_path)


def probe_video(path: PathLike) -> Dict[str, float]:
    """Duration (s), fps, size and frame count of the first video stream, via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe not found")
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,codec_name,pix_fmt",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out on {path}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()[-400:]}")
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return {
        "duration_s": float(data["format"]["duration"]),
        "fps": num / den if den else 0.0,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_read_frames", 0)),
        "codec": stream.get("codec_name", ""),
        "pix_fmt": stream.get("pix_fmt", ""),
    }


# ---------------------------------------------------------------- solved.json
def _vec(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _build(cls: type, data: Dict[str, Any], vec_fields: Sequence[str] = ()) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    for name in vec_fields:
        if name in kwargs:
            kwargs[name] = _vec(kwargs[name])
    return cls(**kwargs)


def _frame_from_dict(data: Dict[str, Any]) -> FrameState:
    entities = {}
    for eid, ef in data.get("entities", {}).items():
        entity = _build(EntityFrame, ef, ("position", "rotation_deg", "velocity", "size"))
        entity.trajectory = [_vec(p) for p in ef.get("trajectory", [])]
        entities[eid] = entity
    cam = data.get("camera")
    fields = dict(data, entities=entities,
                  camera=_build(CameraState, cam, ("position", "look_at")) if cam else None)
    return _build(FrameState, fields)


def load_solved(source: Union[PathLike, Dict[str, Any]]) -> SolvedPlan:
    """Rebuild a SolvedPlan from solved.json (or its parsed dict); unknown keys are ignored."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    return SolvedPlan(
        frames=[_frame_from_dict(f) for f in data.get("frames", [])],
        events={k: _build(SolvedEvent, v) for k, v in data.get("events", {}).items()},
        shots=[_build(SolvedShot, s) for s in data.get("shots", [])],
        total_duration_s=float(data.get("total_duration_s", 0.0)),
        warnings=list(data.get("warnings", [])),
    )


def ordered_frames(solved: SolvedPlan) -> List[FrameState]:
    """Solved keyframes in edited-time order (stable for equal times)."""
    return sorted(solved.frames, key=lambda f: f.t_s)
