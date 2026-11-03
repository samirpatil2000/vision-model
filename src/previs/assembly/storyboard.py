"""
Storyboard pages: one panel per solved keyframe in edited-time order, `cols` x `rows` per page.

Each panel shows the frame (or a grey "pending" placeholder), a camera-move arrow drawn over the
image, and below it the shot id, keyframe number, timecode, size/angle/lens/move, the action and
any dialogue. Pages are saved as PNGs plus one multi-page storyboard.pdf.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from src import config
from src.previs.assembly import PathLike, font, format_timecode, load_frame, ordered_frames, placeholder
from src.previs.schema import ScenePlan, Shot
from src.previs.types import FrameState, SolvedPlan

PAGE_WIDTH = 1754            # A4 landscape width at 150 dpi; page height follows the grid
PDF_DPI = 150
MARGIN = 48
GAP = 28
HEADER_H = 84
TEXT_SIZE = 17
LINE_H = 22
ACTION_LINES = 3
DIALOGUE_LINES = 2
INK = (30, 30, 30)
MUTED = (110, 110, 110)
PAPER = (250, 250, 247)
ASPECTS = {"16:9": 16 / 9, "2.39:1": 2.39}

MOVE_LABELS = {
    "pan_left": "PAN L", "pan_right": "PAN R", "tilt_up": "TILT UP", "tilt_down": "TILT DOWN",
    "dolly_in": "DOLLY IN", "dolly_out": "DOLLY OUT", "truck_left": "TRUCK L", "truck_right": "TRUCK R",
    "crane_up": "CRANE UP", "crane_down": "CRANE DOWN", "orbit_left": "ORBIT L", "orbit_right": "ORBIT R",
    "handheld": "HANDHELD",
}


# ---------------------------------------------------------------- move arrows
def _arrow(draw: ImageDraw.ImageDraw, p0: Tuple[float, float], p1: Tuple[float, float], width: int) -> None:
    """Straight arrow p0 -> p1, white with a dark outline so it reads on any frame."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    head = width * 3.2
    base = (p1[0] - ux * head, p1[1] - uy * head)
    tri = [p1, (base[0] - uy * head * 0.6, base[1] + ux * head * 0.6), (base[0] + uy * head * 0.6, base[1] - ux * head * 0.6)]
    for color, extra in (((20, 20, 20), 3), ((255, 255, 255), 0)):
        draw.line([p0, base], fill=color, width=width + extra)
        draw.polygon(tri, fill=color, outline=color, width=1 + extra // 2)


def draw_move_arrow(img: Image.Image, move: str) -> Image.Image:
    """Storyboard convention: an arrow over the frame showing the camera move (none for static)."""
    if move not in MOVE_LABELS:
        return img
    img = img.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    lw = max(3, w // 110)
    cx, cy = w / 2, h / 2
    if move in ("pan_left", "truck_left"):
        y = cy if move == "pan_left" else h * 0.78
        _arrow(draw, (w * 0.68, y), (w * 0.32, y), lw)
    elif move in ("pan_right", "truck_right"):
        y = cy if move == "pan_right" else h * 0.78
        _arrow(draw, (w * 0.32, y), (w * 0.68, y), lw)
    elif move in ("tilt_up", "crane_up"):
        x = cx if move == "tilt_up" else w * 0.85
        _arrow(draw, (x, h * 0.72), (x, h * 0.28), lw)
    elif move in ("tilt_down", "crane_down"):
        x = cx if move == "tilt_down" else w * 0.85
        _arrow(draw, (x, h * 0.28), (x, h * 0.72), lw)
    elif move in ("dolly_in", "dolly_out"):
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            outer = (cx + sx * w * 0.42, cy + sy * h * 0.40)
            inner = (cx + sx * w * 0.26, cy + sy * h * 0.24)
            if move == "dolly_in":
                _arrow(draw, outer, inner, lw)
            else:
                _arrow(draw, inner, outer, lw)
    elif move in ("orbit_left", "orbit_right"):
        rx, ry = w * 0.30, h * 0.12
        ccy = h * 0.80
        pts = [(cx + rx * math.cos(a), ccy + ry * math.sin(a)) for a in [math.radians(20 + 14 * i) for i in range(11)]]
        if move == "orbit_left":
            pts = pts[::-1]
        for color, extra in (((20, 20, 20), 3), ((255, 255, 255), 0)):
            draw.line(pts[:-1], fill=color, width=lw + extra, joint="curve")
        _arrow(draw, pts[-3], pts[-1], lw)
    elif move == "handheld":
        pts = [(w * (0.30 + 0.04 * i), cy + (h * 0.05 if i % 2 else -h * 0.05)) for i in range(11)]
        for color, extra in (((20, 20, 20), 3), ((255, 255, 255), 0)):
            draw.line(pts, fill=color, width=lw + extra, joint="curve")
    label_font = font(max(11, h // 16))
    draw.text((w - 8, 8), MOVE_LABELS[move], font=label_font, fill=(255, 255, 255), anchor="ra",
              stroke_width=1, stroke_fill=(20, 20, 20))
    return img


# ---------------------------------------------------------------- text layout
def wrap_text(text: str, max_width: float, max_lines: int, size: int = TEXT_SIZE) -> List[str]:
    """Greedy word wrap to `max_width` pixels; the last kept line ends with an ellipsis when cut."""
    fnt = font(size)
    lines: List[str] = []
    current = ""
    words = text.split()
    for i, word in enumerate(words):
        trial = f"{current} {word}".strip()
        if fnt.getlength(trial) <= max_width or not current:
            current = trial
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    else:
        if current:
            lines.append(current)
        return lines
    last = lines[-1]
    while last and fnt.getlength(last + "…") > max_width:
        last = last[:-1]
    lines[-1] = last.rstrip() + "…"
    return lines


def camera_line(shot: Shot) -> str:
    parts = [shot.size.upper(), shot.angle, f"{shot.lens_mm:g}mm", shot.move.replace("_", " ")]
    if shot.time_scale != "real":
        parts.append(shot.time_scale.replace("_", " "))
    return " · ".join(parts)


def _panel_height(image_h: int) -> int:
    return image_h + 10 + LINE_H * (2 + ACTION_LINES + DIALOGUE_LINES)


def _draw_panel(page: Image.Image, origin: Tuple[int, int], image_size: Tuple[int, int], number: int,
                frame: FrameState, shot: Optional[Shot], action: str, dialogue: str, kf_count: int,
                image_path: Optional[PathLike]) -> None:
    x, y = origin
    iw, ih = image_size
    img = load_frame(image_path, image_size)
    img = placeholder(image_size, frame.kf_id) if img is None else img
    if shot is not None:
        img = draw_move_arrow(img, shot.move)
    page.paste(img, (x, y))
    draw = ImageDraw.Draw(page)
    draw.rectangle([x - 1, y - 1, x + iw, y + ih], outline=INK, width=2)
    badge = font(TEXT_SIZE)
    draw.rectangle([x, y, x + 34, y + 26], fill=INK)
    draw.text((x + 17, y + 13), str(number), font=badge, fill=(255, 255, 255), anchor="mm")

    fnt = font(TEXT_SIZE)
    ty = y + ih + 10
    shot_label = frame.shot_id.upper() + f"  ·  {frame.order_in_shot}/{kf_count}"
    draw.text((x, ty), shot_label, font=fnt, fill=INK, stroke_width=1, stroke_fill=INK)
    draw.text((x + iw, ty), format_timecode(frame.t_s, config.FPS), font=fnt, fill=INK, anchor="ra")
    ty += LINE_H
    if shot is not None:
        draw.text((x, ty), camera_line(shot), font=fnt, fill=MUTED)
    ty += LINE_H
    for line in wrap_text(action, iw, ACTION_LINES):
        draw.text((x, ty), line, font=fnt, fill=INK)
        ty += LINE_H
    if dialogue.strip():
        ty = max(ty, y + ih + 10 + LINE_H * (2 + 1))
        for line in wrap_text(f"“{dialogue.strip()}”", iw, DIALOGUE_LINES):
            draw.text((x, ty), line, font=fnt, fill=(40, 70, 120))
            ty += LINE_H


def build_storyboard(run_dir: PathLike, plan: ScenePlan, solved: SolvedPlan, frames: Dict[str, PathLike],
                     cols: int = 3, rows: int = 3) -> List[str]:
    """
    Write storyboard/page_<nn>.png and storyboard/storyboard.pdf.
    Returns the page PNG paths followed by the PDF path.
    """
    if cols < 1 or rows < 1:
        raise ValueError(f"cols and rows must be positive, got {cols}x{rows}")
    keyframes = ordered_frames(solved)
    if not keyframes:
        raise ValueError("Solved plan has no keyframes to board")
    out_dir = Path(run_dir) / "storyboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("page_*.png"):
        stale.unlink()

    # 1. Lookups from the plan.
    shots = {s.id: s for s in plan.bible.shots}
    kf_defs = {kf.id: kf for st in plan.timeline.shots for kf in st.keyframes}
    kf_counts: Dict[str, int] = {}
    for f in keyframes:
        kf_counts[f.shot_id] = kf_counts.get(f.shot_id, 0) + 1

    # 2. Page geometry: fixed width, height follows the grid.
    aspect = ASPECTS.get(plan.bible.scene.aspect, 16 / 9)
    cell_w = (PAGE_WIDTH - 2 * MARGIN - (cols - 1) * GAP) // cols
    image_size = (cell_w, int(round(cell_w / aspect)))
    cell_h = _panel_height(image_size[1])
    page_h = MARGIN + HEADER_H + rows * cell_h + (rows - 1) * GAP + MARGIN
    per_page = cols * rows
    n_pages = math.ceil(len(keyframes) / per_page)
    scene = plan.bible.scene

    # 3. Render pages.
    pages: List[Image.Image] = []
    paths: List[str] = []
    for p in range(n_pages):
        page = Image.new("RGB", (PAGE_WIDTH, page_h), PAPER)
        draw = ImageDraw.Draw(page)
        title = wrap_text(scene.logline, PAGE_WIDTH - 2 * MARGIN - 200, 1, size=24)
        draw.text((MARGIN, MARGIN), title[0] if title else "", font=font(24), fill=INK)
        draw.text((PAGE_WIDTH - MARGIN, MARGIN), f"page {p + 1}/{n_pages}", font=font(20), fill=MUTED, anchor="ra")
        summary = (f"{len(shots)} shots · {len(keyframes)} keyframes · {format_timecode(solved.total_duration_s)} "
                   f"· {scene.aspect} · {scene.time_of_day}")
        draw.text((MARGIN, MARGIN + 34), summary, font=font(18), fill=MUTED)
        draw.line([(MARGIN, MARGIN + HEADER_H - 14), (PAGE_WIDTH - MARGIN, MARGIN + HEADER_H - 14)], fill=MUTED, width=1)

        for slot, frame in enumerate(keyframes[p * per_page:(p + 1) * per_page]):
            r, c = divmod(slot, cols)
            origin = (MARGIN + c * (cell_w + GAP), MARGIN + HEADER_H + r * (cell_h + GAP))
            kf = kf_defs.get(frame.kf_id)
            _draw_panel(page, origin, image_size, p * per_page + slot + 1, frame, shots.get(frame.shot_id),
                        kf.action if kf else "", kf.dialogue if kf else "", kf_counts[frame.shot_id],
                        frames.get(frame.kf_id))
        path = out_dir / f"page_{p + 1:02d}.png"
        page.save(path)
        pages.append(page)
        paths.append(str(path))

    # 4. One multi-page PDF.
    pdf_path = out_dir / "storyboard.pdf"
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:], resolution=PDF_DPI)
    return paths + [str(pdf_path)]
