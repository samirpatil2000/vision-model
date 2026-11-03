"""
Evaluation metrics for a previs run.

Model-free (always available):
  background_drift      mean Lab ΔE76 outside entity masks vs the location plate, grouped by ancestry hop
  adjacent_change       mean abs diff + SSIM between consecutive keyframes of the same shot
  physics_invariant_rate, fact_accuracy, timing_summary, memory_summary
CLIP-dependent (skipped when no embedder / segmenter is passed):
  identity_consistency  CLIP(person crop) vs CLIP(full-body sheet), plus cross-shot crop similarity (no quadrupeds)
  layout_iou            CLIPSeg mask vs blocking id-mask per visible entity
ClipEmbedder reuses the CLIPSeg checkpoint the segmentation stack already downloads.
"""
import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

from src.previs.conditioning import is_quadruped
from src.previs.schema import ScenePlan
from src.previs.state import SceneIndex
from src.previs.types import BlockingResult

PathLike = Union[str, Path]
Embedder = Callable[[List[Image.Image]], np.ndarray]    # images -> L2-normalised (N, D) embeddings

# Validator check codes (src/previs/validator.py); the invariant rate is the share with no error.
CHECK_CODES: Tuple[str, ...] = (
    "ref_integrity", "keyframe_ids", "duration", "timing", "ranges", "custody", "irreversible", "count",
    "fixed", "supported", "event_feasible", "phase_order", "line_side", "setup_location", "subject_visible",
)
IDENTITY_THRESHOLD = 0.6
LAYOUT_THRESHOLD = 0.25
MIN_VISIBLE = 0.05
MIN_CROP_PX = 16


# ---------------------------------------------------------------- helpers
def _skipped(reason: str) -> Dict[str, Any]:
    return {"skipped": True, "reason": reason}


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return round(float(np.mean(values)), 4) if values else None


def _load_rgb(path: PathLike, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None and img.size != tuple(size):
            img = img.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(img)


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to (width, height)."""
    if mask.shape[::-1] == tuple(size):
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def as_dict(item: Any) -> Dict[str, Any]:
    return dataclasses.asdict(item) if dataclasses.is_dataclass(item) else dict(item)


def blocking_size(blocking: BlockingResult, default: Tuple[int, int]) -> Tuple[int, int]:
    """(width, height) of the blocking render, read from the id map header."""
    try:
        with Image.open(blocking.id_path) as img:
            return img.size
    except (OSError, ValueError):
        return default


def entity_masks(blocking: BlockingResult) -> Dict[str, np.ndarray]:
    """Boolean visible-pixel mask per entity from the blocking id map."""
    with Image.open(blocking.id_path) as img:
        ids = np.asarray(img)
    if ids.ndim == 3:
        ids = ids[..., 0]
    return {eid: ids == value for value, eid in blocking.id_map.items() if np.any(ids == value)}


def _scaled_box(bbox: Sequence[int], src: Tuple[int, int], dst: Tuple[int, int], pad: float = 0.08) -> Tuple[int, int, int, int]:
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(x0 - px)), max(0, int(y0 - py)), min(dst[0], int(np.ceil(x1 + px))), min(dst[1], int(np.ceil(y1 + py))))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIE L*a*b* (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Mean SSIM of two grayscale images (Gaussian window, Wang et al. 2004 constants)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = ndimage.gaussian_filter(a, sigma), ndimage.gaussian_filter(b, sigma)
    var_a = ndimage.gaussian_filter(a * a, sigma) - mu_a ** 2
    var_b = ndimage.gaussian_filter(b * b, sigma) - mu_b ** 2
    cov = ndimage.gaussian_filter(a * b, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


# ---------------------------------------------------------------- CLIP (optional)
class ClipEmbedder:
    """CLIP image embeddings from CLIPSeg's CLIP tower (`model.clip.get_image_features`); loads lazily."""

    def __init__(self, segmenter=None, device: Optional[str] = None):
        self._segmenter = segmenter
        self.device = device

    @property
    def segmenter(self):
        if self._segmenter is None:
            from src.segmentation.clipseg_segmenter import CLIPSegSegmenter
            self._segmenter = CLIPSegSegmenter(device=self.device)
        return self._segmenter

    def __call__(self, images: List[Image.Image]) -> np.ndarray:
        import torch

        seg = self.segmenter
        inputs = seg.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with torch.no_grad():
            out = seg.model.clip.get_image_features(pixel_values=inputs["pixel_values"].to(seg.device))
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        feats = feats.float().cpu().numpy()
        return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)


def entity_prompt(index: SceneIndex, entity_id: str) -> str:
    """
    Segmentation prompt: wardrobe for people (CLIPSeg cannot know names), prop name for props, and for a
    quadruped the same appearance text the QA gate segments with (a dog is no "person wearing its collar").
    """
    if is_quadruped(index, entity_id):
        from src.previs.qa import segmentation_prompt
        return segmentation_prompt(index, entity_id)
    if entity_id in index.characters:
        return f"a person wearing {index.characters[entity_id].wardrobe}"
    return index.props[entity_id].name


# ---------------------------------------------------------------- metrics
def identity_consistency(frames: Dict[str, PathLike], plan: ScenePlan, index: SceneIndex,
                         blocking_by_kf: Dict[str, BlockingResult], sheets: Dict[str, PathLike],
                         embedder: Optional[Embedder] = None, threshold: float = IDENTITY_THRESHOLD,
                         min_visible: float = MIN_VISIBLE) -> Dict[str, Any]:
    """
    CLIP cosine of each visible person's crop vs its full-body sheet, and crop vs crop across shots.
    Quadrupeds are skipped, as in the QA gate: an animal crop vs its side-view sheet is not comparable at the
    human threshold. They are listed under "skipped_characters".
    """
    if embedder is None:
        return _skipped("models disabled")
    people = [ch for ch in index.characters if not is_quadruped(index, ch)]
    skipped_characters = [ch for ch in index.characters if ch not in people]
    # 1. Character crops from accepted frames, boxes taken from the blocking pass.
    crops: List[Image.Image] = []
    owners: List[Tuple[str, str, str]] = []
    for kf_id, path in frames.items():
        blk = blocking_by_kf.get(kf_id)
        if blk is None or not path or not Path(path).is_file():
            continue
        with Image.open(path) as opened:
            img = opened.convert("RGB")
        src = blocking_size(blk, img.size)
        for ch in people:
            if blk.visible_frac.get(ch, 0.0) < min_visible or ch not in blk.bboxes:
                continue
            box = _scaled_box(blk.bboxes[ch], src, img.size)
            if box[2] - box[0] < MIN_CROP_PX or box[3] - box[1] < MIN_CROP_PX:
                continue
            crops.append(img.crop(box))
            owners.append((kf_id, ch, index.shot_of_keyframe.get(kf_id, "none")))
    if not crops:
        return {"skipped": False, "crops": 0, "sheet_mean": None, "cross_shot_mean": None, "pass_rate": None,
                "per_keyframe": {}, "per_character": {}, "skipped_characters": skipped_characters}

    # 2. Embeddings.
    emb = embedder(crops)
    sheet_ids = [ch for ch, p in sheets.items() if p and Path(p).is_file()]
    sheet_images = []
    for ch in sheet_ids:
        with Image.open(sheets[ch]) as opened:
            sheet_images.append(opened.convert("RGB"))
    sheet_emb = dict(zip(sheet_ids, embedder(sheet_images))) if sheet_images else {}

    # 3. Sheet similarity per crop, cross-shot similarity per character.
    per_kf: Dict[str, Dict[str, float]] = {}
    sheet_scores: Dict[str, List[float]] = {}
    for (kf_id, ch, _shot), e in zip(owners, emb):
        if ch in sheet_emb:
            score = round(float(np.dot(e, sheet_emb[ch])), 4)
            per_kf.setdefault(kf_id, {})[ch] = score
            sheet_scores.setdefault(ch, []).append(score)
    cross: Dict[str, List[float]] = {}
    for i in range(len(owners)):
        for j in range(i + 1, len(owners)):
            if owners[i][1] == owners[j][1] and owners[i][2] != owners[j][2]:
                cross.setdefault(owners[i][1], []).append(float(np.dot(emb[i], emb[j])))
    per_char = {
        ch: {"crops": sum(1 for o in owners if o[1] == ch), "sheet_mean": mean_or_none(sheet_scores.get(ch, [])),
             "sheet_min": round(min(sheet_scores[ch]), 4) if sheet_scores.get(ch) else None,
             "cross_shot_mean": mean_or_none(cross.get(ch, []))}
        for ch in sorted({o[1] for o in owners})
    }
    all_sheet = [s for v in sheet_scores.values() for s in v]
    return {
        "skipped": False, "crops": len(crops), "threshold": threshold,
        "sheet_mean": mean_or_none(all_sheet), "cross_shot_mean": mean_or_none([s for v in cross.values() for s in v]),
        "pass_rate": mean_or_none([float(s >= threshold) for s in all_sheet]),
        "per_keyframe": per_kf, "per_character": per_char, "skipped_characters": skipped_characters,
    }


def layout_iou(frames: Dict[str, PathLike], plan: ScenePlan, index: SceneIndex,
               blocking_by_kf: Dict[str, BlockingResult], segmenter=None, threshold: float = LAYOUT_THRESHOLD,
               min_visible: float = MIN_VISIBLE) -> Dict[str, Any]:
    """IoU of a text-prompted CLIPSeg mask in the frame vs the entity's blocking id-mask."""
    if segmenter is None:
        return _skipped("models disabled")
    per_kf: Dict[str, Dict[str, float]] = {}
    for kf_id, path in frames.items():
        blk = blocking_by_kf.get(kf_id)
        if blk is None or not path or not Path(path).is_file():
            continue
        with Image.open(path) as opened:
            img = opened.convert("RGB")
        for eid, mask in entity_masks(blk).items():
            if blk.visible_frac.get(eid, 1.0) < min_visible or eid not in index.entities:
                continue
            target = _resize_mask(mask, img.size)
            pred = _resize_mask(np.asarray(segmenter.segment(img, entity_prompt(index, eid)).target_mask) > 0, img.size)
            union = np.logical_or(pred, target).sum()
            per_kf.setdefault(kf_id, {})[eid] = round(float(np.logical_and(pred, target).sum() / union), 4) if union else 0.0
    scores = [s for v in per_kf.values() for s in v.values()]
    return {"skipped": False, "threshold": threshold, "pairs": len(scores), "mean": mean_or_none(scores),
            "pass_rate": mean_or_none([float(s >= threshold) for s in scores]), "per_keyframe": per_kf}


def background_drift(frames: Dict[str, PathLike], plate: Union[PathLike, Dict[str, PathLike]],
                     masks: Dict[str, np.ndarray], ancestry: Optional[Dict[str, int]] = None,
                     dilate_px: int = 8) -> Dict[str, Any]:
    """
    Mean CIE76 ΔE between each frame and its plate outside the (dilated) entity mask.
    `plate` is one path or kf_id -> path; `ancestry` (kf_id -> continuity depth) groups the result by hop.
    """
    per_kf: Dict[str, float] = {}
    for kf_id, path in frames.items():
        plate_path = plate.get(kf_id) if isinstance(plate, dict) else plate
        if not path or not plate_path or not Path(path).is_file() or not Path(plate_path).is_file():
            continue
        img = _load_rgb(path)
        size = (img.shape[1], img.shape[0])
        ref = _load_rgb(plate_path, size)
        background = np.ones(img.shape[:2], dtype=bool)
        if kf_id in masks and masks[kf_id] is not None:
            fg = _resize_mask(masks[kf_id], size)
            if dilate_px > 0:
                fg = ndimage.binary_dilation(fg, iterations=dilate_px)
            background = ~fg
        if not background.any():
            continue
        delta = np.linalg.norm(rgb_to_lab(img) - rgb_to_lab(ref), axis=-1)
        per_kf[kf_id] = round(float(delta[background].mean()), 4)
    by_depth: Dict[str, List[float]] = {}
    for kf_id, value in per_kf.items():
        if ancestry and kf_id in ancestry:
            by_depth.setdefault(str(ancestry[kf_id]), []).append(value)
    return {"metric": "delta_e76", "mean": mean_or_none(per_kf.values()), "per_keyframe": per_kf,
            "by_depth": {d: mean_or_none(v) for d, v in sorted(by_depth.items())}}


def adjacent_change(frames: Dict[str, PathLike], shot_of: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Mean abs diff (0..255) and SSIM between consecutive frames (in dict order) within the same shot."""
    ids = [k for k, p in frames.items() if p and Path(p).is_file()]
    pairs = []
    for a, b in zip(ids, ids[1:]):
        if shot_of is not None and shot_of.get(a) != shot_of.get(b):
            continue
        img_a = _load_rgb(frames[a])
        img_b = _load_rgb(frames[b], (img_a.shape[1], img_a.shape[0]))
        mad = float(np.mean(np.abs(img_a.astype(np.int16) - img_b.astype(np.int16))))
        pairs.append({"from": a, "to": b, "shot_id": shot_of.get(a, "none") if shot_of else "none",
                      "mad": round(mad, 4), "ssim": round(ssim(_luma(img_a), _luma(img_b)), 4)})
    return {"pairs": pairs, "mean_mad": mean_or_none(p["mad"] for p in pairs), "mean_ssim": mean_or_none(p["ssim"] for p in pairs)}


def physics_invariant_rate(violations: Iterable[Any], n_keyframes: Optional[int] = None) -> Dict[str, Any]:
    """Share of validator checks with no error (+ share of keyframes no error points at)."""
    items = [as_dict(v) for v in violations]
    errors = [v for v in items if v.get("severity", "error") == "error"]
    by_code: Dict[str, Dict[str, int]] = {}
    for v in items:
        bucket = by_code.setdefault(v.get("code", "unknown"), {"error": 0, "warning": 0})
        bucket["error" if v.get("severity", "error") == "error" else "warning"] += 1
    failed = sorted({v.get("code", "unknown") for v in errors})
    codes = list(CHECK_CODES) + sorted(c for c in by_code if c not in CHECK_CODES)
    out: Dict[str, Any] = {
        "errors": len(errors), "warnings": len(items) - len(errors), "by_code": by_code,
        "checks": len(codes), "checks_passed": len(codes) - len(failed), "failed_checks": failed,
        "invariant_rate": round((len(codes) - len(failed)) / len(codes), 4),
    }
    if n_keyframes:
        bad = {v.get("kf_id") for v in errors if v.get("kf_id", "none") != "none"}
        out["clean_keyframe_rate"] = round(1.0 - len(bad) / n_keyframes, 4)
    return out


def decided_fact_accuracy(answers: Sequence[str]) -> Optional[float]:
    """
    Share of "yes" among the decided ("yes"/"no") answers to the state-derived fact questions. "unsure" (and
    anything else) is no evidence either way and stays out of the denominator; None when nothing was decided.
    """
    counts = Counter(str(a).strip().lower() for a in answers)
    decided = counts["yes"] + counts["no"]
    return counts["yes"] / decided if decided else None


def _fact_counts(answers: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(str(a).strip().lower() for a in answers)
    accuracy = decided_fact_accuracy(answers)
    return {"questions": len(answers), "yes": counts["yes"], "no": counts["no"], "unsure": counts["unsure"],
            "accuracy": round(accuracy, 4) if accuracy is not None else None}


def fact_accuracy(answers: Union[Sequence[str], Dict[str, Sequence[str]]]) -> Dict[str, Any]:
    """Share of "yes" among the decided fact answers (overall, and per keyframe for a dict); "unsure" is excluded."""
    if isinstance(answers, dict):
        out = _fact_counts([a for v in answers.values() for a in v])
        out["per_keyframe"] = {kf: _fact_counts(v)["accuracy"] for kf, v in answers.items()}
        return out
    return _fact_counts(list(answers))


def timing_summary(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Engine request timings, memory, fallbacks and QA retake rates from render_log / result.json entries."""
    results = [as_dict(r) for r in results]
    secs = [float(r["seconds"]) for r in results if r.get("seconds") is not None]
    rss = [int(r["peak_rss_bytes"]) for r in results if r.get("peak_rss_bytes")]
    phases: Dict[str, List[float]] = {}
    for r in results:
        for name, value in (r.get("phase_seconds") or {}).items():
            phases.setdefault(name, []).append(float(value))
    by_kf: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        if r.get("kf_id"):
            by_kf.setdefault(r["kf_id"], []).append(r)
    first_pass = [bool(sorted(rs, key=lambda r: r.get("attempt", 0))[0].get("qa_passed")) for rs in by_kf.values()
                  if any("qa_passed" in r for r in rs)]
    final_pass = [any(r.get("qa_passed") for r in rs) for rs in by_kf.values() if any("qa_passed" in r for r in rs)]
    return {
        "requests": len(results),
        "total_s": round(sum(secs), 3),
        "mean_s": mean_or_none(secs),
        "median_s": round(float(np.median(secs)), 4) if secs else None,
        "p95_s": round(float(np.percentile(secs, 95)), 4) if secs else None,
        "max_s": round(max(secs), 4) if secs else None,
        "peak_rss_gb": round(max(rss) / 1024 ** 3, 3) if rss else None,
        "phase_mean_s": {k: mean_or_none(v) for k, v in sorted(phases.items())},
        "fallback_warnings": sum(len(r.get("fallback_warnings") or []) for r in results),
        "keyframes": len(by_kf),
        "mean_attempts": mean_or_none(len(v) for v in by_kf.values()),
        "first_pass_rate": mean_or_none(float(x) for x in first_pass),
        "final_pass_rate": mean_or_none(float(x) for x in final_pass),
    }


def memory_summary(results: Union[Iterable[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """Keyframe-memory records: count, approvals, continuity ancestry depth histogram, mean QA score."""
    if isinstance(results, dict):
        records = results.get("records", list(results.values()))
    else:
        records = list(results)
    records = [as_dict(r) for r in records if isinstance(r, dict) or dataclasses.is_dataclass(r)]
    depths = [int(r.get("ancestry_depth", 0)) for r in records]
    scores = [float(r["score"]) for r in records if r.get("score") is not None]
    return {
        "records": len(records),
        "approved": sum(1 for r in records if r.get("approved")),
        "depth_histogram": {str(d): n for d, n in sorted(Counter(depths).items())},
        "max_depth": max(depths) if depths else None,
        "mean_depth": mean_or_none(depths),
        "mean_score": mean_or_none(scores),
        "locations": dict(Counter(r.get("location_id", "none") for r in records)),
    }
