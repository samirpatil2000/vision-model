"""
Quality gate for generated keyframes.

Checks, each reported as a score (higher is better, about 0..1) plus a named failure:
  identity:<ch>  cosine of CLIP embeddings, crop of the character's blocking bbox vs its full-body sheet
                 (humans only: a quadruped's side-view sheet is not comparable with the same threshold;
                 and only when the head and torso are mostly in view: a crop of legs, or of a sliver behind
                 a foreground person, says nothing about who it is).
  layout:<id>    IoU of the CLIPSeg mask for the entity and its blocking id-mask.
  copy:<id>      the plan moves the entity >= 24 px on screen since the previous keyframe of the shot
                 (its solved state changed or the camera moved, and its id-mask centroid shifted),
                 but the pixels inside its mask barely changed (the model copied the previous frame).
  lock           static-camera lock: every pixel outside the lock mask must equal the original exactly
                 (an independent numpy diff, not the compositor's own report).

CLIPSeg (src/segmentation/clipseg_segmenter.py) is loaded lazily and its CLIP vision tower provides the
embeddings, so no extra weights are needed. With use_models=False identity and layout are skipped.
fact_questions() turns solved state into atomic yes/no questions whose correct answer is "yes". The judge
sees only the image, so characters are named by what it can see (wardrobe, or an animal's appearance).
"""
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from src.previs.conditioning import (
    bbox_of,
    cameras_match,
    compact,
    entity_changed,
    entity_label,
    entity_mask,
    is_quadruped,
    load_id_mask,
    load_sibling_blocking,
    location_kind,
    on_screen_ids,
    previous_keyframe_in_shot,
    previous_solved_frame,
    shown_flags,
    surface_name,
)
from src.previs.schema import NONE, ScenePlan
from src.previs.state import SceneIndex
from src.previs.types import BlockingResult, FrameState

IDENTITY_THRESHOLD = 0.6
LAYOUT_THRESHOLD = 0.25
COPY_MOTION_PX = 24.0
COPY_MIN_CHANGE = 3.0            # mean absolute 8-bit difference inside the entity mask
DEFORMING_FLAGS = frozenset({"shattered", "cracked", "crumpled", "torn", "spilled", "dented"})
QA_VISIBLE_FRAC = 0.05           # entities less visible than this in the blocking are not checked
LAYOUT_MIN_AREA_FRAC = 0.005     # tiny on-screen entities give meaningless IoUs
IDENTITY_MIN_CROP_PX = 32
IDENTITY_MIN_KEY_VISIBLE = 0.5   # head + torso visible fraction (camera.visible_fractions) needed for identity
CROP_PAD_FRACTION = 0.08
CLIP_INPUT_PX = 224              # CLIP ViT-B/16 native resolution
MAX_FACT_QUESTIONS = 10
WARDROBE_WORDS = 12              # how much of a wardrobe names a character in a fact question
APPEARANCE_WORDS = 8
_DANGLING = frozenset({"a", "an", "the", "and", "or", "with", "of", "in", "on", "at", "to", "for"})

FLAG_QUESTIONS: Dict[str, str] = {
    "cracked": "Is {label} cracked?", "shattered": "Is {label} shattered into pieces?",
    "dented": "Is {label} dented?", "torn": "Is {label} torn?", "crumpled": "Is {label} crumpled?",
    "spilled": "Have the contents of {label} spilled out?", "stained": "Is {label} stained?",
    "wet": "Is {label} wet?", "dusty": "Is {label} dusty?", "fill_empty": "Is {label} empty?",
    "fill_half": "Is {label} roughly half full?", "fill_full": "Is {label} full?",
}
EXPRESSION_WORDS: Dict[str, str] = {
    "happy": "happy", "sad": "sad", "angry": "angry", "surprised": "surprised", "afraid": "afraid",
    "focused": "focused", "pained": "in pain",
}


@dataclass
class QAResult:
    passed: bool
    scores: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if xs.size else None


def _rgb_array(path: str, size) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != tuple(size):
        img = img.resize(tuple(size), Image.Resampling.LANCZOS)
    return np.asarray(img)


class QualityGate:
    """Per-keyframe QA. Models load on first use and stay resident until close()."""

    def __init__(self, device: Optional[str] = None, use_models: bool = True,
                 identity_threshold: float = IDENTITY_THRESHOLD, layout_threshold: float = LAYOUT_THRESHOLD,
                 copy_motion_px: float = COPY_MOTION_PX, copy_min_change: float = COPY_MIN_CHANGE):
        self.device = device
        self.use_models = use_models
        self.identity_threshold = identity_threshold
        self.layout_threshold = layout_threshold
        self.copy_motion_px = copy_motion_px
        self.copy_min_change = copy_min_change
        self._segmenter = None
        self._sheet_embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ models
    def segmenter(self):
        if self._segmenter is None:
            from src.segmentation.clipseg_segmenter import CLIPSegSegmenter
            self._segmenter = CLIPSegSegmenter(device=self.device)
        return self._segmenter

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """L2-normalised CLIP image embeddings (CLIPSeg's CLIP vision tower), shape (N, D)."""
        import torch

        seg = self.segmenter()
        inputs = seg.processor.image_processor(
            images=[im.convert("RGB") for im in images],
            size={"height": CLIP_INPUT_PX, "width": CLIP_INPUT_PX},
            return_tensors="pt",
        )
        with torch.no_grad():
            out = seg.model.clip.get_image_features(pixel_values=inputs["pixel_values"].to(seg.device))
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        return torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()

    def close(self) -> None:
        """Release the models (they reload on the next check)."""
        self._segmenter = None
        self._sheet_embeddings.clear()
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------ checks
    def check(
        self,
        image_path: str,
        frame: FrameState,
        blocking: BlockingResult,
        plan: ScenePlan,
        index: SceneIndex,
        sheet_paths: Dict[str, str],
        prev_path: Optional[str],
        lock_mask_path: Optional[str],
        original_for_lock: Optional[str],
        prev_blocking: Optional[BlockingResult] = None,
        prev_frame: Optional[FrameState] = None,
    ) -> QAResult:
        """
        Score one generated keyframe. `prev_path` is the previous accepted frame of the same shot;
        `prev_blocking` its blocking (read from graybox/<prev_kf_id>/ when omitted) and `prev_frame`
        its solved state with camera (re-solved from the plan when omitted).
        """
        image = Image.open(image_path).convert("RGB")
        size = image.size
        pixels = np.asarray(image)
        ids = load_id_mask(blocking.id_path, size)
        shot = index.shots.get(frame.shot_id)
        on_screen = set(on_screen_ids(frame, blocking, shot.subject_id if shot else NONE))
        visible = [eid for eid in index.entities
                   if eid in on_screen and blocking.visible_frac.get(eid, 0.0) >= QA_VISIBLE_FRAC]
        scores: Dict[str, float] = {}
        failures: List[str] = []

        # 1. Static-camera lock: outside the mask the frame must be the original, bit for bit.
        if lock_mask_path and original_for_lock:
            original = Image.open(original_for_lock).convert("RGB")
            allowed = np.asarray(Image.open(lock_mask_path).convert("L").resize(size)) > 127
            if original.size != size:
                scores["lock"] = 0.0
                failures.append("lock")
            else:
                diff = np.abs(pixels.astype(np.int16) - np.asarray(original).astype(np.int16)).max(axis=2)
                protected = ~allowed
                changed = int(np.count_nonzero((diff > 0) & protected))
                scores["lock"] = 1.0 - changed / max(1, int(protected.sum()))
                if changed:
                    failures.append("lock")

        # 2. Copy detection against the previous keyframe of the shot.
        prev_kf_id = previous_keyframe_in_shot(index, frame.kf_id)
        if prev_path and prev_kf_id is not None:
            if prev_blocking is None:
                prev_blocking = load_sibling_blocking(blocking, prev_kf_id)
            if prev_blocking is not None:
                prev_pixels = _rgb_array(prev_path, size).astype(np.int16)
                prev_ids = load_id_mask(prev_blocking.id_path, size)
                # An entity moves on screen only when its own state changed or the camera moved. A still
                # entity's visible centroid also shifts when a mover occludes it; that is not motion (and
                # under the static lock its pixels are copied from the previous frame by design).
                prev_frame = previous_solved_frame(plan, index, frame, prev_frame)
                camera_moved = prev_frame is None or not cameras_match(prev_frame.camera, frame.camera)
                for eid in visible:
                    if not camera_moved and not entity_changed(prev_frame.entities.get(eid), frame.entities.get(eid)):
                        continue
                    mask = entity_mask(blocking, eid, ids=ids)
                    c_now = _centroid(mask)
                    c_prev = _centroid(entity_mask(prev_blocking, eid, ids=prev_ids))
                    if c_now is None or c_prev is None or np.hypot(*(c_now - c_prev)) < self.copy_motion_px:
                        continue
                    change = float(np.abs(pixels.astype(np.int16)[mask] - prev_pixels[mask]).mean())
                    scores[f"copy:{eid}"] = min(1.0, change / self.copy_min_change)
                    if change < self.copy_min_change:
                        failures.append(f"copy:{eid}")

        # 3. Identity (characters whose head and torso are in view) and layout (models).
        if self.use_models:
            identifiable = identifiable_characters(frame, plan, index, visible)
            self._identity(image, ids, blocking, identifiable, index, sheet_paths, scores, failures)
            self._layout(image, ids, blocking, visible, index, scores, failures, frame)

        return QAResult(passed=not failures, scores=scores, failures=failures)

    def _identity(self, image: Image.Image, ids: np.ndarray, blocking: BlockingResult, visible: List[str],
                  index: SceneIndex, sheet_paths: Dict[str, str], scores: Dict[str, float],
                  failures: List[str]) -> None:
        crops, ch_ids = [], []
        for ch_id in visible:
            if ch_id not in index.characters or not sheet_paths.get(ch_id) or is_quadruped(index, ch_id):
                continue
            box = bbox_of(entity_mask(blocking, ch_id, ids=ids))
            if box is None:
                continue
            x0, y0, x1, y1 = box
            pad_x, pad_y = int((x1 - x0) * CROP_PAD_FRACTION), int((y1 - y0) * CROP_PAD_FRACTION)
            crop = image.crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                               min(image.width, x1 + 1 + pad_x), min(image.height, y1 + 1 + pad_y)))
            if min(crop.size) < IDENTITY_MIN_CROP_PX:
                continue
            crops.append(crop)
            ch_ids.append(ch_id)
        if not crops:
            return
        crop_emb = self.embed(crops)
        for ch_id, emb in zip(ch_ids, crop_emb):
            sheet = sheet_paths[ch_id]
            if sheet not in self._sheet_embeddings:
                with Image.open(sheet) as img:
                    self._sheet_embeddings[sheet] = self.embed([img.convert("RGB")])[0]
            cosine = float(np.dot(emb, self._sheet_embeddings[sheet]))
            scores[f"identity:{ch_id}"] = cosine
            if cosine < self.identity_threshold:
                failures.append(f"identity:{ch_id}")

    def _layout(self, image: Image.Image, ids: np.ndarray, blocking: BlockingResult, visible: List[str],
                index: SceneIndex, scores: Dict[str, float], failures: List[str],
                frame: Optional[FrameState] = None) -> None:
        """
        IoU of text segmentation vs the planned mask. Only characters and the shot's subject can fail
        the frame: set pieces seen side-on (a table's dark side panel) segment unreliably, so they are
        scored but not gated. Props whose state deforms them (shattered, spilled, ...) are skipped,
        because the intact proxy shape no longer describes them.
        """
        shot = index.shots.get(frame.shot_id) if frame is not None else None
        gated = set(index.characters) | ({shot.subject_id} if shot is not None else set())
        for eid in visible:
            ef = frame.entities.get(eid) if frame is not None else None
            if ef is not None and set(ef.flags) & DEFORMING_FLAGS:
                continue
            planned = entity_mask(blocking, eid, ids=ids)
            if planned.mean() < LAYOUT_MIN_AREA_FRAC:
                continue
            found = self.segmenter().segment(image, segmentation_prompt(index, eid)).target_mask > 0
            score = iou(planned, found)
            scores[f"layout:{eid}"] = score
            if score < self.layout_threshold and eid in gated:
                failures.append(f"layout:{eid}")


def identifiable_characters(frame: FrameState, plan: ScenePlan, index: SceneIndex, visible: Sequence[str],
                            min_key_visible: float = IDENTITY_MIN_KEY_VISIBLE) -> List[str]:
    """
    `visible` without the human characters whose head and torso are mostly hidden (or too small to
    measure) by the solved camera: camera.visible_fractions on a low-res id render, legs and hips not
    counted. Without a solved camera nothing is removed.
    """
    humans = [eid for eid in visible if eid in index.characters and not is_quadruped(index, eid)]
    if frame.camera is None or not humans:
        return list(visible)
    from src.previs.blocking import build_scene      # lazy: the blocking package pulls in the renderers
    from src.previs.camera import visible_fractions

    key_visible = visible_fractions(build_scene(frame, plan, index), frame.camera, humans, key_parts_only=True)
    hidden = {eid for eid in humans if (key_visible.get(eid) or 0.0) < min_key_visible}
    return [eid for eid in visible if eid not in hidden]


def segmentation_prompt(index: SceneIndex, entity_id: str) -> str:
    """CLIPSeg text for an entity: the prop name, 'person wearing <wardrobe>' or, for a quadruped, its
    appearance (names mean nothing to CLIP)."""
    if is_quadruped(index, entity_id):
        return compact(index.characters[entity_id].appearance, 6) or "animal"
    if entity_id in index.characters:
        return f"person wearing {_descriptor(index.characters[entity_id].wardrobe, 8)}"
    if entity_id in index.props:
        return index.props[entity_id].name
    return entity_id


# ---------------------------------------------------------------- fact questions
def _descriptor(text: str, max_words: int) -> str:
    """Leading clauses of `text` within `max_words`, without a dangling 'with a' / 'and' at the cut."""
    words = compact(text, max_words).split()
    while words and words[-1].lower() in _DANGLING:
        words.pop()
    phrase = " ".join(words)
    if len(phrase) > 1 and phrase[0].isupper() and phrase[1].islower():
        phrase = phrase[0].lower() + phrase[1:]
    return phrase


def visual_label(index: SceneIndex, entity_id: str) -> str:
    """
    How a judge who sees only the image can identify an entity: props by name, people by wardrobe,
    animals by appearance; never a character's bible name.
    """
    ch = index.characters.get(entity_id)
    if ch is None:
        return entity_label(index, entity_id)
    if ch.body == "quadruped":
        look = _descriptor(ch.appearance, APPEARANCE_WORDS)
        return f"the animal ({look})" if look else "the animal"
    wardrobe = _descriptor(ch.wardrobe, WARDROBE_WORDS)
    if wardrobe:
        return f"the person wearing {wardrobe}"
    look = _descriptor(ch.appearance, APPEARANCE_WORDS)
    return f"the person ({look})" if look else "the person"


def fact_questions(plan: ScenePlan, index: SceneIndex, frame: FrameState, visible_ids: Sequence[str]) -> List[str]:
    """Atomic yes/no questions about the solved state of the visible entities; every true answer is "yes"."""
    kind = location_kind(index, frame)
    questions: List[str] = []
    asked_held = set()
    entities = [eid for eid in index.entities
                if eid in visible_ids and eid in frame.entities and frame.entities[eid].present]
    for eid in entities:
        ef = frame.entities[eid]
        label = visual_label(index, eid)
        # 1. Presence and custody.
        if eid in index.characters:
            questions.append(f"Is {label} visible in the image?")
        if ef.holding_id != NONE and ef.holding_id in index.entities:
            questions.append(f"Is {label} holding {visual_label(index, ef.holding_id)}?")
            asked_held.add(ef.holding_id)
        # 2. Damage and fill state.
        for flag in shown_flags(ef.flags):
            if flag in FLAG_QUESTIONS:
                questions.append(FLAG_QUESTIONS[flag].format(label=label))
        # 3. Support of movable props.
        prop = index.props.get(eid)
        if prop is not None and prop.movable and eid not in asked_held:
            if ef.support == "airborne":
                questions.append(f"Is {label} in mid-air?")
            elif ef.support == "on_entity" and ef.support_id in index.entities:
                questions.append(f"Is {label} resting on {visual_label(index, ef.support_id)}?")
            elif ef.support == "ground":
                questions.append(f"Is {label} on the {surface_name(index, None, kind)}?")
        # 4. Expression (people only: an animal's mood is not reliably judged from a still).
        if eid in index.characters and not is_quadruped(index, eid) and ef.expression in EXPRESSION_WORDS:
            questions.append(f"Does {label} look {EXPRESSION_WORDS[ef.expression]}?")
    return [_capitalize(q) for q in dict.fromkeys(questions)][:MAX_FACT_QUESTIONS]


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text else text
