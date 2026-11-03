"""
Keyframe conditioning: the prompt and the engine request for one keyframe.

The image model only renders. Everything it needs is decided upstream and compiled here:
  - compile_prompt(): a <= 120-word prompt built from enums through fixed phrase tables
    (style, shot grammar, cast with screen thirds, the keyframe action, physics phrases,
    persistent damage/fill flags, location and light). No free-form physics text.
  - build(): reference images (cast sheet, continuity frame), the init image and, for a static
    camera after the shot's first keyframe, a change mask M so only entities whose state changed
    are regenerated while everything else is copied from the previous accepted frame. "Static"
    means the solved cameras of both keyframes are identical (the solver re-aims a shot whose
    subject moves even when its move is "static"); otherwise the frame is regenerated from blocking.

Change mask M: union of the id-mask pixels (current and previous keyframe) of every entity whose
solved state changed, dilated by 16 px and snapped to the 16 px grid (one latent patch), so the
engine's mask and the bit-exact lock agree on cell boundaries.
"""
import dataclasses
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

from src import config
from src.config import GenerationDefaults, Tier
from src.engine.base import GenerationRequest
from src.previs import seeds
from src.previs.schema import FILL_FLAGS, NONE, Keyframe, ScenePlan, Shot
from src.previs.state import SELF_KINDS, SceneIndex, event_mover
from src.previs.types import BlockingResult, CameraState, EntityFrame, FrameState

if TYPE_CHECKING:
    from src.previs.assets import AssetBank
    from src.previs.memory import KeyframeMemory

MAX_PROMPT_WORDS = 120
MAX_APPEARANCE_WORDS = 12
MAX_ACTION_WORDS = 25
MAX_PHYSICS_WORDS = 25
MAX_REFS = 3
VISIBLE_MIN_FRAC = 0.01        # blocking visible fraction above which an entity counts as on screen
MIN_CHARACTER_AREA_FRAC = 0.02 # a non-subject character smaller on screen (bbox / frame) is not described,
                               # referenced or checked: describing an edge sliver makes the model paint extra people
MIN_PROP_AREA_FRAC = 0.002
MOTION_BLUR_SPEED_MS = 1.5     # solved speed above which a moving entity is described as blurred
LOCK_DILATE_PX = 16
LOCK_GRID_PX = 16
NOISE_STEP_PER_ATTEMPT = 0.03
NOISE_CAP = 0.9
CONTINUITY_BLUR_RADIUS = 0.5
POS_EPS_M = 0.01
ANGLE_EPS_DEG = 1.0
# Solved cameras closer than this count as the same camera (the static lock copies pixels, so even a few
# millimetres of drift misplace the set; 0.1 mm stays under half a pixel at hero size for an 85 mm ECU).
CAMERA_POS_EPS_M = 1e-4
CAMERA_ANGLE_EPS_DEG = 1e-3
CAMERA_LENS_EPS_MM = 1e-3
NEGATIVE_PROMPT = "text, watermark, extra limbs, duplicate people, deformed hands, blurry faces"
CLAUSE_GRACE_WORDS = 1         # compact(): a first clause this much over budget is kept whole, not cut mid-phrase
# Words a cut phrase must not end on ("terrace on", "with a", "in her").
_DANGLING_WORDS = frozenset({"a", "an", "the", "and", "or", "with", "of", "in", "on", "at", "to", "for", "by",
                             "from", "into", "over", "under", "her", "his", "its", "their", "very", "quite",
                             "slightly", "rather"})

ORDINALS = {1: "first", 2: "second", 3: "third"}

SIZE_PHRASES: Dict[str, str] = {
    "ews": "Extreme wide", "ws": "Wide", "fs": "Full", "ms": "Medium", "mcu": "Medium close-up",
    "cu": "Close-up", "ecu": "Extreme close-up", "insert": "Insert",
}
ANGLE_PHRASES: Dict[str, str] = {
    "eye": "eye-level angle", "high": "high angle looking down", "low": "low angle looking up",
    "overhead": "overhead top-down angle", "dutch": "dutch tilted angle", "ground": "ground-level angle",
}
# The angle phrase describes the SOLVED camera (camera.py may lift it above shot.angle to see the subject,
# clamp it above the floor, or tilt it), so the prompt never says "low angle" over a high-angle init image.
# Pitch > 0 looks up (camera.py conventions).
OVERHEAD_PITCH_DEG = -65.0     # at or below: looking (nearly) straight down
HIGH_PITCH_DEG = -10.0         # at or below: high angle
LOW_PITCH_DEG = 5.0            # at or above: low angle
GROUND_LEVEL_M = 0.5           # a level camera lower than this above the floor is a ground-level view
DUTCH_ROLL_MIN_DEG = 6.0       # a dutch angle rolls 12 deg; handheld shake rolls under 1 deg
TIME_OF_DAY_PHRASES: Dict[str, str] = {
    "dawn": "at dawn", "day": "in daytime", "dusk": "at dusk", "night": "at night",
}
KEY_LIGHT_PHRASES: Dict[str, str] = {
    "window_left": "soft window light from the left", "window_right": "soft window light from the right",
    "overhead": "overhead ceiling light", "practical": "warm practical lamp light",
    "sun_high": "hard high sunlight", "sun_low": "low warm raking sunlight", "moon": "cool moonlight",
}

# Deterministic phrases from plan enums. Templates use {label} (entity) and {surface} (what it rests on).
# Rules: present-tense instants, no numbers or units, no "about to", no causal clauses.
PHYSICS_PHRASES: Dict[str, Dict[str, str]] = {
    "phase": {
        "anticipation": "{label} poised and coiled for the motion",
        "contact": "{label} at the instant of contact",
        "release": "{label} just let go",
        "rising": "{label} rising through the air",
        "apex": "{label} hanging at the top of its arc",
        "falling": "{label} falling through the air",
        "impact": "{label} hitting the {surface}",
        "rebound": "{label} bouncing up off the {surface}",
        "aftermath": "{label} settling on the {surface}",
        "rest": "{label} at rest on the {surface}",
        "in_progress": "{label} mid-motion",
        "none": "",
    },
    # Phase phrases specialised by event kind ("<phase>/<kind>"), used before the generic ones.
    "phase_kind": {
        "release/fall": "{label} tipping over the edge",
        "release/drop": "{label} slipping from the hand",
        "release/throw": "{label} just leaving the hand",
        "impact/collide": "{label} colliding",
        "impact/catch": "{label} at the instant of the catch",
        "contact/push": "{label} at the instant of contact",
        "aftermath/sit_down": "{label} settling onto the {surface}",
        "rest/sit_down": "{label} settled on the {surface}",
    },
    "support": {
        "airborne": "{label} in mid-air",
        "shadow": "a detached shadow on the {surface} below",
        "held": "{label} held in a firm grip",
    },
    "motion": {
        "fast": "{label} motion-blurred",
    },
    "secondary": {
        "hair_trail": "hair trailing behind the motion",
        "cloth_flutter": "clothing fluttering",
        "liquid_arc": "an arc of liquid streaming through the air",
        "splash": "a crown splash of liquid",
        "droplets": "droplets scattered in the air",
        "puddle": "a puddle spreading on the {surface}",
        "shards": "sharp shards scattered across the {surface}",
        "debris": "debris strewn on the {surface}",
        "dust": "a puff of dust hanging in the air",
    },
    "flags": {
        "cracked": "cracked", "shattered": "shattered into pieces", "dented": "dented", "torn": "torn",
        "crumpled": "crumpled", "spilled": "spilled", "stained": "stained", "wet": "wet", "dusty": "dusty",
        "fill_empty": "empty", "fill_half": "half full", "fill_full": "full",
    },
    "weight_cue": {
        "none": "", "leaning": "leaning in", "bracing": "bracing", "straining": "straining with effort",
        "off_balance": "off balance", "recoil": "recoiling",
    },
    "expression": {
        "none": "", "neutral": "neutral expression", "happy": "smiling", "sad": "sad expression",
        "angry": "angry scowl", "surprised": "startled wide-eyed expression", "afraid": "fearful expression",
        "focused": "focused expression", "pained": "wincing in pain",
    },
    "pose": {
        "none": "", "stand": "standing", "walk": "walking", "run": "running", "sit": "sitting",
        "crouch": "crouching", "kneel": "kneeling", "lie": "lying down", "reach": "reaching forward",
        "lift": "lifting", "carry": "carrying", "throw": "throwing", "push": "pushing", "pull": "pulling",
        "fall": "falling", "brace": "braced",
    },
    # Character.body == "quadruped": animal body language, never human pose/expression words.
    "pose_quadruped": {
        "none": "", "stand": "standing on all four legs", "walk": "walking on all fours", "run": "running at full stride",
        "sit": "sitting on its haunches", "crouch": "crouching low", "kneel": "crouching low", "lie": "lying down",
        "reach": "stretching its neck forward", "lift": "picking up", "carry": "carrying", "throw": "tossing",
        "push": "nudging", "pull": "tugging", "fall": "tumbling over", "brace": "bracing on all four legs",
    },
    "expression_quadruped": {
        "none": "", "neutral": "", "happy": "playful and relaxed", "sad": "head and ears low", "angry": "teeth bared",
        "surprised": "ears pricked up", "afraid": "cowering", "focused": "alert, ears forward", "pained": "flinching",
    },
}
# Poses that take the held object as their direct object ("carrying the glass").
_TRANSITIVE_POSES = frozenset({"lift", "carry", "throw", "push", "pull"})
# Quadrupeds hold what they lift or carry in the mouth.
_MOUTH_POSES = frozenset({"lift", "carry"})
# Ballistic phases already say the entity is in the air.
_AIRBORNE_PHASES = frozenset({"release", "rising", "apex", "falling", "rebound"})
# Phases whose generic phrase puts the mover on a surface ("at rest on the floor").
_SURFACE_PHASES = frozenset({"impact", "rebound", "aftermath", "rest"})
QUADRUPED_NOUN = "a four-legged animal"

# Compression levels tried in order until the prompt fits MAX_PROMPT_WORDS ("flags": how many entities'
# damage/fill sentences are stated, most important entity first).
_LEVELS: List[Dict[str, int]] = [
    {"appearance": 12, "action": 25, "physics": 25, "location": 25, "style": 14, "flags": 8},
    {"appearance": 10, "action": 25, "physics": 20, "location": 16, "style": 10, "flags": 6},
    {"appearance": 8, "action": 22, "physics": 15, "location": 12, "style": 8, "flags": 4},
    {"appearance": 6, "action": 18, "physics": 10, "location": 8, "style": 6, "flags": 3},
    {"appearance": 4, "action": 15, "physics": 8, "location": 5, "style": 5, "flags": 2},
]


@dataclass
class ConditioningPlan:
    request: GenerationRequest
    static_lock: bool
    lock_mask_path: Optional[str]
    continuity_source: str
    ancestry_depth: int


# ---------------------------------------------------------------- text helpers
def word_count(text: str) -> int:
    return len(text.split())


def compact(text: str, max_words: int) -> str:
    """
    Leading comma/semicolon clauses of `text` that fit `max_words`. A longer first clause is kept whole
    when it is at most CLAUSE_GRACE_WORDS over, else cut without a dangling 'on' / 'with a' / 'her'.
    """
    clauses = [c.strip() for c in re.split(r"[,;]", text.strip().rstrip(".")) if c.strip()]
    out: List[str] = []
    count = 0
    for clause in clauses:
        n = word_count(clause)
        if count + n > max_words:
            break
        out.append(clause)
        count += n
    if not out and clauses and max_words > 0:
        words = clauses[0].split()
        if len(words) > max_words + CLAUSE_GRACE_WORDS:
            words = words[:max_words]
            while len(words) > 1 and words[-1].lower() in _DANGLING_WORDS:
                words.pop()
        out = [" ".join(words)]
    return ", ".join(out)


def _clip_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    return " ".join(words[:max_words]).rstrip(".,;:")


def _sentence(text: str) -> str:
    text = text.strip().rstrip(".,;:")
    return f"{text[0].upper()}{text[1:]}." if text else ""


def _join_and(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _lens(lens_mm: float) -> str:
    return f"{int(round(lens_mm))}" if abs(lens_mm - round(lens_mm)) < 0.05 else f"{lens_mm:.1f}"


def angle_phrase(shot: Shot, camera: Optional[CameraState]) -> str:
    """
    Camera angle as the solved camera films it: pitch bands (overhead / high / low), a level camera near
    the floor is ground-level, otherwise eye-level; a rolled camera is dutch. Without a solved camera the
    shot's planned angle is used.
    """
    if camera is None:
        return ANGLE_PHRASES.get(shot.angle, shot.angle)
    if camera.pitch_deg <= OVERHEAD_PITCH_DEG:
        key = "overhead"
    elif camera.pitch_deg <= HIGH_PITCH_DEG:
        key = "high"
    elif camera.pitch_deg >= LOW_PITCH_DEG:
        key = "low"
    elif camera.position[1] < GROUND_LEVEL_M:
        key = "ground"
    else:
        key = "eye"
    if abs(camera.roll_deg) >= DUTCH_ROLL_MIN_DEG:
        return ANGLE_PHRASES["dutch"] if key == "eye" else f"dutch tilted {ANGLE_PHRASES[key]}"
    return ANGLE_PHRASES[key]


# ---------------------------------------------------------------- entity phrases
def entity_label(index: SceneIndex, entity_id: str) -> str:
    """Characters by name, props as 'the <name>'."""
    if entity_id in index.characters:
        return index.characters[entity_id].name
    if entity_id in index.props:
        return f"the {index.props[entity_id].name}"
    return entity_id


def is_quadruped(index: SceneIndex, entity_id: str) -> bool:
    ch = index.characters.get(entity_id)
    return ch is not None and ch.body == "quadruped"


def surface_name(index: SceneIndex, ef: Optional[EntityFrame], kind: str) -> str:
    """What an entity rests on or lands on: its support entity, else the floor/ground."""
    if ef is not None and ef.support == "on_entity" and ef.support_id in index.props:
        return index.props[ef.support_id].name
    return "floor" if kind == "interior" else "ground"


def location_kind(index: SceneIndex, frame: FrameState) -> str:
    location = index.locations.get(frame.location_id)
    return location.kind if location is not None else "interior"


def shown_flags(flags: Sequence[str]) -> List[str]:
    """Flags worth stating: a shattered container no longer has a fill level."""
    shown = list(dict.fromkeys(flags))
    if "shattered" in shown:
        shown = [f for f in shown if f not in FILL_FLAGS]
    return [f for f in shown if PHYSICS_PHRASES["flags"].get(f)]


def flag_phrase(flags: Sequence[str]) -> str:
    """'shattered into pieces and spilled' for a flag list (empty when nothing is worth stating)."""
    return _join_and([PHYSICS_PHRASES["flags"][f] for f in shown_flags(flags)])


def screen_third(bbox: Tuple[int, int, int, int], width: int) -> str:
    centre = (bbox[0] + bbox[2]) / 2.0
    if centre < width / 3.0:
        return "left third of frame"
    if centre > 2.0 * width / 3.0:
        return "right third of frame"
    return "centre of frame"


def _projected_bbox(frame: FrameState, ef: EntityFrame) -> Optional[Tuple[int, int, int, int]]:
    """Screen bbox of an entity's bounding box through the solved camera (camera.project)."""
    if frame.camera is None:
        return None
    from src.previs.camera import project

    x, y, z = ef.position
    sx, sy, sz = ef.size
    corners = np.array([(x + dx * sx / 2, y + dy * sy, z + dz * sz / 2)
                        for dx in (-1, 1) for dy in (0, 1) for dz in (-1, 1)], dtype=float)
    uv, depth = project(frame.camera, corners)
    uv = np.asarray(uv)[np.asarray(depth) > 0]
    if not len(uv):
        return None
    return (int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max()))


def _pose_clause(index: SceneIndex, ef: EntityFrame) -> str:
    quadruped = is_quadruped(index, ef.entity_id)
    pose = PHYSICS_PHRASES["pose_quadruped" if quadruped else "pose"].get(ef.pose, "")
    if ef.holding_id != NONE and ef.holding_id in index.entities:
        held = entity_label(index, ef.holding_id)
        if ef.pose in _TRANSITIVE_POSES:
            pose = f"{pose} {held}" + (" in its mouth" if quadruped and ef.pose in _MOUTH_POSES else "")
        else:
            holding = f"holding {held}" + (" in its mouth" if quadruped else "")
            pose = f"{pose}, {holding}" if pose else holding
    cue = PHYSICS_PHRASES["weight_cue"].get(ef.weight_cue, "")
    return ", ".join(p for p in (pose, cue) if p)


def _phase_phrase(index: SceneIndex, frame: FrameState, event, mover: str, kind: str) -> str:
    """The focus event's mover at the depicted phase (empty when the phase says nothing true about it)."""
    ef = frame.entities[mover]
    label, surface = entity_label(index, mover), surface_name(index, ef, kind)
    specific = PHYSICS_PHRASES["phase_kind"].get(f"{frame.depicted_phase}/{event.kind}")
    if specific:
        return specific.format(label=label, surface=surface)
    if frame.depicted_phase in _SURFACE_PHASES:
        if ef.support == "held":           # lifted, caught or handed over: it is in a grip, not on the floor
            return PHYSICS_PHRASES["support"]["held"].format(label=label)
        if event.kind in SELF_KINDS:       # a walk/gesture ends however the pose says, not "at rest on the floor"
            return ""
    template = PHYSICS_PHRASES["phase"].get(frame.depicted_phase, "")
    return template.format(label=label, surface=surface) if template else ""


def physics_phrases(index: SceneIndex, frame: FrameState, visible_ids: Sequence[str],
                    max_words: int = MAX_PHYSICS_WORDS) -> List[str]:
    """Physics cues for the visible entities, in priority order, within `max_words` in total."""
    kind = location_kind(index, frame)
    visible = [eid for eid in visible_ids if eid in frame.entities and frame.entities[eid].present]
    candidates: List[str] = []

    # 1. The focus event's mover at the depicted phase.
    mover = None
    event = index.events.get(frame.focus_event_id)
    if event is not None:
        mover = event_mover(event)
        if mover in visible:
            phrase = _phase_phrase(index, frame, event, mover, kind)
            if phrase:
                candidates.append(phrase)

    # 2. Airborne entities and their detached shadows.
    airborne = [eid for eid in visible if frame.entities[eid].support == "airborne"]
    for eid in airborne:
        if not (eid == mover and frame.depicted_phase in _AIRBORNE_PHASES):
            candidates.append(PHYSICS_PHRASES["support"]["airborne"].format(label=entity_label(index, eid)))
    if airborne:
        candidates.append(PHYSICS_PHRASES["support"]["shadow"].format(surface=surface_name(index, None, kind)))

    # 3. Fast movers.
    for eid in visible:
        ef = frame.entities[eid]
        if ef.in_motion and math.sqrt(sum(v * v for v in ef.velocity)) >= MOTION_BLUR_SPEED_MS:
            candidates.append(PHYSICS_PHRASES["motion"]["fast"].format(label=entity_label(index, eid)))

    # 4. Secondary effects (each once).
    seen = set()
    for eid in visible:
        ef = frame.entities[eid]
        for effect in ef.secondary:
            if effect in seen or effect not in PHYSICS_PHRASES["secondary"]:
                continue
            seen.add(effect)
            candidates.append(PHYSICS_PHRASES["secondary"][effect].format(surface=surface_name(index, None, kind)))

    out: List[str] = []
    used = 0
    for phrase in dict.fromkeys(candidates):
        n = word_count(phrase)
        if used + n <= max_words:
            out.append(phrase)
            used += n
    return out


# ---------------------------------------------------------------- prompt
def _ordered_visible(index: SceneIndex, frame: FrameState, shot: Shot, visible_ids: Sequence[str]) -> List[str]:
    """Visible, present entities: the shot subject first, then the secondary, then plan order."""
    present = {eid for eid in visible_ids if eid in frame.entities and frame.entities[eid].present}
    lead = [eid for eid in (shot.subject_id, shot.secondary_id) if eid in present]
    return lead + [eid for eid in index.entities if eid in present and eid not in lead]


def _character_sentence(index: SceneIndex, frame: FrameState, ch_id: str, appearance_words: int,
                        cast_image: Optional[int], bboxes: Optional[Dict[str, Tuple[int, int, int, int]]],
                        image_width: Optional[int]) -> str:
    ch = index.characters[ch_id]
    ef = frame.entities[ch_id]
    quadruped = is_quadruped(index, ch_id)
    look = compact(ch.appearance, appearance_words)
    if cast_image in ORDINALS:
        look = f"{look} as in the {ORDINALS[cast_image]} image" if look else f"as in the {ORDINALS[cast_image]} image"
    expression = PHYSICS_PHRASES["expression_quadruped" if quadruped else "expression"].get(ef.expression, "")
    parts = [ch.name, QUADRUPED_NOUN if quadruped else "", look, _pose_clause(index, ef), expression]
    bbox = (bboxes or {}).get(ch_id)
    width = image_width or (frame.camera.width if frame.camera is not None else None)
    if bbox is None and bboxes is None:
        bbox = _projected_bbox(frame, ef)
    if bbox is not None and width:
        parts.append(screen_third(bbox, width))
    return _sentence(", ".join(p for p in parts if p))


def _flagged_by_priority(index: SceneIndex, frame: FrameState, shot: Shot, entity_ids: Sequence[str]) -> List[str]:
    """Entities with damage/fill flags worth stating: subject, secondary, focus mover first, then plan order."""
    event = index.events.get(frame.focus_event_id)
    lead = [shot.subject_id, shot.secondary_id, event_mover(event) if event is not None else NONE]
    ordered = list(dict.fromkeys([eid for eid in lead if eid in entity_ids] + list(entity_ids)))
    return [eid for eid in ordered if flag_phrase(frame.entities[eid].flags)]


def _flags_sentences(index: SceneIndex, frame: FrameState, entity_ids: Sequence[str]) -> List[str]:
    out = []
    for eid in entity_ids:
        phrase = flag_phrase(frame.entities[eid].flags)
        if phrase:
            out.append(_sentence(f"{entity_label(index, eid)} is {phrase}"))
    return out


def _continuity_sentence(continuity_image: Optional[int]) -> str:
    return f"Same set and lighting as the {ORDINALS[continuity_image]} image." if continuity_image in ORDINALS else ""


def _assemble(plan: ScenePlan, index: SceneIndex, frame: FrameState, shot: Shot, keyframe: Keyframe,
              entity_ids: List[str], characters: List[str], flagged: List[str], level: Dict[str, int],
              bboxes, image_width, continuity_image, cast_image) -> str:
    scene = plan.bible.scene
    parts: List[str] = [
        _sentence(compact(scene.style, level["style"])),
        _sentence(f"{SIZE_PHRASES.get(shot.size, shot.size)} shot, {angle_phrase(shot, frame.camera)}, "
                  f"{_lens(shot.lens_mm)}mm lens"),
    ]
    parts += [_character_sentence(index, frame, cid, level["appearance"], cast_image, bboxes, image_width)
              for cid in characters]
    parts.append(_sentence(_clip_words(keyframe.action, min(level["action"], MAX_ACTION_WORDS))))
    physics = physics_phrases(index, frame, entity_ids, min(level["physics"], MAX_PHYSICS_WORDS))
    if physics:
        parts.append(_sentence("; ".join(physics)))
    parts += _flags_sentences(index, frame, flagged)
    location = index.locations.get(frame.location_id) or index.locations.get(shot.location_id)
    if location is not None:
        setting = compact(location.description, level["location"]) or location.name
        light = ", ".join(p for p in (TIME_OF_DAY_PHRASES.get(scene.time_of_day, ""),
                                      KEY_LIGHT_PHRASES.get(location.key_light, "")) if p)
        parts.append(_sentence(f"{setting}, {light}" if light else setting))
    parts.append(_continuity_sentence(continuity_image))
    return " ".join(p for p in parts if p)


def _screen_area(frame: FrameState, entity_id: str, bboxes: Optional[Dict[str, Tuple[int, int, int, int]]]) -> float:
    """On-screen bbox area of an entity: blocking bboxes when given, else its projected box (clipped to the
    image), else 0."""
    if bboxes is not None:
        box = bboxes.get(entity_id)
    else:
        box = _projected_bbox(frame, frame.entities[entity_id])
        if box is not None and frame.camera is not None:
            w, h = frame.camera.width, frame.camera.height
            box = (min(max(box[0], 0), w), min(max(box[1], 0), h), min(max(box[2], 0), w), min(max(box[3], 0), h))
    if box is None:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _least_prominent(frame: FrameState, shot: Shot, characters: Sequence[str],
                     bboxes: Optional[Dict[str, Tuple[int, int, int, int]]]) -> str:
    """The character to stop describing first: never the subject, the secondary only after the others,
    otherwise the smallest on screen (ties: the last in prompt order)."""
    rank = {shot.subject_id: 2, shot.secondary_id: 1}
    return min(enumerate(characters),
               key=lambda ic: (rank.get(ic[1], 0), _screen_area(frame, ic[1], bboxes), -ic[0]))[1]


def compile_prompt(
    plan: ScenePlan,
    index: SceneIndex,
    frame: FrameState,
    shot: Shot,
    keyframe: Keyframe,
    visible_ids: Sequence[str],
    *,
    bboxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    image_width: Optional[int] = None,
    continuity_image: Optional[int] = None,
    cast_image: Optional[int] = 1,
) -> str:
    """
    Prompt of at most MAX_PROMPT_WORDS words for one keyframe.

    `bboxes` (blocking bboxes) and `image_width` place characters in screen thirds; without them the
    solved camera is used when present. The angle phrase describes the solved camera (frame.camera),
    which may film from higher than shot.angle (see angle_phrase). `cast_image` / `continuity_image` are
    the 1-based positions of the cast sheet and continuity frame among the reference images (None: not used).
    """
    entity_ids = _ordered_visible(index, frame, shot, visible_ids)
    characters = [eid for eid in entity_ids if eid in index.characters]
    flagged = _flagged_by_priority(index, frame, shot, entity_ids)

    def assemble(level: Dict[str, int], chars: List[str], n_flags: int, continuity: Optional[int]) -> str:
        return _assemble(plan, index, frame, shot, keyframe, entity_ids, chars, flagged[:n_flags], level,
                         bboxes, image_width, continuity, cast_image)

    # 1. Progressively compress text fields until the prompt fits.
    for level in _LEVELS:
        prompt = assemble(level, characters, level["flags"], continuity_image)
        if word_count(prompt) <= MAX_PROMPT_WORDS:
            return prompt
    # 2. Crowded frames: state fewer damage flags, then describe fewer characters: the least prominent on
    #    screen first, the secondary next, the subject never.
    level = _LEVELS[-1]
    for n_flags in range(level["flags"] - 1, 0, -1):
        prompt = assemble(level, characters, n_flags, continuity_image)
        if word_count(prompt) <= MAX_PROMPT_WORDS:
            return prompt
    n_flags = min(1, level["flags"])
    while len(characters) > 1:
        victim = _least_prominent(frame, shot, characters, bboxes)
        characters = [c for c in characters if c != victim]
        prompt = assemble(level, characters, n_flags, continuity_image)
        if word_count(prompt) <= MAX_PROMPT_WORDS:
            return prompt
    # 3. Last resort: hard cut of everything but the continuity sentence, which always ends the prompt.
    tail = _continuity_sentence(continuity_image)
    body = assemble(level, characters, n_flags, None).split()[:MAX_PROMPT_WORDS - word_count(tail)]
    return " ".join(body + tail.split())


# ---------------------------------------------------------------- state change + masks
def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def entity_changed(prev: Optional[EntityFrame], cur: Optional[EntityFrame]) -> bool:
    """True when an entity looks different between two keyframes (pose, place, flags, motion...)."""
    if prev is None or cur is None:
        return (prev is not None and prev.present) or (cur is not None and cur.present)
    if prev.present != cur.present:
        return True
    if not cur.present:
        return False
    if prev.in_motion or cur.in_motion:
        return True
    if max(abs(a - b) for a, b in zip(prev.position, cur.position)) > POS_EPS_M:
        return True
    if _angle_delta(prev.yaw_deg, cur.yaw_deg) > ANGLE_EPS_DEG:
        return True
    if any(_angle_delta(a, b) > ANGLE_EPS_DEG for a, b in zip(prev.rotation_deg, cur.rotation_deg)):
        return True
    for name in ("pose", "expression", "holding_id", "gaze_id", "support", "support_id", "weight_cue"):
        if getattr(prev, name) != getattr(cur, name):
            return True
    return set(prev.flags) != set(cur.flags) or set(prev.secondary) != set(cur.secondary)


def changed_entities(prev_frame: FrameState, frame: FrameState) -> List[str]:
    ids = list(frame.entities) + [eid for eid in prev_frame.entities if eid not in frame.entities]
    return [eid for eid in ids if entity_changed(prev_frame.entities.get(eid), frame.entities.get(eid))]


def previous_keyframe_in_shot(index: SceneIndex, kf_id: str) -> Optional[str]:
    """The keyframe before `kf_id` in the same shot, or None for a shot's first keyframe."""
    shot_id = index.shot_of_keyframe.get(kf_id)
    ids = [kf.id for kf in index.shot_keyframes(shot_id)] if shot_id else []
    pos = ids.index(kf_id) if kf_id in ids else 0
    return ids[pos - 1] if pos > 0 else None


def cameras_match(a: Optional[CameraState], b: Optional[CameraState]) -> bool:
    """True when two solved cameras film the same image (pose, lens and resolution within a tiny tolerance).
    An unknown (None) camera never matches."""
    if a is None or b is None or (a.width, a.height) != (b.width, b.height):
        return False
    if max(abs(p - q) for p, q in zip(a.position, b.position)) > CAMERA_POS_EPS_M:
        return False
    angles = ((a.yaw_deg, b.yaw_deg), (a.pitch_deg, b.pitch_deg), (a.roll_deg, b.roll_deg), (a.vfov_deg, b.vfov_deg))
    if any(_angle_delta(p, q) > CAMERA_ANGLE_EPS_DEG for p, q in angles):
        return False
    return abs(a.lens_mm - b.lens_mm) <= CAMERA_LENS_EPS_MM


def previous_solved_frame(plan: ScenePlan, index: SceneIndex, frame: FrameState,
                          prev_frame: Optional[FrameState] = None) -> Optional[FrameState]:
    """
    Solved state of the keyframe before `frame` in its shot (None for a shot's first keyframe).
    `prev_frame` is used when given; when it is missing, or lacks the camera that `frame` has, it is
    re-solved deterministically (physics, then cameras at `frame`'s resolution).
    """
    prev_kf_id = previous_keyframe_in_shot(index, frame.kf_id)
    if prev_kf_id is None:
        return None
    if prev_frame is not None and (prev_frame.camera is not None or frame.camera is None):
        return prev_frame
    from src.previs import physics
    solved = physics.solve(plan)
    if frame.camera is not None:
        from src.previs import camera
        camera.solve_cameras(plan, solved, index, frame.camera.width, frame.camera.height)
    derived = solved.frame(prev_kf_id)
    return derived if prev_frame is None else dataclasses.replace(prev_frame, camera=derived.camera)


def load_id_mask(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """id.png as a 2-D integer array, nearest-resized to `size` (w, h) when given."""
    img = Image.open(path)
    if img.mode not in ("L", "I", "I;16", "P"):
        img = img.convert("L")
    if size is not None and img.size != tuple(size):
        img = img.resize(tuple(size), Image.Resampling.NEAREST)
    arr = np.array(img)
    return arr[..., 0] if arr.ndim == 3 else arr


def id_values(blocking: BlockingResult) -> Dict[str, int]:
    """entity id -> pixel value in id.png."""
    return {eid: int(value) for value, eid in blocking.id_map.items()}


def entity_mask(blocking: BlockingResult, entity_id: str, size: Optional[Tuple[int, int]] = None,
                ids: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean pixel mask of one entity in a blocking id map (empty when it is not drawn)."""
    if ids is None:
        ids = load_id_mask(blocking.id_path, size)
    value = id_values(blocking).get(entity_id)
    return ids == value if value is not None else np.zeros(ids.shape, dtype=bool)


def bbox_of(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1), int(y1)


def load_sibling_blocking(blocking: BlockingResult, kf_id: str) -> Optional[BlockingResult]:
    """
    Blocking of another keyframe rendered next to `blocking` (graybox/<kf_id>/, run-dir layout).
    The id map comes from that keyframe's meta.json when it records one, else from `blocking`.
    """
    folder = Path(blocking.id_path).parent.parent / kf_id
    id_path = folder / "id.png"
    if not id_path.is_file():
        return None
    id_map = dict(blocking.id_map)
    meta_path = folder / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}
        if isinstance(meta, dict) and isinstance(meta.get("id_map"), dict):
            id_map = {int(k): str(v) for k, v in meta["id_map"].items()}
    ids = load_id_mask(str(id_path))
    bboxes = {}
    for value, eid in id_map.items():
        box = bbox_of(ids == value)
        if box is not None:
            bboxes[eid] = box
    return BlockingResult(
        blocking_path=str(folder / "blocking.png"), id_path=str(id_path), depth_path=str(folder / "depth.npy"),
        id_map=id_map, bboxes=bboxes, visible_frac={},
    )


def snap_to_grid(mask: np.ndarray, grid: int = LOCK_GRID_PX) -> np.ndarray:
    """Every grid cell touched by the mask becomes fully masked."""
    h, w = mask.shape
    gh, gw = -(-h // grid), -(-w // grid)
    padded = np.zeros((gh * grid, gw * grid), dtype=bool)
    padded[:h, :w] = mask
    cells = padded.reshape(gh, grid, gw, grid).any(axis=(1, 3))
    return np.repeat(np.repeat(cells, grid, axis=0), grid, axis=1)[:h, :w]


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0 or not mask.any():
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) <= radius_px


def change_mask(entity_ids: Sequence[str], blocking: BlockingResult, prev_blocking: Optional[BlockingResult],
                size: Tuple[int, int], dilate_px: int = LOCK_DILATE_PX, grid_px: int = LOCK_GRID_PX) -> np.ndarray:
    """Union of the entities' id-mask pixels now and at the previous keyframe, dilated and grid-snapped."""
    w, h = size
    union = np.zeros((h, w), dtype=bool)
    sources = [blocking] + ([prev_blocking] if prev_blocking is not None else [])
    for source in sources:
        ids = load_id_mask(source.id_path, size)
        for eid in entity_ids:
            union |= entity_mask(source, eid, ids=ids)
    return snap_to_grid(dilate(union, dilate_px), grid_px)


def _rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img if img.size == tuple(size) else img.resize(tuple(size), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------- request
def on_screen_ids(frame: FrameState, blocking: BlockingResult, subject_id: str = NONE) -> List[str]:
    """
    Entities significant enough on screen to describe in the prompt, put on the cast sheet and QA-check.
    The shot's subject counts whenever it is visible at all; others also need a minimum bbox area.
    """
    with Image.open(blocking.id_path) as id_img:
        frame_area = float(id_img.size[0] * id_img.size[1])
    ids = []
    for eid, frac in blocking.visible_frac.items():
        ef = frame.entities.get(eid)
        if frac < VISIBLE_MIN_FRAC or ef is None or not ef.present:
            continue
        if eid != subject_id:
            x0, y0, x1, y1 = blocking.bboxes.get(eid, (0, 0, 0, 0))
            area = max(0, x1 - x0) * max(0, y1 - y0) / frame_area
            if area < (MIN_CHARACTER_AREA_FRAC if ef.kind == "character" else MIN_PROP_AREA_FRAC):
                continue
        ids.append(eid)
    return ids


def _visible_ids(frame: FrameState, blocking: BlockingResult, subject_id: str = NONE) -> List[str]:
    return on_screen_ids(frame, blocking, subject_id)


def build(
    plan: ScenePlan,
    index: SceneIndex,
    frame: FrameState,
    shot: Shot,
    keyframe: Keyframe,
    blocking: BlockingResult,
    assets: "AssetBank",
    memory: "KeyframeMemory",
    tier: Tier,
    defaults: GenerationDefaults,
    attempt: int,
    out_dir: Path,
    prev_accepted_path: Optional[str],
    prev_frame: Optional[FrameState] = None,
    prev_blocking: Optional[BlockingResult] = None,
) -> ConditioningPlan:
    """
    Conditioning for one keyframe attempt. Writes init.png, mask.png (static lock only),
    continuity.png and request.json into `out_dir`.

    `prev_frame` / `prev_blocking` (previous keyframe of the same shot) are derived when omitted:
    the frame by re-solving the plan (deterministic), the blocking from graybox/<prev_kf_id>/.
    The static lock needs a "static" shot AND identical solved cameras at both keyframes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (tier.width, tier.height)
    visible = _visible_ids(frame, blocking, shot.subject_id)
    ordered = _ordered_visible(index, frame, shot, visible)
    characters = [eid for eid in ordered if eid in index.characters]

    # 1. References: cast sheet of the visible characters, then the continuity frame.
    refs: List[str] = []
    cast_image = None
    if characters:
        for ch_id in characters:
            assets.ensure_character(index.characters[ch_id])      # cached after the first call
        sheet = assets.cast_sheet(characters, tier.sheet_size, name=f"{frame.kf_id}_cast")
        if sheet and Path(sheet).is_file():
            refs.append(str(sheet))
            cast_image = len(refs)
    location = index.locations.get(frame.location_id) or index.locations.get(shot.location_id)
    plate = assets.ensure_plate(location, plan.bible.scene.time_of_day) if location is not None else None
    shot_kf_ids = [kf.id for kf in index.shot_keyframes(shot.id)]
    kf_order = shot_kf_ids.index(frame.kf_id) + 1 if frame.kf_id in shot_kf_ids else 1
    camera_yaw = frame.camera.yaw_deg if frame.camera is not None else 0.0
    cont_path, cont_source, cont_depth = memory.continuity_for(
        shot.id, kf_order, shot.setup_id, frame.location_id, camera_yaw, plate)
    continuity_image = None
    if cont_path and Path(cont_path).is_file() and len(refs) < MAX_REFS:
        continuity = ImageOps.fit(Image.open(cont_path).convert("RGB"), tuple(tier.continuity_size),
                                  Image.Resampling.LANCZOS)
        continuity = continuity.filter(ImageFilter.GaussianBlur(CONTINUITY_BLUR_RADIUS))
        continuity_file = out_dir / "continuity.png"
        continuity.save(continuity_file)
        refs.append(str(continuity_file))
        continuity_image = len(refs)
    else:
        cont_source, cont_depth = "none", 0

    # 2. Init image (+ change mask for a static camera after the shot's first keyframe).
    strength = min(defaults.init_noise + NOISE_STEP_PER_ATTEMPT * attempt, NOISE_CAP)
    init_path = out_dir / "init.png"
    mask_path: Optional[Path] = None
    changed: List[str] = []
    prev_kf_id = previous_keyframe_in_shot(index, frame.kf_id)
    static_lock = False
    if (shot.move == "static" and kf_order > 1 and prev_kf_id is not None
            and bool(prev_accepted_path) and Path(prev_accepted_path).is_file()):
        # The solver re-aims even a "static" shot to follow its subject; the previous frame's pixels are
        # only valid where the camera really did not move.
        prev_frame = previous_solved_frame(plan, index, frame, prev_frame)
        static_lock = prev_frame is not None and cameras_match(prev_frame.camera, frame.camera)
    proxy = _rgb(blocking.blocking_path, size)
    if static_lock:
        if prev_blocking is None:
            prev_blocking = load_sibling_blocking(blocking, prev_kf_id)
        changed = changed_entities(prev_frame, frame)
        mask = change_mask(changed, blocking, prev_blocking, size)
        init = np.array(_rgb(prev_accepted_path, size))
        init[mask] = np.array(proxy)[mask]
        Image.fromarray(init).save(init_path)
        mask_path = out_dir / "mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    else:
        proxy.save(init_path)

    # 3. Prompt and request.
    prompt = compile_prompt(plan, index, frame, shot, keyframe, visible, bboxes=blocking.bboxes,
                            image_width=tier.width, continuity_image=continuity_image, cast_image=cast_image)
    request = GenerationRequest(
        prompt=prompt,
        output_path=str(out_dir / "output.png"),
        width=tier.width,
        height=tier.height,
        seed=seeds.seed_for(assets.project, shot.id, frame.kf_id, attempt),
        steps=defaults.steps,
        cfg_scale=defaults.cfg_scale,
        sampler=defaults.sampler,
        scheduler=defaults.scheduler,
        sigmas=list(config.VIGGLE_SIGMAS) if defaults.use_custom_sigmas else None,
        ref_images=refs[:MAX_REFS],
        init_image=str(init_path),
        mask_image=str(mask_path) if mask_path else None,
        strength=strength,
        strength_as_noise_level=True,
        negative_prompt=NEGATIVE_PROMPT if defaults.cfg_scale > 1.0 else "",
        tag=f"{frame.kf_id}_attempt{attempt}",
    )
    cond = ConditioningPlan(request=request, static_lock=static_lock,
                            lock_mask_path=str(mask_path) if mask_path else None,
                            continuity_source=cont_source, ancestry_depth=cont_depth)
    record = {**dataclasses.asdict(request), "static_lock": static_lock, "lock_mask_path": cond.lock_mask_path,
              "continuity_source": cont_source, "ancestry_depth": cont_depth, "visible_ids": ordered,
              "changed_entities": changed, "prompt_words": word_count(prompt)}
    (out_dir / "request.json").write_text(json.dumps(record, indent=2))
    return cond
