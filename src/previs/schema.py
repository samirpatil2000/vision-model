"""
Scene-state schema shared by the Claude physics planner, the solver, and the renderer.

Two documents come out of the single physics pass:
  - SceneBible (call A): entities, locations, marks, beats, shots, physics bodies, initial state.
  - Timeline   (call B): invariants, then per-shot keyframes, physics events, state snapshots, cause links.

Design rules (they keep one schema valid for Claude structured output):
  - Flat records that reference each other by ID. No recursion, no unions, no optional fields.
  - Absent values use sentinels: "none" for IDs/enums, [] for lists, 0 for numbers.
  - IDs are lowercase with a type prefix (ch_, pr_, lo_, mk_, bt_, sh_, su_, kf_, ev_, iv_, cl_).
  - Claude states intent (what happens, outcomes, classes). Code derives geometry and time.
  - Ranges and counts are enforced in validator.py, never in the JSON Schema
    (structured output rejects minimum/maximum/maxItems).
"""
import copy
import typing
from typing import Any, Dict, List, Literal, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

NONE = "none"

ID_PREFIXES = {
    "character": "ch_",
    "prop": "pr_",
    "location": "lo_",
    "mark": "mk_",
    "beat": "bt_",
    "shot": "sh_",
    "setup": "su_",
    "keyframe": "kf_",
    "event": "ev_",
    "invariant": "iv_",
    "cause_link": "cl_",
}


def id_pattern(*prefixes: str, allow_none: bool = False) -> str:
    alternatives = "|".join(p.rstrip("_") for p in prefixes)
    core = f"(?:{alternatives})_[a-z0-9_]{{1,28}}"
    return f"^(?:{core}|none)$" if allow_none else f"^{core}$"


ENTITY_ID = id_pattern("ch_", "pr_")
ENTITY_ID_OR_NONE = id_pattern("ch_", "pr_", allow_none=True)

# ---------------------------------------------------------------- enums
Aspect = Literal["16:9", "2.39:1"]
TimeOfDay = Literal["dawn", "day", "dusk", "night"]
KeyLight = Literal["window_left", "window_right", "overhead", "practical", "sun_high", "sun_low", "moon"]
LocationKind = Literal["interior", "exterior"]
PropShape = Literal["box", "cylinder", "sphere", "plane"]
Body = Literal["human", "quadruped"]
ShotSize = Literal["ews", "ws", "fs", "ms", "mcu", "cu", "ecu", "insert"]
Angle = Literal["eye", "high", "low", "overhead", "dutch"]
FrameSide = Literal["left", "center", "right"]
LineSide = Literal["a", "b"]
Move = Literal[
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down", "dolly_in", "dolly_out",
    "truck_left", "truck_right", "crane_up", "crane_down", "orbit_left", "orbit_right", "handheld",
]
Transition = Literal["cut", "dissolve", "match_cut"]
DurationClass = Literal["short", "medium", "long"]
TimeScale = Literal["real", "slow_2x", "slow_4x", "ramp"]
Regime = Literal["rigid", "articulated", "cloth", "rope", "liquid", "granular", "static"]
Material = Literal[
    "wood", "metal", "glass", "ceramic", "plastic", "rubber", "fabric", "paper",
    "stone", "flesh", "water", "sand", "food",
]
MassClass = Literal["xs", "s", "m", "l", "xl", "fixed"]
Friction = Literal["low", "med", "high"]
Restitution = Literal["dead", "low", "med", "high"]
FailureMode = Literal["none", "shatter", "crack", "tear", "dent", "crumple", "spill"]
ContainerFill = Literal["none", "empty", "half", "full"]
Pose = Literal[
    "none", "stand", "walk", "run", "sit", "crouch", "kneel", "lie", "reach", "lift",
    "carry", "throw", "push", "pull", "fall", "brace",
]
Expression = Literal["none", "neutral", "happy", "sad", "angry", "surprised", "afraid", "focused", "pained"]
Support = Literal["ground", "on_entity", "held", "attached", "airborne", "none"]
Flag = Literal[
    "cracked", "shattered", "dented", "torn", "crumpled", "spilled", "stained",
    "wet", "dusty", "fill_empty", "fill_half", "fill_full",
]
WeightCue = Literal["none", "leaning", "bracing", "straining", "off_balance", "recoil"]
Secondary = Literal[
    "hair_trail", "cloth_flutter", "liquid_arc", "splash", "droplets", "puddle", "shards", "debris", "dust",
]
EventKind = Literal[
    "push", "pull", "lift", "place", "throw", "drop", "fall", "catch", "hand_off", "collide",
    "impact", "bounce", "slide", "roll", "break", "spill", "pour", "tear", "settle",
    "walk_to", "run_to", "sit_down", "stand_up", "enter", "exit", "gesture", "none",
]
ContactPart = Literal["hand_left", "hand_right", "both_hands", "foot", "body", "top", "bottom", "side", "none"]
DirRef = Literal[
    "toward_target", "away_from_actor", "up", "down", "screen_left", "screen_right",
    "toward_camera", "away_from_camera", "none",
]
Strength = Literal["gentle", "moderate", "hard", "violent"]
DepictedPhase = Literal[
    "anticipation", "contact", "release", "rising", "apex", "falling", "impact",
    "rebound", "aftermath", "rest", "in_progress", "none",
]
OutcomeRest = Literal["rest", "rolling", "airborne", "held", "none"]
OutcomeFlag = Literal[
    "none", "cracked", "shattered", "dented", "torn", "crumpled", "spilled", "stained", "wet", "dusty",
    "fill_empty", "fill_half", "fill_full",
]
InvariantKind = Literal[
    "supported", "held_with_holder", "irreversible_flag", "count_conserved", "volume_conserved",
    "continuous_path", "no_penetration", "fixed_stays_fixed", "occluded_persists",
]
CauseKind = Literal["causes", "enables", "prevents"]

IRREVERSIBLE_FLAGS: Set[str] = {"cracked", "shattered", "dented", "torn", "crumpled", "spilled", "stained"}
DECAYING_FLAGS: Set[str] = {"wet", "dusty"}
FILL_FLAGS: Set[str] = {"fill_empty", "fill_half", "fill_full"}


class StrictModel(BaseModel):
    """Forbids unknown keys and normalizes enum/ID casing (structured output may vary capitalization)."""
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_case(cls, value: Any, info):
        name = info.field_name or ""
        annotation = cls.model_fields[name].annotation if name in cls.model_fields else None
        is_enum = typing.get_origin(annotation) is Literal
        is_enum_list = (
            typing.get_origin(annotation) in (list, List)
            and typing.get_args(annotation)
            and typing.get_origin(typing.get_args(annotation)[0]) is Literal
        )
        is_id = name == "id" or name.endswith("_id") or name.endswith("_ids")
        if isinstance(value, str) and (is_enum or is_id):
            return value.strip().lower()
        if isinstance(value, list) and (is_enum_list or is_id):
            return [v.strip().lower() if isinstance(v, str) else v for v in value]
        return value


# ---------------------------------------------------------------- call A: scene bible
class SceneInfo(StrictModel):
    logline: str
    style: str = Field(description="Visual style phrase for every frame, e.g. 'gritty 35mm film still'")
    aspect: Aspect
    time_of_day: TimeOfDay
    target_duration_s: float = Field(description="Total runtime in seconds, at most 30")


class Character(StrictModel):
    id: str = Field(pattern=id_pattern("ch_"))
    name: str
    body: Body = Field("human", description="Body plan for blocking: human, or quadruped for animals such as dogs")
    height_m: float = Field(description="Standing height; for a quadruped the height at the shoulder")
    appearance: str = Field(description="Frozen physical description, at most 50 words; no clothing")
    wardrobe: str
    palette_hex: List[str] = Field(description="Two #rrggbb colours: a human's wardrobe top then bottom; an animal's main coat then legs/underside")


class Prop(StrictModel):
    id: str = Field(pattern=id_pattern("pr_"))
    name: str
    shape: PropShape
    size_x_m: float
    size_y_m: float
    size_z_m: float
    color_hex: str
    movable: bool


class Location(StrictModel):
    id: str = Field(pattern=id_pattern("lo_"))
    name: str
    kind: LocationKind
    width_m: float = Field(description="Extent along x, marks use 0..width_m")
    depth_m: float = Field(description="Extent along z, marks use 0..depth_m")
    key_light: KeyLight
    palette_hex: List[str] = Field(description="Floor color first, then walls")
    description: str


class Mark(StrictModel):
    id: str = Field(pattern=id_pattern("mk_"))
    location_id: str = Field(pattern=id_pattern("lo_"))
    name: str
    x_m: float
    z_m: float


class Beat(StrictModel):
    id: str = Field(pattern=id_pattern("bt_"))
    order: int
    summary: str
    cause_beat_id: str = Field(pattern=id_pattern("bt_", allow_none=True))


class Shot(StrictModel):
    id: str = Field(pattern=id_pattern("sh_"))
    order: int
    location_id: str = Field(pattern=id_pattern("lo_"))
    beat_ids: List[str]
    setup_id: str = Field(pattern=id_pattern("su_"), description="Shots sharing a camera setup reuse continuity")
    size: ShotSize
    angle: Angle
    subject_id: str = Field(pattern=ENTITY_ID)
    secondary_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    frame_side: FrameSide
    line_side: LineSide
    move: Move
    lens_mm: float
    duration_class: DurationClass
    time_scale: TimeScale
    transition: Transition
    keyframe_count: int
    motion_critical: bool


class PhysBody(StrictModel):
    entity_id: str = Field(pattern=ENTITY_ID)
    regime: Regime
    material: Material
    mass_class: MassClass
    friction: Friction
    restitution: Restitution
    failure_mode: FailureMode
    container_fill: ContainerFill


class EntityState(StrictModel):
    """Complete blocking snapshot of one entity at one keyframe. Unlisted entities carry forward."""
    kf_id: str = Field(pattern=r"^(?:kf_[a-z0-9_]{1,28}|init)$")
    entity_id: str = Field(pattern=ENTITY_ID)
    present: bool
    mark_id: str = Field(pattern=id_pattern("mk_", allow_none=True))
    offset_x_m: float
    offset_z_m: float
    facing_id: str = Field(pattern=id_pattern("ch_", "pr_", "mk_", allow_none=True))
    pose: Pose
    expression: Expression
    holding_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    gaze_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    support: Support
    support_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    flags: List[Flag]
    weight_cue: WeightCue
    secondary: List[Secondary]


class SceneBible(StrictModel):
    scene: SceneInfo
    characters: List[Character]
    props: List[Prop]
    locations: List[Location]
    marks: List[Mark]
    beats: List[Beat]
    shots: List[Shot]
    phys_bodies: List[PhysBody]
    initial_states: List[EntityState]


# ---------------------------------------------------------------- call B: timeline
class Invariant(StrictModel):
    id: str = Field(pattern=id_pattern("iv_"))
    kind: InvariantKind
    subject_id: str = Field(pattern=ENTITY_ID)
    object_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    from_kf_id: str = Field(pattern=r"^kf_[a-z0-9_]{1,28}$")
    until_kf_id: str = Field(pattern=r"^kf_[a-z0-9_]{1,28}$")


class Keyframe(StrictModel):
    id: str = Field(pattern=r"^kf_[a-z0-9_]{1,28}$")
    order: int
    beat_id: str = Field(pattern=id_pattern("bt_"))
    action: str = Field(description="What the frame shows, at most 25 words, present tense")
    dialogue: str
    move_progress: float = Field(description="Camera move progress 0..1 within the shot")
    focus_event_id: str = Field(pattern=id_pattern("ev_", allow_none=True))
    depicted_phase: DepictedPhase


class PhysEvent(StrictModel):
    justification: str = Field(description="Physical rationale, at most 40 words")
    id: str = Field(pattern=id_pattern("ev_"))
    start_kf_id: str = Field(pattern=r"^kf_[a-z0-9_]{1,28}$")
    kind: EventKind
    actor_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    target_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    cause_event_id: str = Field(pattern=id_pattern("ev_", allow_none=True))
    contact_part: ContactPart
    dir_ref: DirRef
    strength: Strength
    outcome_support_id: str = Field(pattern=ENTITY_ID_OR_NONE)
    outcome_mark_id: str = Field(pattern=id_pattern("mk_", allow_none=True))
    outcome_rest: OutcomeRest
    outcome_flag: OutcomeFlag


class CauseLink(StrictModel):
    id: str = Field(pattern=id_pattern("cl_"))
    cause_event_id: str = Field(pattern=id_pattern("ev_"))
    effect_event_id: str = Field(pattern=id_pattern("ev_"))
    kind: CauseKind


class ShotTimeline(StrictModel):
    shot_id: str = Field(pattern=id_pattern("sh_"))
    keyframes: List[Keyframe]
    events: List[PhysEvent]
    states: List[EntityState]
    cause_links: List[CauseLink]


class Timeline(StrictModel):
    invariants: List[Invariant]
    shots: List[ShotTimeline]


class ScenePlan(StrictModel):
    """The full physics pass: bible + timeline. Persisted as plan.json."""
    bible: SceneBible
    timeline: Timeline


# ---------------------------------------------------------------- JSON Schema export
def _inline_schema(model: type) -> Dict[str, Any]:
    """Pydantic schema with $refs inlined, titles/defaults dropped, every object closed."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            out = {k: resolve(v) for k, v in node.items() if k not in ("title", "default")}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}).keys())
            if "const" in out and "enum" not in out:
                out["enum"] = [out.pop("const")]
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(raw)


def bible_json_schema() -> Dict[str, Any]:
    """Static schema for call A (compiled once per release)."""
    return _inline_schema(SceneBible)


def keyframe_ids_for_shot(shot: Shot) -> List[str]:
    """Pre-created keyframe IDs for a shot, with one slot of slack (at least 4)."""
    suffix = shot.id[len("sh_"):]
    count = max(shot.keyframe_count + 1, 4)
    return [f"kf_{suffix}_{n}" for n in range(1, count + 1)]


def timeline_json_schema(bible: SceneBible) -> Dict[str, Any]:
    """
    Schema for call B with per-scene ID enums, so referential integrity is enforced while
    Claude writes. Event IDs are created inside call B, so they stay pattern-checked and
    validator.py verifies their references.
    """
    schema = _inline_schema(Timeline)
    entities = [c.id for c in bible.characters] + [p.id for p in bible.props]
    keyframes = [kf for s in bible.shots for kf in keyframe_ids_for_shot(s)]
    enums: Dict[str, List[str]] = {
        "entity_id": entities,
        "subject_id": entities,
        "object_id": entities + [NONE],
        "actor_id": entities + [NONE],
        "target_id": entities + [NONE],
        "holding_id": entities + [NONE],
        "gaze_id": entities + [NONE],
        "support_id": entities + [NONE],
        "outcome_support_id": entities + [NONE],
        "facing_id": entities + [m.id for m in bible.marks] + [NONE],
        "mark_id": [m.id for m in bible.marks] + [NONE],
        "outcome_mark_id": [m.id for m in bible.marks] + [NONE],
        "beat_id": [b.id for b in bible.beats],
        "shot_id": [s.id for s in bible.shots],
        "kf_id": keyframes,
        "start_kf_id": keyframes,
        "from_kf_id": keyframes,
        "until_kf_id": keyframes,
    }
    # Keyframe.id itself must be one of the pre-created IDs.
    keyframe_obj = schema["properties"]["shots"]["items"]["properties"]["keyframes"]["items"]
    keyframe_obj["properties"]["id"] = {"type": "string", "enum": keyframes}

    def apply(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.get("properties", {}).items():
                if key in enums and isinstance(child, dict) and child.get("type") == "string":
                    node["properties"][key] = {"type": "string", "enum": sorted(set(enums[key]))}
                else:
                    apply(child)
            if "items" in node:
                apply(node["items"])

    apply(schema)
    return schema


def entity_ids(bible: SceneBible) -> List[str]:
    return [c.id for c in bible.characters] + [p.id for p in bible.props]
