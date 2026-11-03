"""
Derived runtime types. Claude never produces these: the reducer, solver, camera solver,
and blocking renderer compute them from the ScenePlan.

Coordinates: right-handed, metres, y up. A location's floor spans x in [0, width_m] and
z in [0, depth_m]. Yaw 0 faces +z; positive yaw turns toward +x.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class EntityFrame:
    """Solved state of one entity at one keyframe instant."""
    entity_id: str
    kind: str                      # "character" or "prop"
    present: bool
    position: Vec3                 # base centre (feet for characters, bottom centre for props)
    yaw_deg: float
    rotation_deg: Vec3             # extra tumble for airborne props (x, y, z Euler)
    velocity: Vec3                 # m/s, for motion-blur and secondary cues
    pose: str
    expression: str
    holding_id: str
    gaze_id: str
    support: str
    support_id: str
    flags: List[str]
    weight_cue: str
    secondary: List[str]
    size: Vec3                     # x, y, z extent in metres (characters: shoulder width, height, depth)
    color_hex: str                 # primary blocking colour
    secondary_color_hex: str       # characters: lower-body colour; props: same as primary
    shape: str                     # "character", "box", "cylinder", "sphere", "plane"
    in_motion: bool = False        # an active physics event moves this entity at this instant
    trajectory: List[Vec3] = field(default_factory=list)  # sampled path of the active event (for proxies)


@dataclass
class CameraState:
    position: Vec3
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    lens_mm: float
    vfov_deg: float
    width: int
    height: int
    look_at: Vec3


@dataclass
class FrameState:
    """Everything needed to block and condition one keyframe."""
    kf_id: str
    shot_id: str
    order_in_shot: int
    t_s: float                     # time on the edited timeline (seconds from scene start)
    t_physical_s: float            # time on the physical clock (before time_scale)
    location_id: str
    entities: Dict[str, EntityFrame]
    focus_event_id: str
    depicted_phase: str
    active_event_ids: List[str]
    camera: Optional[CameraState] = None


@dataclass
class SolvedEvent:
    event_id: str
    kind: str
    shot_id: str
    start_t_physical: float
    end_t_physical: float
    phase_times: Dict[str, float]  # depicted_phase -> physical time offset from start
    feasible: bool
    note: str = ""


@dataclass
class SolvedShot:
    shot_id: str
    start_t_s: float
    end_t_s: float
    keyframe_ids: List[str]


@dataclass
class SolvedPlan:
    frames: List[FrameState]                   # in timeline order
    events: Dict[str, SolvedEvent]
    shots: List[SolvedShot]
    total_duration_s: float
    warnings: List[str] = field(default_factory=list)

    def frame(self, kf_id: str) -> FrameState:
        for f in self.frames:
            if f.kf_id == kf_id:
                return f
        raise KeyError(kf_id)


@dataclass
class Violation:
    code: str
    message: str
    shot_id: str = "none"
    kf_id: str = "none"
    entity_id: str = "none"
    severity: str = "error"        # "error" blocks rendering; "warning" is reported only


@dataclass
class BlockingResult:
    """Outputs of the blocking renderer for one keyframe."""
    blocking_path: str             # RGB proxy render (palette colours, blurred) used as the init image
    id_path: str                   # PNG, pixel value = entity index + 1 (0 = background)
    depth_path: str                # .npy float32 metres, inf where empty
    id_map: Dict[int, str]         # pixel value -> entity id
    bboxes: Dict[str, Tuple[int, int, int, int]]   # entity id -> (x0, y0, x1, y1), visible pixels only
    visible_frac: Dict[str, float] # entity id -> visible pixels / unoccluded pixels
    screen_velocity: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # px/s
