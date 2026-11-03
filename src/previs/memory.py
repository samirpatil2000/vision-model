"""
Keyframe memory: accepted frames and where each one may be reused as a continuity reference.

Every accepted keyframe is recorded with its shot, camera setup, location, camera yaw and
ancestry depth (how many generation hops separate it from an approved frame or a plate).
continuity_for() picks the reference image for the next keyframe:

  - inside a shot (kf_order > 1): the shot anchor, i.e. the first accepted frame of the shot;
  - at a shot's first keyframe: the latest accepted frame of the same camera setup, else the
    best-scored accepted frame at the same location with the nearest camera yaw, else the plate.

Candidates at or beyond `max_depth` hops are skipped so drift cannot compound through chains of
references. Approved frames are reset to depth 0.

memory.json is read tolerantly: keys this version does not know are kept in MemoryRecord.extra and
written back on save (a newer writer's fields survive an older reader), and missing optional fields
take neutral defaults.
"""
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SOURCE_SHOT_ANCHOR = "shot_anchor"
SOURCE_SETUP = "setup"
SOURCE_LOCATION = "location"
SOURCE_PLATE = "plate"
SOURCE_NONE = "none"

YAW_BUCKET_DEG = 15.0          # yaw differences inside one bucket count as equally near (score breaks ties)
MAX_YAW_DEG = 60.0             # beyond this a frame shows a different part of the set: use the plate instead


@dataclass
class MemoryRecord:
    kf_id: str
    shot_id: str
    setup_id: str
    location_id: str
    camera_yaw: float
    path: str
    ancestry_depth: int
    score: float
    approved: bool
    extra: Dict[str, Any] = field(default_factory=dict)   # unknown keys from memory.json, written back on save

    def to_dict(self) -> Dict[str, Any]:
        """Flat JSON record: the known fields win over any extra key of the same name."""
        known = {k: v for k, v in asdict(self).items() if k != "extra"}
        return {**self.extra, **known}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """Record from a JSON dict; unknown keys go to `extra`, missing optional fields get defaults."""
        missing = [k for k in _REQUIRED_FIELDS if k not in data]
        if missing:
            raise ValueError(f"memory record {data.get('kf_id', '?')} lacks {', '.join(missing)}")
        values = {**_FIELD_DEFAULTS, **{k: v for k, v in data.items() if k in _KNOWN_FIELDS}}
        return cls(**{**values, "camera_yaw": float(values["camera_yaw"]), "score": float(values["score"]),
                      "ancestry_depth": int(values["ancestry_depth"]), "approved": bool(values["approved"]),
                      "extra": {k: v for k, v in data.items() if k not in _KNOWN_FIELDS}})


_KNOWN_FIELDS = frozenset(f.name for f in fields(MemoryRecord) if f.name != "extra")
_REQUIRED_FIELDS = ("kf_id", "shot_id", "setup_id", "location_id", "path")
# A record without provenance is treated like the sequencer's neutral record: one hop from a source, unscored.
_FIELD_DEFAULTS: Dict[str, Any] = {"camera_yaw": 0.0, "ancestry_depth": 1, "score": 0.0, "approved": False}


def yaw_distance(a: float, b: float) -> float:
    """Smallest absolute angle between two yaws in degrees (0..180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


class KeyframeMemory:
    """Accepted keyframes in acceptance order (the most recent add is the latest)."""

    def __init__(self, records: Optional[List[MemoryRecord]] = None):
        self.records: List[MemoryRecord] = []
        for record in records or []:
            self.add(record)

    # ------------------------------------------------------------ records
    def add(self, record: MemoryRecord) -> None:
        """Record an accepted frame. A retake of the same keyframe replaces (and becomes the latest)."""
        if record.approved:
            record = replace(record, ancestry_depth=0)
        self.records = [r for r in self.records if r.kf_id != record.kf_id]
        self.records.append(record)

    def get(self, kf_id: str) -> Optional[MemoryRecord]:
        for record in self.records:
            if record.kf_id == kf_id:
                return record
        return None

    def approve(self, kf_id: str) -> None:
        """Mark a frame as human-approved: it becomes a depth-0 continuity source."""
        record = self.get(kf_id)
        if record is None:
            raise KeyError(kf_id)
        record.approved, record.ancestry_depth = True, 0

    # ------------------------------------------------------------ continuity
    def continuity_for(
        self,
        shot: Union[str, Any],
        kf_order: int,
        setup_id: str,
        location_id: str,
        camera_yaw: float,
        plate_path: Optional[str],
        max_depth: int = 2,
        max_yaw_deg: float = MAX_YAW_DEG,
    ) -> Tuple[Optional[str], str, int]:
        """
        (path, source, depth) of the continuity reference for a keyframe, or (None, "none", 0).
        `shot` is a shot id or a Shot; `kf_order` is the keyframe's 1-based position in its shot.
        """
        shot_id = shot if isinstance(shot, str) else shot.id
        usable = [r for r in self.records if r.ancestry_depth < max_depth and Path(r.path).is_file()]

        # 1. Inside a shot: the shot anchor (first accepted frame of this shot).
        if kf_order > 1:
            anchors = [r for r in self.records if r.shot_id == shot_id]
            if anchors and anchors[0] in usable:
                return anchors[0].path, SOURCE_SHOT_ANCHOR, anchors[0].ancestry_depth

        # 2. Latest accepted frame of the same camera setup.
        same_setup = [r for r in usable if r.setup_id == setup_id]
        if same_setup:
            latest = same_setup[-1]
            return latest.path, SOURCE_SETUP, latest.ancestry_depth

        # 3. Best-scored frame at the same location with the nearest camera yaw.
        same_location = [r for r in usable if r.location_id == location_id
                         and yaw_distance(r.camera_yaw, camera_yaw) <= max_yaw_deg]
        if same_location:
            best = min(same_location,
                       key=lambda r: (int(yaw_distance(r.camera_yaw, camera_yaw) // YAW_BUCKET_DEG), -r.score))
            return best.path, SOURCE_LOCATION, best.ancestry_depth

        # 4. The location plate.
        if plate_path and Path(plate_path).is_file():
            return str(plate_path), SOURCE_PLATE, 0
        return None, SOURCE_NONE, 0

    # ------------------------------------------------------------ persistence
    def to_json(self) -> str:
        return json.dumps({"records": [r.to_dict() for r in self.records]}, indent=2)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "KeyframeMemory":
        data = json.loads(Path(path).read_text())
        records = [MemoryRecord.from_dict(r) for r in data.get("records", [])]
        memory = cls()
        memory.records = records          # keep stored order and depths exactly as saved
        return memory
