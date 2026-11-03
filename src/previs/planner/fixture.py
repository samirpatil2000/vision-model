"""
Offline planner that returns the canonical kitchen-glass ScenePlan.

Used by tests and by runs that must not call Claude. Repair and amend return the plan unchanged.
"""
from pathlib import Path
from typing import List

from src.previs.planner.base import PhysicsPlanner
from src.previs.schema import ScenePlan

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "kitchen_glass.json"


class FixturePlanner(PhysicsPlanner):
    def __init__(self, path: Path = DEFAULT_FIXTURE):
        self.path = Path(path)

    def plan(self, scene_text: str, work_dir: Path) -> ScenePlan:
        return ScenePlan.model_validate_json(self.path.read_text())

    def repair(self, plan: ScenePlan, violations_text: str, shot_ids: List[str], work_dir: Path) -> ScenePlan:
        return plan

    def amend(self, plan: ScenePlan, facts_text: str, kf_id: str, work_dir: Path) -> ScenePlan:
        return plan
