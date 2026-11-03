"""Physics planners: the fixture planner (offline) and the Claude CLI planner + fact judge."""
from typing import Any

from src.previs.planner.base import (
    PhysicsPlanner,
    PlannerError,
    PlannerTimeoutError,
    PlannerTruncatedError,
    replace_shots,
)
from src.previs.planner.fixture import FixturePlanner
from src.previs.planner.claude_cli import ClaudeCliJudge, ClaudeCliPlanner

__all__ = [
    "PhysicsPlanner", "PlannerError", "PlannerTimeoutError", "PlannerTruncatedError", "replace_shots",
    "FixturePlanner", "ClaudeCliPlanner", "ClaudeCliJudge", "make_planner",
]


def make_planner(kind: str = "claude", **kw: Any) -> PhysicsPlanner:
    """'fixture' -> FixturePlanner, 'claude' -> ClaudeCliPlanner (keyword arguments are passed through)."""
    if kind == "fixture":
        return FixturePlanner(**kw)
    if kind in ("claude", "claude_cli"):
        return ClaudeCliPlanner(**kw)
    raise ValueError(f"unknown planner kind {kind!r} (expected 'fixture' or 'claude')")
