"""
Deterministic seeds.

Every generation seed is a pure function of (project, shot, keyframe, attempt), so re-running a
stage reproduces the same images and a retake always gets a fresh but repeatable seed.
"""
import hashlib

SEED_MODULUS = 2 ** 31


def seed_for(project: str, shot_id: str, kf_id: str, attempt: int) -> int:
    """Seed in [0, 2**31) from sha256 over the four identifiers."""
    key = f"{project}|{shot_id}|{kf_id}|{int(attempt)}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % SEED_MODULUS
