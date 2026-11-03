"""
Engine version pinning.

The repo is pinned to one stable-diffusion.cpp commit (engine/UPSTREAM_SHA). The binaries
report the commit they were built from via `--version`
("stable-diffusion.cpp version master-955-b7a74e6-2-g2988060, commit 2988060"); a mismatch
means flags, log lines or defaults may differ from what this code was written against.
A binary built outside a git checkout reports "commit unknown" (the engine's CMake fallback).
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src import config
from src.engine.base import EngineError

UNKNOWN_COMMIT = "unknown"
_COMMIT_RE = re.compile(r"\bcommit\s+([0-9a-fA-F]{7,40}|unknown)\b")
MIN_PREFIX = 7


class UnknownEngineVersion(EngineError):
    """The binary ran, but its `--version` output names no commit."""
    pass


def engine_commit(sd_cli: Path = config.SD_CLI, timeout_s: float = 60) -> str:
    """Short commit hash reported by `<binary> --version` (works for sd-cli and sd-server).

    Returns UNKNOWN_COMMIT when the binary says "commit unknown". Raises EngineError when the
    binary cannot be run, and UnknownEngineVersion when its output has no commit at all.
    """
    try:
        proc = subprocess.run(
            [str(sd_cli), "--version"], capture_output=True, text=True, timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise EngineError(f"Engine binary not found at {sd_cli}. Run ./setup.sh first.") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(f"Could not run {sd_cli} --version: {exc}") from exc
    match = _COMMIT_RE.search(proc.stdout + "\n" + proc.stderr)
    if not match:
        raise UnknownEngineVersion(
            f"Could not parse a commit from `{sd_cli} --version` (exit {proc.returncode}): "
            f"{(proc.stdout + proc.stderr).strip()[:300]!r}"
        )
    return match.group(1).lower()


def commits_match(reported: str, pinned: str) -> bool:
    """True when one hash is a prefix of the other (short vs full SHA), at least 7 chars."""
    a, b = reported.strip().lower(), pinned.strip().lower()
    if len(a) < MIN_PREFIX or len(b) < MIN_PREFIX:
        return False
    return a.startswith(b) or b.startswith(a)


def verify_engine(
    sd_cli: Path = config.SD_CLI,
    pinned_sha: Optional[str] = None,
    allow_mismatch: bool = False,
) -> str:
    """Return the binary's commit; raise EngineError if it differs from the pinned SHA.

    A mismatch is tolerated when `allow_mismatch` is set or env PREVIS_ALLOW_ENGINE_MISMATCH=1;
    with either, a binary that reports no commit also runs (UNKNOWN_COMMIT is returned). A
    binary that cannot be run always raises.
    """
    pinned = pinned_sha if pinned_sha is not None else config.pinned_engine_sha()
    allowed = allow_mismatch or os.environ.get("PREVIS_ALLOW_ENGINE_MISMATCH") == "1"
    try:
        reported = engine_commit(sd_cli)
    except UnknownEngineVersion as exc:
        if allowed:
            return UNKNOWN_COMMIT
        raise UnknownEngineVersion(
            f"{exc}. Set PREVIS_ALLOW_ENGINE_MISMATCH=1 (previs.py --allow-engine-mismatch) to run anyway."
        ) from exc
    if commits_match(reported, pinned):
        return reported
    if allowed:
        return reported
    built = ("reports no build commit (built outside a git checkout)" if reported == UNKNOWN_COMMIT
             else f"reports commit {reported}")
    raise EngineError(
        f"Engine binary {sd_cli} {built}, but the repo is pinned to {pinned} "
        f"(engine/UPSTREAM_SHA). Rebuild with `PIN_ENGINE=1 ./setup.sh`, or set "
        f"PREVIS_ALLOW_ENGINE_MISMATCH=1 (previs.py --allow-engine-mismatch) to run anyway."
    )


def read_build_info(engine_dir: Path = config.ENGINE_DIR) -> Optional[Dict[str, Any]]:
    """Contents of build/build-info.json written by setup.sh, or None when absent/invalid."""
    path = Path(engine_dir) / "build" / "build-info.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
