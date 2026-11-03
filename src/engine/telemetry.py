"""
Engine telemetry: peak memory, per-phase timings and memory-fallback detection.

Everything here parses text the engine (or /usr/bin/time) already prints, so it works the
same for sd-cli runs and for the log segment of one sd-server job.
"""
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Engine log lines that mean a requested speed/memory feature was silently dropped.
# "Flash Attention unavailable" is LOG_DEBUG at the pinned commit (needs --log-level debug);
# the prefix-cache lines are LOG_WARN.
FALLBACK_PATTERNS = (
    "Flash Attention unavailable",
    "insufficient memory for prefix caching",
    "incomplete prefix cache",
    "Qwen Image 2.1 cache type",
)

_MAC_MAX_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
_GNU_MAX_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_MAC_FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$", re.MULTILINE)
# Engine lines look like "[I] sampling completed, taking 23.30s --- image.cpp:899"; a progress
# bar may precede the "[I]" tag on the same physical line, so the name starts after the tag.
_PHASE = re.compile(
    r"(?:^|\[[DVIWE?]\]\s*)(?P<name>[A-Za-z_][^\[\]|]*?) completed, taking "
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)\b"
)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def time_prefix() -> List[str]:
    """argv prefix that makes the OS report peak memory of the wrapped command on stderr."""
    if not Path("/usr/bin/time").exists():
        return []
    if sys.platform == "darwin":
        return ["/usr/bin/time", "-l"]
    if sys.platform.startswith("linux"):
        return ["/usr/bin/time", "-v"]
    return []


def parse_time_l(stderr: str) -> Optional[int]:
    """Peak RSS in bytes from `/usr/bin/time -l` (macOS, bytes) or GNU `time -v` (kbytes)."""
    matches = _MAC_MAX_RSS.findall(stderr)
    if matches:
        return int(matches[-1])
    matches = _GNU_MAX_RSS.findall(stderr)
    if matches:
        return int(matches[-1]) * 1024
    return None


def parse_peak_footprint(stderr: str) -> Optional[int]:
    """macOS `peak memory footprint` in bytes (includes GPU/IOKit memory that RSS misses)."""
    matches = _MAC_FOOTPRINT.findall(stderr)
    return int(matches[-1]) if matches else None


def _phase_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_phase_timings(log: str) -> Dict[str, float]:
    """Seconds per engine phase from lines like '[I] sampling completed, taking 12.34s'.

    Keys are normalised phase names ("loading_tensors", "get_learned_condition", "sampling",
    "decode_first_stage", ...). A phase that completes several times in one log (e.g. the
    positive and negative condition graphs) is summed.
    """
    phases: Dict[str, float] = {}
    for line in re.split(r"[\r\n]+", _ANSI.sub("", log)):
        match = _PHASE.search(line.strip())
        if not match:
            continue
        key = _phase_key(match.group("name"))
        if not key:
            continue
        value = float(match.group("value"))
        if match.group("unit") == "ms":
            value /= 1000.0
        phases[key] = round(phases.get(key, 0.0) + value, 4)
    return phases


def find_fallbacks(log: str) -> List[str]:
    """Distinct log lines reporting a flash-attention or prefix-cache fallback, in order."""
    found: List[str] = []
    lowered = [p.lower() for p in FALLBACK_PATTERNS]
    for line in re.split(r"[\r\n]+", _ANSI.sub("", log)):
        text = line.strip()
        if text and any(p in text.lower() for p in lowered) and text not in found:
            found.append(text)
    return found


def process_rss_bytes(pid: int) -> Optional[int]:
    """Current resident set size of a running process (via `ps`), or None if unavailable."""
    try:
        proc = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return int(value) * 1024 if proc.returncode == 0 and value.isdigit() else None


def tail(text: str, lines: int = 40) -> str:
    """Last `lines` lines of a log, for error messages."""
    return "\n".join(re.split(r"[\r\n]+", text.strip())[-lines:])


def safe_log_name(name: str) -> str:
    """Filesystem-safe log file stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "generation"
