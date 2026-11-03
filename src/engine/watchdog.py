"""
Parent-death watchdog for engine processes.

Engine children run in their own session so that stopping one kills its whole process group,
which also means a terminal hang-up never reaches them. If Python dies without running
close()/atexit (SIGHUP from a closed terminal or dropped SSH session, SIGKILL, a crash), a
resident sd-server would keep its weights loaded indefinitely. arm() therefore starts a tiny
stdlib-only process (this file, run as a script) that holds the read end of a pipe from this
process: when the pipe reaches EOF or the watchdog is re-parented, the parent is gone, so it
SIGTERMs then SIGKILLs the engine's process group and removes its private temp dir. disarm()
tells it to exit without touching anything after a clean shutdown.

Only the standard library may be imported here: the watchdog runs as `python -I <this file>`.
"""
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

DISARM = b"x"
POLL_S = 0.5
KILL_GRACE_S = 10.0


def arm(pgid: int, cleanup_dir: Optional[str] = None) -> subprocess.Popen:
    """Start a watchdog that kills process group `pgid` (and removes `cleanup_dir`) if we die."""
    argv = [sys.executable, "-I", str(Path(__file__).resolve()), str(os.getpid()), str(pgid)]
    if cleanup_dir:
        argv.append(str(cleanup_dir))
    return subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def disarm(proc: Optional[subprocess.Popen], timeout_s: float = 5.0) -> None:
    """Stop a watchdog without letting it kill anything (the engine was stopped cleanly)."""
    if proc is None:
        return
    try:
        proc.stdin.write(DISARM)
        proc.stdin.close()
    except (OSError, ValueError):
        pass                                   # watchdog already gone, or pipe already closed
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def kill_group(pgid: int, grace_s: float = KILL_GRACE_S) -> None:
    """SIGTERM the process group, then SIGKILL it if it is still there after `grace_s`."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def wait_for_parent_exit(parent_pid: int, poll_s: float = POLL_S) -> bool:
    """Block until the parent is gone (True) or it disarms us (False). Reads stdin (fd 0)."""
    while True:
        if os.getppid() != parent_pid:
            return True                        # re-parented: the parent died
        ready, _, _ = select.select([0], [], [], poll_s)
        if ready:
            return os.read(0, 1) != DISARM     # b"" (EOF) means the parent's end closed


def main(argv: List[str]) -> int:
    parent_pid, pgid = int(argv[0]), int(argv[1])
    cleanup_dir = argv[2] if len(argv) > 2 else None
    for sig in (signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)     # outlive the hang-up / Ctrl-C that kills the parent
    if not wait_for_parent_exit(parent_pid):
        return 0
    kill_group(pgid)
    if cleanup_dir:
        shutil.rmtree(cleanup_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
