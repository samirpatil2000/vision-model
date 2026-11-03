"""
Claude Code CLI (`claude -p`) transport for the physics planner and the image fact judge.

Single-pass protocol:
  A       fresh session (--session-id), static bible schema            -> SceneBible
  bible_  --resume <A> --fork-session, same bible schema                 -> a SceneBible that passes the checks
  repair  only when bible_violations() finds defects no timeline repair can fix; one fresh fallback
  B       --resume <A> --fork-session, per-scene timeline schema        -> Timeline
          on any failure: one FRESH session (B's stdin already carries the bible)
  repair  --resume <last good> --fork-session, B's schema                -> only the listed shots
          FRESH session with the whole plan when no matching session exists or the fork fails
  amend   FRESH session, B's schema, whole plan + observed facts         -> the dependent shots
  facts   ephemeral session, Haiku, Read tool for the image              -> yes/no/unsure answers

Every call sends its input on stdin only, uses an allowlisted environment and a fixed empty cwd,
streams stream-json under a watchdog, checks the result strictly (error, truncation, refusal,
model reroute, compaction), writes raw records to work_dir/claude/, and goes through a replay
cache keyed by the logical request (PREVIS_CLAUDE_REPLAY=record|replay|off, default record).
"""
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from pydantic import ValidationError

from src import config
from src.previs.planner import prompts
from src.previs.planner.base import (
    PhysicsPlanner,
    PlannerError,
    PlannerTimeoutError,
    PlannerTruncatedError,
    ordered_shot_ids,
    replace_shots,
    shot_keyframe_ids,
)
from src.previs.schema import (
    NONE,
    SceneBible,
    ScenePlan,
    Timeline,
    bible_json_schema,
    keyframe_ids_for_shot,
    timeline_json_schema,
)
from src.previs.state import INIT
from src.previs.validator import HEIGHT_RANGE_M, LENS_RANGE_MM, LINE_FREE_MOVE_PREFIXES

T = TypeVar("T")

PLANNER_MODEL = "claude-opus-5-5"
JUDGE_MODEL = "claude-haiku-5-5"
MAX_OUTPUT_A = 64000
MAX_OUTPUT_TIMELINE = 128000
MAX_OUTPUT_FACTS = 8000
MAX_API_TIMEOUT_MS = 2147483647
STDIN_LIMIT_BYTES = 10 * 1024 * 1024
SCHEMA_ARG_LIMIT_BYTES = 120 * 1024           # Linux caps one argv element at 128 KiB
REPLAY_MODES = ("record", "replay", "off")
SETTINGS = {"switchModelsOnFlag": False, "permissions": {"defaultMode": "dontAsk"}}
ENV_PASSTHROUGH = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "SHELL", "TERM",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
)
DEFAULT_CWD = Path.home() / ".cache" / "previs" / "claude_cwd"
DEFAULT_CACHE_DIR = config.PREVIS_OUTPUT_DIR / "_claude_replay"
FACT_ANSWERS = ("yes", "no", "unsure")
FACTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "q_index": {"type": "integer"},
                    "answer": {"type": "string", "enum": list(FACT_ANSWERS)},
                },
                "required": ["q_index", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}
PROBE_SYSTEM_PROMPT = "You are a connectivity probe. Do exactly what the message asks and reply only with the JSON object the schema requires.\n"

_KF_ID = re.compile(r"^kf_[a-z0-9_]{1,28}$")
_UNKNOWN_OPTION = re.compile(r"unknown option '([^']+)'")
_USAGE_LIMIT = re.compile(r"\b(usage|session|weekly) limit\b", re.IGNORECASE)
_VERSION = re.compile(r"\d+\.\d+\.\d+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_model(reported: str, pinned: str) -> bool:
    """True when the reported model id is the pinned one (allowing date/context suffixes)."""
    reported = reported.strip().lower()
    pinned = pinned.strip().lower()
    return reported == pinned or reported.startswith(pinned + "-") or reported.startswith(pinned + "[")


@dataclass
class CallSpec:
    """One `claude -p` invocation."""
    role: str                       # a | bible_repair | b | repair | amend | facts | probe
    model: str
    effort: str
    body: str                       # stdin
    schema: Dict[str, Any]
    system_prompt_file: Path
    settings_file: Path
    max_output_tokens: int
    resume_id: Optional[str] = None  # --resume <id> --fork-session
    persist: bool = True             # False -> --no-session-persistence (judge calls)
    tools: str = ""
    add_dir: Optional[Path] = None
    max_turns: int = 3
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def mode(self) -> str:
        if self.resume_id:
            return "resume"
        return "fresh" if self.persist else "ephemeral"


@dataclass
class StreamRun:
    """What one CLI process emitted."""
    events: List[Dict[str, Any]] = field(default_factory=list)   # every non-delta event
    init: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    main_models: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    saw_output: bool = False
    deltas: int = 0
    non_json: List[str] = field(default_factory=list)
    compacted: bool = False
    abort_reason: Optional[str] = None
    returncode: Optional[int] = None
    stderr: str = ""
    seconds: float = 0.0

    def feed(self, raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            event = json.loads(text)
        except ValueError:
            self.non_json.append(text[:500])
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "stream_event":
            self.deltas += 1
            delta = (event.get("event") or {}).get("delta") or {}
            if delta.get("type") in ("text_delta", "input_json_delta"):
                self.saw_output = True
            return
        self.events.append(event)
        if kind == "system":
            if event.get("subtype") == "init":
                self.init = event
                self.session_id = event.get("session_id") or self.session_id
            elif event.get("subtype") == "compact_boundary":
                self.compacted = True
        elif kind == "assistant":
            message = event.get("message") or {}
            if not event.get("parent_tool_use_id") and message.get("model"):
                self.main_models.append(str(message["model"]))
            if any(isinstance(b, dict) and b.get("type") in ("text", "tool_use") for b in message.get("content") or []):
                self.saw_output = True
        elif kind == "result":
            self.result = event
            if event.get("session_id"):
                self.session_id = event["session_id"]


def _write_stdin(pipe: Any, data: bytes) -> None:
    try:
        pipe.write(data)
        pipe.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except (OSError, ValueError):
            pass


def _pump_lines(pipe: Any, out: "queue.Queue[Optional[bytes]]") -> None:
    try:
        for line in iter(pipe.readline, b""):
            out.put(line)
    except (OSError, ValueError):
        pass
    finally:
        out.put(None)


def _drain(pipe: Any, chunks: List[bytes]) -> None:
    try:
        for chunk in iter(lambda: pipe.read(65536), b""):
            chunks.append(chunk)
    except (OSError, ValueError):
        pass


def _signal(proc: Any, sig: int) -> None:
    """Send `sig` to the CLI's whole process group (start_new_session made the CLI its leader), so wrapper
    children and helpers that inherited stdout/stderr stop with it. Without a pid, signal the process alone."""
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int) and pid > 0 and hasattr(os, "killpg"):
        try:
            os.killpg(pid, sig)
            return
        except ProcessLookupError:
            return                      # nobody is left in the group
        except PermissionError:
            pass
    proc.send_signal(sig)


class ClaudeCliRunner:
    """Builds, runs, checks, records and caches `claude -p` calls. Shared by the planner and the judge."""

    def __init__(
        self,
        claude_bin: str = config.CLAUDE_CLI,
        auth: str = "subscription",
        config_dir: Optional[str] = None,
        timeout_s: float = 2700,
        restricted: bool = True,
        idle_timeout_s: float = 1200.0,
        first_text_timeout_s: Optional[float] = None,
        total_timeout_s: Optional[float] = None,
        replay: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cwd: Optional[Path] = None,
        force_idle_timeout_ms: Optional[int] = None,
        kill_grace_s: float = 10.0,
    ):
        if auth not in ("subscription", "api_key"):
            raise ValueError(f"auth must be 'subscription' or 'api_key', got {auth!r}")
        mode = (replay or os.environ.get("PREVIS_CLAUDE_REPLAY") or "record").strip().lower()
        if mode not in REPLAY_MODES:
            raise ValueError(f"PREVIS_CLAUDE_REPLAY must be one of {REPLAY_MODES}, got {mode!r}")
        self.claude_bin = str(claude_bin)
        self.auth = auth
        self.config_dir = config_dir
        self.timeout_s = float(timeout_s)
        self.restricted = restricted
        self.idle_timeout_s = float(idle_timeout_s)
        self.first_text_timeout_s = first_text_timeout_s
        self.total_timeout_s = float(total_timeout_s) if total_timeout_s is not None else 2.0 * self.timeout_s
        self.replay = mode
        self.cache_dir = Path(cache_dir or os.environ.get("PREVIS_CLAUDE_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.cwd = Path(cwd) if cwd else DEFAULT_CWD
        self.force_idle_timeout_ms = force_idle_timeout_ms
        self.kill_grace_s = float(kill_grace_s)
        self._version: Optional[str] = None

    # ------------------------------------------------------------ files, env, argv
    def prepare(self, work_dir: Path, system_prompt: str, prompt_name: str) -> Tuple[Path, Path]:
        """Write the system prompt and settings files once per run dir; return their absolute paths."""
        out = Path(work_dir) / "claude"
        out.mkdir(parents=True, exist_ok=True)
        files = ((out / prompt_name, system_prompt), (out / "settings.json", json.dumps(SETTINGS, indent=2) + "\n"))
        for path, text in files:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
        return files[0][0].resolve(), files[1][0].resolve()

    def _base_env(self) -> Dict[str, str]:
        source = os.environ
        env = {k: source[k] for k in ENV_PASSTHROUGH if source.get(k)}
        config_dir = self.config_dir or source.get("CLAUDE_CONFIG_DIR")
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = str(Path(config_dir).expanduser())
        return env

    def build_env(self, max_output_tokens: int) -> Dict[str, str]:
        """Allowlisted environment: nothing else from the parent shell reaches the CLI."""
        env = self._base_env()
        if self.auth == "subscription":
            if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
                env["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
        else:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise PlannerError("auth='api_key' needs ANTHROPIC_API_KEY in the environment", retryable=False)
            env["ANTHROPIC_API_KEY"] = key
        env.update({
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(int(max_output_tokens)),
            "API_TIMEOUT_MS": str(min(int(self.timeout_s * 1000), MAX_API_TIMEOUT_MS)),
            "CLAUDE_CODE_MAX_RETRIES": "1",
            "DISABLE_AUTOUPDATER": "1",
        })
        if self.force_idle_timeout_ms is not None:
            env["API_FORCE_IDLE_TIMEOUT"] = str(int(self.force_idle_timeout_ms))
        return env

    def build_argv(self, spec: CallSpec) -> List[str]:
        """Flags in the contract order. No positional prompt: the body goes on stdin only."""
        schema_json = json.dumps(spec.schema, separators=(",", ":"))
        if len(schema_json.encode("utf-8")) > SCHEMA_ARG_LIMIT_BYTES:
            raise PlannerError(f"{spec.role} JSON schema is {len(schema_json)} bytes, over the argv limit", retryable=False)
        argv = [self.claude_bin, "-p", "--model", spec.model, "--effort", spec.effort, "--safe-mode"]
        if self.restricted:
            argv.append("--restricted")
        if self.auth == "api_key":
            argv.append("--bare")
        argv += ["--strict-mcp-config", "--tools", spec.tools]
        if spec.tools:
            argv += ["--allowedTools", spec.tools]
        if spec.add_dir is not None:
            argv += ["--add-dir", str(spec.add_dir)]
        argv += [
            "--disable-slash-commands", "--permission-prompts", "none", "--prompt-suggestions", "false", "--no-chrome",
            "--settings", str(spec.settings_file), "--system-prompt-file", str(spec.system_prompt_file),
            "--json-schema", schema_json, "--max-turns", str(spec.max_turns),
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
        ]
        if spec.resume_id:
            argv += ["--resume", spec.resume_id, "--fork-session"]
        elif spec.persist:
            argv += ["--session-id", spec.session_id]
        else:
            argv.append("--no-session-persistence")
        return argv

    def version(self) -> str:
        """CLI version (part of every replay key)."""
        if self._version is None:
            try:
                out = subprocess.run(
                    [self.claude_bin, "--version"], capture_output=True, text=True, timeout=60,
                    env=self._base_env(), stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise PlannerError(f"cannot run {self.claude_bin} --version: {exc}", retryable=False) from exc
            match = _VERSION.search(out.stdout or "")
            if out.returncode != 0 or not match:
                raise PlannerError(f"{self.claude_bin} --version failed: {(out.stderr or out.stdout).strip()[:300]}",
                                   retryable=False)
            self._version = match.group(0)
        return self._version

    # ------------------------------------------------------------ process + watchdog
    def stream(self, argv: List[str], body: str, env: Dict[str, str]) -> StreamRun:
        """Run one CLI process, feeding `body` on stdin and reading stream-json under the watchdog."""
        data = body.encode("utf-8")
        if len(data) > STDIN_LIMIT_BYTES:
            raise PlannerError(f"stdin body is {len(data)} bytes, over the CLI's 10 MB limit", retryable=False)
        self.cwd.mkdir(parents=True, exist_ok=True)
        run = StreamRun()
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(self.cwd), env=env, start_new_session=True,
            )
        except OSError as exc:
            raise PlannerError(f"cannot start {argv[0]}: {exc}", retryable=False) from exc

        lines: "queue.Queue[Optional[bytes]]" = queue.Queue()
        err_chunks: List[bytes] = []
        threads = [
            threading.Thread(target=_write_stdin, args=(proc.stdin, data), daemon=True),
            threading.Thread(target=_pump_lines, args=(proc.stdout, lines), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True),
        ]
        for t in threads:
            t.start()

        poll_s = self._poll_interval()
        last_event = started
        finished = False
        try:
            while True:
                reason = self._watchdog(time.monotonic(), started, last_event, run)
                if reason:
                    run.abort_reason = reason
                    break
                try:
                    raw = lines.get(timeout=poll_s)
                except queue.Empty:
                    if run.result is not None and proc.poll() is not None:
                        # The CLI replied and exited; a descendant still holds stdout open.
                        finished = True
                        break
                    continue
                if raw is None:
                    finished = True
                    break
                last_event = time.monotonic()
                run.feed(raw)
                if run.compacted:
                    break
        finally:
            run.returncode = self._wait(proc) if finished else self._terminate(proc)
            readers = threads[1:]
            for t in readers:
                t.join(timeout=min(self.kill_grace_s, 1.0))
            if any(t.is_alive() for t in readers):
                # Descendants that inherited stdout/stderr outlive the CLI: end the rest of its process group.
                _signal(proc, signal.SIGKILL)
                for t in readers:
                    t.join(timeout=self.kill_grace_s)
            while True:
                try:
                    raw = lines.get_nowait()
                except queue.Empty:
                    break
                if raw is None:
                    break
                run.feed(raw)
            for t, pipe in zip(readers, (proc.stdout, proc.stderr)):
                if t.is_alive():
                    continue        # a reader blocked in read() holds the pipe's lock; close() would block too
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
            run.stderr = b"".join(err_chunks).decode("utf-8", errors="replace")
            run.seconds = round(time.monotonic() - started, 3)
        return run

    def _poll_interval(self) -> float:
        limits = [t for t in (self.idle_timeout_s, self.first_text_timeout_s, self.total_timeout_s) if t]
        return max(0.01, min([1.0] + [t / 10.0 for t in limits]))

    def _watchdog(self, now: float, started: float, last_event: float, run: StreamRun) -> Optional[str]:
        if now - started > self.total_timeout_s:
            return f"total time limit of {self.total_timeout_s:g} s exceeded"
        if now - last_event > self.idle_timeout_s:
            return f"no stream event for {self.idle_timeout_s:g} s"
        if self.first_text_timeout_s is not None and not run.saw_output and now - started > self.first_text_timeout_s:
            return f"no output text within {self.first_text_timeout_s:g} s"
        return None

    def _wait(self, proc: Any) -> Optional[int]:
        try:
            return proc.wait(timeout=self.kill_grace_s)
        except subprocess.TimeoutExpired:
            return self._terminate(proc)

    def _terminate(self, proc: Any) -> Optional[int]:
        """SIGINT (ends the turn cleanly), then SIGTERM, then SIGKILL, each to the CLI's whole process group."""
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                break
            try:
                _signal(proc, sig)
            except OSError:
                break
            try:
                proc.wait(timeout=self.kill_grace_s)
                break
            except subprocess.TimeoutExpired:
                continue
        return proc.poll()

    # ------------------------------------------------------------ result checks
    def check(self, run: StreamRun, model: str) -> Dict[str, Any]:
        """Return structured_output or raise a PlannerError describing why the call is unusable."""
        if run.compacted:
            raise PlannerError("the CLI compacted the conversation (compact_boundary); the reply cannot be trusted")
        if run.abort_reason and run.result is None:
            # A result is the stream's last message: one that arrived before the abort is judged on its own.
            raise PlannerTimeoutError(f"watchdog aborted the call: {run.abort_reason}")
        unknown = _UNKNOWN_OPTION.search(run.stderr)
        if unknown:
            raise PlannerError(
                f"claude CLI {self._version or ''} rejected flag {unknown.group(1)}; update the CLI or drop the flag",
                retryable=False,
            )
        res = run.result
        if res is None:
            raise PlannerError(f"no result message (exit code {run.returncode}); stderr: {run.stderr.strip()[-500:]}")
        self._check_model(run, model)
        text = str(res.get("result") or "")
        subtype, stop = res.get("subtype"), res.get("stop_reason")
        truncated = stop == "max_tokens" or "may be incomplete" in text.lower()
        if res.get("is_error") or subtype != "success":
            if _USAGE_LIMIT.search(text):
                raise PlannerError(f"Claude usage limit reached: {text.strip()[:300]}", retryable=False)
            if truncated:
                raise PlannerTruncatedError(f"reply hit the output-token cap ({subtype}); partial JSON is not stitched")
            raise PlannerError(
                f"claude call failed (subtype={subtype}, is_error={res.get('is_error')}, "
                f"exit code {run.returncode}): {text.strip()[:500]}"
            )
        if truncated:
            raise PlannerTruncatedError("reply hit the output-token cap (stop_reason=max_tokens); partial JSON is not stitched")
        if stop == "refusal":
            raise PlannerError("the model refused the request (stop_reason=refusal)")
        structured = res.get("structured_output")
        if not isinstance(structured, dict):
            raise PlannerError("result has no structured_output object")
        return structured

    @staticmethod
    def _check_model(run: StreamRun, model: str) -> None:
        """The main assistant messages must come from the pinned model; auxiliary calls are ignored."""
        main = [m for m in run.main_models if m and m != "<synthetic>"]
        if main:
            wrong = sorted({m for m in main if not same_model(m, model)})
            if wrong:
                raise PlannerError(f"model rerouted: expected {model}, got {', '.join(wrong)}", retryable=False)
            return
        usage = (run.result or {}).get("modelUsage") or (run.result or {}).get("model_usage") or {}
        if usage and not any(same_model(k, model) for k in usage):
            raise PlannerError(f"model rerouted: expected {model}, model usage shows {sorted(usage)}", retryable=False)

    # ------------------------------------------------------------ replay cache + records
    def replay_key(
        self, model: str, effort: str, role: str, max_output_tokens: int, system_prompt: str,
        schema: Dict[str, Any], body: str, context: str = "", tools: str = "",
    ) -> str:
        """sha256 of the logical request. `context` binds inputs the body does not carry
        (the plan a repair resumes from, the image bytes a judge reads)."""
        payload = {
            "cli_version": self.version(), "model": model, "effort": effort, "role": role,
            "max_output_tokens": max_output_tokens, "system_prompt_sha256": sha256_text(system_prompt),
            "schema": schema, "body": body, "context": context, "tools": tools,
        }
        return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return entry if isinstance(entry, dict) and isinstance(entry.get("structured_output"), dict) else None

    def cache_put(self, key: str, spec: CallSpec, run: StreamRun, structured: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key, "role": spec.role, "model": spec.model, "effort": spec.effort, "cli_version": self._version,
            "session_id": run.session_id, "structured_output": structured, "result": run.result,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, path)

    def cache_drop(self, key: str) -> None:
        """Forget a cached reply that turned out to be unusable (record mode only; replay mode is read-only)."""
        if self.replay == "record":
            try:
                self._cache_path(key).unlink()
            except FileNotFoundError:
                pass

    def write_record(
        self, work_dir: Path, spec: CallSpec, key: str, *, argv: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None, run: Optional[StreamRun] = None,
        error: Optional[BaseException] = None, cached: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write work_dir/claude/<role>_<n>.json (+ .events.jsonl). Env values are never recorded."""
        out = Path(work_dir) / "claude"
        out.mkdir(parents=True, exist_ok=True)
        taken = [int(m.group(1)) for p in out.glob(f"{spec.role}_*.json")
                 if (m := re.fullmatch(rf"{re.escape(spec.role)}_(\d+)\.json", p.name))]
        stem = f"{spec.role}_{max(taken, default=0) + 1}"
        result = (run.result if run else (cached or {}).get("result")) or {}
        prompt_sha = sha256_file(spec.system_prompt_file) if Path(spec.system_prompt_file).exists() else None
        record = {
            "role": spec.role,
            "mode": "replay" if cached is not None else spec.mode,
            "replayed": cached is not None,
            "replay_key": key,
            "cli_version": self._version,
            "model": spec.model,
            "effort": spec.effort,
            "max_output_tokens": spec.max_output_tokens,
            "argv": argv,                      # exact; the system prompt is a file, so no prompt text here
            "cwd": str(self.cwd),
            "env_keys": sorted(env) if env is not None else None,
            "system_prompt_file": str(spec.system_prompt_file),
            "system_prompt_sha256": prompt_sha,
            "schema": spec.schema,
            "stdin": spec.body,
            "requested_session_id": spec.session_id if spec.mode == "fresh" else None,
            "resumed_from": spec.resume_id,
            "session_id": run.session_id if run else (cached or {}).get("session_id"),
            "seconds": run.seconds if run else 0.0,
            "returncode": run.returncode if run else None,
            "abort_reason": run.abort_reason if run else None,
            "init": run.init if run else None,
            "main_models": run.main_models if run else None,
            "stream_deltas": run.deltas if run else None,
            "total_cost_usd": result.get("total_cost_usd"),
            "usage": result.get("usage"),
            "model_usage": result.get("modelUsage") or result.get("model_usage"),
            "result": result or None,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            "stderr_tail": run.stderr[-4000:] if run else None,
            "non_json_lines": run.non_json[:20] if run else None,
        }
        path = out / f"{stem}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        if run is not None:
            (out / f"{stem}.events.jsonl").write_text("".join(json.dumps(e, default=str) + "\n" for e in run.events))
        return path

    # ------------------------------------------------------------ one logical call
    def run(
        self, work_dir: Path, key: str, attempts: Sequence[CallSpec], accept: Callable[[Dict[str, Any]], T],
    ) -> Tuple[T, Optional[str]]:
        """
        Serve a logical call from the replay cache, else try `attempts` in order until `accept`
        (parse + validate) succeeds. Returns (value, session_id). Only accepted replies are cached.
        """
        role = attempts[0].role
        # 1. Replay cache.
        if self.replay != "off":
            entry = self.cache_get(key)
            if entry is not None:
                try:
                    value = accept(entry["structured_output"])
                except (PlannerError, ValidationError) as exc:
                    if self.replay == "replay":
                        raise PlannerError(f"cached {role} reply {key[:12]} no longer validates: {exc}",
                                           retryable=False) from exc
                else:
                    self.write_record(work_dir, attempts[0], key, cached=entry)
                    return value, entry.get("session_id")
            elif self.replay == "replay":
                raise PlannerError(f"replay miss for the {role} call (key {key[:12]}); use PREVIS_CLAUDE_REPLAY=record",
                                   retryable=False)

        # 2. Live attempts.
        errors: List[str] = []
        last: Optional[PlannerError] = None
        for spec in attempts:
            argv: Optional[List[str]] = None
            env: Optional[Dict[str, str]] = None
            run: Optional[StreamRun] = None
            error: Optional[PlannerError] = None
            try:
                argv = self.build_argv(spec)
                env = self.build_env(spec.max_output_tokens)
                run = self.stream(argv, spec.body, env)
                structured = self.check(run, spec.model)
                value = accept(structured)
            except PlannerError as exc:
                error = exc
            except ValidationError as exc:
                error = PlannerError(f"{role} reply does not match the plan schema: {exc}")
            self.write_record(work_dir, spec, key, argv=argv, env=env, run=run, error=error)
            if error is None:
                if self.replay != "off":
                    self.cache_put(key, spec, run, structured)
                return value, run.session_id
            errors.append(f"[{spec.mode}] {error}")
            last = error
            if not error.retryable:
                break
        raise type(last)(f"{role} call failed after {len(errors)} attempt(s): " + " | ".join(errors),
                         retryable=last.retryable)


# ---------------------------------------------------------------- reply parsing
def parse_bible(structured: Dict[str, Any]) -> SceneBible:
    """Validate call A's reply against the schema. Content defects are found by bible_violations()."""
    bible = SceneBible.model_validate(structured)
    if not bible.shots or not bible.beats or not (bible.characters or bible.props):
        raise PlannerError("bible needs at least one shot, one beat and one character or prop")
    return bible


@dataclass(frozen=True)
class BibleProblem:
    """One bible-level defect found by the planner's own checks; `ids` are the ids it is about."""
    code: str
    text: str
    ids: Tuple[str, ...] = ()

    def line(self) -> str:
        return f"[{self.code}] {self.text}"


def bible_violations(bible: SceneBible) -> List[str]:
    """
    Bible-level defects as "[code] fact" lines (empty when the bible is sound). No timeline repair can
    fix these: repairs replace shot timelines only.

    The authoritative source is validator.validate_bible (the validator's own bible-only checks, including
    the "init"-state custody, support and fill checks), looked up lazily so the planner works before and
    after it exists. The planner's local checks (local_bible_problems) always run as well: they cover
    what call B and the image stages need (derivable keyframe ids, keyframe_count 1-4, one phys_body and
    one "init" state per entity, props that can be told apart), and they are the fallback for the
    validator's checks when validate_bible is missing or fails. A local problem the validator already
    reports (same code, naming all of its ids) is not repeated.
    """
    local = local_bible_problems(bible)
    shared = shared_bible_violations(bible)
    if shared is None:
        return [p.line() for p in local]
    out = [_violation_line(v) for v in shared]
    out += [p.line() for p in local if not any(_same_defect(p, v) for v in shared)]
    return out


def shared_bible_violations(bible: SceneBible) -> Optional[List[Any]]:
    """Errors from validator.validate_bible, or None when it is not available or fails on this bible."""
    try:
        from src.previs import validator
    except ImportError:
        return None
    check = getattr(validator, "validate_bible", None)
    if not callable(check):
        return None
    try:
        found = list(check(bible))
    except Exception:  # noqa: BLE001 - a crash in the shared checks must not cost the local ones
        return None
    return [v for v in found if getattr(v, "severity", "error") == "error"]


def _violation_line(v: Any) -> str:
    where = [f"{label} {value}" for label, value in (("shot", v.shot_id), ("keyframe", v.kf_id), ("entity", v.entity_id))
             if value not in (NONE, "", None)]
    return f"[{v.code}] " + (f"{', '.join(where)}: " if where else "") + v.message


def _same_defect(problem: BibleProblem, v: Any) -> bool:
    """True when a validator violation already reports this local problem: same code, and it names every id
    the problem is about (so another defect of the same shot or entity is still listed)."""
    if v.code != problem.code:
        return False
    named = {v.shot_id, v.kf_id, v.entity_id} | set(re.findall(r"\b[a-z]{2}_[a-z0-9_]+", v.message))
    return set(problem.ids) <= named


def _rgb(hex_colour: str) -> Optional[Tuple[int, int, int]]:
    text = hex_colour.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", text):
        return None
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


# Two props closer than this in every RGB channel, with the same shape and a similar size, look alike.
SAME_COLOUR_MAX_DIFF = 48
SAME_SIZE_MAX_RATIO = 1.5


def look_alike_props(bible: SceneBible) -> List[Tuple[Any, Any]]:
    """
    Pairs of movable props a viewer, the image model and the frame judge cannot tell apart: same shape,
    similar size and nearly the same colour. Names cannot fix that (the image only shows colour and form).
    """
    def similar(a: float, b: float) -> bool:
        lo, hi = sorted((max(a, 1e-6), max(b, 1e-6)))
        return hi / lo <= SAME_SIZE_MAX_RATIO

    movable = [p for p in bible.props if p.movable]
    pairs = []
    for i, a in enumerate(movable):
        for b in movable[i + 1:]:
            ca, cb = _rgb(a.color_hex), _rgb(b.color_hex)
            if a.shape != b.shape or ca is None or cb is None:
                continue
            if max(abs(x - y) for x, y in zip(ca, cb)) > SAME_COLOUR_MAX_DIFF:
                continue
            if all(similar(getattr(a, f), getattr(b, f)) for f in ("size_x_m", "size_y_m", "size_z_m")):
                pairs.append((a, b))
    return pairs


def local_bible_problems(bible: SceneBible) -> List[BibleProblem]:
    """
    The planner's own bible checks: ids and references, ranges, marks inside their location, target
    duration, 180-degree rule, one location per setup (mirrors of the validator's bible-only checks),
    plus the call A rules call B depends on and props that must look different.
    """
    out: List[BibleProblem] = []

    def add(code: str, text: str, *ids: str) -> None:
        out.append(BibleProblem(code, text, tuple(i for i in ids if i and i != NONE)))

    entity_list = [c.id for c in bible.characters] + [p.id for p in bible.props]
    entities = set(entity_list)
    locations = {lo.id: lo for lo in bible.locations}
    marks = {m.id for m in bible.marks}
    beats = {b.id for b in bible.beats}

    # 1. IDs: unique per namespace; keyframe ids derivable from every shot id.
    for label, ids in (("entity", entity_list), ("location", [lo.id for lo in bible.locations]),
                       ("mark", [m.id for m in bible.marks]), ("beat", [b.id for b in bible.beats]),
                       ("shot", [s.id for s in bible.shots])):
        for dupe in sorted({i for i in ids if ids.count(i) > 1}):
            add("ref_integrity", f"{label} id {dupe} is used more than once", dupe)
    for shot in bible.shots:
        bad = [k for k in keyframe_ids_for_shot(shot) if not _KF_ID.match(k)]
        if bad:
            add("keyframe_ids", f"shot id {shot.id} is too long to derive valid keyframe ids (e.g. {bad[-1]}); "
                f"use at most 12 characters after sh_", shot.id)
        if not 1 <= shot.keyframe_count <= 4:
            add("keyframe_ids", f"shot {shot.id} keyframe_count is {shot.keyframe_count}; it must be 1 to 4", shot.id)

    # 2. References.
    for mark in bible.marks:
        if mark.location_id not in locations:
            add("ref_integrity", f"mark {mark.id} is in unknown location {mark.location_id}", mark.id, mark.location_id)
    for beat in bible.beats:
        if beat.cause_beat_id != NONE and beat.cause_beat_id not in beats:
            add("ref_integrity", f"beat {beat.id} has unknown cause_beat_id {beat.cause_beat_id}",
                beat.id, beat.cause_beat_id)
    for shot in bible.shots:
        if shot.location_id not in locations:
            add("ref_integrity", f"shot {shot.id} is in unknown location {shot.location_id}", shot.id, shot.location_id)
        for beat_id in shot.beat_ids:
            if beat_id not in beats:
                add("ref_integrity", f"shot {shot.id} lists unknown beat {beat_id}", shot.id, beat_id)
        if shot.subject_id not in entities:
            add("ref_integrity", f"shot {shot.id} subject {shot.subject_id} does not exist", shot.id, shot.subject_id)
        if shot.secondary_id != NONE and shot.secondary_id not in entities:
            add("ref_integrity", f"shot {shot.id} secondary {shot.secondary_id} does not exist",
                shot.id, shot.secondary_id)

    # 3. Exactly one phys_body and one "init" state per entity, each with valid references.
    for label, ids in (("phys_body", [b.entity_id for b in bible.phys_bodies]),
                       ("initial state", [s.entity_id for s in bible.initial_states])):
        for eid in entity_list:
            if eid not in ids:
                add("ref_integrity", f"{eid} has no {label}", eid)
            elif ids.count(eid) > 1:
                add("ref_integrity", f"{eid} has {ids.count(eid)} {label}s; it needs exactly one", eid)
        for eid in sorted(set(ids) - entities):
            add("ref_integrity", f"{label} for unknown entity {eid}", eid)
    for st in bible.initial_states:
        refs = [("kf_id", st.kf_id, {INIT}), ("mark_id", st.mark_id, marks | {NONE}),
                ("facing_id", st.facing_id, entities | marks | {NONE})]
        refs += [(name, getattr(st, name), entities | {NONE}) for name in ("holding_id", "gaze_id", "support_id")]
        for name, value, valid in refs:
            if value not in valid:
                add("ref_integrity", f"initial state of {st.entity_id}: {name} '{value}' "
                    + ("must be \"init\"" if name == "kf_id" else "does not exist"), st.entity_id)

    # 4. Ranges and runtime.
    lo_h, hi_h = HEIGHT_RANGE_M
    for ch in bible.characters:
        if not lo_h <= ch.height_m <= hi_h:
            add("ranges", f"{ch.id} height {ch.height_m:g} m is outside {lo_h:g}-{hi_h:g} m", ch.id)
    for prop in bible.props:
        if min(prop.size_x_m, prop.size_y_m, prop.size_z_m) <= 0:
            add("ranges", f"{prop.id} has a non-positive size", prop.id)
    for loc in bible.locations:
        if loc.width_m <= 0 or loc.depth_m <= 0:
            add("ranges", f"location {loc.id} has a non-positive extent", loc.id)
    for mark in bible.marks:
        loc = locations.get(mark.location_id)
        if loc is not None and not (0 <= mark.x_m <= loc.width_m and 0 <= mark.z_m <= loc.depth_m):
            add("ranges", f"mark {mark.id} ({mark.x_m:g}, {mark.z_m:g}) is outside {loc.id} "
                f"(0..{loc.width_m:g} x 0..{loc.depth_m:g} m)", mark.id)
    lo_l, hi_l = LENS_RANGE_MM
    for shot in bible.shots:
        if not lo_l <= shot.lens_mm <= hi_l:
            add("ranges", f"shot {shot.id} lens {shot.lens_mm:g} mm is outside {lo_l:g}-{hi_l:g} mm", shot.id)
    target, limit = bible.scene.target_duration_s, config.MAX_VIDEO_SECONDS
    if not 0 < target <= limit:
        add("duration", f"target_duration_s is {target:g} s; it must be more than 0 and at most {limit:g}")

    # 5. Camera grammar: the 180-degree rule within a run of same-location shots; one location per setup.
    shots = sorted(bible.shots, key=lambda s: s.order)
    last: Dict[frozenset, Any] = {}
    location = None
    for shot in shots:
        if shot.location_id != location:
            last, location = {}, shot.location_id
        if shot.secondary_id in (NONE, shot.subject_id):
            continue
        pair = frozenset((shot.subject_id, shot.secondary_id))
        prev = last.get(pair)
        free = any(m.startswith(LINE_FREE_MOVE_PREFIXES) for m in (shot.move, prev.move if prev else "static"))
        if prev is not None and prev.line_side != shot.line_side and not free:
            add("line_side", f"{shot.id} puts the camera on side '{shot.line_side}' of the "
                f"{shot.subject_id}/{shot.secondary_id} line but {prev.id} used side '{prev.line_side}' "
                f"(180-degree rule; cross only with an orbit or crane move)", shot.id)
        last[pair] = shot
    setup_location: Dict[str, str] = {}
    for shot in shots:
        first = setup_location.setdefault(shot.setup_id, shot.location_id)
        if first != shot.location_id:
            add("setup_location", f"setup {shot.setup_id} is used in {first} and in {shot.location_id} "
                f"({shot.id}); one setup belongs to one location", shot.id, shot.setup_id)

    # 6. Props the image model and the frame judge must tell apart (names alone cannot).
    for a, b in look_alike_props(bible):
        add("distinct_props", f"props {a.id} ('{a.name}', {a.color_hex}) and {b.id} ('{b.name}', {b.color_hex}) "
            f"are the same shape, size and colour, so no frame can show which is which; give one a clearly "
            f"different color_hex and name each by its look (e.g. 'white coffee cup', 'blue coffee cup')", a.id, b.id)
    return out


def parse_checked_bible(structured: Dict[str, Any]) -> SceneBible:
    """Accept a bible repair only when the repaired bible passes every bible check."""
    bible = parse_bible(structured)
    problems = bible_violations(bible)
    if problems:
        raise PlannerError(f"the repaired bible still fails {len(problems)} check(s): " + "; ".join(problems[:8]))
    return bible


def assemble_plan(bible: SceneBible, structured: Dict[str, Any]) -> ScenePlan:
    """Validate call B's reply: exactly one ShotTimeline per bible shot, returned in shot order."""
    timeline = Timeline.model_validate(structured)
    order = {s.id: s.order for s in bible.shots}
    got = [st.shot_id for st in timeline.shots]
    missing = [s for s in order if s not in got]
    extra = sorted(set(got) - set(order))
    if missing or extra or len(set(got)) != len(got):
        raise PlannerError(f"timeline shots {got} do not match the bible shots {list(order)}")
    shots = sorted(timeline.shots, key=lambda st: order[st.shot_id])
    return ScenePlan(bible=bible, timeline=Timeline(invariants=timeline.invariants, shots=shots))


def amend_shot_set(plan: ScenePlan, kf_id: str) -> List[str]:
    """The keyframe's shot plus every later shot that references an entity of that shot."""
    kf = kf_id.strip().lower()
    owner = next((st.shot_id for st in plan.timeline.shots if any(k.id == kf for k in st.keyframes)), None)
    if owner is None:
        raise ValueError(f"keyframe {kf_id} is not in the plan")
    timelines = {st.shot_id: st for st in plan.timeline.shots}

    def entities(shot_id: str) -> set:
        shot = next(s for s in plan.bible.shots if s.id == shot_id)
        ids = {shot.subject_id, shot.secondary_id}
        st = timelines.get(shot_id)
        if st is not None:
            ids |= {s.entity_id for s in st.states}
            for e in st.events:
                ids |= {e.actor_id, e.target_id, e.outcome_support_id}
        ids.discard("none")
        return ids

    shots = sorted(plan.bible.shots, key=lambda s: s.order)
    owner_order = next(s.order for s in shots if s.id == owner)
    base = entities(owner)
    return [owner] + [s.id for s in shots if s.order > owner_order and entities(s.id) & base]


def parse_answers(structured: Dict[str, Any], count: int) -> List[str]:
    """Map judge answers (q_index is 1-based; a clean 0-based reply is accepted) onto the questions."""
    items = structured.get("answers")
    if not isinstance(items, list):
        raise PlannerError("facts reply has no answers list")
    pairs = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("q_index"), int) and not isinstance(item.get("q_index"), bool):
            answer = str(item.get("answer", "")).strip().lower()
            pairs.append((item["q_index"], answer if answer in FACT_ANSWERS else "unsure"))
    indices = {q for q, _ in pairs}
    base = 0 if (0 in indices and count not in indices) else 1
    answers = ["unsure"] * count
    seen = set()
    for q, answer in pairs:
        pos = q - base
        if 0 <= pos < count and pos not in seen:
            answers[pos] = answer
            seen.add(pos)
    return answers


# ---------------------------------------------------------------- planner
class ClaudeCliPlanner(PhysicsPlanner):
    """
    Claude (via `claude -p`) as the physics director. Extra keyword arguments go to ClaudeCliRunner
    (restricted, idle_timeout_s, first_text_timeout_s, total_timeout_s, replay, cache_dir, cwd,
    force_idle_timeout_ms, kill_grace_s).
    """

    def __init__(
        self,
        model: str = PLANNER_MODEL,
        effort: str = "high",
        timeout_s: float = 2700,
        config_dir: Optional[str] = None,
        auth: str = "subscription",
        claude_bin: str = config.CLAUDE_CLI,
        **runner_kw: Any,
    ):
        self.model = model
        self.effort = effort
        self.cli = ClaudeCliRunner(claude_bin=claude_bin, auth=auth, config_dir=config_dir, timeout_s=timeout_s, **runner_kw)
        self._session: Optional[Dict[str, str]] = None

    def _spec(self, role: str, body: str, schema: Dict[str, Any], files: Tuple[Path, Path], max_out: int,
              resume_id: Optional[str] = None) -> CallSpec:
        return CallSpec(role=role, model=self.model, effort=self.effort, body=body, schema=schema,
                        system_prompt_file=files[0], settings_file=files[1], max_output_tokens=max_out,
                        resume_id=resume_id)

    def _key(self, role: str, max_out: int, schema: Dict[str, Any], body: str, context: str = "") -> str:
        return self.cli.replay_key(self.model, self.effort, role, max_out, prompts.SYSTEM_PROMPT, schema, body, context)

    def _remember(self, work_dir: Path, session_id: Optional[str], plan: ScenePlan) -> None:
        """The last good session and the plan it ends with; repairs resume it only for that exact plan."""
        if not session_id:
            return
        self._session = {"session_id": session_id, "plan_sha256": sha256_text(plan.model_dump_json())}
        (Path(work_dir) / "claude" / "session.json").write_text(json.dumps(self._session, indent=2))

    def _session_for(self, work_dir: Path, plan_sha: str) -> Optional[str]:
        state = self._session
        if state is None:
            try:
                state = json.loads((Path(work_dir) / "claude" / "session.json").read_text())
            except (OSError, ValueError):
                return None
        if isinstance(state, dict) and state.get("plan_sha256") == plan_sha:
            return state.get("session_id")
        return None

    def plan(self, scene_text: str, work_dir: Path) -> ScenePlan:
        work_dir = Path(work_dir)
        files = self.cli.prepare(work_dir, prompts.SYSTEM_PROMPT, "system_prompt.md")

        # 1. Call A: the bible, fresh session.
        body_a = prompts.call_a_input(scene_text)
        schema_a = bible_json_schema()
        key_a = self._key("a", MAX_OUTPUT_A, schema_a, body_a)
        bible, sid_a = self.cli.run(work_dir, key_a, [self._spec("a", body_a, schema_a, files, MAX_OUTPUT_A)], parse_bible)

        # 2. Bible checks. Repairs only replace timelines, so bible defects get one bible repair round now,
        #    before B; a bible that cannot be repaired is dropped from the replay cache so a rerun starts over.
        problems = bible_violations(bible)
        if problems:
            try:
                bible, sid_a = self._repair_bible(work_dir, files, scene_text, bible, problems, sid_a)
            except PlannerError as exc:
                if exc.retryable:
                    self.cli.cache_drop(key_a)
                raise

        # 3. Call B: the timeline, forked from A or the bible repair; one fresh fallback (the body carries the bible).
        kf_ids = {s.id: keyframe_ids_for_shot(s) for s in sorted(bible.shots, key=lambda s: s.order)}
        body_b = prompts.call_b_input(bible.model_dump_json(), kf_ids)
        schema_b = timeline_json_schema(bible)
        key_b = self._key("b", MAX_OUTPUT_TIMELINE, schema_b, body_b)
        attempts = [self._spec("b", body_b, schema_b, files, MAX_OUTPUT_TIMELINE, resume_id=sid_a)] if sid_a else []
        attempts.append(self._spec("b", body_b, schema_b, files, MAX_OUTPUT_TIMELINE))
        plan, sid_b = self.cli.run(work_dir, key_b, attempts, lambda so: assemble_plan(bible, so))
        self._remember(work_dir, sid_b, plan)
        return plan

    def _repair_bible(self, work_dir: Path, files: Tuple[Path, Path], scene_text: str, bible: SceneBible,
                      problems: List[str], sid_a: Optional[str]) -> Tuple[SceneBible, Optional[str]]:
        """One bible repair round: fork of call A with the same bible schema, one fresh fallback (the body
        carries the scene and the bible). The reply is accepted only when it passes every bible check."""
        schema = bible_json_schema()
        body = prompts.bible_repair_input(scene_text, bible.model_dump_json(), problems)
        key = self._key("bible_repair", MAX_OUTPUT_A, schema, body)
        attempts = [self._spec("bible_repair", body, schema, files, MAX_OUTPUT_A, resume_id=sid_a)] if sid_a else []
        attempts.append(self._spec("bible_repair", body, schema, files, MAX_OUTPUT_A))
        return self.cli.run(work_dir, key, attempts, parse_checked_bible)

    def repair(self, plan: ScenePlan, violations_text: str, shot_ids: List[str], work_dir: Path) -> ScenePlan:
        work_dir = Path(work_dir)
        if not violations_text.strip():
            raise ValueError("violations_text is empty")
        shots = ordered_shot_ids(plan, shot_ids)
        files = self.cli.prepare(work_dir, prompts.SYSTEM_PROMPT, "system_prompt.md")
        plan_json = plan.model_dump_json()
        plan_sha = sha256_text(plan_json)
        kf_ids = shot_keyframe_ids(plan)
        schema = timeline_json_schema(plan.bible)
        body = prompts.repair_input(violations_text, shots, kf_ids)

        # 1. Fork the last good session when it ended on exactly this plan; else (or on failure) go fresh.
        attempts = []
        session_id = self._session_for(work_dir, plan_sha)
        if session_id:
            attempts.append(self._spec("repair", body, schema, files, MAX_OUTPUT_TIMELINE, resume_id=session_id))
        fresh_body = prompts.repair_input(violations_text, shots, kf_ids, plan_json=plan_json)
        attempts.append(self._spec("repair", fresh_body, schema, files, MAX_OUTPUT_TIMELINE))

        # 2. Replace only the listed shots; replies touching other shots are rejected. The key binds the
        #    session being continued, so a round that left the plan unchanged is never replayed as the next one.
        key = self._key("repair", MAX_OUTPUT_TIMELINE, schema, body, context=f"{plan_sha}:{session_id or 'fresh'}")
        merged, new_session = self.cli.run(
            work_dir, key, attempts, lambda so: replace_shots(plan, Timeline.model_validate(so), shots))
        self._remember(work_dir, new_session, merged)
        return merged

    def amend(self, plan: ScenePlan, facts_text: str, kf_id: str, work_dir: Path) -> ScenePlan:
        work_dir = Path(work_dir)
        if not facts_text.strip():
            raise ValueError("facts_text is empty")
        shots = amend_shot_set(plan, kf_id)
        files = self.cli.prepare(work_dir, prompts.SYSTEM_PROMPT, "system_prompt.md")
        schema = timeline_json_schema(plan.bible)
        body = prompts.amend_input(plan.model_dump_json(), facts_text, kf_id.strip().lower(), shots, shot_keyframe_ids(plan))
        key = self._key("amend", MAX_OUTPUT_TIMELINE, schema, body)
        merged, session_id = self.cli.run(
            work_dir, key, [self._spec("amend", body, schema, files, MAX_OUTPUT_TIMELINE)],
            lambda so: replace_shots(plan, Timeline.model_validate(so), shots))
        self._remember(work_dir, session_id, merged)
        return merged


# ---------------------------------------------------------------- fact judge
class ClaudeCliJudge:
    """
    Yes/no/unsure fact checks on a rendered frame. The image is read through the Read tool
    (`--tools Read --allowedTools Read --add-dir <image dir>`). Extra keyword arguments go to ClaudeCliRunner.
    """

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        effort: str = "low",
        timeout_s: float = 600,
        config_dir: Optional[str] = None,
        auth: str = "subscription",
        claude_bin: str = config.CLAUDE_CLI,
        max_turns: int = 4,
        **runner_kw: Any,
    ):
        runner_kw.setdefault("idle_timeout_s", 300.0)
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.cli = ClaudeCliRunner(claude_bin=claude_bin, auth=auth, config_dir=config_dir, timeout_s=timeout_s, **runner_kw)

    def facts(self, image_path: str, questions: List[str], work_dir: Path) -> List[str]:
        if not questions:
            return []
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        files = self.cli.prepare(Path(work_dir), prompts.JUDGE_SYSTEM_PROMPT, "judge_system_prompt.md")
        body = prompts.facts_input(str(image), questions)
        spec = CallSpec(role="facts", model=self.model, effort=self.effort, body=body, schema=FACTS_SCHEMA,
                        system_prompt_file=files[0], settings_file=files[1], max_output_tokens=MAX_OUTPUT_FACTS,
                        persist=False, tools="Read", add_dir=image.parent, max_turns=self.max_turns)
        key = self.cli.replay_key(self.model, self.effort, "facts", MAX_OUTPUT_FACTS, prompts.JUDGE_SYSTEM_PROMPT,
                                  FACTS_SCHEMA, body, context=sha256_file(image), tools="Read")
        answers, _ = self.cli.run(Path(work_dir), key, [spec], lambda so: parse_answers(so, len(questions)))
        return answers


# ---------------------------------------------------------------- P0 probe (explicit use only)
def probe(work_dir: Path, model: str = PLANNER_MODEL, effort: str = "low", **runner_kw: Any) -> Dict[str, Any]:
    """
    Cheap P0 checks against the real CLI: two tiny calls (fresh, then resume+fork with a different schema).
    Reports init isolation (tools, MCP servers, model, auth source), whether --settings was applied
    (permissionMode dontAsk), flag acceptance, structured output, and resume+fork with a changed schema.
    Spends real usage: never call it from unit tests. Writes work_dir/claude/probe_report.json.
    """
    runner_kw.setdefault("replay", "off")
    cli = ClaudeCliRunner(**runner_kw)
    work_dir = Path(work_dir)
    files = cli.prepare(work_dir, PROBE_SYSTEM_PROMPT, "probe_system_prompt.md")
    report: Dict[str, Any] = {"cli_version": cli.version(), "model": model, "effort": effort, "checks": {}}
    checks = report["checks"]

    def call(body: str, schema: Dict[str, Any], resume_id: Optional[str] = None):
        spec = CallSpec(role="probe", model=model, effort=effort, body=body, schema=schema, system_prompt_file=files[0],
                        settings_file=files[1], max_output_tokens=4000, resume_id=resume_id)
        argv = env = run = structured = error = None
        try:
            argv, env = cli.build_argv(spec), cli.build_env(spec.max_output_tokens)
            run = cli.stream(argv, spec.body, env)
            structured = cli.check(run, model)
        except PlannerError as exc:
            error = exc
        cli.write_record(work_dir, spec, "probe", argv=argv, env=env, run=run, error=error)
        return run, structured, error

    def put(name: str, passed: bool, detail: Any = "") -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    schema_1 = {"type": "object", "properties": {"ok": {"type": "boolean"}, "word": {"type": "string"}},
                "required": ["ok", "word"], "additionalProperties": False}
    run_1, out_1, err_1 = call('Set ok to true and word to "probe".', schema_1)
    init = run_1.init if run_1 else {}
    result_1 = (run_1.result if run_1 else None) or {}
    report["init"] = {k: init.get(k) for k in ("model", "tools", "mcp_servers", "apiKeySource", "permissionMode", "cwd")}
    put("flags_accepted", err_1 is None or "rejected flag" not in str(err_1), str(err_1 or ""))
    put("call_succeeded", err_1 is None, str(err_1 or ""))
    put("isolation_tools", set(init.get("tools") or []) <= {"StructuredOutput"}, init.get("tools"))
    put("isolation_mcp", not init.get("mcp_servers"), init.get("mcp_servers"))
    put("model_pinned", same_model(str(init.get("model") or ""), model), init.get("model"))
    put("auth_source", cli.auth == "api_key" or init.get("apiKeySource") in (None, "none"), init.get("apiKeySource"))
    put("settings_applied", init.get("permissionMode") == "dontAsk", init.get("permissionMode"))
    put("structured_output", bool(out_1) and out_1.get("ok") is True, {"output": out_1, "num_turns": result_1.get("num_turns")})
    required = ("is_error", "subtype", "session_id", "total_cost_usd", "usage", "structured_output")
    optional = ("stop_reason", "num_turns", "modelUsage", "model_usage")
    put("result_fields", all(k in result_1 for k in required),
        {"present": [k for k in required + optional if k in result_1]})

    if run_1 is not None and run_1.session_id and err_1 is None:
        schema_2 = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": False}
        run_2, out_2, err_2 = call("Set n to 7.", schema_2, resume_id=run_1.session_id)
        usage_2 = ((run_2.result if run_2 else None) or {}).get("usage") or {}
        forked = bool(run_2 and run_2.session_id and run_2.session_id != run_1.session_id)
        put("resume_fork_schema_change", err_2 is None and forked and bool(out_2) and out_2.get("n") == 7,
            {"error": str(err_2 or ""), "new_session": forked, "cache_read_input_tokens": usage_2.get("cache_read_input_tokens")})
    else:
        put("resume_fork_schema_change", False, "skipped: first call failed")
    (work_dir / "claude" / "probe_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
