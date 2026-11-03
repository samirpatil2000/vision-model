"""
sd-cli backend: one stable-diffusion.cpp process per request.

Weights reload on every call, so this is the simple, crash-isolated path. Each run is wrapped
in /usr/bin/time to record peak memory, and the engine log is parsed for phase timings and
flash-attention / prefix-cache fallbacks.

The module also holds the request validation, model-flag and LoRA helpers that the sd-server
backend shares.
"""
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from src import config
from src.config import GenerationDefaults
from src.engine import telemetry, version, watchdog
from src.engine.base import EngineClient, EngineError, GenerationRequest, GenerationResult

DEFAULT_LORA_DIR = config.MODELS_DIR / "loras"
LORA_EXTENSIONS = (".safetensors", ".gguf", ".pt", ".ckpt")
REF_IMAGE_ARGS = "resize_before_vae=false"
REF_IMAGE_PREPROCESS = "target=ref,mode=none"
NOISE_LEVEL_ARGS = "strength_as_noise_level=true,force_first_sigma=true"
_PREFIX_CACHE_OFF = {"off", "none", "false", "disabled"}


# ---------------------------------------------------------------------------
# Shared helpers (also used by sd_server_client)
# ---------------------------------------------------------------------------

def format_float(value: float) -> str:
    """Shortest exact decimal for a float flag value (1.0 -> '1.0', 0.9375 -> '0.9375')."""
    return repr(float(value))


def format_sigmas(sigmas: List[float]) -> str:
    return ",".join(format_float(s) for s in sigmas)


def validate_request(req: GenerationRequest) -> None:
    """Reject requests the engine would fail on (or silently misinterpret) before loading weights."""
    if not req.prompt.strip():
        raise ValueError("prompt must not be empty")
    if not req.output_path:
        raise ValueError("output_path must be set")
    for name, dim in (("width", req.width), ("height", req.height)):
        if dim <= 0 or dim % 16:
            raise ValueError(f"{name} must be a positive multiple of 16, got {dim}")
    if req.sigmas:
        if len(req.sigmas) < 2:
            raise ValueError("sigmas needs at least two values")
        if any(s < 0 for s in req.sigmas) or any(b > a for a, b in zip(req.sigmas, req.sigmas[1:])):
            raise ValueError(f"sigmas must be non-negative and non-increasing: {req.sigmas}")
    elif req.steps < 1:
        raise ValueError(f"steps must be >= 1, got {req.steps}")
    if req.mask_image and not req.init_image:
        raise ValueError("mask_image requires init_image")
    if req.init_image and not 0.0 < req.strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {req.strength}")
    for label, path in [("ref image", p) for p in req.ref_images] + [
        ("init image", req.init_image), ("mask image", req.mask_image)
    ]:
        if path and not Path(path).is_file():
            raise ValueError(f"{label} not found: {path}")
    for name in req.loras:
        if not name or any(c in name for c in "<>:"):
            raise ValueError(f"invalid LoRA name {name!r}")


def check_model_files(paths: Dict[str, Path]) -> None:
    missing = [f"{key}={path}" for key, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise EngineError("Model weights missing: " + ", ".join(missing) + ". Run ./download_models.sh.")


def resolve_lora(lora_dir: Path, name: str) -> Path:
    """LoRA file for `name` inside `lora_dir` (exact file name, or name + a known extension)."""
    base = Path(lora_dir) / name
    for candidate in [base] + [base.with_name(base.name + ext) for ext in LORA_EXTENSIONS]:
        if candidate.is_file():
            return candidate
    raise ValueError(f"LoRA {name!r} not found in {lora_dir}")


def prefix_cache_model_args(cache_type: str) -> Optional[str]:
    """--model-args value for the Qwen-Image 2.1 prefix cache ('' -> engine default)."""
    if not cache_type:
        return None
    if cache_type.lower() in _PREFIX_CACHE_OFF:
        return "qwen_image_2_1_prefix_cache=false"
    return f"qwen_image_2_1_prefix_cache_type={cache_type}"


def model_flags(defaults: GenerationDefaults, use_base_model: bool) -> List[str]:
    """Weights + engine-level flags shared by sd-cli and sd-server."""
    paths = config.model_paths(base=use_base_model)
    flags = [
        "--diffusion-model", str(paths["diffusion_model"]),
        "--llm", str(paths["llm"]),
        "--llm_vision", str(paths["llm_vision"]),
        "--vae", str(paths["vae"]),
    ]
    if defaults.flash_attention:
        flags.append("--diffusion-fa")
    cache_args = prefix_cache_model_args(defaults.prefix_cache_type)
    if cache_args:
        flags += ["--model-args", cache_args]
    flags += [str(f) for f in defaults.extra_model_flags]
    return flags


def prompt_with_loras(prompt: str, loras: Dict[str, float]) -> str:
    """sd-cli prompt syntax: '<lora:name:multiplier>' tags appended to the prompt."""
    tags = " ".join(f"<lora:{name}:{format_float(mult)}>" for name, mult in loras.items())
    return f"{prompt} {tags}" if tags else prompt


@dataclass
class ProcessOutput:
    returncode: Optional[int]
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False


def terminate_process_tree(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """SIGTERM then SIGKILL the process group of a child started with start_new_session=True."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


def run_process(argv: List[str], timeout_s: float) -> ProcessOutput:
    """Run argv in its own process group; on timeout, interrupt or our own death kill the whole group."""
    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", start_new_session=True,
    )
    guard = None
    try:
        guard = watchdog.arm(proc.pid)
        out, err = proc.communicate(timeout=timeout_s)
        return ProcessOutput(proc.returncode, out, err, time.monotonic() - t0)
    except subprocess.TimeoutExpired:
        terminate_process_tree(proc, grace_s=5.0)
        out, err = proc.communicate()
        return ProcessOutput(proc.returncode, out or "", err or "", time.monotonic() - t0, timed_out=True)
    except BaseException:
        terminate_process_tree(proc, grace_s=5.0)
        raise
    finally:
        watchdog.disarm(guard)


class RequestLogWriter:
    """Writes one log file per request into a directory, never overwriting within a session."""

    def __init__(self, log_dir: Optional[str]):
        self.log_dir = Path(log_dir) if log_dir else None
        self._used: Set[str] = set()

    def write(self, req: GenerationRequest, text: str) -> Optional[str]:
        if self.log_dir is None:
            return None
        stem = telemetry.safe_log_name(req.tag or Path(req.output_path).stem)
        name, n = stem, 1
        while name in self._used:
            n += 1
            name = f"{stem}_{n}"
        self._used.add(name)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{name}.log"
        path.write_text(text, encoding="utf-8")
        return str(path)


def pinned_sha_or_none() -> Optional[str]:
    return config.pinned_engine_sha() if config.UPSTREAM_SHA_FILE.exists() else None


def checked_engine_commit(binary: Path, allow_mismatch: bool = False) -> str:
    """Commit of an engine binary, verified against engine/UPSTREAM_SHA when the repo pins one."""
    pinned = pinned_sha_or_none()
    if pinned:
        return version.verify_engine(binary, pinned, allow_mismatch=allow_mismatch)
    return version.engine_commit(binary)


# ---------------------------------------------------------------------------
# sd-cli client
# ---------------------------------------------------------------------------

class SdCliClient(EngineClient):
    """Runs `sd-cli` once per request.

    allow_mismatch lets a binary built from a commit other than engine/UPSTREAM_SHA run
    (same as env PREVIS_ALLOW_ENGINE_MISMATCH=1).
    """

    def __init__(
        self,
        defaults: Optional[GenerationDefaults] = None,
        sd_cli: Path = config.SD_CLI,
        strict_fallbacks: bool = False,
        timeout_s: float = 1800,
        verbose_log_dir: Optional[str] = None,
        lora_dir: Optional[Path] = None,
        log_level: str = "debug",
        allow_mismatch: bool = False,
    ):
        self.defaults = defaults or GenerationDefaults()
        self.sd_cli = Path(sd_cli)
        self.strict_fallbacks = strict_fallbacks
        self.allow_mismatch = allow_mismatch
        self.timeout_s = timeout_s
        self.lora_dir = Path(lora_dir) if lora_dir else DEFAULT_LORA_DIR
        self.log_level = log_level
        self._logs = RequestLogWriter(verbose_log_dir)
        self._commit: Optional[str] = None

    def engine_commit(self) -> str:
        """Commit of the sd-cli binary, verified against engine/UPSTREAM_SHA once per client."""
        if self._commit is None:
            self._commit = checked_engine_commit(self.sd_cli, self.allow_mismatch)
        return self._commit

    def build_command(self, req: GenerationRequest) -> List[str]:
        """sd-cli argv for one request (no /usr/bin/time wrapper)."""
        cmd = [str(self.sd_cli)] + model_flags(self.defaults, req.use_base_model)
        cmd += [
            "-p", prompt_with_loras(req.prompt, req.loras),
            "-W", str(req.width),
            "-H", str(req.height),
            "-s", str(req.seed),
            "--cfg-scale", format_float(req.cfg_scale),
            "--sampling-method", req.sampler,
        ]
        if req.sigmas:
            # --steps is implied by the sigma count; passing it avoids the engine's mismatch warning.
            cmd += ["--sigmas", format_sigmas(req.sigmas), "--steps", str(len(req.sigmas) - 1)]
        else:
            cmd += ["--steps", str(req.steps), "--scheduler", req.scheduler]
        for ref in req.ref_images:
            cmd += ["-r", str(ref)]
        if req.ref_images:
            cmd += ["--ref-image-args", REF_IMAGE_ARGS, "--image-preprocess", REF_IMAGE_PREPROCESS]
        if req.init_image:
            cmd += ["-i", str(req.init_image), "--strength", format_float(req.strength)]
            if req.strength_as_noise_level:
                cmd += ["--extra-sample-args", NOISE_LEVEL_ARGS]
        if req.mask_image:
            cmd += ["--mask", str(req.mask_image)]
        if req.negative_prompt.strip():
            cmd += ["-n", req.negative_prompt.strip()]
        if req.loras:
            cmd += ["--lora-model-dir", str(self.lora_dir)]
        cmd += ["--log-level", self.log_level]
        cmd += ["-o", str(req.output_path)]
        return cmd

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # 1. Validate inputs before paying for a multi-GB model load
        validate_request(request)
        check_model_files(config.model_paths(base=request.use_base_model))
        for name in request.loras:
            resolve_lora(self.lora_dir, name)
        commit = self.engine_commit()

        # 2. Build the command; clear any stale output so success is unambiguous
        cmd = self.build_command(request)
        output = Path(request.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)

        # 3. Run under /usr/bin/time for peak RSS
        proc = run_process(telemetry.time_prefix() + cmd, self.timeout_s)
        log_text = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
        log_path = self._logs.write(request, log_text)
        where = f" (log: {log_path})" if log_path else ""

        # 4. Check the outcome
        if proc.timed_out:
            raise EngineError(f"sd-cli timed out after {self.timeout_s:.0f}s{where}\n{telemetry.tail(log_text)}")
        if proc.returncode != 0:
            raise EngineError(f"sd-cli exited with code {proc.returncode}{where}\n{telemetry.tail(log_text)}")
        if not output.is_file() or output.stat().st_size == 0:
            raise EngineError(f"sd-cli succeeded but wrote no image at {output}{where}\n{telemetry.tail(log_text)}")
        fallbacks = telemetry.find_fallbacks(log_text)
        if fallbacks and self.strict_fallbacks:
            raise EngineError(f"Engine fell back from a requested feature{where}: " + " | ".join(fallbacks))

        return GenerationResult(
            image_path=str(output),
            seconds=round(proc.seconds, 3),
            command=cmd,
            engine_commit=commit,
            peak_rss_bytes=telemetry.parse_time_l(proc.stderr),
            phase_seconds=telemetry.parse_phase_timings(log_text),
            fallback_warnings=fallbacks,
            log_path=log_path,
            extra={"backend": "sd-cli", "peak_footprint_bytes": telemetry.parse_peak_footprint(proc.stderr)},
        )
