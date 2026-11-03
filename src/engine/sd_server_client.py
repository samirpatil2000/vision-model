"""
sd-server backend: one resident stable-diffusion.cpp server, requests over loopback.

Weights stay loaded between requests. The server is bound to 127.0.0.1 on a free port and
driven through the native async API (POST /sdcpp/v1/img_gen, then poll
GET /sdcpp/v1/jobs/{id}); images travel as base64 in both directions. The server's log is
captured to a file so each job's log segment can be parsed for phase timings and fallbacks.
The DiT is fixed at server start, so a request for the other variant (turbo/base) restarts it.
"""
import atexit
import base64
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src import config
from src.config import GenerationDefaults
from src.engine import telemetry, watchdog
from src.engine.base import EngineClient, EngineError, GenerationRequest, GenerationResult
from src.engine.sd_cli_client import (
    DEFAULT_LORA_DIR, NOISE_LEVEL_ARGS, REF_IMAGE_ARGS, REF_IMAGE_PREPROCESS,
    RequestLogWriter, check_model_files, checked_engine_commit, model_flags, resolve_lora,
    terminate_process_tree, validate_request,
)

IMG_GEN_PATH = "/sdcpp/v1/img_gen"
JOBS_PATH = "/sdcpp/v1/jobs/"
READY_PATH = "/sdcpp/v1/capabilities"
_OUTPUT_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


class SdServerClient(EngineClient):
    """Keeps one `sd-server` resident and posts each request to its native img_gen API.

    GenerationResult.command is the server launch argv; the request body (image paths instead
    of base64) is in result.extra["request"]. peak_rss_bytes is the server's peak RSS sampled
    while the job ran. allow_mismatch lets a binary built from a commit other than
    engine/UPSTREAM_SHA run (same as env PREVIS_ALLOW_ENGINE_MISMATCH=1). A watchdog process
    stops the server if this Python process dies without calling close().
    """

    def __init__(
        self,
        defaults: Optional[GenerationDefaults] = None,
        sd_server: Path = config.SD_SERVER,
        strict_fallbacks: bool = False,
        timeout_s: float = 1800,
        verbose_log_dir: Optional[str] = None,
        lora_dir: Optional[Path] = None,
        log_level: str = "debug",
        host: str = "127.0.0.1",
        startup_timeout_s: float = 600,
        poll_interval_s: float = 0.5,
        allow_mismatch: bool = False,
    ):
        self.defaults = defaults or GenerationDefaults()
        self.sd_server = Path(sd_server)
        self.strict_fallbacks = strict_fallbacks
        self.allow_mismatch = allow_mismatch
        self.timeout_s = timeout_s
        self.verbose_log_dir = verbose_log_dir
        self.lora_dir = Path(lora_dir) if lora_dir else DEFAULT_LORA_DIR
        self.log_level = log_level
        self.host = host
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s
        self._logs = RequestLogWriter(verbose_log_dir)
        self._http = requests.Session()
        self._http.trust_env = False          # loopback only: never route through a proxy
        self._commit: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._watchdog: Optional[subprocess.Popen] = None
        self._server_cmd: List[str] = []
        self._use_base: Optional[bool] = None
        self._port: Optional[int] = None
        self._log_file = None
        self._log_path: Optional[Path] = None
        self._tmp_dir: Optional[str] = None
        atexit.register(self.close)

    # -- server lifecycle -------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self._port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def engine_commit(self) -> str:
        """Commit of the sd-server binary, verified against engine/UPSTREAM_SHA once per client."""
        if self._commit is None:
            self._commit = checked_engine_commit(self.sd_server, self.allow_mismatch)
        return self._commit

    def build_server_command(self, port: int, use_base_model: bool) -> List[str]:
        cmd = [str(self.sd_server), "--listen-ip", self.host, "--listen-port", str(port)]
        cmd += model_flags(self.defaults, use_base_model)
        if self.lora_dir.is_dir():
            cmd += ["--lora-model-dir", str(self.lora_dir)]
        cmd += ["--log-level", self.log_level]
        return cmd

    def start(self, use_base_model: bool = False) -> float:
        """Start the server (stopping any running one) and wait until it answers. Returns seconds."""
        self.close()
        t0 = time.monotonic()
        try:
            # 1. Log file: kept in verbose_log_dir when given, else a private temp dir
            if self.verbose_log_dir:
                log_dir = Path(self.verbose_log_dir)
            else:
                self._tmp_dir = tempfile.mkdtemp(prefix="sd-server-")
                log_dir = Path(self._tmp_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._port = free_port(self.host)
            self._log_path = log_dir / f"sd-server_{self._port}.log"
            self._log_file = open(self._log_path, "ab")

            # 2. Launch in its own process group so close() can kill everything it spawned; the
            #    group is outside the terminal's job, so a watchdog stops it if we die uncleanly
            self._server_cmd = self.build_server_command(self._port, use_base_model)
            self._proc = subprocess.Popen(
                self._server_cmd, stdin=subprocess.DEVNULL, stdout=self._log_file,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            self._watchdog = watchdog.arm(self._proc.pid, self._tmp_dir)
            self._use_base = use_base_model

            # 3. Wait for the HTTP API
            self._wait_ready(t0 + self.startup_timeout_s)
        except BaseException:
            self.close()
            raise
        return time.monotonic() - t0

    def _wait_ready(self, deadline: float) -> None:
        url = self.base_url + READY_PATH
        while True:
            if self._proc.poll() is not None:
                raise EngineError(
                    f"sd-server exited during startup with code {self._proc.returncode}\n{self._log_tail()}"
                )
            try:
                if self._http.get(url, timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise EngineError(
                    f"sd-server not ready after {self.startup_timeout_s:.0f}s\n{self._log_tail()}"
                )
            time.sleep(self.poll_interval_s)

    def close(self) -> None:
        """Terminate the server (whole process group) and release its log file."""
        if self._proc is not None:
            terminate_process_tree(self._proc, grace_s=10.0)
            self._proc = None
        watchdog.disarm(self._watchdog)          # after the kill: it guards the whole shutdown
        self._watchdog = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
            self._log_path = None
        self._use_base = None

    # -- requests ----------------------------------------------------------

    def build_request_body(self, req: GenerationRequest, encode_images: bool = True) -> Dict[str, Any]:
        """Native /sdcpp/v1/img_gen body. With encode_images=False image fields hold paths (for logs)."""
        image = encode_image if encode_images else str
        sample_params: Dict[str, Any] = {
            "sample_method": req.sampler,
            "guidance": {"txt_cfg": float(req.cfg_scale)},
        }
        if req.sigmas:
            sample_params["custom_sigmas"] = [float(s) for s in req.sigmas]
            sample_params["sample_steps"] = len(req.sigmas) - 1
        else:
            sample_params["sample_steps"] = int(req.steps)
            sample_params["scheduler"] = req.scheduler
        if req.init_image and req.strength_as_noise_level:
            sample_params["extra_sample_args"] = NOISE_LEVEL_ARGS

        body: Dict[str, Any] = {
            "prompt": req.prompt,               # server APIs ignore <lora:...> tags by design
            "negative_prompt": req.negative_prompt.strip(),
            "width": int(req.width),
            "height": int(req.height),
            "seed": int(req.seed),
            "batch_count": 1,
            "strength": float(req.strength),
            "sample_params": sample_params,
            "ref_images": [image(p) for p in req.ref_images],
            "init_image": image(req.init_image) if req.init_image else None,
            "mask_image": image(req.mask_image) if req.mask_image else None,
            "lora": [
                {"path": resolve_lora(self.lora_dir, name).relative_to(self.lora_dir).as_posix(),
                 "multiplier": float(mult)}
                for name, mult in req.loras.items()
            ],
            "preview": "none",
            "output_format": _OUTPUT_FORMATS.get(Path(req.output_path).suffix.lower(), "png"),
        }
        if req.ref_images:
            body["ref_image_args"] = REF_IMAGE_ARGS
            body["image_preprocess"] = REF_IMAGE_PREPROCESS
        return body

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # 1. Validate inputs and make sure the right model variant is resident
        validate_request(request)
        check_model_files(config.model_paths(base=request.use_base_model))
        commit = self.engine_commit()
        body = self.build_request_body(request)          # also resolves LoRA files (ValueError)
        startup_s = 0.0
        needs_lora_dir = bool(request.loras) and "--lora-model-dir" not in self._server_cmd
        if not self.running or self._use_base != request.use_base_model or needs_lora_dir:
            startup_s = self.start(request.use_base_model)
        output = Path(request.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)

        # 2. Submit and wait; a dead or hung server is torn down so the next call restarts it
        log_offset = self._log_size()
        t0 = time.monotonic()
        try:
            job_id = self._submit(body)
            job, peak_rss = self._wait_for_job(job_id, t0 + self.timeout_s)
        except EngineError:
            raise
        except requests.RequestException as exc:
            tail = self._log_tail()
            self.close()
            raise EngineError(f"sd-server request failed: {exc}\n{tail}") from exc
        except BaseException:
            self.close()
            raise
        seconds = time.monotonic() - t0

        # 3. Per-request log segment and telemetry
        log_text = self._read_log(log_offset)
        log_path = self._logs.write(request, log_text)
        where = f" (log: {log_path})" if log_path else ""
        if job.get("status") != "completed":
            error = job.get("error") or {}
            raise EngineError(
                f"sd-server job {job_id} {job.get('status')}: {error.get('message', error)}{where}\n"
                f"{telemetry.tail(log_text)}"
            )

        # 4. Decode the image
        images = (job.get("result") or {}).get("images") or []
        if not images or not images[0].get("b64_json"):
            raise EngineError(f"sd-server job {job_id} completed without an image{where}")
        data = base64.b64decode(images[0]["b64_json"])
        tmp = output.with_name(output.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(output)

        fallbacks = telemetry.find_fallbacks(log_text)
        if fallbacks and self.strict_fallbacks:
            raise EngineError(f"Engine fell back from a requested feature{where}: " + " | ".join(fallbacks))

        return GenerationResult(
            image_path=str(output),
            seconds=round(seconds, 3),
            command=list(self._server_cmd),
            engine_commit=commit,
            peak_rss_bytes=peak_rss,
            phase_seconds=telemetry.parse_phase_timings(log_text),
            fallback_warnings=fallbacks,
            log_path=log_path,
            extra={
                "backend": "sd-server",
                "job_id": job_id,
                "server_start_seconds": round(startup_s, 3),
                "request": self.build_request_body(request, encode_images=False),
            },
        )

    def _submit(self, body: Dict[str, Any]) -> str:
        response = self._http.post(self.base_url + IMG_GEN_PATH, json=body, timeout=60)
        if response.status_code != 202:
            raise EngineError(f"sd-server rejected img_gen ({response.status_code}): {response.text[:500]}")
        job_id = response.json().get("id")
        if not job_id:
            raise EngineError(f"sd-server returned no job id: {response.text[:500]}")
        return job_id

    def _wait_for_job(self, job_id: str, deadline: float) -> Tuple[Dict[str, Any], Optional[int]]:
        """Poll until the job is terminal. Returns (job json, peak server RSS seen while polling)."""
        peak_rss: Optional[int] = None
        url = self.base_url + JOBS_PATH + job_id
        while True:
            if not self.running:
                code = self._proc.returncode if self._proc is not None else None
                tail = self._log_tail()
                self.close()
                raise EngineError(f"sd-server exited (code {code}) while running job {job_id}\n{tail}")
            rss = telemetry.process_rss_bytes(self._proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss or 0, rss)
            response = self._http.get(url, timeout=30)
            if response.status_code != 200:
                raise EngineError(f"sd-server job {job_id} poll failed ({response.status_code}): {response.text[:500]}")
            job = response.json()
            if job.get("status") in ("completed", "failed", "cancelled"):
                return job, peak_rss
            if time.monotonic() > deadline:
                tail = self._log_tail()
                self.close()
                raise EngineError(f"sd-server job {job_id} timed out after {self.timeout_s:.0f}s\n{tail}")
            time.sleep(self.poll_interval_s)

    # -- log helpers -------------------------------------------------------

    def _log_size(self) -> int:
        if self._log_file is not None:
            self._log_file.flush()
        return self._log_path.stat().st_size if self._log_path and self._log_path.exists() else 0

    def _read_log(self, offset: int) -> str:
        if not self._log_path or not self._log_path.exists():
            return ""
        with open(self._log_path, "rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")

    def _log_tail(self) -> str:
        return telemetry.tail(self._read_log(0)) if self._log_path else ""
