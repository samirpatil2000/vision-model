import base64
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List
from unittest import mock

from PIL import Image

from src import config
from src.config import GenerationDefaults
from src.engine import SdCliClient, SdServerClient, make_engine, telemetry, version, watchdog
from src.engine.base import EngineError, GenerationRequest
from src.engine.sd_cli_client import (
    ProcessOutput, format_sigmas, prompt_with_loras, run_process, validate_request,
)

ROOT = Path(__file__).resolve().parents[1]
VIGGLE = list(config.VIGGLE_SIGMAS)
PINNED = "2988060a4338a5dcd94721c964a8e87003c1ac8b"
VERSION_OUT = "stable-diffusion.cpp version master-955-b7a74e6-2-g2988060, commit 2988060\n"

ENGINE_LOG = "\n".join([
    "[I] Using flash attention in the diffusion model --- diffusion_engine.cpp:1204",
    "[I] loading tensors completed, taking 7.11s (read: 6.99s, memcpy: 0.00s) --- model_loader.cpp:1390",
    "[V] computing condition graph completed, taking 8243 ms --- conditioner.hpp:3210",
    "[I] get_learned_condition completed, taking 8.24s --- image.cpp:529",
    "[I] loading tensors completed, taking 8.25s (read: 8.10s) --- model_loader.cpp:1390",
    "  |==========>      | 5/6 - 2.33s/it\x1b[K[D] qwen_image_2_1 executing segment 1/1: graph",
    "  |=================| 6/6 - 2.33s/it\x1b[K\r[I] sampling completed, taking 23.30s --- image.cpp:899",
    "[I] decode_first_stage completed, taking 3.27s --- image.cpp:624",
])
FA_FALLBACK = ("[D] Qwen Image 2.1: Flash Attention unavailable; using F32 prefix caching for this "
               "sampling run --- qwen_image_2_1.hpp:569")
CACHE_FALLBACK = ("[W] Qwen Image 2.1: insufficient memory for prefix caching; retrying without it for "
                  "this sampling run --- qwen_image_2_1.hpp:577")
MAC_TIME = """       49.07 real         1.50 user         7.60 sys
         11096080384  maximum resident set size
                   0  average shared memory size
         11277689568  peak memory footprint
"""
GNU_TIME = """\tCommand being timed: "sd-cli"
\tMaximum resident set size (kbytes): 2048
\tExit status: 0
"""


def png_bytes(color=(10, 20, 30), size=(32, 16)) -> bytes:
    path = Path(tempfile.mkdtemp()) / "x.png"
    Image.new("RGB", size, color).save(path)
    return path.read_bytes()


def wait_gone(pid: int, timeout_s: float = 10.0) -> bool:
    """True once `pid` no longer exists (for processes that are not our own children)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def kill_if_alive(pid: int) -> None:
    for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
        try:
            kill()
        except (ProcessLookupError, PermissionError):
            pass


def flag_value(cmd: List[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def flag_values(cmd: List[str], flag: str) -> List[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag]


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def image(self, name: str, color=(10, 20, 30)) -> str:
        path = self.tmp / name
        Image.new("RGB", (32, 16), color).save(path)
        return str(path)

    def request(self, **kw) -> GenerationRequest:
        base = dict(prompt="a glass on a table", output_path=str(self.tmp / "out" / "output.png"),
                    width=512, height=288, seed=42)
        base.update(kw)
        return GenerationRequest(**base)


class TestTelemetry(unittest.TestCase):
    def test_parse_time_l_macos_bytes(self):
        self.assertEqual(telemetry.parse_time_l("sd-cli stderr\n" + MAC_TIME), 11096080384)
        self.assertEqual(telemetry.parse_peak_footprint(MAC_TIME), 11277689568)

    def test_parse_time_gnu_kbytes(self):
        self.assertEqual(telemetry.parse_time_l(GNU_TIME), 2048 * 1024)
        self.assertIsNone(telemetry.parse_peak_footprint(GNU_TIME))

    def test_parse_time_absent(self):
        self.assertIsNone(telemetry.parse_time_l("no timing here"))

    def test_phase_timings_units_sums_and_progress_bars(self):
        phases = telemetry.parse_phase_timings(ENGINE_LOG)
        self.assertAlmostEqual(phases["loading_tensors"], 15.36)        # two models summed
        self.assertAlmostEqual(phases["computing_condition_graph"], 8.243)  # ms -> s
        self.assertAlmostEqual(phases["get_learned_condition"], 8.24)
        self.assertAlmostEqual(phases["sampling"], 23.30)                # after a progress bar
        self.assertAlmostEqual(phases["decode_first_stage"], 3.27)
        self.assertEqual(len(phases), 5)

    def test_phase_timing_after_bar_on_same_line(self):
        line = "  |=====| 6/6 - 2.33s/it\x1b[K[I] sampling completed, taking 23.30s --- image.cpp:899"
        self.assertEqual(telemetry.parse_phase_timings(line), {"sampling": 23.3})

    def test_find_fallbacks(self):
        log = "\n".join([ENGINE_LOG, FA_FALLBACK, CACHE_FALLBACK, FA_FALLBACK])
        found = telemetry.find_fallbacks(log)
        self.assertEqual(len(found), 2)                                  # de-duplicated, ordered
        self.assertIn("Flash Attention unavailable", found[0])
        self.assertIn("insufficient memory for prefix caching", found[1])

    def test_no_fallbacks_in_clean_log(self):
        self.assertEqual(telemetry.find_fallbacks(ENGINE_LOG), [])

    def test_ignored_cache_type_is_a_fallback(self):
        line = "[W] ignoring unknown Qwen Image 2.1 cache type 'q9_9' --- qwen_image_2_1.hpp:420"
        self.assertEqual(telemetry.find_fallbacks(line), [line])

    def test_safe_log_name_and_tail(self):
        self.assertEqual(telemetry.safe_log_name("kf a/1:attempt 2"), "kf_a_1_attempt_2")
        self.assertEqual(telemetry.safe_log_name("///"), "generation")
        self.assertEqual(telemetry.tail("a\nb\nc", lines=2), "b\nc")

    def test_time_prefix_matches_platform(self):
        prefix = telemetry.time_prefix()
        if sys.platform == "darwin" and Path("/usr/bin/time").exists():
            self.assertEqual(prefix, ["/usr/bin/time", "-l"])
        self.assertIsInstance(prefix, list)


class TestVersion(unittest.TestCase):
    def _run(self, stdout="", stderr="", rc=0):
        return mock.patch("src.engine.version.subprocess.run",
                          return_value=subprocess.CompletedProcess([], rc, stdout, stderr))

    def test_engine_commit_parses_version_line(self):
        with self._run(VERSION_OUT) as run:
            self.assertEqual(version.engine_commit(Path("/x/sd-cli")), "2988060")
            self.assertEqual(run.call_args.args[0], ["/x/sd-cli", "--version"])
            self.assertIn("timeout", run.call_args.kwargs)

    def test_engine_commit_missing_binary(self):
        with self.assertRaises(EngineError):
            version.engine_commit(Path("/nonexistent/dir/sd-cli"))

    def test_engine_commit_unparseable(self):
        with self._run("garbage"), self.assertRaises(EngineError):
            version.engine_commit(Path("/x/sd-cli"))

    def test_commits_match_prefix(self):
        self.assertTrue(version.commits_match("2988060", PINNED))
        self.assertTrue(version.commits_match(PINNED.upper(), "2988060a43"))
        self.assertFalse(version.commits_match("1d0a58a", PINNED))
        self.assertFalse(version.commits_match("2988", PINNED))           # too short to trust

    def test_verify_engine_ok(self):
        with self._run(VERSION_OUT):
            self.assertEqual(version.verify_engine(Path("/x/sd-cli"), PINNED), "2988060")

    def test_verify_engine_mismatch_raises(self):
        out = "stable-diffusion.cpp version master-962, commit 1d0a58a\n"
        with self._run(out), mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            with self.assertRaises(EngineError) as ctx:
                version.verify_engine(Path("/x/sd-cli"), PINNED)
            self.assertIn("PIN_ENGINE=1", str(ctx.exception))

    def test_verify_engine_mismatch_allowed(self):
        out = "stable-diffusion.cpp version master-962, commit 1d0a58a\n"
        with self._run(out):
            self.assertEqual(version.verify_engine(Path("/x/sd-cli"), PINNED, allow_mismatch=True), "1d0a58a")
            with mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": "1"}):
                self.assertEqual(version.verify_engine(Path("/x/sd-cli"), PINNED), "1d0a58a")

    def test_engine_commit_unknown_build(self):
        # The engine's CMake fallback when the sources are not a git checkout.
        with self._run("stable-diffusion.cpp version unknown, commit unknown\n"):
            self.assertEqual(version.engine_commit(Path("/x/sd-cli")), version.UNKNOWN_COMMIT)

    def test_verify_engine_unknown_commit_runs_when_allowed(self):
        out = "stable-diffusion.cpp version unknown, commit unknown\n"
        with self._run(out), mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            with self.assertRaises(EngineError) as ctx:
                version.verify_engine(Path("/x/sd-cli"), PINNED)
            self.assertIn("no build commit", str(ctx.exception))
            self.assertEqual(version.verify_engine(Path("/x/sd-cli"), PINNED, allow_mismatch=True), "unknown")
            with mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": "1"}):
                self.assertEqual(version.verify_engine(Path("/x/sd-cli"), PINNED), "unknown")

    def test_verify_engine_unparseable_version_runs_when_allowed(self):
        with self._run("usage: sd-cli [options]", rc=1), \
                mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            with self.assertRaises(version.UnknownEngineVersion) as ctx:
                version.verify_engine(Path("/x/sd-cli"), PINNED)
            self.assertIn("PREVIS_ALLOW_ENGINE_MISMATCH=1", str(ctx.exception))
            self.assertEqual(version.verify_engine(Path("/x/sd-cli"), PINNED, allow_mismatch=True), "unknown")

    def test_verify_engine_missing_binary_raises_even_when_allowed(self):
        with self.assertRaises(EngineError) as ctx:
            version.verify_engine(Path("/nonexistent/dir/sd-cli"), PINNED, allow_mismatch=True)
        self.assertIn("not found", str(ctx.exception))

    def test_verify_engine_defaults_to_repo_pin(self):
        with self._run(VERSION_OUT):
            self.assertEqual(version.verify_engine(Path("/x/sd-cli")), "2988060")

    def test_read_build_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(version.read_build_info(Path(tmp)))
            (Path(tmp) / "build").mkdir()
            (Path(tmp) / "build" / "build-info.json").write_text(json.dumps({"commit": PINNED}))
            self.assertEqual(version.read_build_info(Path(tmp))["commit"], PINNED)


class TestValidateRequest(TempDirCase):
    def test_valid_request_passes(self):
        validate_request(self.request(sigmas=VIGGLE, ref_images=[self.image("r.png")],
                                      init_image=self.image("i.png"), mask_image=self.image("m.png")))

    def test_rejections(self):
        cases = [
            dict(prompt="  "),
            dict(width=500),
            dict(height=0),
            dict(sigmas=[1.0]),
            dict(sigmas=[0.5, 0.9, 0.0]),                      # increasing
            dict(sigmas=None, steps=0),
            dict(mask_image=self.image("m.png")),              # mask without init
            dict(init_image=self.image("i.png"), strength=0.0),
            dict(ref_images=[str(self.tmp / "missing.png")]),
            dict(loras={"bad:name": 1.0}),
        ]
        for kw in cases:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                validate_request(self.request(**kw))


class TestCliCommand(TempDirCase):
    def setUp(self):
        super().setUp()
        self.defaults = GenerationDefaults(flash_attention=True, prefix_cache_type="q8_0", extra_model_flags=[])
        self.client = SdCliClient(defaults=self.defaults, sd_cli=Path("/opt/sd-cli"), lora_dir=self.tmp / "loras")

    def test_t2i_with_viggle_sigmas(self):
        cmd = self.client.build_command(self.request(sigmas=VIGGLE))
        self.assertEqual(cmd[0], "/opt/sd-cli")
        self.assertEqual(flag_value(cmd, "--diffusion-model"), str(config.TURBO_MODEL))
        self.assertEqual(flag_value(cmd, "--llm"), str(config.LLM_MODEL))
        self.assertEqual(flag_value(cmd, "--llm_vision"), str(config.LLM_VISION))
        self.assertEqual(flag_value(cmd, "--vae"), str(config.VAE_MODEL))
        self.assertEqual(flag_value(cmd, "-p"), "a glass on a table")
        self.assertEqual((flag_value(cmd, "-W"), flag_value(cmd, "-H"), flag_value(cmd, "-s")), ("512", "288", "42"))
        self.assertEqual(flag_value(cmd, "--cfg-scale"), "1.0")
        self.assertEqual(flag_value(cmd, "--sampling-method"), "euler")
        self.assertEqual(flag_value(cmd, "--sigmas"), "1.0,0.9375,0.875,0.75,0.5,0.25,0.0")
        self.assertEqual(flag_value(cmd, "--steps"), "6")          # implied by the 7 sigmas
        self.assertNotIn("--scheduler", cmd)
        for absent in ("-r", "-i", "--strength", "--extra-sample-args", "--mask", "-n",
                       "--ref-image-args", "--lora-model-dir"):
            self.assertNotIn(absent, cmd)
        self.assertEqual(cmd[-2:], ["-o", str(self.tmp / "out" / "output.png")])

    def test_steps_and_scheduler_without_sigmas(self):
        cmd = self.client.build_command(self.request(steps=4, scheduler="simple", sampler="euler_a", cfg_scale=2.5))
        self.assertEqual(flag_value(cmd, "--steps"), "4")
        self.assertEqual(flag_value(cmd, "--scheduler"), "simple")
        self.assertEqual(flag_value(cmd, "--sampling-method"), "euler_a")
        self.assertEqual(flag_value(cmd, "--cfg-scale"), "2.5")
        self.assertNotIn("--sigmas", cmd)

    def test_refs_keep_order_and_disable_resizing(self):
        refs = [self.image("cast.png"), self.image("continuity.png"), self.image("third.png")]
        cmd = self.client.build_command(self.request(ref_images=refs))
        self.assertEqual(flag_values(cmd, "-r"), refs)
        self.assertEqual(flag_value(cmd, "--ref-image-args"), "resize_before_vae=false")
        self.assertEqual(flag_value(cmd, "--image-preprocess"), "target=ref,mode=none")

    def test_init_strength_and_noise_level_args(self):
        init = self.image("blocking.png")
        cmd = self.client.build_command(self.request(init_image=init, strength=0.6))
        self.assertEqual(flag_value(cmd, "-i"), init)
        self.assertEqual(flag_value(cmd, "--strength"), "0.6")
        self.assertEqual(flag_value(cmd, "--extra-sample-args"), "strength_as_noise_level=true,force_first_sigma=true")

    def test_init_without_noise_level(self):
        cmd = self.client.build_command(self.request(init_image=self.image("i.png"), strength_as_noise_level=False))
        self.assertIn("--strength", cmd)
        self.assertNotIn("--extra-sample-args", cmd)

    def test_mask(self):
        mask = self.image("mask.png")
        cmd = self.client.build_command(self.request(init_image=self.image("i.png"), mask_image=mask))
        self.assertEqual(flag_value(cmd, "--mask"), mask)

    def test_negative_prompt_only_when_non_empty(self):
        self.assertEqual(flag_value(self.client.build_command(self.request(negative_prompt=" blurry ")), "-n"), "blurry")
        self.assertNotIn("-n", self.client.build_command(self.request(negative_prompt="   ")))

    def test_flash_attention_and_prefix_cache_flags(self):
        cmd = self.client.build_command(self.request())
        self.assertIn("--diffusion-fa", cmd)
        self.assertNotIn("--fa", cmd)
        self.assertEqual(flag_value(cmd, "--model-args"), "qwen_image_2_1_prefix_cache_type=q8_0")

        self.client.defaults = GenerationDefaults(flash_attention=False, prefix_cache_type="", extra_model_flags=[])
        cmd = self.client.build_command(self.request())
        self.assertNotIn("--diffusion-fa", cmd)
        self.assertNotIn("--model-args", cmd)

        self.client.defaults = GenerationDefaults(prefix_cache_type="off", extra_model_flags=["--vae-tiling"])
        cmd = self.client.build_command(self.request())
        self.assertEqual(flag_value(cmd, "--model-args"), "qwen_image_2_1_prefix_cache=false")
        self.assertIn("--vae-tiling", cmd)

    def test_base_model_variant(self):
        cmd = self.client.build_command(self.request(use_base_model=True))
        self.assertEqual(flag_value(cmd, "--diffusion-model"), str(config.BASE_MODEL))

    def test_lora_prompt_syntax(self):
        cmd = self.client.build_command(self.request(loras={"film_grain": 0.8, "style": 1.0}))
        self.assertEqual(flag_value(cmd, "-p"), "a glass on a table <lora:film_grain:0.8> <lora:style:1.0>")
        self.assertEqual(flag_value(cmd, "--lora-model-dir"), str(self.tmp / "loras"))
        self.assertEqual(prompt_with_loras("x", {}), "x")

    def test_log_level(self):
        self.assertEqual(flag_value(self.client.build_command(self.request()), "--log-level"), "debug")
        client = SdCliClient(sd_cli=Path("/opt/sd-cli"), log_level="verbose")
        self.assertEqual(flag_value(client.build_command(self.request()), "--log-level"), "verbose")

    def test_format_sigmas(self):
        self.assertEqual(format_sigmas([1, 0.5, 0]), "1.0,0.5,0.0")


class TestCliGenerate(TempDirCase):
    def setUp(self):
        super().setUp()
        self.logs = self.tmp / "logs"
        self.client = SdCliClient(sd_cli=Path("/opt/sd-cli"), verbose_log_dir=str(self.logs))
        for target, kw in [
            ("src.engine.sd_cli_client.check_model_files", {}),
            ("src.engine.version.engine_commit", {"return_value": "2988060"}),
        ]:
            patcher = mock.patch(target, **kw)
            self.addCleanup(patcher.stop)
            setattr(self, target.rsplit(".", 1)[1], patcher.start())

    def _run_with(self, returncode=0, stdout=ENGINE_LOG, stderr=MAC_TIME, write=True, timed_out=False):
        calls = []

        def fake_run(argv, timeout_s):
            calls.append((argv, timeout_s))
            if write:
                out = Path(argv[argv.index("-o") + 1])
                out.write_bytes(png_bytes())
            return ProcessOutput(returncode, stdout, stderr, 12.5, timed_out)

        patcher = mock.patch("src.engine.sd_cli_client.run_process", side_effect=fake_run)
        self.addCleanup(patcher.stop)
        patcher.start()
        return calls

    def test_success_records_telemetry_and_log(self):
        calls = self._run_with()
        result = self.client.generate(self.request(sigmas=VIGGLE, tag="kf_a_1/attempt_1"))
        argv, timeout = calls[0]
        self.assertEqual(argv[:len(argv) - len(result.command)], telemetry.time_prefix())
        self.assertEqual(argv[len(argv) - len(result.command):], result.command)
        self.assertEqual(timeout, 1800)
        self.assertTrue(Path(result.image_path).is_file())
        self.assertEqual(result.engine_commit, "2988060")
        self.assertEqual(result.peak_rss_bytes, 11096080384)
        self.assertEqual(result.extra["peak_footprint_bytes"], 11277689568)
        self.assertAlmostEqual(result.phase_seconds["sampling"], 23.3)
        self.assertEqual(result.fallback_warnings, [])
        self.assertEqual(result.seconds, 12.5)
        self.assertEqual(Path(result.log_path), self.logs / "kf_a_1_attempt_1.log")
        self.assertIn("sampling completed", Path(result.log_path).read_text())

    def test_log_names_do_not_collide(self):
        self._run_with()
        first = self.client.generate(self.request())
        second = self.client.generate(self.request())
        self.assertEqual(Path(first.log_path).name, "output.log")
        self.assertEqual(Path(second.log_path).name, "output_2.log")

    def test_fallbacks_reported(self):
        self._run_with(stdout=ENGINE_LOG + "\n" + FA_FALLBACK)
        result = self.client.generate(self.request())
        self.assertEqual(len(result.fallback_warnings), 1)

    def test_strict_fallbacks_raise(self):
        self._run_with(stdout=ENGINE_LOG + "\n" + CACHE_FALLBACK)
        self.client.strict_fallbacks = True
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("insufficient memory for prefix caching", str(ctx.exception))

    def test_nonzero_exit_raises_with_log_tail(self):
        self._run_with(returncode=1, stderr="[E] new_sd_ctx_t failed\n", write=False)
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("code 1", str(ctx.exception))
        self.assertIn("new_sd_ctx_t failed", str(ctx.exception))

    def test_timeout_raises(self):
        self._run_with(returncode=-9, timed_out=True, write=False)
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("timed out", str(ctx.exception))

    def test_stale_output_is_not_mistaken_for_success(self):
        self._run_with(write=False)
        out = self.tmp / "out" / "output.png"
        out.parent.mkdir(parents=True)
        out.write_bytes(png_bytes())
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("wrote no image", str(ctx.exception))

    def test_version_mismatch_blocks_generation(self):
        calls = self._run_with()
        self.engine_commit.return_value = "1d0a58a"
        with mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            with self.assertRaises(EngineError):
                self.client.generate(self.request())
        self.assertEqual(calls, [])

    def test_version_mismatch_allowed_by_env(self):
        self._run_with()
        self.engine_commit.return_value = "1d0a58a"
        with mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": "1"}):
            self.assertEqual(self.client.generate(self.request()).engine_commit, "1d0a58a")

    def test_version_mismatch_allowed_by_constructor(self):
        self._run_with()
        self.engine_commit.return_value = "1d0a58a"
        client = SdCliClient(sd_cli=Path("/opt/sd-cli"), allow_mismatch=True)
        with mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            self.assertEqual(client.generate(self.request()).engine_commit, "1d0a58a")

    def test_unknown_commit_needs_allow_mismatch(self):
        self._run_with()
        self.engine_commit.return_value = version.UNKNOWN_COMMIT
        with mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            with self.assertRaises(EngineError):
                self.client.generate(self.request())
            client = SdCliClient(sd_cli=Path("/opt/sd-cli"), allow_mismatch=True)
            self.assertEqual(client.generate(self.request()).engine_commit, "unknown")

    def test_commit_checked_once_per_client(self):
        self._run_with()
        self.client.generate(self.request())
        self.client.generate(self.request())
        self.assertEqual(self.engine_commit.call_count, 1)

    def test_missing_lora_rejected_before_running(self):
        calls = self._run_with()
        with self.assertRaises(ValueError):
            self.client.generate(self.request(loras={"nope": 1.0}))
        self.assertEqual(calls, [])

    def test_existing_lora_accepted(self):
        lora_dir = self.tmp / "loras"
        lora_dir.mkdir()
        (lora_dir / "style.safetensors").write_bytes(b"x")
        self.client.lora_dir = lora_dir
        calls = self._run_with()
        self.client.generate(self.request(loras={"style": 0.5}))
        self.assertIn("<lora:style:0.5>", flag_value(calls[0][0], "-p"))


class TestRunProcess(unittest.TestCase):
    def test_success(self):
        out = run_process([sys.executable, "-c", "import sys; print('hi'); print('err', file=sys.stderr)"], 30)
        self.assertEqual((out.returncode, out.stdout.strip(), out.stderr.strip(), out.timed_out), (0, "hi", "err", False))

    def test_watchdog_guards_the_child(self):
        for argv, timeout in [([sys.executable, "-c", "print('ok')"], 30),
                              ([sys.executable, "-c", "import time; time.sleep(30)"], 0.5)]:
            with self.subTest(argv=argv), mock.patch("src.engine.watchdog.arm", return_value="guard") as arm, \
                    mock.patch("src.engine.watchdog.disarm") as disarm:
                out = run_process(argv, timeout)
                arm.assert_called_once()
                self.assertIsInstance(arm.call_args.args[0], int)          # the child's pid (= its pgid)
                disarm.assert_called_once_with("guard")
                self.assertEqual(out.timed_out, timeout < 1)

    def test_timeout_kills_whole_process_group(self):
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        t0 = time.monotonic()
        out = run_process([sys.executable, "-c", script], timeout_s=1.0)
        self.assertTrue(out.timed_out)
        self.assertLess(time.monotonic() - t0, 15)
        grandchild = int(out.stdout.split()[0])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("grandchild process survived the timeout kill")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeProc:
    instances: List["FakeProc"] = []

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4_000_000 + len(FakeProc.instances)
        self.returncode = None
        FakeProc.instances.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class FakeSession:
    """Stands in for requests.Session: ready check, img_gen submit, job polling."""

    def __init__(self, image_b64: str, final_status="completed", submit_status=202, log_lines=""):
        self.image_b64 = image_b64
        self.final_status = final_status
        self.submit_status = submit_status
        self.log_lines = log_lines
        self.posts = []
        self.polls = 0
        self.trust_env = True
        self.client = None

    def get(self, url, timeout=None):
        if url.endswith("/sdcpp/v1/capabilities"):
            return FakeResponse(200, {"current_mode": "img_gen"})
        self.polls += 1
        if self.polls == 1:
            return FakeResponse(200, {"id": "job_1", "status": "generating"})
        if self.final_status == "completed":
            return FakeResponse(200, {"id": "job_1", "status": "completed",
                                      "result": {"output_format": "png",
                                                 "images": [{"index": 0, "b64_json": self.image_b64}]}})
        return FakeResponse(200, {"id": "job_1", "status": self.final_status,
                                  "error": {"code": "generation_failed", "message": "boom"}})

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if self.log_lines and self.client is not None:
            with open(self.client._log_path, "a") as handle:   # what the server would log
                handle.write(self.log_lines)
        if self.submit_status != 202:
            return FakeResponse(self.submit_status, {"error": "invalid generation parameters"})
        return FakeResponse(202, {"id": "job_1", "status": "queued", "poll_url": "/sdcpp/v1/jobs/job_1"})


class TestServerClient(TempDirCase):
    def setUp(self):
        super().setUp()
        FakeProc.instances = []
        self.png = png_bytes((200, 10, 10))
        self.session = FakeSession(base64.b64encode(self.png).decode(), log_lines=ENGINE_LOG + "\n")
        self.terminated = []

        def fake_terminate(proc, grace_s=10.0):
            self.terminated.append(proc)
            proc.returncode = -15

        for target, kw in [
            ("src.engine.sd_server_client.requests.Session", {"return_value": self.session}),
            ("src.engine.sd_server_client.subprocess.Popen", {"side_effect": FakeProc}),
            ("src.engine.sd_server_client.terminate_process_tree", {"side_effect": fake_terminate}),
            ("src.engine.sd_server_client.check_model_files", {}),
            ("src.engine.telemetry.process_rss_bytes", {"return_value": 5_000_000_000}),
            ("src.engine.version.engine_commit", {"return_value": "2988060"}),
        ]:
            patcher = mock.patch(target, **kw)
            self.addCleanup(patcher.stop)
            patcher.start()
        arm = mock.patch("src.engine.watchdog.arm", return_value=mock.sentinel.guard)
        disarm = mock.patch("src.engine.watchdog.disarm")
        self.watchdog_arm, self.watchdog_disarm = arm.start(), disarm.start()
        self.addCleanup(arm.stop)
        self.addCleanup(disarm.stop)
        self.client = SdServerClient(sd_server=Path("/opt/sd-server"), lora_dir=self.tmp / "loras",
                                     verbose_log_dir=str(self.tmp / "logs"), poll_interval_s=0)
        self.session.client = self.client
        self.addCleanup(self.client.close)

    def test_server_command(self):
        cmd = self.client.build_server_command(5555, use_base_model=False)
        self.assertEqual(cmd[:5], ["/opt/sd-server", "--listen-ip", "127.0.0.1", "--listen-port", "5555"])
        self.assertEqual(flag_value(cmd, "--diffusion-model"), str(config.TURBO_MODEL))
        self.assertEqual(flag_value(cmd, "--llm_vision"), str(config.LLM_VISION))
        self.assertIn("--diffusion-fa", cmd)
        self.assertEqual(flag_value(cmd, "--model-args"), "qwen_image_2_1_prefix_cache_type=q8_0")
        self.assertEqual(flag_value(cmd, "--log-level"), "debug")
        self.assertNotIn("--lora-model-dir", cmd)                   # dir does not exist
        (self.tmp / "loras").mkdir()
        cmd = self.client.build_server_command(5555, use_base_model=True)
        self.assertEqual(flag_value(cmd, "--diffusion-model"), str(config.BASE_MODEL))
        self.assertEqual(flag_value(cmd, "--lora-model-dir"), str(self.tmp / "loras"))

    def test_request_body_t2i_sigmas(self):
        body = self.client.build_request_body(self.request(sigmas=VIGGLE, negative_prompt=" noise "))
        sp = body["sample_params"]
        self.assertEqual(sp["custom_sigmas"], VIGGLE)
        self.assertEqual(sp["sample_steps"], 6)
        self.assertNotIn("scheduler", sp)
        self.assertNotIn("extra_sample_args", sp)
        self.assertEqual(sp["sample_method"], "euler")
        self.assertEqual(sp["guidance"], {"txt_cfg": 1.0})
        self.assertEqual((body["width"], body["height"], body["seed"], body["batch_count"]), (512, 288, 42, 1))
        self.assertEqual(body["negative_prompt"], "noise")
        self.assertEqual((body["ref_images"], body["init_image"], body["mask_image"], body["lora"]), ([], None, None, []))
        self.assertNotIn("ref_image_args", body)
        self.assertEqual(body["output_format"], "png")

    def test_request_body_steps_and_scheduler(self):
        sp = self.client.build_request_body(self.request(steps=4, scheduler="simple"))["sample_params"]
        self.assertEqual((sp["sample_steps"], sp["scheduler"]), (4, "simple"))
        self.assertNotIn("custom_sigmas", sp)

    def test_request_body_images_base64_in_order(self):
        refs = [self.image("a.png", (1, 2, 3)), self.image("b.png", (4, 5, 6))]
        init, mask = self.image("init.png", (7, 8, 9)), self.image("mask.png", (255, 255, 255))
        body = self.client.build_request_body(self.request(ref_images=refs, init_image=init, mask_image=mask, strength=0.6))
        self.assertEqual([base64.b64decode(r) for r in body["ref_images"]], [Path(p).read_bytes() for p in refs])
        self.assertEqual(base64.b64decode(body["init_image"]), Path(init).read_bytes())
        self.assertEqual(base64.b64decode(body["mask_image"]), Path(mask).read_bytes())
        self.assertEqual(body["strength"], 0.6)
        self.assertEqual(body["sample_params"]["extra_sample_args"], "strength_as_noise_level=true,force_first_sigma=true")
        self.assertEqual(body["ref_image_args"], "resize_before_vae=false")
        self.assertEqual(body["image_preprocess"], "target=ref,mode=none")
        paths = self.client.build_request_body(self.request(ref_images=refs, init_image=init), encode_images=False)
        self.assertEqual(paths["ref_images"], refs)
        self.assertEqual(paths["init_image"], init)

    def test_request_body_structured_lora(self):
        (self.tmp / "loras" / "sub").mkdir(parents=True)
        (self.tmp / "loras" / "sub" / "grain.safetensors").write_bytes(b"x")
        body = self.client.build_request_body(self.request(loras={"sub/grain": 0.7}))
        self.assertEqual(body["lora"], [{"path": "sub/grain.safetensors", "multiplier": 0.7}])
        self.assertNotIn("<lora:", body["prompt"])

    def test_generate_starts_once_and_decodes_image(self):
        first = self.client.generate(self.request(sigmas=VIGGLE, tag="t2i"))
        self.assertEqual(len(FakeProc.instances), 1)
        proc = FakeProc.instances[0]
        self.assertTrue(proc.kwargs["start_new_session"])
        self.assertEqual(proc.argv, first.command)
        self.assertEqual(Path(first.image_path).read_bytes(), self.png)
        url, body = self.session.posts[0]
        self.assertTrue(url.endswith("/sdcpp/v1/img_gen"))
        self.assertEqual(body["sample_params"]["custom_sigmas"], VIGGLE)
        self.assertEqual(first.extra["job_id"], "job_1")
        self.assertEqual(first.peak_rss_bytes, 5_000_000_000)
        self.assertAlmostEqual(first.phase_seconds["sampling"], 23.3)
        self.assertEqual(Path(first.log_path).name, "t2i.log")
        self.assertFalse(self.session.trust_env)

        self.session.polls = 0
        second = self.client.generate(self.request(output_path=str(self.tmp / "second.png")))
        self.assertEqual(len(FakeProc.instances), 1)                 # server reused
        self.assertEqual(second.extra["server_start_seconds"], 0.0)
        self.assertEqual(second.phase_seconds["sampling"], 23.3)    # only this job's log segment

    def test_model_variant_change_restarts_server(self):
        self.client.generate(self.request())
        self.session.polls = 0
        self.client.generate(self.request(use_base_model=True))
        self.assertEqual(len(FakeProc.instances), 2)
        self.assertEqual(self.terminated, [FakeProc.instances[0]])
        self.assertEqual(flag_value(FakeProc.instances[1].argv, "--diffusion-model"), str(config.BASE_MODEL))

    def test_lora_request_restarts_server_without_lora_dir(self):
        self.client.generate(self.request())
        (self.tmp / "loras").mkdir()
        (self.tmp / "loras" / "grain.safetensors").write_bytes(b"x")
        self.session.polls = 0
        self.client.generate(self.request(loras={"grain": 0.5}))
        self.assertEqual(len(FakeProc.instances), 2)
        self.assertEqual(flag_value(FakeProc.instances[1].argv, "--lora-model-dir"), str(self.tmp / "loras"))
        self.assertEqual(self.session.posts[-1][1]["lora"], [{"path": "grain.safetensors", "multiplier": 0.5}])

    def test_close_terminates_server(self):
        self.client.generate(self.request())
        self.assertTrue(self.client.running)
        self.client.close()
        self.assertFalse(self.client.running)
        self.assertEqual(self.terminated, [FakeProc.instances[0]])

    def test_watchdog_armed_with_server_and_disarmed_after_close(self):
        self.client.generate(self.request())
        self.watchdog_arm.assert_called_once_with(FakeProc.instances[0].pid, None)
        self.assertNotIn(mock.call(mock.sentinel.guard), self.watchdog_disarm.call_args_list)
        self.client.close()
        self.assertEqual(self.watchdog_disarm.call_args, mock.call(mock.sentinel.guard))
        self.assertEqual(self.watchdog_disarm.call_args_list.count(mock.call(mock.sentinel.guard)), 1)
        self.assertEqual(self.terminated, [FakeProc.instances[0]])

    def test_watchdog_cleans_up_private_temp_log_dir(self):
        client = SdServerClient(sd_server=Path("/opt/sd-server"), poll_interval_s=0)
        self.session.client = client
        client.generate(self.request())
        self.assertEqual(self.watchdog_arm.call_args.args[1], str(client._log_path.parent))
        client.close()

    def test_version_mismatch_allowed_by_constructor(self):
        client = SdServerClient(sd_server=Path("/opt/sd-server"), poll_interval_s=0, allow_mismatch=True)
        self.addCleanup(client.close)
        self.session.client = client
        with mock.patch("src.engine.version.engine_commit", return_value="1d0a58a"), \
                mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
            self.assertEqual(client.generate(self.request()).engine_commit, "1d0a58a")
            strict = SdServerClient(sd_server=Path("/opt/sd-server"), poll_interval_s=0)
            with self.assertRaises(EngineError):
                strict.generate(self.request())

    def test_failed_job_raises_and_keeps_server(self):
        self.session.final_status = "failed"
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("boom", str(ctx.exception))
        self.assertTrue(self.client.running)

    def test_rejected_submission_raises(self):
        self.session.submit_status = 400
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("400", str(ctx.exception))

    def test_server_dying_during_startup(self):
        def dying(argv, **kw):
            proc = FakeProc(argv, **kw)
            proc.returncode = 1
            return proc

        with mock.patch("src.engine.sd_server_client.subprocess.Popen", side_effect=dying):
            with self.assertRaises(EngineError) as ctx:
                self.client.generate(self.request())
        self.assertIn("exited during startup", str(ctx.exception))
        self.assertFalse(self.client.running)

    def test_job_timeout_tears_down_server(self):
        self.session.final_status = "generating"
        self.client.timeout_s = 0
        with self.assertRaises(EngineError) as ctx:
            self.client.generate(self.request())
        self.assertIn("timed out", str(ctx.exception))
        self.assertFalse(self.client.running)
        self.assertEqual(len(self.terminated), 1)

    def test_strict_fallbacks(self):
        self.session.log_lines = ENGINE_LOG + "\n" + FA_FALLBACK + "\n"
        self.client.strict_fallbacks = True
        with self.assertRaises(EngineError):
            self.client.generate(self.request())

    def test_temp_log_dir_removed_on_close(self):
        client = SdServerClient(sd_server=Path("/opt/sd-server"), poll_interval_s=0)
        self.session.client = client
        client.generate(self.request())
        log_path = client._log_path
        self.assertTrue(log_path.exists())
        client.close()
        self.assertFalse(log_path.exists())


class TestMakeEngine(unittest.TestCase):
    def test_kinds(self):
        self.assertIsInstance(make_engine(), SdCliClient)
        cli = make_engine("cli", timeout_s=5, strict_fallbacks=True)
        self.assertEqual((cli.timeout_s, cli.strict_fallbacks), (5, True))
        server = make_engine("server")
        self.assertIsInstance(server, SdServerClient)
        server.close()
        with self.assertRaises(ValueError):
            make_engine("gpu")

    def test_allow_mismatch_forwarded(self):
        for kind in ("cli", "server"):
            with self.subTest(kind=kind):
                default, allowed = make_engine(kind), make_engine(kind, allow_mismatch=True)
                self.assertEqual((default.allow_mismatch, allowed.allow_mismatch), (False, True))
                with mock.patch("src.engine.version.engine_commit", return_value="1d0a58a"), \
                        mock.patch.dict(os.environ, {"PREVIS_ALLOW_ENGINE_MISMATCH": ""}):
                    self.assertEqual(allowed.engine_commit(), "1d0a58a")
                    with self.assertRaises(EngineError):
                        default.engine_commit()
                default.close()
                allowed.close()


FAKE_SD_SERVER = """
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):                      # answers the /sdcpp/v1/capabilities readiness probe
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--listen-port") + 1])), Handler).serve_forever()
"""


class TestWatchdog(TempDirCase):
    """Real processes: an engine child must not outlive a Python parent that dies uncleanly."""

    def _driver(self, body: str) -> subprocess.Popen:
        script = f"import os, signal, subprocess, sys, time\nsys.path.insert(0, {str(ROOT)!r})\n{body}"
        return subprocess.Popen([sys.executable, "-c", script], cwd=ROOT, stdout=subprocess.PIPE, text=True)

    def test_parent_sigkill_kills_group_and_removes_dir(self):
        doomed = self.tmp / "sd-server-tmp"
        doomed.mkdir()
        driver = self._driver(
            "from src.engine import watchdog\n"
            "target = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
            f"watchdog.arm(target.pid, {str(doomed)!r})\n"
            "print(target.pid, flush=True)\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n"
        )
        target = int(driver.stdout.readline())
        self.addCleanup(kill_if_alive, target)
        driver.communicate(timeout=10)
        self.assertEqual(driver.returncode, -signal.SIGKILL)
        self.assertTrue(wait_gone(target), "engine child survived its parent's SIGKILL")
        deadline = time.monotonic() + 5
        while doomed.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(doomed.exists())

    def test_disarm_leaves_target_alone(self):
        target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self.addCleanup(target.wait)
        self.addCleanup(kill_if_alive, target.pid)
        guard = watchdog.arm(target.pid, str(self.tmp))
        watchdog.disarm(guard)
        self.assertEqual(guard.returncode, 0)
        self.assertIsNone(target.poll())
        self.assertTrue(self.tmp.is_dir())
        watchdog.disarm(guard)                                       # idempotent

    def test_sd_server_stopped_when_python_dies_from_sighup(self):
        # A closed terminal / dropped SSH session: SIGHUP kills Python without atexit, and the
        # server (in its own session) never sees the hang-up itself.
        server_py = self.tmp / "fake_sd_server.py"
        server_py.write_text(FAKE_SD_SERVER)
        server = self.tmp / "sd-server"
        server.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{server_py}" "$@"\n')
        server.chmod(0o755)
        driver = self._driver(
            "from pathlib import Path\n"
            "from src.engine.sd_server_client import SdServerClient\n"
            f"client = SdServerClient(sd_server=Path({str(server)!r}), poll_interval_s=0.05, startup_timeout_s=20)\n"
            "client.start()\n"
            "print(client._proc.pid, client._tmp_dir, flush=True)\n"
            "os.kill(os.getpid(), signal.SIGHUP)\n"
            "time.sleep(30)\n"
        )
        pid_text, tmp_dir = driver.stdout.readline().split()
        server_pid = int(pid_text)
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        self.addCleanup(kill_if_alive, server_pid)
        driver.communicate(timeout=10)
        self.assertEqual(driver.returncode, -signal.SIGHUP)
        self.assertTrue(wait_gone(server_pid), "sd-server survived its parent's SIGHUP")
        deadline = time.monotonic() + 5
        while Path(tmp_dir).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(Path(tmp_dir).exists(), "sd-server temp log dir left behind")


FAKE_CMAKE = """#!/bin/sh
# Stand-in for cmake: configure makes the build dir, build writes an sd-cli that prints a version.
if [ "$1" = "-B" ]; then mkdir -p "$2"; exit 0; fi
if [ "$1" = "--build" ]; then
  mkdir -p "$2/bin"
  printf '#!/bin/sh\\necho "stable-diffusion.cpp version unknown, commit unknown"\\n' > "$2/bin/sd-cli"
  chmod +x "$2/bin/sd-cli"
  exit 0
fi
exit 1
"""


@unittest.skipUnless(shutil.which("git") and shutil.which("bash"), "needs git and bash")
class TestSetupScript(TempDirCase):
    """setup.sh in a sandbox repo with a stand-in cmake (no clone, no compiler, no network)."""

    def git(self, cwd: Path, *args: str) -> str:
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=/dev/null", *args]
        return subprocess.run(cmd, cwd=cwd, env=self.env, check=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()

    def setUp(self):
        super().setUp()
        fakebin = self.tmp / "bin"
        fakebin.mkdir()
        (fakebin / "cmake").write_text(FAKE_CMAKE)
        (fakebin / "cmake").chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "PIN_ENGINE"}
        self.env["PATH"] = f"{fakebin}{os.pathsep}{self.env.get('PATH', '')}"
        # Parent repo laid out like vision-model: the engine dir is ignored, not a submodule.
        self.repo = self.tmp / "repo"
        (self.repo / "engine").mkdir(parents=True)
        shutil.copy2(ROOT / "setup.sh", self.repo / "setup.sh")
        (self.repo / "engine" / "UPSTREAM_SHA").write_text(PINNED + "\n")
        (self.repo / ".gitignore").write_text("stable-diffusion.cpp/\n")
        self.git(self.repo, "init", "-q")
        self.git(self.repo, "add", "-A")
        self.git(self.repo, "commit", "-q", "-m", "parent")
        self.parent_head = self.git(self.repo, "rev-parse", "HEAD")
        self.engine = self.repo / "stable-diffusion.cpp"
        self.engine.mkdir()
        (self.engine / "CMakeLists.txt").write_text("project(fake)\n")

    def run_setup(self, **env) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.repo / "setup.sh")], cwd=self.repo, env={**self.env, **env},
                              capture_output=True, text=True, timeout=60)

    def build_info(self) -> dict:
        return json.loads((self.engine / "build" / "build-info.json").read_text())

    def test_sources_without_git_record_unknown_commit(self):
        proc = self.run_setup()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        info = self.build_info()
        self.assertEqual(info["commit"], "unknown")
        self.assertIsNone(info["dirty"])
        self.assertEqual(info["pinned_sha"], PINNED)
        self.assertNotIn(self.parent_head, proc.stdout + proc.stderr)
        self.assertIn("not a git checkout", proc.stdout)

    def test_pin_without_git_checkout_fails_without_touching_parent_repo(self):
        proc = self.run_setup(PIN_ENGINE="1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PIN_ENGINE=1 needs a git checkout", proc.stdout)
        self.assertEqual(self.git(self.repo, "rev-parse", "HEAD"), self.parent_head)
        self.assertFalse((self.engine / "build").exists())

    def test_dir_without_sources_fails_clearly(self):
        (self.engine / "CMakeLists.txt").unlink()
        (self.engine / "build" / "bin").mkdir(parents=True)          # what the Drive notebook pre-creates
        proc = self.run_setup()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no stable-diffusion.cpp sources", proc.stdout)

    def test_own_checkout_records_its_own_commit(self):
        self.git(self.engine, "init", "-q")
        self.git(self.engine, "add", "-A")
        self.git(self.engine, "commit", "-q", "-m", "engine")
        engine_head = self.git(self.engine, "rev-parse", "HEAD")
        proc = self.run_setup()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(f"is at {engine_head} but engine/UPSTREAM_SHA pins {PINNED}", proc.stdout)
        info = self.build_info()
        self.assertEqual((info["commit"], info["dirty"]), (engine_head, False))


def _version_tuple(text: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:3])


class TestRequirements(unittest.TestCase):
    # Oldest release of each package that has an API the repo calls.
    API_FLOORS = {
        "pillow": ("10.1", "ImageFont.load_default(size=...) in src/previs/assembly"),
        "huggingface_hub": ("0.34", "the `hf` CLI that download_models.sh runs"),
        "pydantic": ("2.0", "the pydantic v2 API (model_validate, field_validator)"),
    }

    def minimums(self) -> dict:
        found = {}
        for line in (ROOT / "requirements.txt").read_text().splitlines():
            match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*>=\s*([0-9][0-9.]*)", line)
            if match:
                found[match.group(1).lower()] = match.group(2)
        return found

    def test_minimums_cover_used_apis(self):
        minimums = self.minimums()
        for name, (floor, reason) in self.API_FLOORS.items():
            with self.subTest(package=name):
                self.assertIn(name, minimums)
                self.assertGreaterEqual(_version_tuple(minimums[name]), _version_tuple(floor),
                                        f"{name}>={minimums[name]} is too old for {reason}")

    def test_installed_versions_meet_minimums(self):
        for name, minimum in self.minimums().items():
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            with self.subTest(package=name):
                self.assertGreaterEqual(_version_tuple(installed), _version_tuple(minimum))


@unittest.skipUnless(os.environ.get("RUN_ENGINE_TESTS") == "1", "set RUN_ENGINE_TESTS=1 to run the real engine")
class TestRealEngine(unittest.TestCase):
    """Real generations with the local binaries and weights (minutes, ~12 GB RAM)."""

    def _request(self, out_dir: str, name: str) -> GenerationRequest:
        return GenerationRequest(
            prompt="A cinematic film still of a glass of water on a wooden kitchen table, soft window light",
            output_path=str(Path(out_dir) / f"{name}.png"), width=512, height=288, seed=42,
            sigmas=VIGGLE, tag=name,
        )

    def _check(self, result, width=512, height=288):
        with Image.open(result.image_path) as img:
            self.assertEqual(img.size, (width, height))
        self.assertTrue(version.commits_match(result.engine_commit, config.pinned_engine_sha()))
        self.assertIn("sampling", result.phase_seconds)

    def test_cli_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = SdCliClient(verbose_log_dir=tmp).generate(self._request(tmp, "cli"))
            self._check(result)
            if sys.platform == "darwin":
                self.assertGreater(result.peak_rss_bytes or 0, 0)

    def test_server_generation(self):
        with tempfile.TemporaryDirectory() as tmp, SdServerClient(verbose_log_dir=tmp) as client:
            self._check(client.generate(self._request(tmp, "server")))


if __name__ == "__main__":
    unittest.main()
