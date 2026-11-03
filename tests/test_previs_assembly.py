import dataclasses
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src.previs import physics
from src.previs.assembly import format_timecode, load_solved, probe_video
from src.previs.assembly.animatic import _Renderer, build_animatic, build_timeline, ken_burns_box
from src.previs.assembly.graybox import build_graybox_animatic, interpolate_camera
from src.previs.assembly.storyboard import build_storyboard, draw_move_arrow, wrap_text
from src.previs.schema import ScenePlan
from src.previs.state import build_index
from src.previs.types import CameraState

FIXTURE = Path(__file__).resolve().parent.parent / "src" / "previs" / "fixtures" / "kitchen_glass.json"
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40), (200, 40, 200), (40, 200, 200), (230, 230, 230)]
SIZE = (128, 72)


def load_plan() -> ScenePlan:
    return ScenePlan.model_validate_json(FIXTURE.read_text())


def write_frames(directory: Path, solved, skip=(), size=SIZE):
    """Solid-colour stand-ins for accepted keyframes."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = {}
    for n, f in enumerate(solved.frames):
        if f.kf_id in skip:
            continue
        path = directory / f"{f.kf_id}.png"
        Image.new("RGB", size, COLORS[n % len(COLORS)]).save(path)
        frames[f.kf_id] = str(path)
    return frames


def centre_pixel(img: Image.Image):
    w, h = img.size
    return np.array(img.convert("RGB"))[h // 3, w // 2].astype(int)


class TestHelpers(unittest.TestCase):
    def test_format_timecode_uses_frames_within_second(self):
        self.assertEqual(format_timecode(0.0), "00:00.00")
        self.assertEqual(format_timecode(1.75), "00:01.18")
        self.assertEqual(format_timecode(61.5), "01:01.12")
        self.assertEqual(format_timecode(2.0, fps=25), "00:02.00")

    def test_load_solved_round_trips_solved_json(self):
        plan = load_plan()
        solved = physics.solve(plan)
        restored = load_solved(json.loads(json.dumps(dataclasses.asdict(solved))))
        self.assertEqual([f.kf_id for f in restored.frames], [f.kf_id for f in solved.frames])
        self.assertAlmostEqual(restored.total_duration_s, solved.total_duration_s)
        a, b = solved.frames[3].entities["pr_glass"], restored.frames[3].entities["pr_glass"]
        self.assertIsInstance(b.position, tuple)
        self.assertEqual(b.position, a.position)
        self.assertEqual(b.trajectory, a.trajectory)
        self.assertEqual(set(restored.events), set(solved.events))
        self.assertEqual(restored.frame("kf_glass_2").t_s, solved.frame("kf_glass_2").t_s)

    def test_ffmpeg_and_ffprobe_never_read_the_terminal(self):
        # A backgrounded run (`previs.py run ... &`) is stopped by SIGTTOU/SIGTTIN if ffmpeg touches the tty.
        from src.previs import assembly

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if Path(cmd[0]).name == "ffprobe":
                out = {"streams": [{"width": 2, "height": 2, "r_frame_rate": "24/1", "nb_read_frames": "1"}],
                       "format": {"duration": "0.04"}}
                return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")
            Path(cmd[-1]).write_bytes(b"mp4")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        tmp = Path(tempfile.mkdtemp())
        try:
            with mock.patch.object(assembly.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(assembly.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"):
                seq = assembly.FrameSequence(tmp / "frames")
                seq.add(Image.new("RGB", (2, 2)))
                seq.encode(tmp / "out.mp4", fps=24)
                assembly.probe_video(tmp / "out.mp4")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(len(calls), 2)
        ffmpeg_cmd = calls[0][0]
        self.assertIn("-nostdin", ffmpeg_cmd)
        for _cmd, kwargs in calls:
            self.assertIs(kwargs.get("stdin"), subprocess.DEVNULL)


class TestStoryboard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = load_plan()
        self.solved = physics.solve(self.plan)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pages_and_pdf_with_pending_placeholder(self):
        frames = write_frames(self.tmp / "frames", self.solved, skip={"kf_glass_2"})
        paths = build_storyboard(self.tmp, self.plan, self.solved, frames, cols=2, rows=2)
        # 7 keyframes at 4 per page -> 2 PNG pages, then the PDF.
        self.assertEqual([Path(p).name for p in paths], ["page_01.png", "page_02.png", "storyboard.pdf"])
        for p in paths:
            self.assertTrue(Path(p).is_file())
        pdf = Path(paths[-1]).read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page\r") + pdf.count(b"/Type /Page "), 2)

        # Panel 4 (kf_glass_2: second row, second column of page 1) is the grey "pending" placeholder.
        page = Image.open(paths[0]).convert("RGB")
        from src.previs.assembly import storyboard as sb
        cell_w = (sb.PAGE_WIDTH - 2 * sb.MARGIN - sb.GAP) // 2
        image_h = int(round(cell_w / (16 / 9)))
        row2_y = sb.MARGIN + sb.HEADER_H + sb._panel_height(image_h) + sb.GAP
        px = np.array(page)[row2_y + image_h // 4, sb.MARGIN + cell_w + sb.GAP + cell_w // 8]
        self.assertTrue(np.all(np.abs(px.astype(int) - 128) <= 16), px)
        first = np.array(page)[sb.MARGIN + sb.HEADER_H + image_h // 4, sb.MARGIN + cell_w // 8]
        self.assertTrue(np.all(np.abs(first.astype(int) - np.array(COLORS[0])) <= 2), first)

    def test_rebuild_removes_stale_pages(self):
        frames = write_frames(self.tmp / "frames", self.solved)
        build_storyboard(self.tmp, self.plan, self.solved, frames, cols=1, rows=1)
        paths = build_storyboard(self.tmp, self.plan, self.solved, frames, cols=3, rows=3)
        self.assertEqual(sorted(p.name for p in (self.tmp / "storyboard").glob("page_*.png")), ["page_01.png"])
        self.assertEqual(len(paths), 2)

    def test_move_arrow_drawn_except_static(self):
        base = Image.new("RGB", (320, 180), (100, 100, 100))
        self.assertTrue(np.array_equal(np.array(draw_move_arrow(base.copy(), "static")), np.array(base)))
        for move in ("pan_left", "tilt_up", "dolly_in", "orbit_right", "handheld", "crane_down"):
            drawn = np.array(draw_move_arrow(base.copy(), move))
            self.assertGreater(int((drawn == 255).all(axis=-1).sum()), 50, move)

    def test_wrap_text_limits_lines_with_ellipsis(self):
        lines = wrap_text("word " * 60, 200, 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[-1].endswith("…"))
        self.assertEqual(wrap_text("short action", 400, 3), ["short action"])


class TestAnimaticTimeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = load_plan()
        self.solved = physics.solve(self.plan)
        self.frames = write_frames(self.tmp / "frames", self.solved)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_segments_tile_the_timeline(self):
        spans = build_timeline(self.plan, self.solved, self.frames, 24)
        total = int(round(self.solved.total_duration_s * 24))
        self.assertEqual([s.shot_id for s in spans], ["sh_wide", "sh_glass", "sh_jon"])
        self.assertEqual(spans[0].start_frame, 0)
        self.assertEqual(spans[-1].end_frame, total)
        cursor = 0
        for span in spans:
            self.assertEqual(span.start_frame, cursor)
            self.assertEqual(span.segments[0].start_frame, span.start_frame)
            for seg in span.segments:
                self.assertEqual(seg.start_frame, cursor)
                cursor = seg.end_frame
            self.assertEqual(cursor, span.end_frame)
        seg = next(s for sp in spans for s in sp.segments if s.kf_id == "kf_glass_2")
        self.assertEqual(seg.start_frame, int(round(self.solved.frame("kf_glass_2").t_s * 24)))

    def test_hold_crossfade_and_cut(self):
        spans = build_timeline(self.plan, self.solved, self.frames, 24)
        renderer = _Renderer(spans, SIZE, 24)
        wide = spans[0]
        kf2_start = wide.segments[1].start_frame
        # Pure hold right after the first keyframe; 5-frame crossfade ends exactly on the next keyframe.
        np.testing.assert_allclose(centre_pixel(renderer.frame(1)), COLORS[0], atol=2)
        np.testing.assert_allclose(centre_pixel(renderer.frame(kf2_start)), COLORS[1], atol=2)
        mid = centre_pixel(renderer.frame(kf2_start - 2))
        expected = np.array(COLORS[0]) * 0.4 + np.array(COLORS[1]) * 0.6
        np.testing.assert_allclose(mid, expected, atol=3)
        # Hard cut between shots (transition "cut").
        np.testing.assert_allclose(centre_pixel(renderer.frame(wide.end_frame - 1)), COLORS[1], atol=2)
        np.testing.assert_allclose(centre_pixel(renderer.frame(wide.end_frame)), COLORS[2], atol=2)

    def test_dissolve_into_shot(self):
        data = json.loads(FIXTURE.read_text())
        data["bible"]["shots"][2]["transition"] = "dissolve"
        plan = ScenePlan.model_validate(data)
        spans = build_timeline(plan, self.solved, self.frames, 24)
        renderer = _Renderer(spans, SIZE, 24)
        cut = spans[2].start_frame
        before, at = centre_pixel(renderer.frame(cut - 1)), centre_pixel(renderer.frame(cut))
        last_glass, first_jon = np.array(COLORS[4]), np.array(COLORS[5])
        for px in (before, at):
            self.assertFalse(np.allclose(px, last_glass, atol=4))
            self.assertFalse(np.allclose(px, first_jon, atol=4))
        # Far from the cut the pictures are pure again.
        np.testing.assert_allclose(centre_pixel(renderer.frame(cut - 10)), last_glass, atol=3)
        np.testing.assert_allclose(centre_pixel(renderer.frame(cut + 10)), first_jon, atol=3)

    def test_ken_burns_boxes(self):
        self.assertIsNone(ken_burns_box("static", 0.5, 1.0, 768, 448))
        self.assertIsNone(ken_burns_box("dolly_in", 0.0, 0.0, 768, 448))
        x0, y0, x1, y1 = ken_burns_box("dolly_in", 1.0, 1.0, 768, 448)
        self.assertAlmostEqual((x1 - x0) / 768, 1 / 1.10, places=4)
        self.assertAlmostEqual((x0 + x1) / 2, 384, places=4)
        start, end = ken_burns_box("pan_left", 0.0, 0, 768, 448), ken_burns_box("pan_left", 1.0, 2, 768, 448)
        self.assertGreater(start[0], end[0])
        self.assertAlmostEqual(end[0], 0.0)
        top, bottom = ken_burns_box("tilt_down", 0.0, 0, 768, 448), ken_burns_box("tilt_down", 1.0, 2, 768, 448)
        self.assertLess(top[1], bottom[1])
        for t in np.linspace(0, 5, 25):
            box = ken_burns_box("handheld", 0.5, float(t), 768, 448, seed=7)
            self.assertGreaterEqual(box[0], 0.0)
            self.assertLessEqual(box[2], 768.0 + 1e-6)
            self.assertLessEqual(box[3], 448.0 + 1e-6)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
    def test_build_animatic_encodes_mp4_and_timeline(self):
        frames = write_frames(self.tmp / "frames2", self.solved, skip={"kf_jon_2"})
        path = build_animatic(self.tmp, self.plan, self.solved, frames, fps=24, width=128, height=72)
        self.assertEqual(Path(path), self.tmp / "animatic" / "animatic.mp4")
        info = probe_video(path)
        self.assertEqual(info["fps"], 24.0)
        self.assertEqual((info["width"], info["height"]), (128, 72))
        self.assertEqual(info["frames"], int(round(self.solved.total_duration_s * 24)))
        self.assertAlmostEqual(info["duration_s"], self.solved.total_duration_s, delta=1 / 24)
        self.assertEqual(info["pix_fmt"], "yuv420p")
        self.assertFalse((self.tmp / "animatic" / "_frames").exists())
        timeline = json.loads((self.tmp / "animatic" / "timeline.json").read_text())
        status = {s["kf_id"]: s["status"] for s in timeline["segments"]}
        self.assertEqual(status["kf_jon_2"], "pending")
        self.assertEqual(status["kf_wide_1"], "frame")
        # Decode one held frame and check its colour survived encoding.
        still = self.tmp / "still.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", path, "-frames:v", "1", str(still)],
                       check=True, timeout=60)
        np.testing.assert_allclose(centre_pixel(Image.open(still)), COLORS[0], atol=12)

    def test_size_defaults_to_first_frame(self):
        if not HAS_FFMPEG:
            self.skipTest("ffmpeg/ffprobe not installed")
        path = build_animatic(self.tmp, self.plan, self.solved, self.frames)
        self.assertEqual(probe_video(path)["width"], SIZE[0])


class FakeRenderer:
    def __init__(self):
        self.frames = []

    def render_image(self, frame, plan, index):
        self.frames.append(frame)
        glass = frame.entities["pr_glass"].position
        shade = int(np.clip(glass[1] * 200, 0, 255))
        return Image.new("RGB", SIZE, (shade, 80, 255 - shade))


def fake_camera(shot, frame, index, width, height):
    p = index.keyframe_by_id[frame.kf_id].move_progress
    return CameraState(position=(p, 1.6, 0.0), yaw_deg=10 * p, pitch_deg=0.0, roll_deg=0.0, lens_mm=shot.lens_mm,
                       vfov_deg=40.0, width=width, height=height, look_at=(0.0, 1.0, 2.0))


class TestGraybox(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.plan = load_plan()
        self.solved = physics.solve(self.plan)
        self.index = build_index(self.plan)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_interpolate_camera_short_arc(self):
        a = CameraState((0, 0, 0), 170.0, 0.0, 0.0, 50.0, 30.0, 64, 36, (0, 0, 1))
        b = CameraState((2, 0, 0), -170.0, 10.0, 0.0, 50.0, 30.0, 64, 36, (0, 0, 3))
        mid = interpolate_camera(a, b, 0.5)
        self.assertAlmostEqual(abs(((mid.yaw_deg + 180) % 360) - 180), 180.0)
        self.assertEqual(mid.position, (1.0, 0.0, 0.0))
        self.assertEqual(mid.look_at, (0.0, 0.0, 2.0))
        self.assertAlmostEqual(mid.pitch_deg, 5.0)
        self.assertIs(interpolate_camera(a, b, 0.0), a)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
    def test_graybox_samples_every_frame_and_encodes(self):
        renderer = FakeRenderer()
        path = build_graybox_animatic(self.tmp, self.plan, self.solved, self.index, *SIZE, fps=12,
                                      renderer=renderer, solve_camera=fake_camera)
        n = int(round(self.solved.total_duration_s * 12))
        self.assertEqual(len(renderer.frames), n)
        self.assertTrue(all(f.camera is not None for f in renderer.frames))
        info = probe_video(path)
        self.assertEqual(info["frames"], n)
        self.assertEqual(info["fps"], 12.0)
        self.assertEqual(Path(path), self.tmp / "graybox" / "graybox.mp4")

        # sh_jon dollies in from move_progress 0 to 1 between its keyframes: the camera moves smoothly.
        jon = [f for f in renderer.frames if f.shot_id == "sh_jon"]
        xs = [f.camera.position[0] for f in jon]
        self.assertEqual(xs, sorted(xs))
        self.assertTrue(any(0.05 < x < 0.95 for x in xs))
        # The falling glass is sampled mid-air between keyframes.
        heights = [f.entities["pr_glass"].position[1] for f in renderer.frames if f.shot_id == "sh_glass"]
        self.assertGreater(max(heights), 0.3)
        self.assertLess(min(heights), 0.1)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
    def test_graybox_with_real_renderer_and_camera(self):
        try:
            from src.previs.blocking.box_renderer import BoxRenderer  # noqa: F401
            from src.previs.camera import solve_camera  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"blocking/camera not available yet: {exc}")
        path = build_graybox_animatic(self.tmp, self.plan, self.solved, self.index, 192, 112, fps=4)
        info = probe_video(path)
        self.assertEqual(info["frames"], int(round(self.solved.total_duration_s * 4)))
        self.assertEqual((info["width"], info["height"]), (192, 112))


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class TestAssembleCli(unittest.TestCase):
    def setUp(self):
        import signal
        self.tmp = Path(tempfile.mkdtemp())
        self.signals = {s: signal.getsignal(s) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}

    def tearDown(self):
        import signal
        for sig, handler in self.signals.items():
            signal.signal(sig, handler)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_animatic_keeps_the_frame_size_unless_tier_is_given(self):
        # Frames rendered at a non-default tier must not be downscaled to the --tier default by `assemble`.
        import previs
        from contextlib import redirect_stdout
        from io import StringIO
        from src import config
        from src.config import Tier

        plan = load_plan()
        run_dir = self.tmp / "kitchen"
        run_dir.mkdir()
        (run_dir / "plan.json").write_text(plan.model_dump_json())
        write_frames(run_dir / "frames", physics.solve(plan), size=SIZE)
        Image.new("RGB", (32, 32)).save(run_dir / "frames" / "kf_dropped_from_plan.png")

        def assemble(*extra):
            with redirect_stdout(StringIO()):
                previs.main(["assemble", "--run-name", "kitchen", *extra])
            return json.loads((run_dir / "animatic" / "timeline.json").read_text())

        with mock.patch.object(config, "PREVIS_OUTPUT_DIR", self.tmp), \
                mock.patch.dict(config.TIERS, {"mini": Tier("mini", 256, 160, (256, 128), (192, 128))}):
            timeline = assemble()
            self.assertEqual((timeline["width"], timeline["height"]), SIZE)
            timeline = assemble("--tier", "mini")
            self.assertEqual((timeline["width"], timeline["height"]), (256, 160))


if __name__ == "__main__":
    unittest.main()
