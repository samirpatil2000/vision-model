import unittest
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from src.pipeline.segmented_editor import SegmentedEditor
from src.segmentation.base import SegmentationResult

class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.temp_dir, "input.png")
        self.output_path = os.path.join(self.temp_dir, "output.png")

        # Generate a distinct synthetic 512x512 image
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        # Background gradient
        for y in range(512):
            arr[y, :, 0] = y // 2
            arr[y, :, 1] = 100
            arr[y, :, 2] = 200
        # Foreground square object (the target)
        arr[200:300, 200:300, :] = [255, 0, 0]
        self.orig_arr = arr
        Image.fromarray(arr).save(self.image_path, format="PNG")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subprocess.run")
    def test_mock_pipeline_e2e_preservation(self, mock_subproc):
        # Setup mock subprocess to simulate sd-cli generating a patch
        def side_effect(cmd, **kwargs):
            # Find output path from -o argument
            o_idx = cmd.index("-o")
            out_file = cmd[o_idx + 1]
            w_idx = cmd.index("-W")
            w = int(cmd[w_idx + 1])
            h_idx = cmd.index("-H")
            h = int(cmd[h_idx + 1])

            # Generate dummy green patch
            patch_arr = np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)
            Image.fromarray(patch_arr).save(out_file, format="PNG")
            res = MagicMock()
            res.returncode = 0
            res.stdout = "Mock generation completed"
            res.stderr = ""
            return res

        mock_subproc.side_effect = side_effect

        editor = SegmentedEditor(strict_mode=True)

        # Mock segmenter to return the 200:300 square as target
        mock_mask = np.zeros((512, 512), dtype=np.uint8)
        mock_mask[200:300, 200:300] = 255

        mock_seg = MagicMock()
        mock_seg.segment.return_value = SegmentationResult(
            target_mask=mock_mask,
            confidence=0.95,
            bounding_box=(200, 200, 300, 300),
            metadata={"is_valid": True, "model": "mock"},
            ambiguity_detected=False,
        )

        with patch.object(editor, "_get_segmenter", return_value=mock_seg):
            result = editor.edit(
                image_path=self.image_path,
                target_description="red square",
                edit_instruction="change to green square",
                output_path=self.output_path,
                expand_pixels=16,
                feather_pixels=4,
            )

        # 1. Output file must exist
        self.assertTrue(os.path.exists(self.output_path))

        # 2. Invariant verification report must have passed with 0 protected pixel changes
        v = result["verification"]
        self.assertTrue(v.passed)
        self.assertEqual(v.changed_protected_pixels, 0)
        self.assertEqual(v.max_protected_diff, 0)

        # 3. Direct array comparison on saved output
        final_img = Image.open(self.output_path)
        final_arr = np.array(final_img)

        # Verify that all pixels outside the dilated mask match original EXACTLY
        # In setup, mock_mask was 200:300, with expand_pixels=16, bounds are ~184:316
        # Any pixel outside row 184:316 or col 184:316 is definitely protected
        protected_corner = final_arr[:150, :150, :]
        orig_corner = self.orig_arr[:150, :150, :]
        np.testing.assert_array_equal(protected_corner, orig_corner)

        # 4. Verify diagnostic artifacts were generated
        diag = result["diagnostic_artifacts"]
        self.assertTrue(os.path.exists(diag["three_mask_panel"]))
        self.assertTrue(os.path.exists(diag["allowed_overlay"]))
        self.assertTrue(os.path.exists(diag["metadata_json"]))

    def test_preview_mode_generates_diagnostics_without_subprocess(self):
        editor = SegmentedEditor(strict_mode=True)

        mock_mask = np.zeros((512, 512), dtype=np.uint8)
        mock_mask[200:300, 200:300] = 255

        mock_seg = MagicMock()
        mock_seg.segment.return_value = SegmentationResult(
            target_mask=mock_mask,
            confidence=0.90,
            bounding_box=(200, 200, 300, 300),
            metadata={"is_valid": True, "model": "mock"},
            ambiguity_detected=False,
        )

        with patch.object(editor, "_get_segmenter", return_value=mock_seg):
            with patch("subprocess.run") as mock_subproc:
                res = editor.edit(
                    image_path=self.image_path,
                    target_description="red square",
                    edit_instruction="preview check",
                    output_path=self.output_path,
                    preview_only=True,
                )
                # Generation should NEVER be called in preview mode
                mock_subproc.assert_not_called()
                self.assertTrue(res["preview_only"])
                self.assertIn("diagnostic_artifacts", res)

    @patch("subprocess.run")
    def test_negative_prompt_and_custom_model_forwarding(self, mock_subproc):
        def side_effect(cmd, **kwargs):
            out_file = cmd[cmd.index("-o") + 1]
            w = int(cmd[cmd.index("-W") + 1])
            h = int(cmd[cmd.index("-H") + 1])
            patch_arr = np.zeros((h, w, 3), dtype=np.uint8)
            Image.fromarray(patch_arr).save(out_file, format="PNG")
            res = MagicMock()
            res.returncode = 0
            return res

        mock_subproc.side_effect = side_effect
        editor = SegmentedEditor(strict_mode=True)

        mock_mask = np.zeros((512, 512), dtype=np.uint8)
        mock_mask[200:300, 200:300] = 255
        mock_seg = MagicMock()
        mock_seg.segment.return_value = SegmentationResult(
            target_mask=mock_mask,
            confidence=0.95,
            bounding_box=(200, 200, 300, 300),
            metadata={"is_valid": True},
            ambiguity_detected=False,
        )

        custom_model = "/tmp/custom_base_model.gguf"
        negative_text = "mangled hands, saw teeth, extra fingers"

        with patch.object(editor, "_get_segmenter", return_value=mock_seg):
            editor.edit(
                image_path=self.image_path,
                target_description="red square",
                edit_instruction="change to black square",
                output_path=self.output_path,
                negative_prompt=negative_text,
                model_path=custom_model,
            )

        self.assertTrue(mock_subproc.called)
        cmd_called = mock_subproc.call_args[0][0]
        # Verify custom model was passed
        self.assertIn("--diffusion-model", cmd_called)
        self.assertEqual(cmd_called[cmd_called.index("--diffusion-model") + 1], os.path.abspath(custom_model))
        # Verify negative prompt was passed
        self.assertIn("-n", cmd_called)
        self.assertEqual(cmd_called[cmd_called.index("-n") + 1], negative_text)

if __name__ == "__main__":
    unittest.main()
