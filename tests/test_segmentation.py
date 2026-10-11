import unittest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from src.segmentation.base import BaseSegmenter, SegmentationResult
from src.segmentation.mask_utils import generate_three_masks, refine_mask, compute_protected_mask

class TestSegmentationSubsystem(unittest.TestCase):
    def setUp(self):
        self.shape = (200, 200)
        self.dummy_image = Image.new("RGB", (200, 200), color=(128, 128, 128))

    def test_mask_validation_criteria(self):
        # 1. Normal valid mask
        valid_mask = np.zeros(self.shape, dtype=np.uint8)
        valid_mask[50:100, 50:100] = 255
        is_val, msg = BaseSegmenter.validate_mask(valid_mask, self.shape)
        self.assertTrue(is_val)
        self.assertEqual(msg, "Valid mask")

        # 2. Empty mask
        empty_mask = np.zeros(self.shape, dtype=np.uint8)
        is_val, msg = BaseSegmenter.validate_mask(empty_mask, self.shape)
        self.assertFalse(is_val)
        self.assertIn("empty", msg)

        # 3. Shape mismatch
        is_val, msg = BaseSegmenter.validate_mask(valid_mask, (300, 300))
        self.assertFalse(is_val)
        self.assertIn("does not match", msg)

        # 4. Overly large mask (> 85% area)
        large_mask = np.full(self.shape, 255, dtype=np.uint8)
        is_val, msg = BaseSegmenter.validate_mask(large_mask, self.shape, max_area_ratio=0.85)
        self.assertFalse(is_val)
        self.assertIn("exceeding safety ceiling", msg)

    def test_three_masks_mathematical_properties(self):
        # Create an irregular shape mask
        raw_mask = np.zeros(self.shape, dtype=np.uint8)
        raw_mask[60:90, 70:120] = 255
        raw_mask[80:110, 100:140] = 255

        t_mask, a_mask, p_mask = generate_three_masks(raw_mask, expand_pixels=10)

        # Target mask is completely contained in Allowed mask
        self.assertTrue(np.all(a_mask[t_mask > 0] == 255))

        # Allowed mask is larger than target mask
        self.assertGreater(np.sum(a_mask > 0), np.sum(t_mask > 0))

        # Intersection of Allowed and Protected is strictly 0
        overlap = (a_mask > 0) & (p_mask > 0)
        self.assertEqual(np.sum(overlap), 0)

        # Union of Allowed and Protected is exactly 100% of the canvas
        union = (a_mask > 0) | (p_mask > 0)
        self.assertEqual(np.sum(union), self.shape[0] * self.shape[1])

        # Protected mask is bit-for-bit inverse
        np.testing.assert_array_equal(p_mask, 255 - a_mask)

    @patch("transformers.CLIPSegForImageSegmentation.from_pretrained")
    @patch("transformers.CLIPSegProcessor.from_pretrained")
    def test_segmenter_mock_ambiguity_handling(self, mock_proc_init, mock_model_init):
        from src.segmentation.clipseg_segmenter import CLIPSegSegmenter
        import torch

        mock_proc = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1]]), "pixel_values": torch.zeros((1, 3, 352, 352))}
        mock_proc.return_value = mock_inputs
        mock_proc_init.return_value = mock_proc

        mock_model = MagicMock()
        # Mock logits of shape (1, 352, 352) with low confidence values
        fake_logits = torch.full((1, 352, 352), -5.0)  # sigmoid(-5.0) ~ 0.006 < threshold
        mock_output = MagicMock()
        mock_output.logits = fake_logits
        mock_model.return_value = mock_output
        mock_model.to.return_value = mock_model
        mock_model_init.return_value = mock_model

        segmenter = CLIPSegSegmenter(device="cpu")
        result = segmenter.segment(self.dummy_image, "a vague shadow")
        
        # Low confidence should set warning or mark invalid
        self.assertTrue(result.ambiguity_detected or not result.metadata.get("is_valid", False) or result.confidence < 0.25)

if __name__ == "__main__":
    unittest.main()
