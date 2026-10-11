import unittest
import numpy as np
from src.segmentation.base import BaseSegmenter
from src.segmentation.mask_utils import refine_mask, compute_protected_mask, generate_three_masks

class TestMaskProcessing(unittest.TestCase):
    def setUp(self):
        # Create 100x100 synthetic canvas with a 20x20 square in center
        self.shape = (100, 100)
        self.target = np.zeros(self.shape, dtype=np.uint8)
        self.target[40:60, 40:60] = 255

    def test_validation_empty_mask(self):
        empty = np.zeros(self.shape, dtype=np.uint8)
        is_valid, msg = BaseSegmenter.validate_mask(empty, self.shape)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg)

    def test_validation_full_image_mask(self):
        full = np.full(self.shape, 255, dtype=np.uint8)
        is_valid, msg = BaseSegmenter.validate_mask(full, self.shape)
        self.assertFalse(is_valid)
        self.assertIn("exceeding safety ceiling", msg)

    def test_dilation_expands_mask(self):
        target_count = np.sum(self.target > 0)
        expanded = refine_mask(self.target, expand_pixels=5, fill_holes=False, smooth_radius=0)
        expanded_count = np.sum(expanded > 0)
        self.assertGreater(expanded_count, target_count)

    def test_hole_filling(self):
        # Mask with hole inside
        donut = self.target.copy()
        donut[48:52, 48:52] = 0
        filled = refine_mask(donut, expand_pixels=0, fill_holes=True, smooth_radius=0)
        self.assertEqual(np.sum(filled[48:52, 48:52] > 0), 16)

    def test_three_masks_partition(self):
        t, a, p = generate_three_masks(self.target, expand_pixels=4)
        # 1. Target is untouched
        np.testing.assert_array_equal(t, self.target)
        # 2. Allowed contains target
        self.assertTrue(np.all((a >= t)))
        # 3. Allowed and Protected partition canvas
        self.assertEqual(np.sum(a > 0) + np.sum(p > 0), self.shape[0] * self.shape[1])
        self.assertEqual(np.sum((a > 0) & (p > 0)), 0)

if __name__ == "__main__":
    unittest.main()
