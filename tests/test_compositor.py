import unittest
import numpy as np
from PIL import Image
from src.processing.compositor import StrictPixelCompositor, StrictPreservationViolationError

class TestStrictPixelCompositor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic deterministic 256x256 RGB image with random pixel values
        self.orig_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        self.orig_img = Image.fromarray(self.orig_arr, mode="RGB")

        # Candidate image: completely different random pixels
        self.cand_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        # Allowed edit mask: circular region in center
        y, x = np.ogrid[:256, :256]
        self.mask = ((x - 128)**2 + (y - 128)**2 <= 40**2).astype(np.uint8) * 255
        self.compositor = StrictPixelCompositor(strict_mode=True)

    def test_strict_preservation_invariant_rgb(self):
        final_img, report = self.compositor.composite(
            self.orig_img, self.cand_arr, self.mask, feather_pixels=0
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.changed_protected_pixels, 0)
        self.assertEqual(report.max_protected_diff, 0)

        # Check array directly
        final_arr = np.array(final_img)
        protected_bool = (self.mask == 0)
        np.testing.assert_array_equal(final_arr[protected_bool], self.orig_arr[protected_bool])

        # Allowed region should match candidate
        allowed_bool = (self.mask > 0)
        np.testing.assert_array_equal(final_arr[allowed_bool], self.cand_arr[allowed_bool])

    def test_strict_preservation_rgba(self):
        rgba_orig = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
        rgba_cand = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        img = Image.fromarray(rgba_orig, mode="RGBA")
        final_img, report = self.compositor.composite(img, rgba_cand, mask)
        self.assertTrue(report.passed)
        self.assertEqual(report.max_protected_diff, 0)
        self.assertEqual(report.changed_protected_pixels, 0)

    def test_feathering_bounded_within_allowed_mask(self):
        final_img, report = self.compositor.composite(
            self.orig_img, self.cand_arr, self.mask, feather_pixels=8
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.changed_protected_pixels, 0)
        self.assertEqual(report.max_protected_diff, 0)

        final_arr = np.array(final_img)
        protected_bool = (self.mask == 0)
        # Even with feathering, protected pixels must remain bit-for-bit identical
        np.testing.assert_array_equal(final_arr[protected_bool], self.orig_arr[protected_bool])

    def test_regression_deliberate_corruption_detected(self):
        # Manually alter 1 pixel in the protected region and ensure validator fails
        orig_arr = self.orig_arr.copy()
        corrupted_arr = self.orig_arr.copy()
        val = int(corrupted_arr[10, 10, 0])
        corrupted_arr[10, 10, 0] = np.uint8((val + 1) % 256)

        protected_bool = (self.mask == 0)
        report = StrictPixelCompositor.verify_preservation(
            orig_arr, corrupted_arr, protected_bool, (256, 256), "RGB"
        )
        self.assertFalse(report.passed)
        self.assertGreater(report.changed_protected_pixels, 0)
        self.assertGreater(report.max_protected_diff, 0)

    def test_regression_global_resize_detected(self):
        orig_arr = self.orig_arr.copy()
        wrong_size_arr = np.zeros((200, 200, 3), dtype=np.uint8)
        protected_bool = (self.mask == 0)

        report = StrictPixelCompositor.verify_preservation(
            orig_arr, wrong_size_arr, protected_bool, (256, 256), "RGB"
        )
        self.assertFalse(report.passed)
        self.assertFalse(report.dimensions_match)

if __name__ == "__main__":
    unittest.main()
