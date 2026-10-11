import unittest
import numpy as np
from PIL import Image
from src.processing.cropper import ContextCropEngine

class TestContextCropEngine(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 1000, 800
        self.img_arr = np.random.randint(0, 256, (self.h, self.w, 3), dtype=np.uint8)
        self.img = Image.fromarray(self.img_arr)

        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        # Target box at (300, 200) to (450, 350)
        self.target_box = (300, 200, 450, 350)
        self.mask[200:350, 300:450] = 255
        self.engine = ContextCropEngine(context_margin=1.5, snap_multiple=32, min_size=512)

    def test_snap_to_multiple_of_32(self):
        crop_img, crop_mask, meta = self.engine.extract_crop(self.img, self.mask, self.target_box)
        cw, ch = crop_img.size
        self.assertEqual(cw % 32, 0)
        self.assertEqual(ch % 32, 0)
        self.assertEqual(meta.crop_width % 32, 0)
        self.assertEqual(meta.crop_height % 32, 0)

    def test_coordinate_roundtrip_mapping(self):
        crop_img, crop_mask, meta = self.engine.extract_crop(self.img, self.mask, self.target_box)

        # Draw a synthetic patch inside the crop
        patch_arr = np.array(crop_img).copy()
        patch_arr[:, :] = [0, 255, 0]  # Green patch
        green_patch = Image.fromarray(patch_arr)

        canvas = ContextCropEngine.map_patch_to_canvas(self.img, green_patch, meta)

        # 1. Canvas size matches original
        self.assertEqual(canvas.shape[:2], (self.h, self.w))

        # 2. Inside the crop window, canvas is exactly green
        crop_slice = canvas[meta.crop_y0:meta.crop_y1, meta.crop_x0:meta.crop_x1]
        np.testing.assert_array_equal(crop_slice, patch_arr)

        # 3. Outside the crop window, canvas is untouched
        outside_mask = np.ones((self.h, self.w), dtype=bool)
        outside_mask[meta.crop_y0:meta.crop_y1, meta.crop_x0:meta.crop_x1] = False
        np.testing.assert_array_equal(canvas[outside_mask], self.img_arr[outside_mask])

if __name__ == "__main__":
    unittest.main()
