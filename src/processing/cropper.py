import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class CropMetadata:
    orig_width: int
    orig_height: int
    # Bounding box in original image coordinates (x0, y0, x1, y1)
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    crop_width: int
    crop_height: int
    # Working resolution used for model generation (must be multiple of 32)
    working_width: int
    working_height: int

class ContextCropEngine:
    """
    Extracts a local crop around the target/allowed edit mask with surrounding context.
    Tracks exact coordinate transformations to map generated patches back to the original canvas.
    Ensures dimensions are snapped to multiples of 32 for Qwen-Image VAE compatibility.
    """
    def __init__(self, context_margin: float = 1.6, snap_multiple: int = 32, min_size: int = 512):
        self.context_margin = context_margin
        self.snap_multiple = snap_multiple
        self.min_size = min_size

    def extract_crop(
        self,
        image: Image.Image,
        allowed_mask: np.ndarray,
        target_box: Tuple[int, int, int, int],
    ) -> Tuple[Image.Image, np.ndarray, CropMetadata]:
        orig_w, orig_h = image.size
        bx0, by0, bx1, by1 = target_box
        bw = bx1 - bx0
        bh = by1 - by0

        # Calculate padded context dimensions
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0

        padded_w = max(int(bw * self.context_margin), self.min_size)
        padded_h = max(int(bh * self.context_margin), self.min_size)

        # Snap to multiple of 32
        padded_w = ((padded_w + self.snap_multiple - 1) // self.snap_multiple) * self.snap_multiple
        padded_h = ((padded_h + self.snap_multiple - 1) // self.snap_multiple) * self.snap_multiple

        # Make sure crop fits within canvas boundaries if canvas is large enough
        padded_w = min(padded_w, ((orig_w // self.snap_multiple) * self.snap_multiple))
        padded_h = min(padded_h, ((orig_h // self.snap_multiple) * self.snap_multiple))

        crop_x0 = int(round(cx - padded_w / 2.0))
        crop_y0 = int(round(cy - padded_h / 2.0))

        # Clamp crop coordinates to keep inside image
        if crop_x0 < 0:
            crop_x0 = 0
        if crop_y0 < 0:
            crop_y0 = 0
        if crop_x0 + padded_w > orig_w:
            crop_x0 = orig_w - padded_w
        if crop_y0 + padded_h > orig_h:
            crop_y0 = orig_h - padded_h

        crop_x1 = crop_x0 + padded_w
        crop_y1 = crop_y0 + padded_h

        # Extract local crop without altering original image
        crop_img = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        crop_mask = allowed_mask[crop_y0:crop_y1, crop_x0:crop_x1].copy()

        meta = CropMetadata(
            orig_width=orig_w,
            orig_height=orig_h,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            crop_x1=crop_x1,
            crop_y1=crop_y1,
            crop_width=padded_w,
            crop_height=padded_h,
            working_width=padded_w,
            working_height=padded_h,
        )

        return crop_img, crop_mask, meta

    @staticmethod
    def map_patch_to_canvas(
        original_image: Image.Image,
        generated_patch: Image.Image,
        meta: CropMetadata,
    ) -> np.ndarray:
        """
        Maps a generated crop patch back onto the full original canvas dimensions.
        Returns a full canvas numpy array where pixels inside the crop window match
        generated_patch, and pixels outside are from original_image.
        """
        orig_arr = np.array(original_image)
        patch_arr = np.array(generated_patch)

        # Resize patch if the model altered working dimensions
        if patch_arr.shape[:2] != (meta.crop_height, meta.crop_width):
            patch_pil = generated_patch.resize((meta.crop_width, meta.crop_height), Image.Resampling.LANCZOS)
            patch_arr = np.array(patch_pil)

        candidate_canvas = orig_arr.copy()
        candidate_canvas[meta.crop_y0:meta.crop_y1, meta.crop_x0:meta.crop_x1] = patch_arr
        return candidate_canvas
