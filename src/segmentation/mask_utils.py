import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from typing import Tuple, Optional, Dict

def refine_mask(
    target_mask: np.ndarray,
    expand_pixels: int = 12,
    fill_holes: bool = True,
    smooth_radius: int = 2,
    max_allowed_ratio: float = 0.80,
) -> np.ndarray:
    """
    Derives the allowed edit mask from the raw target segmentation mask.
    Applies morphological dilation, hole filling, and boundary cleanup.
    Guarantees that the expansion does not authorize changes across the whole canvas.
    """
    binary = (target_mask > 0).astype(bool)

    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)

    if expand_pixels > 0:
        # Create disk structuring element
        r = expand_pixels
        y, x = np.ogrid[-r:r+1, -r:r+1]
        struct = x*x + y*y <= r*r
        binary = ndimage.binary_dilation(binary, structure=struct)

    if smooth_radius > 0:
        r = smooth_radius
        y, x = np.ogrid[-r:r+1, -r:r+1]
        struct = x*x + y*y <= r*r
        binary = ndimage.binary_closing(binary, structure=struct)

    total_pixels = binary.size
    active_pixels = np.sum(binary)
    ratio = active_pixels / total_pixels

    if ratio > max_allowed_ratio:
        raise ValueError(
            f"Expanded allowed edit mask occupies {ratio*100:.1f}% of canvas, "
            f"exceeding safety ceiling ({max_allowed_ratio*100:.0f}%)."
        )

    return (binary.astype(np.uint8) * 255)

def compute_protected_mask(allowed_edit_mask: np.ndarray) -> np.ndarray:
    """
    Protected mask is the exact binary complement of the allowed edit mask.
    1 where protected, 0 where editing is permitted.
    """
    return ((allowed_edit_mask == 0).astype(np.uint8) * 255)

def generate_three_masks(
    target_mask: np.ndarray,
    expand_pixels: int = 12,
    fill_holes: bool = True,
    smooth_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (target_mask, allowed_edit_mask, protected_mask) as uint8 arrays {0, 255}.
    Ensures target_mask is never modified by downstream operations.
    """
    target = target_mask.copy()
    allowed = refine_mask(
        target,
        expand_pixels=expand_pixels,
        fill_holes=fill_holes,
        smooth_radius=smooth_radius,
    )
    protected = compute_protected_mask(allowed)
    return target, allowed, protected

def create_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blends a colored mask overlay onto an RGB PIL Image for visual inspection.
    """
    img_rgb = image.convert("RGB")
    img_arr = np.array(img_rgb).astype(float)
    color_arr = np.array(color, dtype=float)

    mask_bool = (mask > 0)[:, :, np.newaxis]
    blended = np.where(mask_bool, (1 - alpha) * img_arr + alpha * color_arr, img_arr)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def create_three_mask_diagnostic(
    image: Image.Image,
    target_mask: np.ndarray,
    allowed_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> Image.Image:
    """
    Creates a 4-panel diagnostic comparison image:
    [Original Image] | [Target Mask (Blue)] | [Allowed Edit Mask (Red)] | [Protected Area (Green)]
    """
    target_overlay = create_mask_overlay(image, target_mask, color=(0, 150, 255), alpha=0.5)
    allowed_overlay = create_mask_overlay(image, allowed_mask, color=(255, 50, 50), alpha=0.5)
    protected_overlay = create_mask_overlay(image, protected_mask, color=(50, 220, 50), alpha=0.35)

    w, h = image.size
    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(image, (0, 0))
    canvas.paste(target_overlay, (w, 0))
    canvas.paste(allowed_overlay, (0, h))
    canvas.paste(protected_overlay, (w, h))

    return canvas
