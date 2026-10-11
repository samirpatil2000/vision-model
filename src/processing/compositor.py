import numpy as np
from PIL import Image
from scipy import ndimage
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

class StrictPreservationViolationError(Exception):
    """Raised when any protected pixel was altered during editing."""
    pass

@dataclass
class VerificationReport:
    passed: bool
    total_pixels: int
    allowed_pixels: int
    protected_pixels: int
    changed_protected_pixels: int
    max_protected_diff: int
    dimensions_match: bool
    mode_matches: bool
    error_message: Optional[str] = None

class StrictPixelCompositor:
    """
    Combines the original image and the generated candidate strictly within
    the authorized edit mask.
    Guarantee: final_pixels[protected_mask] == original_pixels[protected_mask]
    """
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def composite(
        self,
        original_image: Image.Image,
        candidate_canvas: np.ndarray,
        allowed_edit_mask: np.ndarray,
        feather_pixels: int = 0,
    ) -> Tuple[Image.Image, VerificationReport]:
        orig_arr = np.array(original_image)
        cand_arr = np.array(candidate_canvas)

        if orig_arr.shape != cand_arr.shape:
            raise ValueError(
                f"Candidate canvas shape {cand_arr.shape} does not match original shape {orig_arr.shape}"
            )

        if orig_arr.shape[:2] != allowed_edit_mask.shape[:2]:
            raise ValueError(
                f"Allowed mask shape {allowed_edit_mask.shape[:2]} does not match image shape {orig_arr.shape[:2]}"
            )

        # Derive protected mask (strict binary complement)
        protected_bool = (allowed_edit_mask == 0)
        allowed_bool = (allowed_edit_mask > 0)

        # Build alpha blending weight array
        if feather_pixels > 0:
            # Feathering is strictly confined INSIDE allowed_edit_mask
            # Distance from the edge of the allowed mask inward
            dist = ndimage.distance_transform_edt(allowed_bool)
            alpha = np.clip(dist / float(feather_pixels), 0.0, 1.0)
            # Ensure alpha is 0 outside allowed mask
            alpha[protected_bool] = 0.0
        else:
            alpha = allowed_bool.astype(float)

        # Expand alpha to match image channels
        if orig_arr.ndim == 3:
            alpha_3d = alpha[:, :, np.newaxis]
        else:
            alpha_3d = alpha

        # Pixel blending: candidate where allowed, original where protected
        final_float = alpha_3d * cand_arr.astype(float) + (1.0 - alpha_3d) * orig_arr.astype(float)
        final_arr = np.clip(np.round(final_float), 0, 255).astype(orig_arr.dtype)

        # Strict preservation enforcement: overwrite protected pixels with exact original bytes
        final_arr[protected_bool] = orig_arr[protected_bool]

        # Verify preservation invariant
        report = self.verify_preservation(orig_arr, final_arr, protected_bool, original_image.size, original_image.mode)

        if self.strict_mode and not report.passed:
            raise StrictPreservationViolationError(
                f"Strict pixel preservation violated: {report.error_message}"
            )

        final_image = Image.fromarray(final_arr, mode=original_image.mode)
        return final_image, report

    @staticmethod
    def verify_preservation(
        orig_arr: np.ndarray,
        final_arr: np.ndarray,
        protected_bool: np.ndarray,
        expected_size: Tuple[int, int],
        expected_mode: str,
    ) -> VerificationReport:
        total_pixels = int(orig_arr.shape[0] * orig_arr.shape[1])
        protected_pixels = int(np.sum(protected_bool))
        allowed_pixels = total_pixels - protected_pixels

        # 1. Dimension check
        h, w = final_arr.shape[:2]
        dims_match = (w, h) == expected_size
        if not dims_match:
            return VerificationReport(
                passed=False,
                total_pixels=total_pixels,
                allowed_pixels=allowed_pixels,
                protected_pixels=protected_pixels,
                changed_protected_pixels=total_pixels,
                max_protected_diff=255,
                dimensions_match=False,
                mode_matches=True,
                error_message=f"Output size ({w}, {h}) does not match original size {expected_size}",
            )

        # 2. Pixel difference check on protected region
        orig_prot = orig_arr[protected_bool]
        final_prot = final_arr[protected_bool]

        diff = np.abs(orig_prot.astype(int) - final_prot.astype(int))
        if diff.size > 0:
            max_diff = int(np.max(diff))
            changed_count = int(np.sum(np.any(diff > 0, axis=-1) if diff.ndim > 1 else (diff > 0)))
        else:
            max_diff = 0
            changed_count = 0

        passed = (max_diff == 0) and (changed_count == 0)
        err = None if passed else f"{changed_count} protected pixels were modified (max diff: {max_diff})"

        return VerificationReport(
            passed=passed,
            total_pixels=total_pixels,
            allowed_pixels=allowed_pixels,
            protected_pixels=protected_pixels,
            changed_protected_pixels=changed_count,
            max_protected_diff=max_diff,
            dimensions_match=True,
            mode_matches=True,
            error_message=err,
        )
