from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

@dataclass
class SegmentationCandidate:
    mask: np.ndarray  # uint8 (0 or 255)
    box: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    confidence: float
    label: str

@dataclass
class SegmentationResult:
    target_mask: np.ndarray  # uint8 (0 or 255), shape (H, W)
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    candidates: List[SegmentationCandidate] = field(default_factory=list)
    ambiguity_detected: bool = False
    warning_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseSegmenter(ABC):
    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        target_prompt: str,
        instance_index: Optional[int] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        point: Optional[Tuple[int, int]] = None,
        threshold: float = 0.3,
    ) -> SegmentationResult:
        """
        Segment the target described by target_prompt in image.
        """
        pass

    @staticmethod
    def validate_mask(
        mask: np.ndarray,
        image_shape: Tuple[int, int],
        min_area_ratio: float = 0.00005,
        max_area_ratio: float = 0.70,
    ) -> Tuple[bool, str]:
        """
        Validates the generated mask. Rejects empty masks, invalid dimensions,
        and masks covering an unexpectedly large portion of the canvas.
        """
        if mask is None:
            return False, "Mask is None"

        if mask.shape[:2] != image_shape[:2]:
            return False, f"Mask shape {mask.shape[:2]} does not match image shape {image_shape[:2]}"

        total_pixels = mask.shape[0] * mask.shape[1]
        active_pixels = np.sum(mask > 0)
        ratio = active_pixels / total_pixels

        if active_pixels == 0 or ratio < min_area_ratio:
            return False, f"Target mask is empty or smaller than minimum threshold ({ratio:.6f} < {min_area_ratio})"

        if ratio > max_area_ratio:
            return False, f"Mask covers {ratio*100:.1f}% of image, exceeding safety ceiling ({max_area_ratio*100:.0f}%). Ambiguity detected."

        return True, "Valid mask"
