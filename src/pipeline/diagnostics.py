import os
import json
import numpy as np
from PIL import Image
from typing import Dict, Any

from src.segmentation.mask_utils import create_mask_overlay, create_three_mask_diagnostic
from src.processing.compositor import VerificationReport

def save_diagnostics(
    output_dir: str,
    base_name: str,
    original_img: Image.Image,
    final_img: Image.Image,
    target_mask: np.ndarray,
    allowed_mask: np.ndarray,
    protected_mask: np.ndarray,
    verification: VerificationReport,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    """
    Saves diagnostic masks, visual overlays, side-by-side comparisons, and JSON metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # 1. Raw masks (grayscale PNG)
    target_mask_path = os.path.join(output_dir, f"{base_name}_target_mask.png")
    allowed_mask_path = os.path.join(output_dir, f"{base_name}_allowed_mask.png")
    protected_mask_path = os.path.join(output_dir, f"{base_name}_protected_mask.png")

    Image.fromarray(target_mask).save(target_mask_path)
    Image.fromarray(allowed_mask).save(allowed_mask_path)
    Image.fromarray(protected_mask).save(protected_mask_path)

    paths["target_mask"] = target_mask_path
    paths["allowed_mask"] = allowed_mask_path
    paths["protected_mask"] = protected_mask_path

    # 2. Visual overlays
    target_ov = create_mask_overlay(original_img, target_mask, color=(0, 150, 255), alpha=0.5)
    allowed_ov = create_mask_overlay(original_img, allowed_mask, color=(255, 50, 50), alpha=0.5)
    protected_ov = create_mask_overlay(original_img, protected_mask, color=(50, 220, 50), alpha=0.35)

    target_ov_path = os.path.join(output_dir, f"{base_name}_target_overlay.png")
    allowed_ov_path = os.path.join(output_dir, f"{base_name}_allowed_overlay.png")
    three_mask_path = os.path.join(output_dir, f"{base_name}_three_mask_panel.png")

    target_ov.save(target_ov_path)
    allowed_ov.save(allowed_ov_path)
    create_three_mask_diagnostic(original_img, target_mask, allowed_mask, protected_mask).save(three_mask_path)

    paths["target_overlay"] = target_ov_path
    paths["allowed_overlay"] = allowed_ov_path
    paths["three_mask_panel"] = three_mask_path

    # 3. Before vs After Side-by-Side
    w, h = original_img.size
    sbs = Image.new("RGB", (w * 2, h))
    sbs.paste(original_img, (0, 0))
    sbs.paste(final_img, (w, 0))
    sbs_path = os.path.join(output_dir, f"{base_name}_comparison_sbs.png")
    sbs.save(sbs_path)
    paths["comparison_sbs"] = sbs_path

    # 4. JSON metadata report
    meta_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    full_meta = {
        "metadata": metadata,
        "verification": {
            "passed": verification.passed,
            "total_pixels": verification.total_pixels,
            "allowed_pixels": verification.allowed_pixels,
            "protected_pixels": verification.protected_pixels,
            "changed_protected_pixels": verification.changed_protected_pixels,
            "max_protected_diff": verification.max_protected_diff,
            "dimensions_match": verification.dimensions_match,
            "mode_matches": verification.mode_matches,
            "error_message": verification.error_message,
        },
        "artifact_paths": paths,
    }

    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)

    paths["metadata"] = meta_path
    paths["metadata_json"] = meta_path
    return paths
