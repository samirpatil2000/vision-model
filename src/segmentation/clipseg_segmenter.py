import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any
from scipy import ndimage
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from src.segmentation.base import BaseSegmenter, SegmentationResult, SegmentationCandidate

class CLIPSegSegmenter(BaseSegmenter):
    """
    Zero-shot text-prompted segmentation using CLIPSeg.
    Runs efficiently on Apple Silicon (MPS) or CPU.
    """
    def __init__(self, model_id: str = "CIDAS/clipseg-rd64-refined", device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def segment(
        self,
        image: Image.Image,
        target_prompt: str,
        instance_index: Optional[int] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        point: Optional[Tuple[int, int]] = None,
        threshold: float = 0.25,
    ) -> SegmentationResult:
        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")

        inputs = self.processor(
            text=[target_prompt],
            images=[img_rgb],
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Output shape: (1, 352, 352) -> resize to original image size
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Resize heatmap back to original canvas
        heatmap_pil = Image.fromarray((probs * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
        heatmap = np.array(heatmap_pil).astype(float) / 255.0

        max_prob = float(np.max(heatmap))
        mean_prob = float(np.mean(heatmap))

        # Dynamic thresholding if max_prob is modest
        effective_thresh = min(threshold, max(0.12, max_prob * 0.65))
        binary_mask = (heatmap >= effective_thresh).astype(np.uint8) * 255

        # Extract connected components (instances)
        labeled_array, num_features = ndimage.label(binary_mask > 0)
        candidates: List[SegmentationCandidate] = []

        if num_features > 0:
            sizes = ndimage.sum(binary_mask > 0, labeled_array, range(1, num_features + 1))
            sorted_indices = np.argsort(sizes)[::-1]  # Largest first

            for idx in sorted_indices:
                comp_idx = idx + 1
                comp_mask = (labeled_array == comp_idx).astype(np.uint8) * 255
                comp_size = sizes[idx]

                # Filter out tiny noise specks (< 0.05% of canvas)
                if comp_size < (orig_w * orig_h * 0.0005):
                    continue

                # Bounding box
                rows = np.any(comp_mask, axis=1)
                cols = np.any(comp_mask, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]

                comp_conf = float(np.mean(heatmap[comp_mask > 0]))
                candidates.append(SegmentationCandidate(
                    mask=comp_mask,
                    box=(int(x_min), int(y_min), int(x_max), int(y_max)),
                    confidence=comp_conf,
                    label=target_prompt,
                ))

        # Ambiguity check
        ambiguity_detected = False
        warning = None
        if len(candidates) > 1:
            diff = abs(candidates[0].confidence - candidates[1].confidence)
            if diff < 0.05 and instance_index is None:
                ambiguity_detected = True
                warning = f"Found {len(candidates)} candidate instances for '{target_prompt}' with close confidence scores. Using largest instance."

        # Select candidate
        if instance_index is not None and 0 <= instance_index < len(candidates):
            selected = candidates[instance_index]
            final_mask = selected.mask
            bbox = selected.box
            conf = selected.confidence
        elif len(candidates) > 0:
            selected = candidates[0]
            final_mask = selected.mask
            bbox = selected.box
            conf = selected.confidence
        else:
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            bbox = (0, 0, 0, 0)
            conf = 0.0
            warning = f"No candidate matching '{target_prompt}' could be localized."

        is_valid, validation_msg = self.validate_mask(final_mask, (orig_h, orig_w))
        if not is_valid:
            ambiguity_detected = True
            warning = validation_msg

        return SegmentationResult(
            target_mask=final_mask,
            confidence=conf,
            bounding_box=bbox,
            candidates=candidates,
            ambiguity_detected=ambiguity_detected,
            warning_message=warning,
            metadata={
                "model": "clipseg-rd64-refined",
                "max_activation": max_prob,
                "mean_activation": mean_prob,
                "threshold_used": effective_thresh,
                "num_candidates": len(candidates),
                "is_valid": is_valid,
            }
        )
