import os
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple

from src.segmentation.base import BaseSegmenter
from src.segmentation.clipseg_segmenter import CLIPSegSegmenter
from src.segmentation.mask_utils import generate_three_masks
from src.processing.cropper import ContextCropEngine, CropMetadata
from src.processing.compositor import StrictPixelCompositor, VerificationReport
from src.processing.formats import load_image_safely, save_lossless_image
from src.pipeline.diagnostics import save_diagnostics

class SegmentedEditor:
    """
    Segmentation-Aware, Pixel-Preserving Image Editing Pipeline.
    Edits only authorized target regions while guaranteeing bit-for-bit exact
    preservation of all protected pixels outside the allowed edit mask.
    """
    def __init__(
        self,
        sd_cli_path: str = "./stable-diffusion.cpp/build/bin/sd-cli",
        model_path: str = "./models/diffusion_models/Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf",
        llm_path: str = "./models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        vision_path: str = "./models/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        vae_path: str = "./models/vae/qwen_image_2.1_vae_bf16.safetensors",
        strict_mode: bool = True,
    ):
        self.sd_cli_path = os.path.abspath(sd_cli_path)
        self.model_path = os.path.abspath(model_path)
        self.llm_path = os.path.abspath(llm_path)
        self.vision_path = os.path.abspath(vision_path)
        self.vae_path = os.path.abspath(vae_path)
        self.strict_mode = strict_mode

        self.segmenter: Optional[BaseSegmenter] = None
        self.cropper = ContextCropEngine(context_margin=1.5, snap_multiple=32, min_size=512)
        self.compositor = StrictPixelCompositor(strict_mode=strict_mode)

    def _get_segmenter(self) -> BaseSegmenter:
        if self.segmenter is None:
            self.segmenter = CLIPSegSegmenter()
        return self.segmenter

    def _release_segmenter(self) -> None:
        """Sequential memory release to minimize memory footprint on Apple Silicon."""
        self.segmenter = None
        import gc
        import torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def edit(
        self,
        image_path: str,
        target_description: str,
        edit_instruction: str,
        output_path: str,
        expand_pixels: int = 16,
        feather_pixels: int = 4,
        steps: int = 4,
        cfg_scale: float = 1.0,
        sampler: str = "euler",
        scheduler: str = "simple",
        seed: int = 42,
        preview_only: bool = False,
        instance_index: Optional[int] = None,
        custom_allowed_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        metrics: Dict[str, Any] = {}

        # 1. Load source image immutably
        orig_img, source_meta = load_image_safely(image_path)
        w, h = orig_img.size
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        output_dir = os.path.dirname(os.path.abspath(output_path))
        diag_dir = os.path.join(output_dir, f"{base_name}_diagnostics")

        # 2. Target Segmentation
        t0 = time.time()
        if custom_allowed_mask is not None:
            target_mask = custom_allowed_mask.copy()
            allowed_mask = custom_allowed_mask.copy()
            from src.segmentation.mask_utils import compute_protected_mask
            protected_mask = compute_protected_mask(allowed_mask)
            seg_result = None
            metrics["segmentation_source"] = "custom_user_mask"
        else:
            seg_engine = self._get_segmenter()
            seg_result = seg_engine.segment(
                orig_img,
                target_description,
                instance_index=instance_index,
            )
            target_mask, allowed_mask, protected_mask = generate_three_masks(
                seg_result.target_mask,
                expand_pixels=expand_pixels,
            )
            metrics["segmentation_source"] = "clipseg-rd64-refined"
            metrics["target_confidence"] = seg_result.confidence
            metrics["ambiguity_detected"] = seg_result.ambiguity_detected

            if not seg_result.metadata.get("is_valid", False):
                if self.strict_mode:
                    raise ValueError(
                        f"Segmentation failed safety validation: {seg_result.warning_message}"
                    )

        metrics["segmentation_time_s"] = round(time.time() - t0, 3)

        # 3. Preview mode: return diagnostics immediately without running generation
        if preview_only:
            paths = save_diagnostics(
                diag_dir,
                base_name,
                orig_img,
                orig_img,
                target_mask,
                allowed_mask,
                protected_mask,
                VerificationReport(
                    passed=True,
                    total_pixels=w * h,
                    allowed_pixels=int(np.sum(allowed_mask > 0)),
                    protected_pixels=int(np.sum(protected_mask > 0)),
                    changed_protected_pixels=0,
                    max_protected_diff=0,
                    dimensions_match=True,
                    mode_matches=True,
                ),
                {"preview_only": True, "target": target_description},
            )
            return {
                "preview_only": True,
                "diagnostic_artifacts": paths,
                "allowed_pixels": int(np.sum(allowed_mask > 0)),
                "protected_pixels": int(np.sum(protected_mask > 0)),
            }

        # 4. Release segmenter memory before running diffusion model
        self._release_segmenter()

        # 5. Extract context crop around allowed edit region
        t_crop = time.time()
        # Find bounding box of allowed mask
        rows = np.any(allowed_mask, axis=1)
        cols = np.any(allowed_mask, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        target_box = (int(x0), int(y0), int(x1), int(y1))

        crop_img, crop_mask, crop_meta = self.cropper.extract_crop(
            orig_img, allowed_mask, target_box
        )
        metrics["crop_time_s"] = round(time.time() - t_crop, 3)
        metrics["crop_box"] = [crop_meta.crop_x0, crop_meta.crop_y0, crop_meta.crop_x1, crop_meta.crop_y1]
        metrics["crop_dimensions"] = [crop_meta.crop_width, crop_meta.crop_height]

        # 6. Run image generation on the local crop
        t_gen = time.time()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_crop_in = os.path.join(tmp_dir, "crop_input.png")
            temp_crop_out = os.path.join(tmp_dir, "crop_output.png")
            crop_img.save(temp_crop_in, format="PNG")

            cmd = [
                self.sd_cli_path,
                "--diffusion-model", self.model_path,
                "--llm", self.llm_path,
                "--llm_vision", self.vision_path,
                "--vae", self.vae_path,
                "-r", temp_crop_in,
                "-p", edit_instruction,
                "--steps", str(steps),
                "--cfg-scale", str(cfg_scale),
                "--sampling-method", sampler,
                "--scheduler", scheduler,
                "-W", str(crop_meta.working_width),
                "-H", str(crop_meta.working_height),
                "-s", str(seed),
                "-o", temp_crop_out,
            ]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Generation failed with exit code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}"
                )

            if not os.path.exists(temp_crop_out):
                raise FileNotFoundError(f"Expected generated output patch at {temp_crop_out} was not created.")

            generated_patch = Image.open(temp_crop_out).convert(orig_img.mode)

        metrics["generation_time_s"] = round(time.time() - t_gen, 3)

        # 7. Map generated patch back onto full canvas
        t_comp = time.time()
        candidate_canvas = self.cropper.map_patch_to_canvas(orig_img, generated_patch, crop_meta)

        # 8. Strict pixel-level compositing & verification
        final_img, verify_report = self.compositor.composite(
            orig_img,
            candidate_canvas,
            allowed_mask,
            feather_pixels=feather_pixels,
        )
        metrics["compositing_time_s"] = round(time.time() - t_comp, 3)
        metrics["total_time_s"] = round(time.time() - start_time, 3)

        # 9. Save lossless output
        save_lossless_image(final_img, output_path)

        # 10. Save diagnostic artifacts
        diag_paths = save_diagnostics(
            diag_dir,
            base_name,
            orig_img,
            final_img,
            target_mask,
            allowed_mask,
            protected_mask,
            verify_report,
            {
                "target_description": target_description,
                "edit_instruction": edit_instruction,
                "metrics": metrics,
            },
        )

        return {
            "output_path": output_path,
            "verification": verify_report,
            "metrics": metrics,
            "diagnostic_artifacts": diag_paths,
        }
