#!/usr/bin/env python3
import argparse
import sys
import os

from src.pipeline.segmented_editor import SegmentedEditor

def main():
    parser = argparse.ArgumentParser(
        description="Segmentation-Aware, Pixel-Preserving Image Editing"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--target", "-t", required=True, help="Natural language description of target object to edit")
    parser.add_argument("--instruction", "-p", required=True, help="Full editing instruction for the model")
    parser.add_argument("--output", "-o", default="./output/segmented_edit.png", help="Path to save output image (PNG/TIFF)")
    parser.add_argument("--expand-pixels", type=int, default=16, help="Mask dilation radius in pixels (default: 16)")
    parser.add_argument("--feather-pixels", type=int, default=4, help="Internal feathering band in pixels (default: 4)")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps (default: 4 for Turbo)")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--preview-masks", action="store_true", help="Generate and preview masks without running image generation")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict preservation check (not recommended)")

    args = parser.parse_args()

    editor = SegmentedEditor(strict_mode=not args.no_strict)

    print("==========================================================")
    print(" Segmentation-Aware Pixel-Preserving Editing")
    print("==========================================================")
    print(f" Source Image: {args.image}")
    print(f" Target Object: {args.target}")
    print(f" Instruction:   {args.instruction}")
    print(f" Output Target: {args.output}")
    print(f" Expand Pixels: {args.expand_pixels}")
    print(f" Steps:         {args.steps}")
    print(f" Mode:          {'PREVIEW ONLY' if args.preview_masks else 'FULL EDIT'}")
    print("==========================================================")

    try:
        res = editor.edit(
            image_path=args.image,
            target_description=args.target,
            edit_instruction=args.instruction,
            output_path=args.output,
            expand_pixels=args.expand_pixels,
            feather_pixels=args.feather_pixels,
            steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            preview_only=args.preview_masks,
        )

        if args.preview_masks:
            print("\n[✓] Mask preview generated successfully!")
            print(f"Diagnostics saved to: {res['diagnostic_artifacts']['allowed_overlay']}")
        else:
            v = res["verification"]
            print("\n[✓] Image edit completed successfully!")
            print(f"Output saved to: {res['output_path']}")
            print(f"Strict Preservation: {'PASSED (0 modified protected pixels)' if v.passed else 'FAILED'}")
            print(f"Allowed Edit Area:   {v.allowed_pixels:,} pixels ({(v.allowed_pixels/v.total_pixels)*100:.2f}%)")
            print(f"Protected Pixels:    {v.protected_pixels:,} pixels ({(v.protected_pixels/v.total_pixels)*100:.2f}%)")
            print(f"Changed Protected:   {v.changed_protected_pixels} pixels (Must be exactly 0)")
            print(f"Max Protected Diff:  {v.max_protected_diff} (Must be exactly 0)")
            print(f"Total Latency:       {res['metrics']['total_time_s']}s")
            print(f"Diagnostics:         {res['diagnostic_artifacts']['three_mask_panel']}")

    except Exception as e:
        print(f"\n[X] Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
