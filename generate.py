"""Generate images using the deployed Qwen-Image-Turbo on Modal.

Deploy once:
    modal deploy qwen_image_turbo.py

Then run anytime instantly without reloading the model (stays warm for 5 minutes):
    python generate.py "a majestic eagle flying over snowy mountains"
    python generate.py "cyberpunk alleyway in rain" --output cyber.png --seed 42
"""
import argparse
import time
import modal


def main():
    parser = argparse.ArgumentParser(description="Call deployed Qwen-Image-Turbo on Modal")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--output", "-o", type=str, default="output.png", help="Output file path (default: output.png)")
    args = parser.parse_args()

    print(f"Connecting to deployed 'qwen-image-turbo' on Modal...")
    try:
        Model = modal.Cls.from_name("qwen-image-turbo", "Model")
    except Exception as e:
        print(f"Error accessing deployed model: {e}")
        print("Please ensure you deployed it first with: modal deploy qwen_image_turbo.py")
        return

    print(f"Sending prompt: '{args.prompt}' (seed={args.seed})...")
    t0 = time.time()
    png = Model().generate.remote(args.prompt, seed=args.seed)
    with open(args.output, "wb") as f:
        f.write(png)
    duration = time.time() - t0
    print(f"✓ Saved {args.output} in {duration:.2f}s")


if __name__ == "__main__":
    main()
