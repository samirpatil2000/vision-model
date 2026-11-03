"""Benchmark parallel image generation on Modal using Qwen-Image-2.1 + Viggle Turbo LoRA."""
import asyncio
import json
import statistics
import time
from pathlib import Path
import modal
from PIL import Image

BENCHMARK_PROMPTS = [
    {
        "id": 1,
        "name": "Chameleon Macro",
        "prompt": "A close-up macro portrait of a vibrant chameleon on a lush jungle branch, vivid green and orange scales, bokeh background, national geographic style",
        "seed": 101,
        "filename": "parallel_1_chameleon.png",
    },
    {
        "id": 2,
        "name": "Tokyo Ramen Shop",
        "prompt": "A cozy Tokyo ramen shop on a rainy night, warm glowing paper lanterns, steam rising, reflections in puddles, cinematic film still",
        "seed": 102,
        "filename": "parallel_2_tokyo_ramen.png",
    },
    {
        "id": 3,
        "name": "Cliffside Castle",
        "prompt": "A majestic medieval fantasy castle perched on high rocky cliffs above a sea of clouds, golden hour sunlight, hyperrealistic",
        "seed": 103,
        "filename": "parallel_3_castle.png",
    },
    {
        "id": 4,
        "name": "Cyberpunk Hypercar",
        "prompt": "A futuristic sleek hypercar driving through a neon cyber city at twilight, light trails, wet asphalt, 8k octane render",
        "seed": 104,
        "filename": "parallel_4_hypercar.png",
    },
    {
        "id": 5,
        "name": "Ancient Library",
        "prompt": "An ancient library with towering wooden bookshelves reaching the vaulted ceiling, warm fireplace, sunbeams through stained glass",
        "seed": 105,
        "filename": "parallel_5_library.png",
    },
]


async def run_prompt(model, item: dict, overall_start: float):
    pid = item["id"]
    name = item["name"]
    prompt = item["prompt"]
    seed = item["seed"]
    filename = item["filename"]

    dispatch_offset = time.time() - overall_start
    print(f"[{pid}/5] 🚀 Dispatched: '{name}' (+{dispatch_offset:.2f}s)")

    t0 = time.time()
    try:
        png_bytes = await model.generate.remote.aio(prompt, seed=seed)
        latency = time.time() - t0
        finish_offset = time.time() - overall_start

        # Save to disk
        with open(filename, "wb") as f:
            f.write(png_bytes)

        # Inspect image
        with Image.open(filename) as img:
            width, height = img.size
            mode = img.mode

        file_size_kb = Path(filename).stat().st_size / 1024

        print(
            f"[{pid}/5] ✅ Done '{name}' in {latency:.2f}s (finish at +{finish_offset:.2f}s) "
            f"[{width}x{height} {mode}, {file_size_kb:.1f} KB]"
        )

        return {
            "id": pid,
            "name": name,
            "filename": filename,
            "prompt": prompt,
            "seed": seed,
            "latency_s": round(latency, 2),
            "dispatch_offset_s": round(dispatch_offset, 2),
            "finish_offset_s": round(finish_offset, 2),
            "resolution": f"{width}x{height}",
            "file_size_kb": round(file_size_kb, 1),
            "success": True,
        }
    except Exception as e:
        latency = time.time() - t0
        print(f"[{pid}/5] ❌ Failed '{name}': {e}")
        return {
            "id": pid,
            "name": name,
            "filename": filename,
            "error": str(e),
            "latency_s": round(latency, 2),
            "success": False,
        }


async def main():
    print("=" * 70)
    print("🚀 Qwen-Image-2.1 + Viggle Turbo: Modal Parallel Inference Benchmark")
    print("=" * 70)
    print(f"Connecting to deployed Modal app 'qwen-image-turbo'...")

    Model = modal.Cls.from_name("qwen-image-turbo", "Model")
    model = Model()

    print(f"Dispatching 5 parallel inference requests across Modal GPU infrastructure...\n")
    wall_start = time.time()

    tasks = [run_prompt(model, item, wall_start) for item in BENCHMARK_PROMPTS]
    results = await asyncio.gather(*tasks)

    wall_duration = time.time() - wall_start
    successful = [r for r in results if r.get("success")]

    latencies = [r["latency_s"] for r in successful]

    mean_latency = statistics.mean(latencies) if latencies else 0
    median_latency = statistics.median(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    throughput = len(successful) / wall_duration if wall_duration > 0 else 0

    report = {
        "benchmark": "Qwen-Image-2.1-viggle-turbo-Modal-Parallel",
        "num_requests": len(BENCHMARK_PROMPTS),
        "successful_requests": len(successful),
        "total_wall_clock_s": round(wall_duration, 2),
        "throughput_images_per_sec": round(throughput, 3),
        "throughput_images_per_min": round(throughput * 60, 1),
        "latency_stats": {
            "min_s": min_latency,
            "max_s": max_latency,
            "mean_s": round(mean_latency, 2),
            "median_s": round(median_latency, 2),
        },
        "runs": results,
    }

    # Save JSON report
    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Images Generated:    {len(successful)} / {len(BENCHMARK_PROMPTS)}")
    print(f"Total Wall-Clock Time:     {wall_duration:.2f} seconds")
    print(f"Overall Throughput:        {throughput:.2f} images/sec ({throughput * 60:.1f} images/min)")
    print(f"Fastest Image (Min):       {min_latency:.2f}s")
    print(f"Slowest Image (Max):       {max_latency:.2f}s")
    print(f"Average Latency:           {mean_latency:.2f}s")
    print(f"Median Latency:            {median_latency:.2f}s")
    print("=" * 70)
    print("Saved detailed metrics to benchmark_results.json\n")


if __name__ == "__main__":
    asyncio.run(main())
