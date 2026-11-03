# Qwen-Image-2.1 + Viggle Turbo LoRA on Modal

High-performance, serverless deployment of **Qwen-Image-2.1** accelerated with **Viggle Turbo LoRA (v0.3, 6-step distillation, rank 256)** running on **Nvidia L40S (48 GB)** GPUs via [Modal](https://modal.com).

Generates photorealistic **1024×1024 RGBA** images in **~2.9 seconds** of GPU inference time (~8–10s end-to-end including network transit) with **zero classifier-free guidance (CFG)**, costing approximately **$0.0022 per image** (~460 images per $1.00).

---

## Key Highlights

- **Ultra-Fast 6-Step Generation**: Distilled from 40 steps down to 6 steps using Viggle Turbo v0.3 LoRA with calibrated flow-matching sigmas.
- **Serverless Cloud GPU (Nvidia L40S)**: Peaks at ~26 GB VRAM during inference, comfortably fitting within the L40S's 48 GB memory at $1.95/hr (less than half the cost of an H100).
- **Zero Reload Overhead (Warm Containers)**: Configured with `scaledown_window=300` (5 minutes) and persistent deployment (`modal deploy`), keeping the model resident in GPU memory so subsequent prompts skip model initialization entirely.
- **Persistent Weight Caching**: Model weights and LoRAs are cached inside a persistent `modal.Volume` (`hf-cache`), eliminating repetitive 20 GB downloads.
- **Parallel Dispatch & Benchmarking**: Supports concurrent async requests (`Model.generate.remote.aio`) to generate image batches simultaneously.
- **FastAPI Web Endpoint**: Includes a built-in HTTPS endpoint for direct web integration.

---

## File Structure

```
vision-model/
├── qwen_image_turbo.py   # Modal app definition, container image, model loading & endpoints
├── generate.py           # Fast local CLI client to query deployed model without reloading
├── generate_batch.py     # Concurrent multi-image batch benchmark suite
├── PAPERCUTS.md          # Environment gotchas, resolution notes, and dependency fixes
├── archive/              # Archived legacy local deployment tools, GGUFs, and previous runs
│   ├── models/           # Local GGUF models & text encoders (Q4_K_M, mmproj)
│   ├── stable-diffusion.cpp/ # Local C++ inference engine build
│   ├── download_models.sh# Legacy GGUF downloader
│   ├── setup.sh          # Legacy local build script
│   ├── run_inference.sh  # Legacy local text-to-image runner
│   ├── edit_segmented.py # Legacy segmentation-aware pixel preservation editor
│   ├── colab_runner*.ipynb # Legacy Colab notebooks
│   └── *.png / *.json    # Test outputs & benchmark artifacts
```

---

## Quickstart

### 1. Prerequisites & Authentication

Install the Modal client in your Python environment and log in:

```bash
pip install modal
modal setup
```

Verify your active workspace profile:
```bash
modal profile current
```

> **Note:** Running GPU functions on Modal requires a payment method on file in your [Modal Settings](https://modal.com/settings/billing). The monthly **$30 free compute credit** applies first before any charges occur.

---

### 2. Run a One-Off Test

To run a single test prompt in an ephemeral container:

```bash
modal run qwen_image_turbo.py --prompt "a red fox in fresh snow, golden hour photo"
```
The resulting image will be downloaded and saved locally to `output.png`.

---

### 3. Interactive CLI Session (Warm Container)

To test multiple prompts back-to-back in your terminal without waiting for the model to reload between prompts:

```bash
modal run qwen_image_turbo.py --interactive
```

This starts a single persistent container on Modal, loads the model once, and lets you type prompts interactively:

```text
⚡ Interactive mode active — container will stay warm in GPU memory between prompts.
Type a prompt and press Enter. Type 'exit' or 'q' to stop.

Enter prompt > a cinematic retro sports car in neon rain
Generating image 1...
✓ Saved output_1.png in 3.42s

Enter prompt > an emerald hummingbird feeding from an orchid
Generating image 2...
✓ Saved output_2.png in 3.18s
```

---

### 4. Deploy as a Persistent Cloud Service

Deploy the app permanently to your Modal workspace:

```bash
modal deploy qwen_image_turbo.py
```

Once deployed, Modal gives you:
1. **A permanent HTTPS Web Endpoint**:
   ```
   https://<your-username>--qwen-image-turbo-model-web.modal.run?prompt=your+prompt+here
   ```
2. **A callable Python Class** that can be queried from any local script without redeploying.

---

### 5. Generate Images via Local Client (`generate.py`)

With the app deployed, use [`generate.py`](generate.py) to trigger generations instantly from your terminal:

```bash
# Basic generation
python3 generate.py "an astronaut floating in deep space next to a glowing nebula"

# Custom output file and seed
python3 generate.py "a majestic eagle flying over snowy mountains" --output eagle.png --seed 42
```

- **Cold Start (first request if idle >5 min)**: Boots container, mounts cache, loads into VRAM, and generates (~10–12s total).
- **Hot Request (within 5 minutes of previous request)**: **~2.9s inference (~8s total roundtrip)** with **zero model reloading**.

---

## Parallel Generation & Benchmark

Modal natively supports concurrent fan-out across GPU containers. Run the benchmark suite to generate 5 complex images in parallel:

```bash
python3 generate_batch.py
```

### Verified Benchmark Performance

| # | Prompt / Concept | Latency | Resolution | File Size |
|---|---|---|---|---|
| **1** | Ancient Library (Vaulted ceiling, fireplace) | **9.58s** | 1024×1024 RGBA | 2.1 MB |
| **2** | Chameleon Macro (Scales, bokeh, NatGeo) | **12.24s** | 1024×1024 RGBA | 1.7 MB |
| **3** | Cyberpunk Hypercar (Neon city, wet asphalt) | **15.82s** | 1024×1024 RGBA | 1.8 MB |
| **4** | Tokyo Ramen Shop (Rainy night, lanterns) | **19.96s** | 1024×1024 RGBA | 1.6 MB |
| **5** | Cliffside Castle (Sea of clouds, golden hour) | **24.25s** | 1024×1024 RGBA | 1.8 MB |

- **Total Wall-Clock Time (All 5 Images):** **24.25 seconds**
- **Effective Throughput:** **~12.4 images / minute** (~0.21 images / second)
- **Fastest Single Image:** **9.58 seconds**
- **Median Latency:** **15.82 seconds**

---

## Cost & Economics (Nvidia L40S)

Modal charges by the exact second of compute used.

| Component | Pricing Rate | Cost Impact |
|---|---|---|
| **Nvidia L40S (48 GB)** | **$1.95 / hour** ($0.00054 / sec) | **~$0.0022 per image** (~460 images / $1.00) |
| **Idle Warm Container** | 5-minute scaledown window | ~$0.16 max to keep GPU warm between requests |
| **Volume Storage (`hf-cache`)** | $0.09 / GiB / month | ~$0.06 / day for ~20 GB model cache |
| **Free Starter Credit** | $30.00 / month | Covers **~13,000+ image generations** per month |

---

## Technical Architecture Details

### Model & Pipeline Configuration
- **Base Pipeline:** `Qwen/Qwen-Image-2.1` using `QwenImage21Pipeline` (`diffusers>=0.37.0.dev0`).
- **Text & Conditioning Encoder:** `Qwen3VLProcessor` + `Qwen3VLForConditionalGeneration` (`transformers>=5.17`). Requires `torchvision` and `sentencepiece`.
- **LoRA Distillation:** `Viggle/Qwen-Image-2.1-viggle-turbo` (`Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors`, 1.3 GB).
- **FlowMatch Scheduler:** `FlowMatchEulerDiscreteScheduler` loaded directly from Viggle's `subfolder="scheduler"` with explicit 6-step sigmas:
  ```python
  STEPS = 6
  SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
  ```
- **Precision:** `torch.bfloat16` throughout DiT and text encoder for high fidelity and reduced VRAM footprint.

---

## Archived Legacy Workflows

All previous local deployment tooling—including GGUF models, the C++ `stable-diffusion.cpp` engine, and the segmentation-aware pixel preservation editor—have been preserved in [`archive/`](archive/):

- **Local C++ Engine & Scripts:** [`archive/setup.sh`](archive/setup.sh), [`archive/run_inference.sh`](archive/run_inference.sh), [`archive/edit_image.sh`](archive/edit_image.sh), [`archive/download_models.sh`](archive/download_models.sh).
- **Segmentation-Aware Editor:** [`archive/edit_segmented.py`](archive/edit_segmented.py) and [`archive/benchmark_preservation.py`](archive/benchmark_preservation.py).
- **Google Colab Notebooks:** [`archive/colab_runner.ipynb`](archive/colab_runner.ipynb) and [`archive/colab_runner_drive.ipynb`](archive/colab_runner_drive.ipynb).
- **Previs Storyboard Pipeline:** [`archive/previs.py`](archive/previs.py) and [`archive/benchmark_previs.py`](archive/benchmark_previs.py).
- **Local GGUF Weights & Engine:** [`archive/models/`](archive/models/) and [`archive/stable-diffusion.cpp/`](archive/stable-diffusion.cpp/).
