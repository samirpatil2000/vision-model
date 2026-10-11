# Viggle Qwen-Image-2.1 Turbo (GGUF) - Standalone Metal Runner

A standalone, ComfyUI-free environment for running **Viggle Qwen-Image-2.1 Turbo** GGUF on Apple Silicon (M-series) Macs using native Metal acceleration. Supports both **Text-to-Image** and **Image Editing** (Image-to-Image).

---

## Showcase

### 1. Photorealistic Generation (1024×1024, 4 Steps)
<p align="center">
  <img src="assets/realistic_craftsman.png" width="600" alt="Photorealistic Craftsman" />
  <br />
  <em>"A raw 35mm photograph of an elderly craftsman carving wood in a sunlit workshop in Florence, fine skin wrinkles and realistic texture, natural silver hair, sawdust in the air, warm directional golden morning sunlight through dusty windows, shallow depth of field, 85mm f/1.4 portrait, photorealistic"</em>
</p>

### 2. Image Editing (Before vs After, 4 Steps)
<p align="center">
  <img src="assets/test_panda.png" width="380" alt="Original Red Panda" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/edited_panda.png" width="380" alt="Edited Red Panda" />
  <br />
  <em>Left: Original ("a cute red panda with aviator goggles") &nbsp;|&nbsp; Right: Edited ("Change the aviator goggles to futuristic glowing blue sunglasses")</em>
</p>

---

## Architecture Overview

- **Diffusion Model**: [realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs](https://huggingface.co/realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs) (`Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf`, ~5.17 GB)
  - 4-step distilled FlowMatch DiT transformer fine-tuned by Viggle.
- **Text Encoder**: [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) (`Qwen3VL-8B-Instruct-Q4_K_M.gguf`, ~4.68 GB)
  - Native Qwen3-VL multimodal text encoder.
- **Vision Projector (for Image Editing)**: [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) (`mmproj-Qwen3VL-8B-Instruct-F16.gguf`, ~1.08 GB)
  - Multimodal projector encoding reference images for semantic image edits.
- **VAE**: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) (`qwen_image_2.1_vae_bf16.safetensors`, ~644 MB)
  - 4-channel RGBA VAE supporting native transparency.
- **Engine**: [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) compiled with `-DSD_METAL=ON` for Apple Silicon GPU/Unified Memory.

---

## Quickstart

### 1. Build Engine
```bash
./setup.sh
```
*Clones `stable-diffusion.cpp` and compiles with Apple Silicon Metal acceleration.*

### 2. Download Weights
```bash
./download_models.sh
```
*Downloads the DiT GGUF, text encoder, vision projector, and VAE into `models/`.*

### 3. Generate Image (Text-to-Image)
```bash
# Default quick run (1024x1024, 4 steps)
./run_inference.sh

# Custom prompt
./run_inference.sh "A cyberpunk street market at night with neon signs and glowing rain reflections" "./output/cyberpunk.png"

# Custom resolution & parameters
WIDTH=1024 HEIGHT=1024 STEPS=4 CFG=1.0 ./run_inference.sh "Portrait of an astronaut on Mars" "./output/mars.png"
```

### 4. Edit Existing Image (Image-to-Image)
```bash
# Usage: ./edit_image.sh <input_image> <prompt> [output_image]
./edit_image.sh "./output/test_panda.png" "Change the aviator goggles to futuristic glowing blue sunglasses" "./output/edited_panda.png"
```

---

## Recommended Viggle Turbo Settings

| Setting | Recommended Value | Notes |
|---|---|---|
| **Steps** | `4` | Distilled 4-step student |
| **CFG Scale** | `1.0` | Locked at 1.0 (no classifier-free guidance needed) |
| **Sampler** | `euler` | FlowMatch Euler |
| **Scheduler** | `simple` | Simple linear timestep schedule |
| **Negative Prompt** | Empty / None | Turbo distilled models do not require negative prompts |
| **Resolutions** | Multiples of 32 | e.g. 512×512, 768×768, 1024×1024 |

---

## Speed Benchmarks (MacBook Pro Apple M5, 24 GB Unified Memory)

| Resolution | Total Latents | 4-Step Sampling Time | Total Run Time |
|---|---|---|---|
| **512 × 512** | 262,144 px | ~25.7s (~3.9s / step) | **~37 seconds** |
| **1024 × 1024** | 1,048,576 px | ~101.7s (~23.5s / step) | **~2.1 minutes** |
