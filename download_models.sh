#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/models/diffusion_models"
mkdir -p "$SCRIPT_DIR/models/text_encoders"
mkdir -p "$SCRIPT_DIR/models/vae"

VARIANT="${1:---turbo-q4}"

echo "=== Downloading Model Weights ==="
echo "Target Variant: $VARIANT"

# Check for hf CLI
if command -v hf &> /dev/null; then
  HF_CMD="hf download"
elif [ -f "$SCRIPT_DIR/.venv/bin/hf" ]; then
  HF_CMD="$SCRIPT_DIR/.venv/bin/hf download"
else
  echo "Installing huggingface_hub..."
  python3 -m pip install --upgrade huggingface_hub
  HF_CMD="hf download"
fi

# 1. Diffusion Model
case "$VARIANT" in
  --turbo-q4)
    echo "Downloading Viggle Turbo DiT (Q4_K_M, ~5.2 GB)..."
    $HF_CMD realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs \
      Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf --local-dir "$SCRIPT_DIR/models/diffusion_models"
    ;;
  --turbo-q8)
    echo "Downloading Viggle Turbo DiT (Q8_0, ~7.2 GB)..."
    $HF_CMD realrebelai/Viggle_Qwen-Image-2.1-Turbo_GGUFs \
      Qwen-Image-2.1-viggle-turbo-Q8_0.gguf --local-dir "$SCRIPT_DIR/models/diffusion_models"
    ;;
  --base|--base-q4)
    echo "Downloading Base Qwen-Image 2.1 DiT (Q4, ~5.5 GB, 30-50 steps)..."
    $HF_CMD realrebelai/Qwen-Image-2.1_GGUFs \
      Qwen-Image-2.1-Q4.gguf --local-dir "$SCRIPT_DIR/models/diffusion_models"
    # Create alias for compatibility with scripts looking for Q4_K_M
    (cd "$SCRIPT_DIR/models/diffusion_models" && ln -sf Qwen-Image-2.1-Q4.gguf Qwen-Image-2.1-Q4_K_M.gguf)
    ;;
  --base-q8)
    echo "Downloading Base Qwen-Image 2.1 DiT (Q8, ~7.2 GB, 30-50 steps)..."
    $HF_CMD realrebelai/Qwen-Image-2.1_GGUFs \
      Qwen-Image-2.1-Q8.gguf --local-dir "$SCRIPT_DIR/models/diffusion_models"
    (cd "$SCRIPT_DIR/models/diffusion_models" && ln -sf Qwen-Image-2.1-Q8.gguf Qwen-Image-2.1-Q8_0.gguf)
    ;;
  *)
    echo "Unknown variant: $VARIANT"
    echo "Options: --turbo-q4 (default), --turbo-q8, --base-q4, --base-q8"
    exit 1
    ;;
esac

# 2. Text Encoder & Vision Projector (Qwen3-VL)
if [ ! -f "$SCRIPT_DIR/models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf" ]; then
  echo "Downloading Qwen3-VL Text Encoder (Q4_K_M, ~4.7 GB)..."
  $HF_CMD Qwen/Qwen3-VL-8B-Instruct-GGUF \
    Qwen3VL-8B-Instruct-Q4_K_M.gguf --local-dir "$SCRIPT_DIR/models/text_encoders"
fi

if [ ! -f "$SCRIPT_DIR/models/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf" ]; then
  echo "Downloading Qwen3-VL Vision Projector (F16, ~1.1 GB)..."
  $HF_CMD Qwen/Qwen3-VL-8B-Instruct-GGUF \
    mmproj-Qwen3VL-8B-Instruct-F16.gguf --local-dir "$SCRIPT_DIR/models/text_encoders"
fi

# 3. VAE (RGBA)
if [ ! -f "$SCRIPT_DIR/models/vae/qwen_image_2.1_vae_bf16.safetensors" ]; then
  echo "Downloading Qwen-Image 2.1 VAE (~644 MB)..."
  curl -L -C - --retry 5 -o "$SCRIPT_DIR/models/vae/qwen_image_2.1_vae_bf16.safetensors" \
    https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors
fi

# Clean up any cache directories
rm -rf "$SCRIPT_DIR/models/diffusion_models/.cache" "$SCRIPT_DIR/models/text_encoders/.cache"

echo ""
echo "=== Weights downloaded and verified! ==="
