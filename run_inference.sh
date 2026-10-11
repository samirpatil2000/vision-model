#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_CLI="$SCRIPT_DIR/stable-diffusion.cpp/build/bin/sd-cli"
MODEL="$SCRIPT_DIR/models/diffusion_models/Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf"
TEXT_ENCODER="$SCRIPT_DIR/models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
VAE="$SCRIPT_DIR/models/vae/qwen_image_2.1_vae_bf16.safetensors"

PROMPT="${1:-A futuristic cybernetic tiger prowling through a neon-lit rain-soaked alley at night, ultra detailed, cinematic lighting}"
OUTPUT="${2:-$SCRIPT_DIR/output/viggle_turbo_output.png}"
STEPS="${STEPS:-4}"
CFG="${CFG:-1.0}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
SAMPLER="${SAMPLER:-euler}"
SCHEDULE="${SCHEDULE:-simple}"
SEED="${SEED:-42}"

mkdir -p "$SCRIPT_DIR/output"

echo "=========================================================="
echo " Running Viggle Qwen-Image-2.1 Turbo (GGUF) via sd-cli"
echo "=========================================================="
echo " Prompt:   $PROMPT"
echo " Output:   $OUTPUT"
echo " Steps:    $STEPS (Viggle 4-step distilled)"
echo " CFG:      $CFG"
echo " Size:     ${WIDTH}x${HEIGHT}"
echo " Sampler:  $SAMPLER"
echo " Schedule: $SCHEDULE"
echo " Seed:     $SEED"
echo " Backend:  Apple Silicon Metal (MPS / GGML)"
echo "=========================================================="

"$SD_CLI" \
  --diffusion-model "$MODEL" \
  --llm "$TEXT_ENCODER" \
  --vae "$VAE" \
  -p "$PROMPT" \
  --steps "$STEPS" \
  --cfg-scale "$CFG" \
  --sampling-method "$SAMPLER" \
  --scheduler "$SCHEDULE" \
  -W "$WIDTH" \
  -H "$HEIGHT" \
  -s "$SEED" \
  -o "$OUTPUT"

echo "=========================================================="
echo " Finished! Image saved to: $OUTPUT"
echo "=========================================================="
