#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_CLI="$SCRIPT_DIR/stable-diffusion.cpp/build/bin/sd-cli"
MODEL="$SCRIPT_DIR/models/diffusion_models/Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf"
TEXT_ENCODER="$SCRIPT_DIR/models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
VISION_PROJECTOR="$SCRIPT_DIR/models/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf"
VAE="$SCRIPT_DIR/models/vae/qwen_image_2.1_vae_bf16.safetensors"

INPUT_IMAGE="${1:-$SCRIPT_DIR/output/test_panda.png}"
PROMPT="${2:-Change the aviator goggles to futuristic glowing blue sunglasses}"
OUTPUT="${3:-$SCRIPT_DIR/output/edited_output.png}"
STEPS="${STEPS:-4}"
CFG="${CFG:-1.0}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
SAMPLER="${SAMPLER:-euler}"
SCHEDULE="${SCHEDULE:-simple}"
SEED="${SEED:-42}"

mkdir -p "$SCRIPT_DIR/output"

echo "=========================================================="
echo " Editing Image via Viggle Qwen-Image-2.1 Turbo (GGUF)"
echo "=========================================================="
echo " Input:    $INPUT_IMAGE"
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
  --llm_vision "$VISION_PROJECTOR" \
  --vae "$VAE" \
  -r "$INPUT_IMAGE" \
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
echo " Finished! Edited image saved to: $OUTPUT"
echo "=========================================================="
