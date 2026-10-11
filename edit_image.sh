#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SD_CLI="$SCRIPT_DIR/stable-diffusion.cpp/build/bin/sd-cli"

MODEL="${MODEL:-$SCRIPT_DIR/models/diffusion_models/Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf}"
TEXT_ENCODER="${TEXT_ENCODER:-$SCRIPT_DIR/models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf}"
VISION_PROJECTOR="${VISION_PROJECTOR:-$SCRIPT_DIR/models/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf}"
VAE="${VAE:-$SCRIPT_DIR/models/vae/qwen_image_2.1_vae_bf16.safetensors}"

INPUT_IMAGE="${1:-$SCRIPT_DIR/output/test_panda.png}"
PROMPT="${2:-Change the aviator goggles to futuristic glowing blue sunglasses}"
OUTPUT="${3:-$SCRIPT_DIR/output/edited_output.png}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
STEPS="${STEPS:-4}"
CFG="${CFG:-1.0}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
SAMPLER="${SAMPLER:-euler}"
SCHEDULE="${SCHEDULE:-simple}"
SEED="${SEED:-42}"
FLASH_ATTN="${FLASH_ATTN:-false}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================================="
echo " Editing Image via Qwen-Image-2.1 (GGUF)"
echo "=========================================================="
echo " Model:    $(basename "$MODEL")"
echo " Input:    $INPUT_IMAGE"
echo " Prompt:   $PROMPT"
if [ -n "$NEGATIVE_PROMPT" ]; then
  echo " Negative: $NEGATIVE_PROMPT"
fi
echo " Output:   $OUTPUT"
echo " Steps:    $STEPS"
echo " CFG:      $CFG"
echo " Size:     ${WIDTH}x${HEIGHT}"
echo " Sampler:  $SAMPLER"
echo " Schedule: $SCHEDULE"
echo " Seed:     $SEED"
echo "=========================================================="

CMD_ARGS=(
  --diffusion-model "$MODEL"
  --llm "$TEXT_ENCODER"
  --llm_vision "$VISION_PROJECTOR"
  --vae "$VAE"
  -r "$INPUT_IMAGE"
  -p "$PROMPT"
  --steps "$STEPS"
  --cfg-scale "$CFG"
  --sampling-method "$SAMPLER"
  --scheduler "$SCHEDULE"
  -W "$WIDTH"
  -H "$HEIGHT"
  -s "$SEED"
  -o "$OUTPUT"
)

if [ -n "$NEGATIVE_PROMPT" ]; then
  CMD_ARGS+=(-n "$NEGATIVE_PROMPT")
fi

if [ "$FLASH_ATTN" = "true" ]; then
  CMD_ARGS+=(--fa)
fi

"$SD_CLI" "${CMD_ARGS[@]}"

echo "=========================================================="
echo " Finished! Edited image saved to: $OUTPUT"
echo "=========================================================="
