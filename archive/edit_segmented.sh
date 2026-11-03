#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

INPUT_IMAGE="${1:-$SCRIPT_DIR/assets/realistic_craftsman.png}"
TARGET="${2:-the hand saw and wooden block}"
PROMPT="${3:-Replace with a chef knife cutting a chocolate cake}"
OUTPUT="${4:-$SCRIPT_DIR/output/segmented_craftsman.png}"

EXPAND_PIXELS="${EXPAND_PIXELS:-16}"
FEATHER_PIXELS="${FEATHER_PIXELS:-4}"
STEPS="${STEPS:-4}"
CFG="${CFG:-1.0}"
SEED="${SEED:-42}"
PREVIEW="${PREVIEW:-false}"
MODEL="${MODEL:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

mkdir -p "$(dirname "$OUTPUT")"

CMD=(
  "$VENV_PYTHON" "$SCRIPT_DIR/edit_segmented.py"
  --image "$INPUT_IMAGE"
  --target "$TARGET"
  --instruction "$PROMPT"
  --output "$OUTPUT"
  --expand-pixels "$EXPAND_PIXELS"
  --feather-pixels "$FEATHER_PIXELS"
  --steps "$STEPS"
  --cfg "$CFG"
  --seed "$SEED"
)

if [ -n "$NEGATIVE_PROMPT" ]; then
  CMD+=(--negative-prompt "$NEGATIVE_PROMPT")
fi

if [ -n "$MODEL" ]; then
  CMD+=(--model "$MODEL")
fi

if [ "$PREVIEW" = "true" ]; then
  CMD+=(--preview-masks)
fi

"${CMD[@]}"
