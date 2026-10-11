#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Setting up Viggle Qwen-Image-2.1 Turbo (GGUF) Standalone Runner ==="

# Check for cmake
if ! command -v cmake &> /dev/null; then
  echo "Error: cmake is required to build stable-diffusion.cpp."
  echo "Install it on macOS via: brew install cmake"
  exit 1
fi

# Clone stable-diffusion.cpp if not present
if [ ! -d "$SCRIPT_DIR/stable-diffusion.cpp" ]; then
  echo "Cloning leejet/stable-diffusion.cpp..."
  git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git "$SCRIPT_DIR/stable-diffusion.cpp"
else
  echo "stable-diffusion.cpp already present."
fi

# Build with Metal acceleration
echo "Building stable-diffusion.cpp with Apple Silicon Metal acceleration..."
cd "$SCRIPT_DIR/stable-diffusion.cpp"
cmake -B build -DSD_METAL=ON
cmake --build build --config Release -j
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/models/diffusion_models"
mkdir -p "$SCRIPT_DIR/models/text_encoders"
mkdir -p "$SCRIPT_DIR/models/vae"
mkdir -p "$SCRIPT_DIR/output"

echo ""
echo "=== Setup complete! ==="
echo "Next step: Run ./download_models.sh to fetch the required GGUF weights."
