#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Setting up Qwen-Image-2.1 Standalone Runner ==="

# Check for cmake
if ! command -v cmake &> /dev/null; then
  echo "Error: cmake is required to build stable-diffusion.cpp."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Install on macOS via: brew install cmake"
  else
    echo "Install on Linux / Colab via: sudo apt-get update && sudo apt-get install -y cmake build-essential"
  fi
  exit 1
fi

# Clone stable-diffusion.cpp if not present
if [ ! -d "$SCRIPT_DIR/stable-diffusion.cpp" ]; then
  echo "Cloning leejet/stable-diffusion.cpp..."
  git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git "$SCRIPT_DIR/stable-diffusion.cpp"
else
  echo "stable-diffusion.cpp already present."
fi

# Detect platform and GPU backend
CMAKE_FLAGS=()
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Platform: macOS detected. Enabling Apple Silicon Metal (-DSD_METAL=ON)..."
  CMAKE_FLAGS+=("-DSD_METAL=ON")
elif command -v nvcc &> /dev/null || command -v nvidia-smi &> /dev/null; then
  echo "Platform: Linux / NVIDIA GPU detected (CUDA). Enabling CUDA (-DSD_CUDA=ON)..."
  CMAKE_FLAGS+=("-DSD_CUDA=ON")
else
  echo "Platform: Generic CPU. Building without GPU acceleration..."
fi

echo "Configuring and compiling stable-diffusion.cpp..."
cd "$SCRIPT_DIR/stable-diffusion.cpp"
cmake -B build "${CMAKE_FLAGS[@]}"
cmake --build build --config Release -j
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/models/diffusion_models"
mkdir -p "$SCRIPT_DIR/models/text_encoders"
mkdir -p "$SCRIPT_DIR/models/vae"
mkdir -p "$SCRIPT_DIR/output"

echo ""
echo "=== Setup complete! ==="
echo "Next step: Run ./download_models.sh to fetch the required GGUF weights."
