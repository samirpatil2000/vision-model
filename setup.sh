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
elif command -v nvcc &> /dev/null || command -v nvidia-smi &> /dev/null || [ -d "/usr/local/cuda" ]; then
  # Ensure CUDA toolchain is in PATH
  if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  fi
  if [ -d "/usr/local/cuda" ]; then
    export CUDA_PATH="/usr/local/cuda"
    export CUDACXX="/usr/local/cuda/bin/nvcc"
  fi

  echo "Platform: Linux / NVIDIA GPU detected (CUDA). Enabling CUDA (-DSD_CUDA=ON)..."
  CMAKE_FLAGS+=("-DSD_CUDA=ON")

  # Detect specific GPU compute capability (e.g. 75 for T4, 80 for A100, 89 for L4)
  # This speeds up build time by 5x-10x and avoids memory spikes on Colab
  if command -v nvidia-smi &> /dev/null; then
    GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.' | tr -d ' ' || echo "")
    if [ -n "$GPU_CAP" ]; then
      echo "Targeting detected GPU compute capability: $GPU_CAP"
      CMAKE_FLAGS+=("-DCMAKE_CUDA_ARCHITECTURES=$GPU_CAP")
    fi
  fi
else
  echo "Platform: Generic CPU. Building without GPU acceleration..."
fi

echo "Configuring and compiling stable-diffusion.cpp..."
cd "$SCRIPT_DIR/stable-diffusion.cpp"
cmake -B build "${CMAKE_FLAGS[@]}"

# Use bounded core count to prevent OOM killer on cloud VMs
NUM_CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)
cmake --build build --config Release -j "$NUM_CORES"
cd "$SCRIPT_DIR"

# Verify binary was created
SD_CLI="$SCRIPT_DIR/stable-diffusion.cpp/build/bin/sd-cli"
if [ ! -f "$SD_CLI" ]; then
  echo ""
  echo "❌ Error: Build completed but binary was not found at $SD_CLI"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/models/diffusion_models"
mkdir -p "$SCRIPT_DIR/models/text_encoders"
mkdir -p "$SCRIPT_DIR/models/vae"
mkdir -p "$SCRIPT_DIR/output"

echo ""
echo "=== Setup complete! ==="
echo "✓ Verified binary: $SD_CLI"
echo "Next step: Run ./download_models.sh to fetch the required GGUF weights."
