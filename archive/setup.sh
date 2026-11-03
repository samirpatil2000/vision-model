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

ENGINE_DIR="$SCRIPT_DIR/stable-diffusion.cpp"
PINNED_SHA=""
if [ -f "$SCRIPT_DIR/engine/UPSTREAM_SHA" ]; then
  PINNED_SHA="$(tr -d '[:space:]' < "$SCRIPT_DIR/engine/UPSTREAM_SHA")"
fi

# True when the engine dir is its own git checkout. Without its own .git, `git -C "$ENGINE_DIR"`
# walks up to this repo and would report (or fetch/check out) vision-model commits as the engine's.
engine_is_checkout() {
  [ -e "$ENGINE_DIR/.git" ]
}

checkout_pinned_engine() {
  if ! git -C "$ENGINE_DIR" cat-file -e "${PINNED_SHA}^{commit}" 2>/dev/null; then
    git -C "$ENGINE_DIR" fetch origin
  fi
  git -C "$ENGINE_DIR" checkout --quiet "$PINNED_SHA"
  git -C "$ENGINE_DIR" submodule update --init --recursive
}

# Clone stable-diffusion.cpp if not present, pinned to engine/UPSTREAM_SHA
if [ ! -d "$ENGINE_DIR" ]; then
  echo "Cloning leejet/stable-diffusion.cpp..."
  git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git "$ENGINE_DIR"
  if [ -n "$PINNED_SHA" ]; then
    echo "Pinning stable-diffusion.cpp to $PINNED_SHA (engine/UPSTREAM_SHA)..."
    checkout_pinned_engine
  fi
elif engine_is_checkout; then
  echo "stable-diffusion.cpp already present."
  CURRENT_SHA="$(git -C "$ENGINE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [ -n "$PINNED_SHA" ] && [ "$CURRENT_SHA" != "$PINNED_SHA" ]; then
    echo "⚠️  Warning: stable-diffusion.cpp is at $CURRENT_SHA but engine/UPSTREAM_SHA pins $PINNED_SHA."
    if [ "${PIN_ENGINE:-0}" = "1" ]; then
      echo "PIN_ENGINE=1: checking out the pinned commit..."
      checkout_pinned_engine
    else
      echo "   Building the current checkout. Re-run with PIN_ENGINE=1 to build the pinned commit."
    fi
  fi
else
  echo "stable-diffusion.cpp already present, but it is not a git checkout."
  if [ ! -f "$ENGINE_DIR/CMakeLists.txt" ]; then
    echo "❌ Error: $ENGINE_DIR holds no stable-diffusion.cpp sources (no CMakeLists.txt)."
    echo "   Remove it and re-run ./setup.sh to clone the pinned engine."
    exit 1
  fi
  if [ "${PIN_ENGINE:-0}" = "1" ]; then
    echo "❌ Error: PIN_ENGINE=1 needs a git checkout to pin. Remove $ENGINE_DIR and re-run ./setup.sh to clone it."
    exit 1
  fi
  echo "⚠️  Warning: cannot verify these sources match engine/UPSTREAM_SHA ($PINNED_SHA); building them as they are."
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

# Record what was built (read by src/engine/version.read_build_info for run provenance);
# sources that are not their own git checkout have an unknown commit and dirty state
BUILD_COMMIT=unknown
BUILD_DIRTY=null
if engine_is_checkout; then
  BUILD_COMMIT="$(git -C "$ENGINE_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  BUILD_DIRTY=false
  if [ -n "$(git -C "$ENGINE_DIR" status --porcelain --untracked-files=no 2>/dev/null || true)" ]; then
    BUILD_DIRTY=true
  fi
fi
BINARY_VERSION="$("$SD_CLI" --version 2>/dev/null | head -n 1 | tr -d '"\\' || true)"
cat > "$ENGINE_DIR/build/build-info.json" <<EOF
{
  "commit": "$BUILD_COMMIT",
  "pinned_sha": "$PINNED_SHA",
  "dirty": $BUILD_DIRTY,
  "build_time_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "binary_version": "$BINARY_VERSION",
  "cmake_flags": "${CMAKE_FLAGS[*]:-}",
  "platform": "$(uname -s)-$(uname -m)"
}
EOF

mkdir -p "$SCRIPT_DIR/models/diffusion_models"
mkdir -p "$SCRIPT_DIR/models/text_encoders"
mkdir -p "$SCRIPT_DIR/models/vae"
mkdir -p "$SCRIPT_DIR/output"

echo ""
echo "=== Setup complete! ==="
echo "✓ Verified binary: $SD_CLI"
echo "Next step: Run ./download_models.sh to fetch the required GGUF weights."
