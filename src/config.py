"""
Single source of truth for paths and generation defaults.

All paths resolve from the repository root, never from the current working
directory, so scripts behave the same no matter where they are launched from.
Environment variables override the defaults (see README "Configuration Reference").
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


ENGINE_DIR = _env_path("SD_CPP_DIR", REPO_ROOT / "stable-diffusion.cpp")
SD_CLI = _env_path("SD_CLI", ENGINE_DIR / "build" / "bin" / "sd-cli")
SD_SERVER = _env_path("SD_SERVER", ENGINE_DIR / "build" / "bin" / "sd-server")
UPSTREAM_SHA_FILE = REPO_ROOT / "engine" / "UPSTREAM_SHA"

MODELS_DIR = _env_path("MODELS_DIR", REPO_ROOT / "models")
TURBO_MODEL = _env_path("MODEL", MODELS_DIR / "diffusion_models" / "Qwen-Image-2.1-viggle-turbo-Q4_K_M.gguf")
BASE_MODEL = _env_path("BASE_MODEL", MODELS_DIR / "diffusion_models" / "qwen_image_2.1-Q4_K.gguf")
LLM_MODEL = _env_path("LLM_MODEL", MODELS_DIR / "text_encoders" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
LLM_VISION = _env_path("LLM_VISION", MODELS_DIR / "text_encoders" / "mmproj-Qwen3VL-8B-Instruct-F16.gguf")
VAE_MODEL = _env_path("VAE_MODEL", MODELS_DIR / "vae" / "qwen_image_2.1_vae_bf16.safetensors")

OUTPUT_DIR = _env_path("OUTPUT_DIR", REPO_ROOT / "output")
PREVIS_OUTPUT_DIR = OUTPUT_DIR / "previs"

CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "claude")
BLENDER = _env_path("BLENDER", Path("/Applications/Blender.app/Contents/MacOS/Blender"))
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# Viggle Turbo v0.3 card sigma list (6 steps). The 4-step "simple" schedule is the A/B alternative.
VIGGLE_SIGMAS: Tuple[float, ...] = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25, 0.0)

MAX_VIDEO_SECONDS = 30.0
FPS = 24


@dataclass(frozen=True)
class Tier:
    """Resolution tier. Every size is a multiple of 32 (Qwen-Image VAE f=16, vision token = 32 px)."""
    name: str
    width: int
    height: int
    sheet_size: Tuple[int, int]
    continuity_size: Tuple[int, int]


TIERS: Dict[str, Tier] = {
    "draft": Tier("draft", 512, 288, (512, 256), (384, 224)),
    "final": Tier("final", 768, 448, (768, 384), (512, 288)),
    "hero": Tier("hero", 1024, 576, (1024, 512), (512, 288)),
}


@dataclass
class GenerationDefaults:
    steps: int = int(os.environ.get("STEPS", "6"))
    cfg_scale: float = float(os.environ.get("CFG", "1.0"))
    sampler: str = os.environ.get("SAMPLER", "euler")
    scheduler: str = os.environ.get("SCHEDULE", "simple")
    use_custom_sigmas: bool = os.environ.get("USE_VIGGLE_SIGMAS", "true").lower() == "true"
    flash_attention: bool = os.environ.get("FLASH_ATTN", "true").lower() == "true"
    prefix_cache_type: str = os.environ.get("PREFIX_CACHE_TYPE", "q8_0")
    init_noise: float = float(os.environ.get("INIT_NOISE", "0.85"))
    max_retakes: int = int(os.environ.get("MAX_RETAKES", "2"))
    extra_model_flags: List[str] = field(default_factory=list)


def model_paths(base: bool = False) -> Dict[str, Path]:
    """Resolved model weight paths for the requested DiT variant."""
    return {
        "diffusion_model": BASE_MODEL if base else TURBO_MODEL,
        "llm": LLM_MODEL,
        "llm_vision": LLM_VISION,
        "vae": VAE_MODEL,
    }


def pinned_engine_sha() -> str:
    """The stable-diffusion.cpp commit this repo is pinned to (engine/UPSTREAM_SHA)."""
    return UPSTREAM_SHA_FILE.read_text().strip()
