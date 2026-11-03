"""
Engine abstraction over stable-diffusion.cpp.

EngineClient is the only place that knows how to talk to the engine. SdCliClient runs one
sd-cli process per request (weights reload each call); SdServerClient keeps one sd-server
resident and posts requests to it over loopback.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class EngineError(RuntimeError):
    """Raised when the engine fails, times out, or reports a memory fallback in strict mode."""
    pass


@dataclass
class GenerationRequest:
    prompt: str
    output_path: str
    width: int
    height: int
    seed: int
    steps: int = 6
    cfg_scale: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    sigmas: Optional[List[float]] = None          # overrides steps/scheduler when set
    ref_images: List[str] = field(default_factory=list)   # ordered: image1, image2, ...
    init_image: Optional[str] = None
    mask_image: Optional[str] = None
    strength: float = 0.75                         # noise level when strength_as_noise_level is on
    strength_as_noise_level: bool = True
    negative_prompt: str = ""
    use_base_model: bool = False
    loras: Dict[str, float] = field(default_factory=dict)   # name -> multiplier
    tag: str = ""                                  # free-form label for logs/provenance


@dataclass
class GenerationResult:
    image_path: str
    seconds: float
    command: List[str]
    engine_commit: str
    peak_rss_bytes: Optional[int] = None
    phase_seconds: Dict[str, float] = field(default_factory=dict)
    fallback_warnings: List[str] = field(default_factory=list)   # FA / prefix-cache fallback log lines
    log_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class EngineClient(ABC):
    """Generates one image per request. Implementations must be safe to call sequentially."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        """Release engine resources (stop servers, free memory)."""
        return None

    def __enter__(self) -> "EngineClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
