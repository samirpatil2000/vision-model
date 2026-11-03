"""
stable-diffusion.cpp engine clients.

make_engine("cli") runs one sd-cli process per request; make_engine("server") keeps one
sd-server resident. Both implement EngineClient.generate(GenerationRequest).
"""
from src.engine.base import EngineClient, EngineError, GenerationRequest, GenerationResult
from src.engine.sd_cli_client import SdCliClient
from src.engine.sd_server_client import SdServerClient

__all__ = [
    "EngineClient", "EngineError", "GenerationRequest", "GenerationResult",
    "SdCliClient", "SdServerClient", "make_engine",
]


def make_engine(kind: str = "cli", allow_mismatch: bool = False, **kw) -> EngineClient:
    """Engine client by name: "cli" (SdCliClient) or "server" (SdServerClient).

    allow_mismatch lets a binary built from a commit other than engine/UPSTREAM_SHA run.
    """
    key = kind.strip().lower()
    if key in ("cli", "sd-cli", "sd_cli"):
        return SdCliClient(allow_mismatch=allow_mismatch, **kw)
    if key in ("server", "sd-server", "sd_server"):
        return SdServerClient(allow_mismatch=allow_mismatch, **kw)
    raise ValueError(f"Unknown engine kind {kind!r}; expected 'cli' or 'server'")
