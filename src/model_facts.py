"""
Real model facts read from the local weight-file headers. No tensor data is loaded.

    python -m src.model_facts                 # table
    python -m src.model_facts --json          # machine-readable
    python -m src.model_facts --models-dir /path/to/models

GGUF files are parsed with the `gguf` package (metadata + tensor-info table; the tensor data stays an
unread memory map). safetensors files are parsed with `safetensors.safe_open` (JSON header only).
GGUF tensor shapes are in ggml order: ne[0] is the INPUT-feature axis of a linear weight, ne[1] the output.
safetensors shapes are in PyTorch order: (out_channels, in_channels, *kernel).
GGUFReader parses all metadata eagerly, so the text encoder's ~152k-entry tokenizer vocab costs ~2 s and
~0.5 GB RAM; the DiT/VAE headers take well under 0.1 s each.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gguf
from safetensors import SafetensorError, safe_open

from src import config

_READ_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, SafetensorError)
_MAX_HPARAM_ITEMS = 64  # skip huge metadata arrays (tokenizer vocab, merges)

_DIT_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")
_VAE_DOWNSAMPLE_RE = re.compile(r"^encoder\.downsamples\.\d+\.downsamples\.\d+\.resample\.\d+\.weight$")
_VAE_HEAD_RE = re.compile(r"^decoder\.head\.(\d+)\.weight$")

# Normalised text-encoder field -> hparam key below the "<arch>." prefix (mmproj keys live under "vision.").
_TEXT_ENCODER_KEYS: Dict[str, str] = {
    "layers": "block_count",
    "hidden_size": "embedding_length",
    "ffn_size": "feed_forward_length",
    "heads": "attention.head_count",
    "kv_heads": "attention.head_count_kv",
    "head_dim": "attention.key_length",
    "context_length": "context_length",
    "rope_sections": "rope.dimension_sections",
    "projector_type": "projector_type",
    "projection_dim": "projection_dim",
    "image_size": "image_size",
    "patch_size": "patch_size",
    "spatial_merge_size": "spatial_merge_size",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dim(shapes: Dict[str, List[int]], name: str, axis: int) -> Optional[int]:
    """One dimension of a tensor shape, or None when the tensor/axis is absent."""
    shape = shapes.get(name)
    return shape[axis] if shape is not None and len(shape) > axis else None


def _dtype_histogram(items: Iterable[Tuple[str, int]]) -> Dict[str, Dict[str, int]]:
    """{dtype: {"tensors": n, "params": p}} from (dtype, n_elements) pairs, largest parameter share first."""
    hist: Dict[str, Dict[str, int]] = {}
    for dtype, params in items:
        bucket = hist.setdefault(dtype, {"tensors": 0, "params": 0})
        bucket["tensors"] += 1
        bucket["params"] += params
    return dict(sorted(hist.items(), key=lambda kv: (-kv[1]["params"], kv[0])))


def _gguf_field(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return None if field is None else field.contents()


def _file_type_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return gguf.LlamaFileType(int(value)).name
    except ValueError:
        return str(value)


def _gguf_common(reader: gguf.GGUFReader) -> Dict[str, Any]:
    """Architecture, quantisation and size facts shared by every GGUF file."""
    tensors = reader.tensors
    params = sum(int(t.n_elements) for t in tensors)
    tensor_bytes = sum(int(t.n_bytes) for t in tensors)
    return {
        "architecture": _gguf_field(reader, "general.architecture"),
        "file_type": _file_type_name(_gguf_field(reader, "general.file_type")),
        "tensor_count": len(tensors),
        "param_count": params,
        "tensor_bytes": tensor_bytes,
        "bits_per_weight": round(tensor_bytes * 8 / params, 3) if params else None,
        "dtypes": _dtype_histogram((t.tensor_type.name, int(t.n_elements)) for t in tensors),
    }


def _embedded_config(reader: gguf.GGUFReader) -> Dict[str, Any]:
    """The diffusers config JSON some converters embed under the "config" key ({} when absent/invalid)."""
    raw = _gguf_field(reader, "config")
    if not isinstance(raw, str):
        return {}
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(cfg, dict) and isinstance(cfg.get("transformer"), dict):
        cfg = cfg["transformer"]
    return cfg if isinstance(cfg, dict) else {}


# ---------------------------------------------------------------------------
# Per-file readers
# ---------------------------------------------------------------------------

def dit_facts(path: Path) -> Dict[str, Any]:
    """Qwen-Image DiT facts from a GGUF header. Tensor shapes are authoritative; an embedded config is cross-checked."""
    reader = gguf.GGUFReader(path, "r")
    shapes = {t.name: [int(d) for d in t.shape] for t in reader.tensors}
    cfg = _embedded_config(reader)

    hidden = _dim(shapes, "img_in.weight", 1)
    head_dim = _dim(shapes, "transformer_blocks.0.attn.norm_q.weight", 0)
    blocks = {int(m.group(1)) for name in shapes if (m := _DIT_BLOCK_RE.match(name))}
    facts: Dict[str, Any] = {
        **_gguf_common(reader),
        "class_name": cfg.get("_class_name"),
        "hidden_size": hidden,
        "num_layers": len(blocks) or None,
        "num_heads": hidden // head_dim if hidden and head_dim else None,
        "head_dim": head_dim,
        "mlp_hidden": _dim(shapes, "transformer_blocks.0.img_mlp.out.weight", 0),
        "in_channels": _dim(shapes, "img_in.weight", 0),
        "out_channels": _dim(shapes, "proj_out.weight", 1),
        "context_in_dim": _dim(shapes, "txt_in.in_layer.weight", 0),
        "patch_size": cfg.get("patch_size"),
        "patch_size_source": "config" if cfg.get("patch_size") is not None else None,
        "rope_axes": cfg.get("axes_dims_rope"),
        "config": cfg or None,
    }
    facts["config_mismatches"] = _config_mismatches(facts, cfg)
    return facts


def _config_mismatches(facts: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    """Fields where the embedded config disagrees with the tensor shapes."""
    expected = {
        "num_layers": cfg.get("num_layers"),
        "num_heads": cfg.get("num_attention_heads"),
        "head_dim": cfg.get("attention_head_dim"),
        "in_channels": cfg.get("in_channels"),
        "out_channels": cfg.get("out_channels"),
        "context_in_dim": cfg.get("context_in_dim"),
    }
    if cfg.get("num_attention_heads") and cfg.get("attention_head_dim"):
        expected["hidden_size"] = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    if cfg.get("mlp_ratio") and facts["hidden_size"]:
        expected["mlp_hidden"] = int(cfg["mlp_ratio"] * facts["hidden_size"])
    mismatches = [f"{key}: tensors {facts[key]} != config {value}" for key, value in expected.items()
                  if value is not None and facts[key] is not None and facts[key] != value]
    rope = cfg.get("axes_dims_rope")
    if rope and facts["head_dim"] and sum(rope) != facts["head_dim"]:
        mismatches.append(f"rope_axes: sum {sum(rope)} != head_dim {facts['head_dim']}")
    return mismatches


def vae_facts(path: Path) -> Dict[str, Any]:
    """VAE facts from a safetensors JSON header (Wan-style VAE layout used by Qwen-Image).

    Strides are not stored in the header: spatial_downsample = 2 ** (number of encoder `resample` convs),
    each of which is a stride-2 downsample in this layout. latent_channels = decoder.conv1 input channels.
    """
    with safe_open(str(path), framework="numpy") as handle:
        metadata = handle.metadata() or {}
        shapes: Dict[str, List[int]] = {}
        dtypes: List[Tuple[str, int]] = []
        for name in handle.keys():
            tensor = handle.get_slice(name)
            shapes[name] = list(tensor.get_shape())
            dtypes.append((tensor.get_dtype(), math.prod(shapes[name])))

    has_encoder = any(name.startswith("encoder.") for name in shapes)
    downsamples = sum(1 for name in shapes if _VAE_DOWNSAMPLE_RE.match(name))
    heads = sorted(int(m.group(1)) for name in shapes if (m := _VAE_HEAD_RE.match(name)))
    latent = _dim(shapes, "decoder.conv1.weight", 1)
    if latent is None:
        latent = _dim(shapes, "conv2.weight", 0)
    params = sum(p for _, p in dtypes)
    return {
        "architecture": metadata.get("modelspec.architecture"),
        "tensor_count": len(shapes),
        "param_count": params,
        "dtypes": _dtype_histogram(dtypes),
        "input_channels": _dim(shapes, "encoder.conv1.weight", 1),
        "output_channels": _dim(shapes, f"decoder.head.{heads[-1]}.weight", 0) if heads else None,
        "latent_channels": latent,
        "spatial_downsample": 2 ** downsamples if has_encoder else None,
        "metadata": metadata,
    }


def text_encoder_facts(path: Path) -> Dict[str, Any]:
    """Text-encoder LLM or vision projector (mmproj) facts from GGUF metadata."""
    reader = gguf.GGUFReader(path, "r")
    arch = _gguf_field(reader, "general.architecture")
    prefix = f"{arch}."
    hparams: Dict[str, Any] = {}
    for key, field in reader.fields.items():
        if arch and key.startswith(prefix) and len(field.data) <= _MAX_HPARAM_ITEMS:
            hparams[key[len(prefix):]] = field.contents()

    model_type = _gguf_field(reader, "general.type")
    facts: Dict[str, Any] = {
        **_gguf_common(reader),
        "kind": "mmproj" if model_type == "mmproj" or arch == "clip" else "llm",
        "name": _gguf_field(reader, "general.name"),
        "size_label": _gguf_field(reader, "general.size_label"),
    }
    for out_key, hp_key in _TEXT_ENCODER_KEYS.items():
        value = hparams.get(hp_key)
        facts[out_key] = value if value is not None else hparams.get(f"vision.{hp_key}")
    facts["hparams"] = hparams
    return facts


# ---------------------------------------------------------------------------
# Whole models directory
# ---------------------------------------------------------------------------

def _configured_roles() -> Dict[Path, List[str]]:
    """Resolved weight path -> config roles it is used as (turbo, base, llm, llm_vision, vae)."""
    roles: Dict[Path, List[str]] = {}
    for role, path in (("turbo", config.TURBO_MODEL), ("base", config.BASE_MODEL), ("llm", config.LLM_MODEL),
                       ("llm_vision", config.LLM_VISION), ("vae", config.VAE_MODEL)):
        roles.setdefault(Path(path).resolve(), []).append(role)
    return roles


def _read_entries(directory: Path, pattern: str, reader: Callable[[Path], Dict[str, Any]],
                  roles: Dict[Path, List[str]]) -> List[Dict[str, Any]]:
    """Read every matching file; a missing directory gives [] and an unreadable file an entry with "error"."""
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(p for p in directory.glob(pattern) if p.is_file()):
        entry: Dict[str, Any] = {"file": path.name, "path": str(path), "size_bytes": path.stat().st_size,
                                 "roles": roles.get(path.resolve(), [])}
        try:
            entry.update(reader(path))
        except _READ_ERRORS as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)
    return entries


def _pick(entries: List[Dict[str, Any]], role: str, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The readable entry configured as `role`, else the first readable one (of `kind` when given)."""
    usable = [e for e in entries if "error" not in e and (kind is None or e.get("kind") == kind)]
    return next((e for e in usable if role in e["roles"]), usable[0] if usable else None)


def _derive(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Fill DiT patch sizes missing from GGUF metadata and cross-check the components against each other."""
    dit = _pick(facts["diffusion_models"], "turbo")
    vae = _pick(facts["vae"], "vae")
    llm = _pick(facts["text_encoders"], "llm", kind="llm")
    mmproj = _pick(facts["text_encoders"], "llm_vision", kind="mmproj")
    latent = vae["latent_channels"] if vae else None

    # 1. A DiT GGUF without an embedded config: in_channels = latent_channels * patch_size^2.
    for entry in facts["diffusion_models"]:
        if "error" in entry or entry["patch_size"] is not None or not latent or not entry["in_channels"]:
            continue
        ratio, rem = divmod(entry["in_channels"], latent)
        if rem == 0 and math.isqrt(ratio) ** 2 == ratio:
            entry["patch_size"] = math.isqrt(ratio)
            entry["patch_size_source"] = "vae_latent_channels"

    # 2. Cross-component checks (None when a component is missing).
    def get(entry: Optional[Dict[str, Any]], key: str) -> Any:
        return entry.get(key) if entry else None

    patch, down = get(dit, "patch_size"), get(vae, "spatial_downsample")
    dit_in, context, llm_hidden = get(dit, "in_channels"), get(dit, "context_in_dim"), get(llm, "hidden_size")
    vis_patch, merge = get(mmproj, "patch_size"), get(mmproj, "spatial_merge_size")
    projection = get(mmproj, "projection_dim")
    return {
        "dit_file": get(dit, "file"),
        "vae_file": get(vae, "file"),
        "llm_file": get(llm, "file"),
        "mmproj_file": get(mmproj, "file"),
        "pixels_per_dit_token": down * patch if down and patch else None,
        "dit_in_matches_vae_latent": dit_in == latent * patch ** 2 if dit_in and latent and patch else None,
        "dit_context_matches_llm_hidden": context == llm_hidden if context and llm_hidden else None,
        "vision_token_px": vis_patch * merge if vis_patch and merge else None,
        "vision_projection_matches_llm_hidden": projection == llm_hidden if projection and llm_hidden else None,
    }


def model_facts(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Header facts for every weight file under models_dir (default config.MODELS_DIR).

    Returns {"models_dir", "diffusion_models": [...], "vae": [...], "text_encoders": [...], "derived": {...}}.
    Missing directories give empty lists; unreadable files give entries with an "error" string.
    """
    root = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    roles = _configured_roles()
    facts: Dict[str, Any] = {
        "models_dir": str(root),
        "diffusion_models": _read_entries(root / "diffusion_models", "*.gguf", dit_facts, roles),
        "vae": _read_entries(root / "vae", "*.safetensors", vae_facts, roles),
        "text_encoders": _read_entries(root / "text_encoders", "*.gguf", text_encoder_facts, roles),
    }
    facts["derived"] = _derive(facts)
    return facts


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

_SECTIONS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("Diffusion transformer", "diffusion_models", (
        "roles", "architecture", "class_name", "file_type", "size_bytes", "param_count", "bits_per_weight", "dtypes",
        "hidden_size", "num_layers", "num_heads", "head_dim", "mlp_hidden", "in_channels", "out_channels",
        "patch_size", "patch_size_source", "rope_axes", "context_in_dim", "config_mismatches")),
    ("VAE", "vae", (
        "roles", "architecture", "size_bytes", "param_count", "dtypes", "input_channels", "output_channels",
        "latent_channels", "spatial_downsample")),
    ("Text encoders", "text_encoders", (
        "roles", "kind", "architecture", "name", "size_label", "file_type", "size_bytes", "param_count",
        "bits_per_weight", "dtypes", "layers", "hidden_size", "ffn_size", "heads", "kv_heads", "head_dim",
        "context_length", "rope_sections", "projector_type", "projection_dim", "image_size", "patch_size",
        "spatial_merge_size")),
)


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if key == "size_bytes":
        return f"{value / 1e9:.2f} GB"
    if key == "param_count":
        return f"{value / 1e9:.3f} B" if value >= 1e8 else f"{value / 1e6:.1f} M"
    if key == "dtypes":
        return " ".join(f"{dtype}:{bucket['tensors']}" for dtype, bucket in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else ("none" if key == "config_mismatches" else "-")
    return str(value)


def _grid(entries: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[str]:
    if any("error" in e for e in entries):
        keys = keys + ("error",)
    rows = [["field"] + [e["file"] for e in entries]]
    rows += [[key] + [_fmt(key, e.get(key)) for e in entries] for key in keys]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  " + "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]


def format_table(facts: Dict[str, Any]) -> str:
    """Human-readable report of model_facts() output."""
    lines = [f"Model facts from file headers (tensor data not loaded): {facts['models_dir']}", ""]
    for title, key, keys in _SECTIONS:
        lines.append(f"== {title} ({key}/) ==")
        lines.extend(_grid(facts[key], keys) if facts[key] else ["  (no files found)"])
        lines.append("")
    lines.append("== Derived ==")
    width = max(len(k) for k in facts["derived"])
    lines.extend(f"  {k.ljust(width)}  {_fmt(k, v)}" for k, v in facts["derived"].items())
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print model facts read from local weight-file headers.")
    parser.add_argument("--models-dir", type=Path, default=None, help=f"models directory (default: {config.MODELS_DIR})")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    facts = model_facts(args.models_dir)
    print(json.dumps(facts, indent=2, default=str) if args.json else format_table(facts))
    failed = any("error" in e for key in ("diffusion_models", "vae", "text_encoders") for e in facts[key])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
