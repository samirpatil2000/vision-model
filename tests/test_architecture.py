"""Tests for src.model_facts: synthetic GGUF/safetensors headers, missing files, and the real local headers."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

import gguf
import numpy as np
from safetensors.numpy import save_file

from src import config
from src import model_facts as mf


def _write_gguf(path: Path, arch: str, tensors: Dict[str, np.ndarray], kv: Optional[Dict[str, object]] = None) -> None:
    """Write a tiny GGUF file. Numpy arrays are in PyTorch order (out, in); GGUF stores them reversed."""
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in (kv or {}).items():
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_uint32(key, value)
    for name, array in tensors.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_dit(path: Path, hidden=64, layers=2, head_dim=16, in_ch=16, out_ch=16, ctx=32, mlp=192,
               cfg: Optional[dict] = None) -> int:
    """Synthetic Qwen-Image DiT GGUF; returns its parameter count."""
    f32, f16 = np.float32, np.float16
    tensors = {
        "img_in.weight": np.zeros((hidden, in_ch), f32),
        "proj_out.weight": np.zeros((out_ch, hidden), f32),
        "txt_in.in_layer.weight": np.zeros((hidden, ctx), f32),
    }
    for b in range(layers):
        tensors[f"transformer_blocks.{b}.attn.norm_q.weight"] = np.zeros((head_dim,), f32)
        tensors[f"transformer_blocks.{b}.attn.to_q.weight"] = np.zeros((hidden, hidden), f16)
        tensors[f"transformer_blocks.{b}.img_mlp.out.weight"] = np.zeros((hidden, mlp), f16)
    kv = {"config": json.dumps({"transformer": cfg})} if cfg is not None else None
    _write_gguf(path, "qwen_image", tensors, kv)
    return sum(a.size for a in tensors.values())


def _write_vae(path: Path, in_ch=4, latent=8, stages=3) -> None:
    """Synthetic Wan-style VAE safetensors header with `stages` stride-2 encoder resample convs."""
    z = lambda *shape: np.zeros(shape, np.float16)  # noqa: E731
    tensors = {
        "encoder.conv1.weight": z(8, in_ch, 1, 3, 3),
        "encoder.head.2.weight": z(2 * latent, 8, 1, 3, 3),
        "conv1.weight": z(2 * latent, 2 * latent, 1, 1, 1),
        "conv2.weight": z(latent, latent, 1, 1, 1),
        "decoder.conv1.weight": z(8, latent, 1, 3, 3),
        "decoder.head.0.gamma": z(8, 1, 1, 1),
        "decoder.head.2.weight": z(in_ch, 8, 1, 3, 3),
    }
    for s in range(stages):
        tensors[f"encoder.downsamples.{s}.downsamples.0.residual.2.weight"] = z(8, 8, 1, 3, 3)
        tensors[f"encoder.downsamples.{s}.downsamples.1.resample.1.weight"] = z(8, 8, 3, 3)
    tensors["encoder.downsamples.9.downsamples.0.residual.2.weight"] = z(8, 8, 1, 3, 3)  # last stage: no resample
    save_file(tensors, str(path), metadata={"modelspec.architecture": "test_vae"})


def _write_llm(path: Path, hidden=32) -> None:
    _write_gguf(path, "qwen3vl", {"token_embd.weight": np.zeros((100, hidden), np.float32)}, {
        "general.name": "Tiny LLM", "general.type": "model",
        "qwen3vl.block_count": 3, "qwen3vl.embedding_length": hidden, "qwen3vl.feed_forward_length": 96,
        "qwen3vl.attention.head_count": 4, "qwen3vl.attention.head_count_kv": 2, "qwen3vl.attention.key_length": 8,
        "qwen3vl.context_length": 4096, "tokenizer.ggml.tokens": [f"t{i}" for i in range(200)],
    })


def _write_mmproj(path: Path, projection=32) -> None:
    _write_gguf(path, "clip", {"v.patch_embd.weight": np.zeros((24, 3, 16, 16), np.float16)}, {
        "general.type": "mmproj", "clip.projector_type": "qwen3vl_merger",
        "clip.vision.block_count": 2, "clip.vision.embedding_length": 24, "clip.vision.patch_size": 16,
        "clip.vision.spatial_merge_size": 2, "clip.vision.projection_dim": projection,
        "clip.vision.attention.head_count": 4,
    })


class TestModelFactsSynthetic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for sub in ("diffusion_models", "vae", "text_encoders"):
            (self.root / sub).mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_dit_facts_from_header(self):
        path = self.root / "diffusion_models" / "dit.gguf"
        cfg = {"_class_name": "QwenImage21Transformer2DModel", "num_layers": 2, "num_attention_heads": 4,
               "attention_head_dim": 16, "in_channels": 16, "out_channels": 16, "patch_size": 1,
               "mlp_ratio": 3, "axes_dims_rope": [4, 6, 6], "context_in_dim": 32}
        params = _write_dit(path, cfg=cfg)
        facts = mf.dit_facts(path)
        self.assertEqual(facts["architecture"], "qwen_image")
        self.assertEqual(facts["class_name"], "QwenImage21Transformer2DModel")
        self.assertEqual((facts["hidden_size"], facts["num_layers"], facts["num_heads"], facts["head_dim"]),
                         (64, 2, 4, 16))
        self.assertEqual((facts["in_channels"], facts["out_channels"], facts["mlp_hidden"], facts["context_in_dim"]),
                         (16, 16, 192, 32))
        self.assertEqual((facts["patch_size"], facts["patch_size_source"]), (1, "config"))
        self.assertEqual(facts["rope_axes"], [4, 6, 6])
        self.assertEqual(facts["config_mismatches"], [])
        self.assertEqual(facts["param_count"], params)
        self.assertEqual(facts["dtypes"]["F16"]["tensors"], 4)
        self.assertEqual(facts["dtypes"]["F32"]["tensors"], 5)

    def test_dit_config_disagreeing_with_tensors_is_reported(self):
        path = self.root / "diffusion_models" / "dit.gguf"
        _write_dit(path, cfg={"num_layers": 3, "num_attention_heads": 4, "attention_head_dim": 16,
                              "axes_dims_rope": [4, 4, 4]})
        mismatches = mf.dit_facts(path)["config_mismatches"]
        self.assertEqual(len(mismatches), 2)
        self.assertTrue(mismatches[0].startswith("num_layers: tensors 2 != config 3"))
        self.assertTrue(mismatches[1].startswith("rope_axes: sum 12 != head_dim 16"))

    def test_vae_facts_from_header(self):
        path = self.root / "vae" / "vae.safetensors"
        _write_vae(path, in_ch=4, latent=8, stages=3)
        facts = mf.vae_facts(path)
        self.assertEqual(facts["architecture"], "test_vae")
        self.assertEqual((facts["input_channels"], facts["output_channels"], facts["latent_channels"]), (4, 4, 8))
        self.assertEqual(facts["spatial_downsample"], 8)
        self.assertEqual(list(facts["dtypes"]), ["F16"])

    def test_text_encoder_and_mmproj_facts(self):
        llm_path, mm_path = self.root / "text_encoders" / "llm.gguf", self.root / "text_encoders" / "mmproj.gguf"
        _write_llm(llm_path)
        _write_mmproj(mm_path)
        llm = mf.text_encoder_facts(llm_path)
        self.assertEqual((llm["kind"], llm["architecture"], llm["name"]), ("llm", "qwen3vl", "Tiny LLM"))
        self.assertEqual((llm["layers"], llm["hidden_size"], llm["heads"], llm["kv_heads"], llm["head_dim"]),
                         (3, 32, 4, 2, 8))
        self.assertNotIn("tokenizer.ggml.tokens", llm["hparams"])
        mm = mf.text_encoder_facts(mm_path)
        self.assertEqual((mm["kind"], mm["architecture"], mm["projector_type"]), ("mmproj", "clip", "qwen3vl_merger"))
        self.assertEqual((mm["layers"], mm["hidden_size"], mm["patch_size"], mm["spatial_merge_size"]), (2, 24, 16, 2))

    def test_model_facts_tree_infers_patch_and_cross_checks(self):
        dit_path = self.root / "diffusion_models" / "dit.gguf"
        _write_dit(dit_path, in_ch=32, out_ch=32, ctx=32)          # no embedded config
        _write_vae(self.root / "vae" / "vae.safetensors", latent=8, stages=3)
        _write_llm(self.root / "text_encoders" / "llm.gguf", hidden=32)
        _write_mmproj(self.root / "text_encoders" / "mmproj.gguf", projection=32)
        with mock.patch.object(config, "TURBO_MODEL", dit_path):
            facts = mf.model_facts(self.root)

        dit = facts["diffusion_models"][0]
        self.assertEqual(dit["roles"], ["turbo"])
        self.assertEqual((dit["patch_size"], dit["patch_size_source"]), (2, "vae_latent_channels"))
        derived = facts["derived"]
        self.assertEqual(derived["dit_file"], "dit.gguf")
        self.assertEqual(derived["pixels_per_dit_token"], 16)      # f=8 VAE x 2x2 patches
        self.assertTrue(derived["dit_in_matches_vae_latent"])
        self.assertTrue(derived["dit_context_matches_llm_hidden"])
        self.assertEqual(derived["vision_token_px"], 32)
        self.assertTrue(derived["vision_projection_matches_llm_hidden"])
        json.dumps(facts)                                            # JSON-serialisable
        table = mf.format_table(facts)
        self.assertIn("dit.gguf", table)
        self.assertIn("pixels_per_dit_token", table)

    def test_missing_models_dir_gives_empty_facts(self):
        missing = self.root / "does_not_exist"
        with mock.patch.object(config, "MODELS_DIR", missing):
            facts = mf.model_facts()
        self.assertEqual(facts["models_dir"], str(missing))
        self.assertEqual((facts["diffusion_models"], facts["vae"], facts["text_encoders"]), ([], [], []))
        self.assertTrue(all(v is None for v in facts["derived"].values()))
        self.assertIn("(no files found)", mf.format_table(facts))

    def test_unreadable_files_are_reported_not_raised(self):
        (self.root / "diffusion_models" / "broken.gguf").write_bytes(b"not a gguf file at all")
        (self.root / "vae" / "empty.safetensors").write_bytes(b"")
        _write_llm(self.root / "text_encoders" / "llm.gguf")
        facts = mf.model_facts(self.root)
        self.assertIn("error", facts["diffusion_models"][0])
        self.assertIn("error", facts["vae"][0])
        self.assertNotIn("error", facts["text_encoders"][0])
        self.assertIsNone(facts["derived"]["dit_file"])
        self.assertIn("error", mf.format_table(facts))

    def test_cli_table_json_and_exit_codes(self):
        _write_llm(self.root / "text_encoders" / "llm.gguf")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(mf.main(["--models-dir", str(self.root), "--json"]), 0)
        self.assertEqual(json.loads(out.getvalue())["text_encoders"][0]["layers"], 3)

        (self.root / "vae" / "bad.safetensors").write_bytes(b"\x00" * 4)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(mf.main(["--models-dir", str(self.root)]), 1)
        self.assertIn("== VAE (vae/) ==", out.getvalue())


@unittest.skipUnless(config.TURBO_MODEL.is_file() and config.VAE_MODEL.is_file(),
                     "local Qwen-Image weights not present")
class TestModelFactsRealHeaders(unittest.TestCase):
    def test_real_headers(self):
        facts = mf.model_facts()
        for key in ("diffusion_models", "vae", "text_encoders"):
            for entry in facts[key]:
                self.assertNotIn("error", entry, entry["file"])

        dit = next(e for e in facts["diffusion_models"] if "turbo" in e["roles"])
        self.assertEqual((dit["hidden_size"], dit["num_layers"], dit["num_heads"], dit["head_dim"]), (4096, 32, 32, 128))
        self.assertEqual((dit["in_channels"], dit["patch_size"]), (64, 1))
        self.assertEqual(dit["config_mismatches"], [])
        vae = next(e for e in facts["vae"] if "vae" in e["roles"])
        self.assertEqual((vae["latent_channels"], vae["spatial_downsample"]), (64, 16))
        derived = facts["derived"]
        self.assertEqual(derived["pixels_per_dit_token"], 16)
        self.assertTrue(derived["dit_in_matches_vae_latent"])
        if derived["llm_file"]:
            self.assertTrue(derived["dit_context_matches_llm_hidden"])
        # Every tier size must be a whole number of DiT tokens.
        for tier in config.TIERS.values():
            self.assertEqual(tier.width % derived["pixels_per_dit_token"], 0, tier.name)
            self.assertEqual(tier.height % derived["pixels_per_dit_token"], 0, tier.name)


if __name__ == "__main__":
    unittest.main()
