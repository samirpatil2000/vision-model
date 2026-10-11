import unittest
import torch
from src.architecture.mask_adapter import LatentInpaintAdapter, SpatialAttentionMaskBias

class TestArchitectureAdapter(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.B, self.C, self.H, self.W = 1, 16, 16, 16
        self.hidden_dim = 128
        self.patch_size = 2

    def test_zero_initialization_identity(self):
        adapter = LatentInpaintAdapter(
            in_channels_base=self.C,
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
        ).to(self.device)

        z_noisy = torch.randn(self.B, self.C, self.H, self.W)
        z_orig = torch.randn(self.B, self.C, self.H, self.W)
        mask = torch.zeros(self.B, 1, self.H, self.W)
        mask[:, :, 4:12, 4:12] = 1.0

        with torch.no_grad():
            out_adapter = adapter(z_noisy, z_orig, mask)
            out_base = adapter.proj_base(z_noisy).flatten(2).transpose(1, 2)
            max_diff = torch.max(torch.abs(out_adapter - out_base)).item()
            self.assertEqual(max_diff, 0.0)

    def test_spatial_attention_bias_shape_and_values(self):
        mask = torch.zeros(self.B, 1, self.H, self.W)
        mask[:, :, 4:12, 4:12] = 1.0

        bias_module = SpatialAttentionMaskBias(penalty_val=-1e4)
        bias = bias_module(mask, patch_size=self.patch_size)

        num_patches = (self.H // self.patch_size) * (self.W // self.patch_size)
        self.assertEqual(bias.shape, (self.B, 1, num_patches, num_patches))
        # Ensure penalties are present for protected attending to edit
        self.assertLess(torch.min(bias).item(), -100.0)

if __name__ == "__main__":
    unittest.main()
