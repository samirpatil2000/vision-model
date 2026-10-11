#!/usr/bin/env python3
"""
Runnable PyTorch prototypes for native mask-conditioned adaptation of DiT models.
Demonstrates:
1. LatentInpaintAdapter: 33-channel patch projection adapter with zero-initialization.
2. SpatialAttentionMaskBias: Additive self-attention bias enforcing boundary isolation.
3. Overfit smoke test verifying gradient flow and identity initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LatentInpaintAdapter(nn.Module):
    """
    Expands a DiT patch embedding projection from 16 channels to 33 channels:
    - 16 channels: Noisy latent z_t
    - 16 channels: Masked original latent z_orig * (1 - mask)
    - 1 channel:   Downsampled binary allowed-edit mask (1=edit, 0=protected)

    Initialized with zero weights on the new 17 channels to guarantee that
    at step 0 of training, the adapter behaves identically to the frozen base DiT.
    """
    def __init__(
        self,
        in_channels_base: int = 16,
        hidden_dim: int = 1152,
        patch_size: int = 2,
    ):
        super().__init__()
        self.in_channels_base = in_channels_base
        self.in_channels_total = in_channels_base + in_channels_base + 1  # 16 + 16 + 1 = 33
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Base patch projection (representing pre-trained weights)
        self.proj_base = nn.Conv2d(
            in_channels_base,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Adapter patch projection for the 17 new conditioning channels
        self.proj_mask_cond = nn.Conv2d(
            in_channels_base + 1,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # Zero-initialization: initially produces exactly 0 delta
        nn.init.zeros_(self.proj_mask_cond.weight)

    def forward(
        self,
        z_noisy: torch.Tensor,
        z_orig: torch.Tensor,
        mask_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_noisy: (B, 16, H, W) noisy latents
            z_orig:  (B, 16, H, W) reference image latents
            mask_latent: (B, 1, H, W) binary edit mask (1 = allowed edit, 0 = protected)
        Returns:
            tokens: (B, num_patches, hidden_dim) flattened patch embeddings
        """
        # Mask original latent: keep only protected context
        masked_orig = z_orig * (1.0 - mask_latent)

        # Condition tensor: (B, 17, H, W)
        cond_tensor = torch.cat([masked_orig, mask_latent], dim=1)

        # Base projection + Zero-initialized adapter branch
        out_base = self.proj_base(z_noisy)
        out_cond = self.proj_mask_cond(cond_tensor)

        out = out_base + out_cond  # (B, hidden_dim, H/p, W/p)

        # Flatten spatial dimensions to patch sequence: (B, num_patches, hidden_dim)
        B, C, H_p, W_p = out.shape
        tokens = out.flatten(2).transpose(1, 2)
        return tokens

class SpatialAttentionMaskBias(nn.Module):
    """
    Computes an additive spatial attention bias matrix B in R^{S x S}
    for DiT self-attention blocks to enforce structural isolation.
    """
    def __init__(self, penalty_val: float = -1e4):
        super().__init__()
        self.penalty_val = penalty_val

    def forward(
        self,
        mask_latent: torch.Tensor,
        patch_size: int = 2,
    ) -> torch.Tensor:
        """
        Args:
            mask_latent: (B, 1, H, W) binary mask in latent space
        Returns:
            bias: (B, 1, num_patches, num_patches) additive attention bias
        """
        # Downsample mask to patch tokens via max-pooling
        mask_patches = F.max_pool2d(
            mask_latent, kernel_size=patch_size, stride=patch_size
        )  # (B, 1, H_p, W_p)
        B, _, H_p, W_p = mask_patches.shape
        flat_mask = mask_patches.flatten(2)  # (B, 1, S) where S = H_p * W_p

        # Query tokens in protected area (flat_mask == 0) attending to key tokens in edit area (flat_mask == 1)
        # receives negative penalty to prevent information leakage
        is_edit = (flat_mask > 0.5).float()  # (B, 1, S)
        is_protected = (flat_mask <= 0.5).float()  # (B, 1, S)

        # Cross penalty: protected query -> edit key
        cross_leakage = torch.bmm(is_protected.transpose(1, 2), is_edit)  # (B, S, S)
        bias = cross_leakage * self.penalty_val
        return bias.unsqueeze(1)  # (B, 1, S, S)

def run_overfit_smoke_test():
    """
    Executable smoke test verifying:
    1. Zero-initialization invariant: Adapter(x, orig, mask) == Base(x)
    2. Gradient flow through adapter weights
    3. 5-step loss minimization on toy latent problem
    """
    print("==========================================================")
    print(" Running DiT Mask-Adapter Smoke Test")
    print("==========================================================")
    device = torch.device("cpu")
    B, C, H, W = 1, 16, 32, 32
    hidden_dim = 256
    patch_size = 2

    adapter = LatentInpaintAdapter(
        in_channels_base=C,
        hidden_dim=hidden_dim,
        patch_size=patch_size,
    ).to(device)

    # 1. Verify Zero-initialization invariant
    z_noisy = torch.randn(B, C, H, W, device=device)
    z_orig = torch.randn(B, C, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    mask[:, :, 8:24, 8:24] = 1.0  # Center edit region

    with torch.no_grad():
        out_adapter = adapter(z_noisy, z_orig, mask)
        out_base_expected = adapter.proj_base(z_noisy).flatten(2).transpose(1, 2)
        diff = torch.max(torch.abs(out_adapter - out_base_expected)).item()
        print(f"[1] Zero-Init Maximum Deviation: {diff:.8f}")
        assert diff == 0.0, "Adapter must produce exactly 0 deviation from base at initialization!"

    # 2. Verify Gradient Flow and Overfit Convergence
    optimizer = torch.optim.AdamW(adapter.proj_mask_cond.parameters(), lr=1e-3)
    target_tokens = torch.randn_like(out_adapter)

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = adapter(z_noisy, z_orig, mask)
        loss = F.mse_loss(out, target_tokens)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"[2] Initial Loss: {losses[0]:.4f} -> Final Loss after 5 steps: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Adapter weights must successfully decrease loss via backprop!"

    # 3. Verify Spatial Attention Bias
    bias_module = SpatialAttentionMaskBias(penalty_val=-1e4)
    attn_bias = bias_module(mask, patch_size=patch_size)
    print(f"[3] Attention Bias Shape: {attn_bias.shape}")
    assert attn_bias.shape == (B, 1, (H // patch_size) * (W // patch_size), (H // patch_size) * (W // patch_size))
    print("[✓] All Architectural Smoke Tests Passed Successfully!")

if __name__ == "__main__":
    run_overfit_smoke_test()
