"""CLIP-conditioned decoder that maps semantic embeddings to H&E images."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.dropout(self.act2(self.norm2(x))))
        return x + residual


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = self.proj_in(x).view(b, c, h * w).permute(0, 2, 1)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output.permute(0, 2, 1).view(b, c, h, w)
        x = self.proj_out(x)
        return x + residual


class ConditioningProjector(nn.Module):
    def __init__(self, clip_dim: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(clip_dim, channels)

    def forward(self, conditioning: torch.Tensor, spatial_shape: torch.Size) -> torch.Tensor:
        b, c = conditioning.shape
        h, w = spatial_shape
        cond = self.linear(conditioning).view(b, c, 1, 1)
        return cond.expand(b, c, h, w)


class CLIPConditionedDecoder(nn.Module):
    """Decodes CLIP-aligned semantic embeddings into H&E images."""

    def __init__(
        self,
        image_size: int,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        dropout: float,
        clip_dim: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        channels = [base_channels * m for m in channel_mults]

        self.init_conv = nn.Conv2d(clip_dim, channels[0], kernel_size=3, padding=1)
        self.conditioning = ConditioningProjector(clip_dim, channels[0])

        downs = []
        in_ch = channels[0]
        current_res = image_size
        for idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                downs.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if current_res in attention_resolutions:
                    downs.append(SpatialTransformer(in_ch))
            if idx != len(channel_mults) - 1:
                downs.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))
                current_res //= 2
        self.down = nn.ModuleList(downs)

        self.mid = nn.Sequential(
            ResidualBlock(in_ch, in_ch, dropout),
            SpatialTransformer(in_ch),
            ResidualBlock(in_ch, in_ch, dropout),
        )

        ups = []
        for idx, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                ups.append(ResidualBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if image_size // (2 ** idx) in attention_resolutions:
                    ups.append(SpatialTransformer(in_ch))
            if idx != 0:
                ups.append(nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1))
                in_ch //= 2
        self.up = nn.ModuleList(ups)
        self.final_norm = nn.GroupNorm(32, in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

    def forward(self, semantic_map: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        b, c, h, w = semantic_map.shape
        x = self.init_conv(semantic_map)
        x = x + self.conditioning(conditioning, (h, w))

        skips = []
        for layer in self.down:
            x = layer(x)
            skips.append(x)

        x = self.mid(x)

        for layer in self.up:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            else:
                x = layer(x)

        x = self.final_conv(self.final_act(self.final_norm(x)))
        return torch.tanh(x)

    def reconstruction_loss(self, images: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(recon, images) + 0.1 * F.mse_loss(recon, images)
