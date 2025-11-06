"""Flow Matching Transformer conditioned on CLIP semantics."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Generates sinusoidal time embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        exponent = exponent / half_dim
        angles = t[:, None] * torch.exp(-math.log(10_000) * exponent)[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class FlowMatchingTransformer(nn.Module):
    """Predicts the semantic velocity field conditioned on omics/image embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        time_embedding_dim: int,
        conditioning_dropout: float = 0.0,
        noise_schedule: Tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.conditioning_dropout = nn.Dropout(conditioning_dropout)
        self.input_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        if noise_schedule is None:
            noise_schedule = (0.0, 0.25, 0.5, 0.75, 1.0)
        self.register_buffer(
            "noise_schedule",
            torch.tensor(noise_schedule, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        noisy_embedding: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim == 2:
            condition = condition[:, None, :]
        condition = self.conditioning_dropout(condition)
        time_embedding = self.time_mlp(self.time_embed(time))
        time_embedding = time_embedding[:, None, :]

        x = torch.cat([noisy_embedding, condition], dim=-1)
        x = self.input_proj(x) + time_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        velocity = self.out(x)
        return velocity

    def loss(
        self,
        noise: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, dim = noise.shape
        if time is None:
            time = torch.rand(batch, device=noise.device)
        noise = noise.view(batch, seq, dim)
        target = target.view(batch, seq, dim)

        z_t = (1 - time)[:, None, None] * noise + time[:, None, None] * target
        velocity_target = target - noise
        velocity_pred = self.forward(z_t, time, condition)

        loss = F.mse_loss(velocity_pred, velocity_target)
        return loss, velocity_pred
