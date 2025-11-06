"""Omics↔CLIP alignment module."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    import open_clip
except ImportError as err:  # pragma: no cover - dependency not available during tests
    open_clip = None
    _IMPORT_ERROR = err
else:
    _IMPORT_ERROR = None


class OmicsProjection(nn.Module):
    """Projects omics vectors into the CLIP embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (1024, 768),
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != output_dim:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OmicsCLIPAligner(nn.Module):
    """Aligns omics embeddings with CLIP image semantics."""

    def __init__(
        self,
        omics_dim: int,
        clip_model_name: str,
        projection_dim: int,
        temperature: float = 0.07,
        hidden_dims: Tuple[int, ...] = (1024, 768),
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if open_clip is None:
            raise RuntimeError(
                "open_clip is required to instantiate OmicsCLIPAligner"
            ) from _IMPORT_ERROR

        model_name, pretrained = clip_model_name.split("/")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        embed_dim = self.clip_model.text_projection.shape[1]
        if projection_dim != embed_dim:
            # Align to CLIP embedding dimension.
            projection_dim = embed_dim

        self.projector = OmicsProjection(
            input_dim=omics_dim,
            output_dim=projection_dim,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(images)
        return F.normalize(image_features, dim=-1)

    def forward(self, omics: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)
        omics_proj = self.projector(omics)
        omics_features = F.normalize(omics_proj, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_omics = logit_scale * omics_features @ image_features.T
        logits_per_image = logits_per_omics.T

        targets = torch.arange(len(omics), device=omics.device)
        loss_omics = F.cross_entropy(logits_per_omics, targets)
        loss_images = F.cross_entropy(logits_per_image, targets)
        loss = (loss_omics + loss_images) / 2

        return omics_features, image_features, loss

    def infer_semantic_pair(self, omics: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            u, v, _ = self.forward(omics, images)
        return u, v
