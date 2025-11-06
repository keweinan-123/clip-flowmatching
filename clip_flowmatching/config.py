"""Configuration dataclasses for the CLIP-conditioned flow matching pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


def _default_schedule() -> Sequence[float]:
    return (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class AlignerConfig:
    """Hyper-parameters for the Omics↔CLIP alignment stage."""

    omics_dim: int = 1024
    clip_model_name: str = "openai/clip-vit-large-patch14"
    projection_dim: int = 768
    temperature: float = 0.07
    mlp_hidden_dims: Sequence[int] = (1024, 768)
    use_batch_norm: bool = True
    dropout: float = 0.1


@dataclass
class FlowMatchingConfig:
    """Hyper-parameters for the Flow Matching Transformer stage."""

    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    time_embedding_dim: int = 128
    conditioning_dropout: float = 0.0
    noise_schedule: Sequence[float] = field(default_factory=_default_schedule)


@dataclass
class DecoderConfig:
    """Hyper-parameters for the CLIP-conditioned H&E decoder."""

    image_size: int = 256
    base_channels: int = 64
    channel_mults: Sequence[int] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Sequence[int] = (16,)
    dropout: float = 0.1
    clip_dim: int = 768


@dataclass
class TrainingConfig:
    """Global training configuration coordinating the three stages."""

    batch_size: int = 8
    num_workers: int = 8
    max_epochs: int = 100
    precision: str = "bf16"
    aligner_lr: float = 5e-5
    flow_matching_lr: float = 1e-4
    decoder_lr: float = 5e-5
    weight_decay: float = 1e-2
    alpha: float = 0.5
    gradient_clip_norm: Optional[float] = 1.0
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
