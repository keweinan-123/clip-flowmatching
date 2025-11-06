"""Dataset utilities for multi-omics to H&E generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class MultiOmicsSample:
    omics: torch.Tensor
    image: torch.Tensor


class OmicsToHEEmbeddingDataset(Dataset[MultiOmicsSample]):
    """Simple in-memory dataset mapping omics vectors to paired images."""

    def __init__(self, omics: torch.Tensor, images: torch.Tensor) -> None:
        assert len(omics) == len(images), "omics and image tensors must align"
        self.omics = omics
        self.images = images

    def __len__(self) -> int:
        return len(self.omics)

    def __getitem__(self, idx: int) -> MultiOmicsSample:
        return MultiOmicsSample(self.omics[idx], self.images[idx])


def collate_fn(batch: Tuple[MultiOmicsSample, ...]) -> Dict[str, torch.Tensor]:
    omics = torch.stack([sample.omics for sample in batch])
    images = torch.stack([sample.image for sample in batch])
    return {"omics": omics, "images": images}
