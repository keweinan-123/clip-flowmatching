"""End-to-end training script for the multi-omics → H&E generator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clip_flowmatching.config import (
    AlignerConfig,
    FlowMatchingConfig,
    DecoderConfig,
    TrainingConfig,
)
from clip_flowmatching.datasets import OmicsToHEEmbeddingDataset
from clip_flowmatching.trainer import MultiOmicsToHETrainer, build_training_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to a npz file with 'omics' and 'images' arrays")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config override")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: Path | None) -> tuple[AlignerConfig, FlowMatchingConfig, DecoderConfig, TrainingConfig]:
    if path is None:
        return AlignerConfig(), FlowMatchingConfig(), DecoderConfig(), TrainingConfig()
    data = json.loads(path.read_text())
    aligner = AlignerConfig(**data.get("aligner", {}))
    flow = FlowMatchingConfig(**data.get("flow_matching", {}))
    decoder = DecoderConfig(**data.get("decoder", {}))
    training = TrainingConfig(**data.get("training", {}))
    return aligner, flow, decoder, training


def main() -> None:
    args = parse_args()
    aligner_cfg, flow_cfg, decoder_cfg, train_cfg = load_config(args.config)

    npz = np.load(args.data)
    omics = torch.from_numpy(npz["omics"]).float()
    images = torch.from_numpy(npz["images"]).float()
    dataset = OmicsToHEEmbeddingDataset(omics, images)

    device = torch.device(args.device)
    state = build_training_state(aligner_cfg, flow_cfg, decoder_cfg, device)
    trainer = MultiOmicsToHETrainer(state.aligner, state.flow_model, state.decoder, train_cfg, device)

    dataloader = trainer.build_dataloader(dataset, train_cfg.batch_size, train_cfg.num_workers)

    print("Stage 1: training Omics–CLIP aligner")
    trainer.train_aligner(dataloader)
    print("Stage 2: training Flow Matching Transformer")
    trainer.train_flow_matching(dataloader)
    print("Stage 3: training CLIP-conditioned decoder")
    trainer.train_decoder(dataloader)


if __name__ == "__main__":
    main()
