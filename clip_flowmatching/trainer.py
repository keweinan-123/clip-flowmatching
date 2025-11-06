"""Training orchestration for the visual-anchored multi-omics → H&E generator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import AlignerConfig, FlowMatchingConfig, DecoderConfig, TrainingConfig
from .datasets import collate_fn
from .models.decoder import CLIPConditionedDecoder
from .models.flow_matching import FlowMatchingTransformer
from .models.omics_clip import OmicsCLIPAligner


@dataclass
class TrainingState:
    aligner: OmicsCLIPAligner
    flow_model: FlowMatchingTransformer
    decoder: CLIPConditionedDecoder


class MultiOmicsToHETrainer:
    def __init__(
        self,
        aligner: OmicsCLIPAligner,
        flow_model: FlowMatchingTransformer,
        decoder: CLIPConditionedDecoder,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.aligner = aligner.to(device)
        self.flow_model = flow_model.to(device)
        self.decoder = decoder.to(device)
        self.config = config
        self.device = device

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        omics = batch["omics"].to(self.device)
        images = batch["images"].to(self.device)
        return omics, images

    def _semantic_targets(self, omics: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.aligner.eval()
            u, v, _ = self.aligner(omics, images)
        semantic_target = 0.5 * (u + v)
        condition = self.config.alpha * v + (1 - self.config.alpha) * u
        return semantic_target, condition, v

    def train_aligner(self, dataloader: DataLoader) -> None:
        self.aligner.train()
        optimizer = optim.AdamW(
            self.aligner.projector.parameters(),
            lr=self.config.aligner_lr,
            weight_decay=self.config.weight_decay,
        )
        for epoch in range(self.config.max_epochs):
            for step, batch in enumerate(dataloader):
                omics, images = self._prepare_batch(batch)
                optimizer.zero_grad()
                _, _, loss = self.aligner(omics, images)
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.aligner.projector.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                if step % self.config.log_every_n_steps == 0:
                    print(f"[Aligner][Epoch {epoch}] Step {step}: loss={loss.item():.4f}")

    def train_flow_matching(self, dataloader: DataLoader) -> None:
        self.aligner.eval()
        self.flow_model.train()
        optimizer = optim.AdamW(
            self.flow_model.parameters(),
            lr=self.config.flow_matching_lr,
            weight_decay=self.config.weight_decay,
        )
        for epoch in range(self.config.max_epochs):
            for step, batch in enumerate(dataloader):
                omics, images = self._prepare_batch(batch)
                semantic_target, condition, _ = self._semantic_targets(omics, images)
                noise = torch.randn_like(semantic_target)
                optimizer.zero_grad()
                loss, _ = self.flow_model.loss(
                    noise[:, None, :],
                    semantic_target[:, None, :],
                    condition,
                )
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                if step % self.config.log_every_n_steps == 0:
                    print(f"[Flow][Epoch {epoch}] Step {step}: loss={loss.item():.4f}")

    def train_decoder(self, dataloader: DataLoader) -> None:
        self.aligner.eval()
        self.flow_model.eval()
        self.decoder.train()
        optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=self.config.decoder_lr,
            weight_decay=self.config.weight_decay,
        )
        for epoch in range(self.config.max_epochs):
            for step, batch in enumerate(dataloader):
                omics, images = self._prepare_batch(batch)
                semantic_target, condition, _ = self._semantic_targets(omics, images)
                pred_semantic = self.sample_semantic(condition)
                conditioning = condition
                recon = self.decoder(pred_semantic, conditioning)
                loss = self.decoder.reconstruction_loss(images, recon)
                optimizer.zero_grad()
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                if step % self.config.log_every_n_steps == 0:
                    print(f"[Decoder][Epoch {epoch}] Step {step}: loss={loss.item():.4f}")

    def sample_semantic(self, condition: torch.Tensor) -> torch.Tensor:
        """Integrates the learned velocity field starting from Gaussian noise."""
        self.flow_model.eval()
        with torch.no_grad():
            batch, dim = condition.shape
            schedule = self.flow_model.noise_schedule.to(self.device)
            noise = torch.randn(batch, 1, dim, device=self.device)
            x = noise
            for i in range(len(schedule) - 1):
                t0, t1 = schedule[i], schedule[i + 1]
                dt = t1 - t0
                t_mid = torch.full((batch,), (t0 + t1) / 2, device=self.device)
                velocity = self.flow_model(x, t_mid, condition)
                x = x + dt * velocity
            semantic = x.squeeze(1)
            spatial = semantic[:, :, None, None]
            return spatial.expand(-1, -1, self.decoder.image_size, self.decoder.image_size)

    @staticmethod
    def build_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def build_training_state(
    aligner_config: AlignerConfig,
    flow_config: FlowMatchingConfig,
    decoder_config: DecoderConfig,
    device: torch.device,
) -> TrainingState:
    aligner = OmicsCLIPAligner(
        omics_dim=aligner_config.omics_dim,
        clip_model_name=aligner_config.clip_model_name,
        projection_dim=aligner_config.projection_dim,
        temperature=aligner_config.temperature,
        hidden_dims=tuple(aligner_config.mlp_hidden_dims),
        use_batch_norm=aligner_config.use_batch_norm,
        dropout=aligner_config.dropout,
    )

    flow_model = FlowMatchingTransformer(
        embedding_dim=flow_config.embedding_dim,
        num_layers=flow_config.num_layers,
        num_heads=flow_config.num_heads,
        mlp_ratio=flow_config.mlp_ratio,
        dropout=flow_config.dropout,
        time_embedding_dim=flow_config.time_embedding_dim,
        conditioning_dropout=flow_config.conditioning_dropout,
        noise_schedule=tuple(flow_config.noise_schedule),
    )

    decoder = CLIPConditionedDecoder(
        image_size=decoder_config.image_size,
        base_channels=decoder_config.base_channels,
        channel_mults=tuple(decoder_config.channel_mults),
        num_res_blocks=decoder_config.num_res_blocks,
        attention_resolutions=tuple(decoder_config.attention_resolutions),
        dropout=decoder_config.dropout,
        clip_dim=decoder_config.clip_dim,
    )

    return TrainingState(aligner=aligner.to(device), flow_model=flow_model.to(device), decoder=decoder.to(device))
