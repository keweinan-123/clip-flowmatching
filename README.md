# Visual-Anchored Multi-Omics → H&E Generation

This repository implements a CLIP-conditioned **Flow Matching Transformer** that fuses omics embeddings with histology semantics to generate hematoxylin and eosin (H&E) images. The design follows the specification:

1. **Omics–CLIP alignment** learns a shared semantic space \((u, v)\) between omics profiles and CLIP image features.
2. **Flow Matching Transformer** conditions on an \(\alpha\)-weighted fusion of omics and image semantics to learn a continuous flow from Gaussian noise to the shared embedding.
3. **CLIP-conditioned decoder** translates the predicted semantic embedding into controllable, high-fidelity H&E imagery.

## Project layout

```
clip_flowmatching/
  __init__.py                # Package exports
  config.py                  # Dataclass configurations
  datasets.py                # Dataset + collate utilities
  models/
    decoder.py               # CLIP-conditioned image decoder
    flow_matching.py         # Flow Matching Transformer
    omics_clip.py            # Omics–CLIP alignment module
  trainer.py                 # Multi-stage training orchestration
scripts/
  train.py                   # CLI for the three-stage training pipeline
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust for your CUDA
pip install open_clip_torch numpy
```

## Data format

The training script expects an `.npz` file containing:

- `omics`: shape `(N, D)` omics feature matrix.
- `images`: shape `(N, C, H, W)` paired H&E tensors in `[0, 1]`.

## Training

```bash
python scripts/train.py path/to/dataset.npz --config configs.json --device cuda
```

The optional JSON configuration can override any field in the dataclasses exposed by `clip_flowmatching.config`. Example:

```json
{
  "training": {
    "batch_size": 4,
    "alpha": 0.6,
    "max_epochs": 10
  },
  "flow_matching": {
    "num_layers": 8,
    "conditioning_dropout": 0.1
  }
}
```

### Pipeline stages

1. **Aligner**: freezes the CLIP vision backbone and trains an omics projection MLP using symmetric CLIP-style contrastive loss.
2. **Flow Matching**: learns a rectified flow field that transports noise to the fused semantics `0.5 * (u + v)` while conditioning on `α * v + (1 - α) * u`.
3. **Decoder**: predicts H&E imagery from the sampled semantic maps using a CLIP-conditioned U-Net with spatial transformers.

## Inference

After training, `MultiOmicsToHETrainer.sample_semantic` and the decoder can be used to synthesize H&E images conditioned on new omics measurements. Provide omics profiles to the aligner to obtain `u`, choose a reference visual anchor `v`, and run the flow sampler followed by the decoder.

## License

[MIT](LICENSE)
