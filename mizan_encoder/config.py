"""
Configuration utilities for Mizan Encoder.

Handles:
- Training hyperparameters
- Model settings
- Dataset paths
- Export paths
"""

from dataclasses import dataclass

@dataclass
class MizanConfig:
    # Model
    backbone: str = "distilbert-base-uncased"
    proj_dim: int = 384
    alpha: float = 0.2           # scale stabilizer exponent

    # Training
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    max_len: int = 256

    # Loss
    margin: float = 0.5
    p: int = 2
    eps: float = 1e-6

    # Datasets
    dataset_path: str = "data/all_pairs.jsonl"

    # Checkpoints
    save_dir: str = "saved/mizan_encoder_v1"
    save_every: int = 1
