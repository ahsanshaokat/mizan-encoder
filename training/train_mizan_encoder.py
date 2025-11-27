"""
Training script for MizanTextEncoder
------------------------------------

Usage:
    python training/train_mizan_encoder.py --data data/all_pairs.jsonl

"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# -------------------------------------------------------------------
# Add project root to sys.path (Fixes ModuleNotFoundError on Windows)
# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from mizan_encoder.encoder import MizanTextEncoder
from mizan_encoder.loss import MizanContrastiveLoss
from training.dataset import PairDataset
from training.collate import collate_batch
from training.trainer import MizanTrainer


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train MizanTextEncoder")

    parser.add_argument("--data", type=str, default="scripts/data/all_pairs.jsonl")
    parser.add_argument("--backbone", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output", type=str, default="saved/mizan_encoder_v1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)

    # ★ NEW ARGUMENT
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples for CPU-fast training")

    return parser.parse_args()


# ---------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------
def main():
    args = get_args()

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== MizanTextEncoder Training ===")
    print(f"Device: {device}")
    print(f"Dataset: {args.data}")
    print(f"Backbone: {args.backbone}")
    print(f"Output Directory: {args.output}\n")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = MizanTextEncoder(backbone=args.backbone).to(device)

    # Loss, optimizer
    criterion = MizanContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Dataset + DataLoader
    dataset = PairDataset(args.data, tokenizer, subset=args.subset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    # Trainer object
    trainer = MizanTrainer(model, optimizer, criterion, device=device)

    # Make output directory
    os.makedirs(args.output, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_one_epoch(loader, epoch)
        print(f"Epoch {epoch} Completed - Avg Loss: {avg_loss:.4f}")

    # Save Model
    trainer.save(args.output)
    print(f"\nModel saved successfully → {args.output}\n")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
