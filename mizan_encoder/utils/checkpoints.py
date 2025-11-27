"""
Checkpoint handling utilities for saving and loading model states.
"""

import os
import torch

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"💾 Saved checkpoint → {path}")


def load_checkpoint(path, model, optimizer=None):
    if not os.path.exists(path):
        print(f"⚠️ No checkpoint at {path}, skipping load.")
        return None

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"🔄 Loaded checkpoint from {path}")
    return ckpt.get("epoch", None)
