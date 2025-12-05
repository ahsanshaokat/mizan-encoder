import torch
import torch.nn as nn

class BalancedMeanPooling(nn.Module):
    """
    Balanced Mean Pooling
    ---------------------
    A scale-aware mean pooling that prevents:
        - long texts from diluting meaning
        - short texts from dominating

    Formula:
        pooled = sum(token_emb * mask) / count(mask)

    But carefully stabilized with clamp for numerical safety.
    """

    def forward(self, hidden, mask):
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1) / denom
        return pooled
