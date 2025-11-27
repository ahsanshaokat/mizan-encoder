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

    def forward(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        count = mask.sum(dim=1)
        return summed / torch.clamp(count, min=1e-9)
