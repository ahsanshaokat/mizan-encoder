import torch
import torch.nn as nn


class BalancedMeanPooling(nn.Module):
    """
    Balanced Mean Pooling - Fixed with numerical stability
    """

    def forward(self, hidden, mask):
        weights = mask.float()
        # Add small epsilon and clamp to prevent division by zero
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1) / denom
        
        # Check for NaN
        if torch.isnan(pooled).any():
            print("Warning: NaN in pooling output")
            # Return zeros instead of NaN
            pooled = torch.zeros_like(pooled)
            
        return pooled