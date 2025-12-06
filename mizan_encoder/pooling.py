import torch
import torch.nn as nn


class BalancedMeanPooling(nn.Module):
    """
    Robust Balanced Mean Pooling with NaN protection
    """

    def forward(self, hidden, mask):
        # Convert mask to float with stability
        weights = mask.float()
        
        # Add small positive bias to prevent zero masks
        weights = weights + 1e-8
        
        # Calculate denominator with safety
        denom = weights.sum(dim=1, keepdim=True)
        denom = torch.clamp(denom, min=1.0)  # Ensure at least 1
        
        # Weighted sum with stability
        weighted_sum = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        
        # Safe division
        pooled = weighted_sum / denom
        
        # Post-process to remove any potential NaN/Inf
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize to prevent extreme values
        pooled_norm = torch.norm(pooled, dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = pooled / pooled_norm
        
        return pooled


class SafeBalancedMeanPooling(nn.Module):
    """
    Even safer version with debugging
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, hidden, mask):
        # Debug: Check inputs
        if torch.isnan(hidden).any():
            print("⚠️ WARNING: NaN in hidden states!")
            hidden = torch.nan_to_num(hidden, nan=0.0)
        
        if torch.isnan(mask).any():
            print("⚠️ WARNING: NaN in mask!")
            mask = torch.nan_to_num(mask, nan=0.0)
        
        # Ensure mask is valid
        mask = mask.float()
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # Add epsilon to prevent all-zero masks
        mask = mask + self.eps
        
        # Sum with stability
        denom = mask.sum(dim=1, keepdim=True)
        denom = torch.clamp(denom, min=1.0)
        
        # Weighted sum
        weighted_sum = torch.zeros_like(hidden[:, 0, :])
        for i in range(hidden.size(1)):
            weighted_sum += hidden[:, i, :] * mask[:, i:i+1]
        
        # Safe division
        pooled = weighted_sum / denom
        
        # Final safety checks
        if torch.isnan(pooled).any():
            print("⚠️ WARNING: NaN after pooling! Using zeros.")
            pooled = torch.zeros_like(pooled)
        
        return pooled