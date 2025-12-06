import torch
import torch.nn as nn
import torch.nn.functional as F


class MizanContrastiveLoss(nn.Module):
    """
    Mizan Contrastive Loss - Fixed robust version
    Uses mizan_similarity with numerical stability
    """

    def __init__(self, margin=0.3, p=2, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps

    def mizan_sim(self, x, y):
        """Numerically stable mizan similarity"""
        # Add small noise to prevent zero norms
        x = x + torch.randn_like(x) * 1e-8
        y = y + torch.randn_like(y) * 1e-8
        
        num = torch.norm(x - y, p=self.p, dim=-1)
        den = torch.norm(x, p=self.p, dim=-1) + torch.norm(y, p=self.p, dim=-1) + self.eps
        
        # Clamp to prevent NaN
        ratio = num / den
        ratio = torch.clamp(ratio, 0.0, 2.0)  # Clamp ratio
        
        return 1 - ratio

    def forward(self, emb1, emb2, label):
        # Check for NaN
        if torch.isnan(emb1).any() or torch.isnan(emb2).any():
            print("Warning: NaN in embeddings, returning zero loss")
            return torch.tensor(0.0, device=emb1.device, requires_grad=True)
        
        sim = self.mizan_sim(emb1, emb2)
        
        # Clamp similarity to valid range
        sim = torch.clamp(sim, -1.0, 1.0)
        
        # Positive pairs: maximize similarity (label ~ 1.0)
        # Negative pairs: minimize similarity (label ~ 0.0)
        pos_loss = (1 - label) * (1 - sim)  # For positive pairs
        neg_loss = label * torch.clamp(self.margin - sim, min=0)  # For negative pairs
        
        loss = pos_loss + neg_loss
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            print("Warning: NaN in loss calculation")
            return torch.tensor(0.0, device=emb1.device, requires_grad=True)
        
        return loss.mean()


class MizanLoss(nn.Module):
    """Alternative loss combining direction and scale"""
    
    def __init__(self, alpha=0.15, beta=0.1):
        super().__init__()
        self.alpha = alpha  # Scale weight
        self.beta = beta    # Regularization
        self.eps = 1e-8

    def forward(self, emb1, emb2, labels):
        # Direction loss (cosine-based)
        cos = F.cosine_similarity(emb1, emb2)
        cos = torch.clamp(cos, -1.0, 1.0)
        dir_loss = (1 - cos) * labels + cos * (1 - labels)

        # Scale loss (difference in norms)
        norm1 = torch.norm(emb1, dim=-1)
        norm2 = torch.norm(emb2, dim=-1)
        scale_loss = torch.abs(norm1 - norm2) * self.alpha

        # Small regularization to prevent collapse
        reg_loss = self.beta * (torch.mean(norm1) + torch.mean(norm2))

        total_loss = dir_loss.mean() + scale_loss.mean() + reg_loss
        
        return total_loss


class StableMizanLoss(nn.Module):
    """Most stable version - use this if you still get NaN"""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.eps = 1e-8
    
    def forward(self, emb1, emb2, labels):
        # Simple cosine-based contrastive loss (most stable)
        cos_sim = F.cosine_similarity(emb1, emb2)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Positive pairs: maximize similarity
        # Negative pairs: minimize similarity
        loss = torch.where(
            labels > 0.5,  # Positive pairs
            1 - cos_sim,   # Maximize similarity
            torch.clamp(cos_sim + self.margin, min=0)  # Minimize similarity
        )
        
        return loss.mean()