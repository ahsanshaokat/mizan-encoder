import torch
import torch.nn as nn
import torch.nn.functional as F

class MizanContrastiveLoss(nn.Module):
    """
    Mizan Contrastive Loss
    ----------------------

    Positive pairs:
        loss = 1 - mizan_sim(x, y)

    Negative pairs:
        loss = max( margin - mizan_sim(x, y), 0 )

    Where:
        mizan_sim = 1 - ( ||x - y||_p / (||x||_p + ||y||_p + eps) )
    """

    def __init__(self, margin=0.5, p=2, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps

    def mizan_sim(self, x, y):
        num = torch.norm(x - y, p=self.p, dim=-1)
        den = torch.norm(x, p=self.p, dim=-1) + torch.norm(y, p=self.p, dim=-1) + self.eps
        return 1 - (num / den)

    def forward(self, emb1, emb2, label):
        sim = self.mizan_sim(emb1, emb2)
        pos_loss = 1 - sim
        neg_loss = torch.clamp(self.margin - sim, min=0)
        return torch.where(label == 1, pos_loss, neg_loss).mean()


class MizanLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, emb1, emb2, labels):
        # direction loss
        cos = F.cosine_similarity(emb1, emb2)
        dir_loss = (1 - cos) * labels + (cos) * (1 - labels)

        # scale loss
        scale1 = emb1.norm(dim=-1)
        scale2 = emb2.norm(dim=-1)
        scale_loss = ((scale1 - scale2).abs()) * self.alpha

        return dir_loss.mean() + scale_loss.mean()
