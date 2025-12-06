import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel
from .pooling import BalancedMeanPooling


class MizanTextEncoder(nn.Module):

    def __init__(self, backbone="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=384, alpha=0.2, load_transformer=True):
        super().__init__()

        self.backbone_name = backbone
        self.proj_dim = proj_dim
        self.alpha = alpha

        if load_transformer:
            self.transformer = AutoModel.from_pretrained(backbone)
            hidden = self.transformer.config.hidden_size
        else:
            self.transformer = None
            hidden = proj_dim

        self.pooler = BalancedMeanPooling()
        self.proj = nn.Linear(hidden, proj_dim)

    # -----------------------------------------
    def scale_stabilize(self, x):
        """Mizan vector normalization: x / norm^alpha"""
        eps = 1e-8  # Increased for stability
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Clamp norm to prevent division by near-zero
        norm = torch.clamp(norm, min=1e-6)
        return x / (norm**self.alpha + eps)

    # -----------------------------------------
    def forward(self, input_ids, attention_mask, token_type_ids=None):

        supports_tti = "token_type_ids" in self.transformer.forward.__code__.co_varnames

        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if supports_tti else None,
        )

        pooled = self.pooler(out.last_hidden_state, attention_mask)
        projected = self.proj(pooled)
        stabilized = self.scale_stabilize(projected)

        return stabilized

    # -----------------------------------------
    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)

        cfg = {
            "backbone_name": self.backbone_name,
            "proj_dim": self.proj_dim,
            "alpha": self.alpha
        }

        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        torch.save(self.state_dict(), os.path.join(directory, "pytorch_model.bin"))

    # -----------------------------------------
    @classmethod
    def from_pretrained(cls, directory):
        cfg = json.load(open(os.path.join(directory, "config.json")))

        model = cls(
            backbone=cfg["backbone_name"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"],
            load_transformer=True
        )

        weights = os.path.join(directory, "pytorch_model.bin")
        sd = torch.load(weights, map_location="cpu")
        model.load_state_dict(sd, strict=False)

        return model