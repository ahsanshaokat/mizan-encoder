import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel
from .pooling import BalancedMeanPooling


class MizanTextEncoder(nn.Module):
    """
    MizanTextEncoder
    A scale-aware embedding model compatible with save_pretrained / from_pretrained.
    """

    def __init__(
        self,
        backbone: str = "distilbert-base-uncased",
        proj_dim: int = 384,
        alpha: float = 0.2,
        load_transformer: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.proj_dim = proj_dim
        self.alpha = alpha

        if load_transformer:
            self.transformer = AutoModel.from_pretrained(backbone)
            hidden_size = self.transformer.config.hidden_size
        else:
            # placeholder for loading state_dict
            self.transformer = None
            hidden_size = None

        self.pooler = BalancedMeanPooling()
        self.proj = nn.Linear(hidden_size, proj_dim)

    # -------------------------------------------------------------
    def scale_stabilize(self, x):
        eps = 1e-6
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + eps) ** self.alpha

    # -------------------------------------------------------------
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Conditional passing of token_type_ids
        supports_tti = "token_type_ids" in self.transformer.forward.__code__.co_varnames

        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if supports_tti else None
        )

        pooled = self.pooler(out.last_hidden_state, attention_mask)
        projected = self.proj(pooled)
        stabilized = self.scale_stabilize(projected)
        return stabilized
    
    # -------------------------------------------------------------
    # Save the Encoder (like HuggingFace)
    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)

        # Save config.json
        config = {
            "backbone": self.backbone,
            "proj_dim": self.proj_dim,
            "alpha": self.alpha
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(directory, "pytorch_model.bin"))
        print(f"Saved model to {directory}")

    # -------------------------------------------------------------
    # Load the Encoder (like HuggingFace)
    @classmethod
    def from_pretrained(cls, directory):
        with open(os.path.join(directory, "config.json"), "r") as f:
            cfg = json.load(f)

        model = cls(
            backbone=cfg["backbone"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"],
            load_transformer=True
        )

        state_dict = torch.load(os.path.join(directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded model from {directory}")
        return model
