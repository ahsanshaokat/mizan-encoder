
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

    def scale_stabilize(self, x):
        """Safe Mizan normalization"""
        eps = 1e-6
        
        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️ WARNING: NaN/Inf in scale_stabilize input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calculate norm with safety
        norm = torch.norm(x, dim=-1, keepdim=True)
        
        # Clamp norm to safe range
        norm = torch.clamp(norm, min=1e-6, max=1e6)
        
        # Apply Mizan normalization
        result = x / (norm**self.alpha + eps)
        
        # Final safety check
        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Safety check inputs
        if torch.isnan(input_ids).any():
            print("⚠️ WARNING: NaN in input_ids")
            return torch.zeros((input_ids.size(0), self.proj_dim), 
                             device=input_ids.device)
        
        try:
            supports_tti = "token_type_ids" in self.transformer.forward.__code__.co_varnames

            out = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if supports_tti else None,
                return_dict=True
            )

            # Check transformer output
            if torch.isnan(out.last_hidden_state).any():
                print("⚠️ WARNING: NaN in transformer output")
                # Return zero embeddings
                return torch.zeros((input_ids.size(0), self.proj_dim), 
                                 device=input_ids.device)

            pooled = self.pooler(out.last_hidden_state, attention_mask)
            
            # Check pooling output
            if torch.isnan(pooled).any():
                print("⚠️ WARNING: NaN after pooling")
                # Initialize projection layer and use it
                projected = self.proj(torch.zeros_like(pooled))
            else:
                projected = self.proj(pooled)
            
            stabilized = self.scale_stabilize(projected)
            
            # Final safety check
            if torch.isnan(stabilized).any():
                print("⚠️ WARNING: NaN in final output, returning zeros")
                stabilized = torch.zeros_like(stabilized)
            
            return stabilized
            
        except Exception as e:
            print(f"⚠️ ERROR in forward pass: {e}")
            # Return safe zero embeddings
            return torch.zeros((input_ids.size(0), self.proj_dim), 
                             device=input_ids.device)


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