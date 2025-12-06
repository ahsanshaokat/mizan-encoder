import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel

from .pooling import BalancedMeanPooling


class MizanEncoderConfig(PretrainedConfig):
    model_type = "mizan-encoder"

    def __init__(self, backbone_name="sentence-transformers/all-MiniLM-L6-v2",
                 pooling="balanced-mean", proj_dim=384, alpha=0.15, **kwargs):
        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.pooling = pooling
        self.proj_dim = proj_dim
        self.alpha = alpha  # CRITICAL: Add alpha parameter


class MizanEncoderHF(PreTrainedModel):
    config_class = MizanEncoderConfig

    def __init__(self, config, **unused):
        super().__init__(config)

        self.backbone = AutoModel.from_pretrained(config.backbone_name)
        hidden = self.backbone.config.hidden_size

        self.pooler = BalancedMeanPooling()
        self.proj = nn.Linear(hidden, config.proj_dim)
        self.alpha = config.alpha  # Use alpha from config

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        supports_tti = "token_type_ids" in self.backbone.forward.__code__.co_varnames

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if supports_tti else None
        )

        pooled = self.pooler(out.last_hidden_state, attention_mask)
        emb = self.proj(pooled)
        
        # MIZAN NORMALIZATION (Same as encoder.py)
        eps = 1e-8
        norm = torch.norm(emb, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-6)  # Clamp to prevent division by near-zero
        emb = emb / (norm**self.alpha + eps)
        
        return emb

    def encode(self, sentences, tokenizer=None, device="cpu"):
        if isinstance(sentences, str):
            sentences = [sentences]

        from transformers import AutoTokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        enc = tokenizer(sentences, return_tensors="pt",
                        padding=True, truncation=True).to(device)

        with torch.no_grad():
            return self.forward(
                enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")
            )