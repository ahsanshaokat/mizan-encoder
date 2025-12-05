"""
HuggingFace-compatible Mizan Encoder
------------------------------------
Allows:
✔ save_pretrained()
✔ from_pretrained()
✔ encode() for inference
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer

from .pooling import BalancedMeanPooling


# ---------------------------------------------------------------------
# Model Config
# ---------------------------------------------------------------------
class MizanEncoderConfig(PretrainedConfig):
    model_type = "mizan-encoder"

    def __init__(self,
                 backbone_name="sentence-transformers/all-MiniLM-L6-v2",
                 pooling="balanced-mean",
                 emb_dim=384,
                 **kwargs):
        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.pooling = pooling
        self.emb_dim = emb_dim


# ---------------------------------------------------------------------
# HuggingFace Model Wrapper
# ---------------------------------------------------------------------
class MizanEncoderHF(PreTrainedModel):
    config_class = MizanEncoderConfig

    def __init__(self, config):
        super().__init__(config)

        # Load backbone transformer
        self.backbone = AutoModel.from_pretrained(config.backbone_name)

        # Decide pooling
        self.pooling = BalancedMeanPooling()

        # Optional projection head
        self.proj = nn.Linear(self.backbone.config.hidden_size,
                              config.emb_dim)

    # --------------------------------------------------------------
    # Forward pass used for training
    # --------------------------------------------------------------
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask)

        pooled = self.pooling(out.last_hidden_state, attention_mask)
        emb = self.proj(pooled)

        return emb

    # --------------------------------------------------------------
    # Encode with text input
    # --------------------------------------------------------------
    def encode(self, sentences, tokenizer=None, device="cpu"):
        if isinstance(sentences, str):
            sentences = [sentences]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        enc = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            emb = self.forward(input_ids, attention_mask)

        return emb.squeeze(0)


# ---------------------------------------------------------------------
# Load from directory
# ---------------------------------------------------------------------
def load_encoder(path):
    return MizanEncoderHF.from_pretrained(path)
