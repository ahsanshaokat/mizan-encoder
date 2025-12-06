"""
HuggingFace-compatible Mizan Encoder
------------------------------------
Allows:
✔ save_pretrained()
✔ from_pretrained()
✔ encode() for inference
✔ Extra config fields (proj_dim, pooling, backbone_name)
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
                 proj_dim=384,
                 **kwargs):

        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.pooling = pooling
        self.proj_dim = proj_dim  # instead of emb_dim (HF safe)


# ---------------------------------------------------------------------
# HuggingFace Model Wrapper
# ---------------------------------------------------------------------
class MizanEncoderHF(PreTrainedModel):
    config_class = MizanEncoderConfig

    def __init__(self, config, **unused_kwargs):
        """
        NOTE: HF may pass additional arguments stored in config.json.
        We MUST accept **unused_kwargs to prevent crashes.
        """
        super().__init__(config)

        # Load backbone transformer
        self.backbone = AutoModel.from_pretrained(config.backbone_name)

        # Decide pooling
        self.pooling = BalancedMeanPooling()

        # Optional projection head
        hidden = self.backbone.config.hidden_size
        proj_dim = config.proj_dim

        self.proj = nn.Linear(hidden, proj_dim)

    # --------------------------------------------------------------
    # Forward pass used for training
    # --------------------------------------------------------------
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Check if backbone supports token_type_ids
        supports_tti = "token_type_ids" in self.backbone.forward.__code__.co_varnames

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if supports_tti else None
        )

        pooled = self.pooling(out.last_hidden_state, attention_mask)
        emb = self.proj(pooled)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

    # --------------------------------------------------------------
    # Encode API for inference
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
        ).to(device)

        with torch.no_grad():
            emb = self.forward(
                enc["input_ids"],
                enc["attention_mask"],
                enc.get("token_type_ids")
            )

        # return (batch, dim)
        return emb

# ---------------------------------------------------------------------
# Load from directory
# ---------------------------------------------------------------------
def load_encoder(path):
    return MizanEncoderHF.from_pretrained(path)
