import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer

from .pooling import BalancedMeanPooling


class MizanEncoderConfig(PretrainedConfig):
    model_type = "mizan-encoder"

    def __init__(self, backbone_name="sentence-transformers/all-MiniLM-L6-v2",
                 pooling="balanced-mean", proj_dim=384, **kwargs):
        super().__init__(**kwargs)

        self.backbone_name = backbone_name
        self.pooling = pooling
        self.proj_dim = proj_dim


class MizanEncoderHF(PreTrainedModel):
    config_class = MizanEncoderConfig

    def __init__(self, config, **unused):
        super().__init__(config)

        self.backbone = AutoModel.from_pretrained(config.backbone_name)
        hidden = self.backbone.config.hidden_size

        self.pooling = BalancedMeanPooling()
        self.proj = nn.Linear(hidden, config.proj_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):

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

    def encode(self, sentences, tokenizer=None, device="cpu"):
        if isinstance(sentences, str):
            sentences = [sentences]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        enc = tokenizer(sentences, return_tensors="pt",
                        padding=True, truncation=True).to(device)

        with torch.no_grad():
            return self.forward(
                enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")
            )


def load_encoder(path):
    return MizanEncoderHF.from_pretrained(path)
