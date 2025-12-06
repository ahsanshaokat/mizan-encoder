import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# =====================================================================
# Debug Mizan Encoder – loads HF backbone + your finetuned projection
# =====================================================================

class DebugMizanEncoder(nn.Module):
    def __init__(self, backbone="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=384, alpha=0.15):
        super().__init__()

        print(f"\n🔧 Loading HF backbone: {backbone}")
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size

        # Projection head (trained)
        self.proj = nn.Linear(hidden, proj_dim)
        self.alpha = alpha

    # -----------------------------------------------------------------
    # Factory loader for finetuned checkpoint folder
    # -----------------------------------------------------------------
    @classmethod
    def load_finetuned(cls, ckpt_dir):
        """
        ckpt_dir must contain:
        - config.json
        - mizan_encoder.pt
        - tokenizer files
        """
        print(f"\n📥 Loading fine-tuned encoder from: {ckpt_dir}")

        config_path = os.path.join(ckpt_dir, "config.json")
        weights_path = os.path.join(ckpt_dir, "mizan_encoder.pt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {ckpt_dir}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing mizan_encoder.pt in {ckpt_dir}")

        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        print(f"Config loaded: {cfg}")

        # Create model
        model = cls(
            backbone=cfg["backbone"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"]
        )

        # Load weights
        print("\n🔍 Loading weights from mizan_encoder.pt...")
        state = torch.load(weights_path, map_location="cpu")

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"   Missing keys   : {missing}")
        print(f"   Unexpected keys: {unexpected}")
        print("✅ Weights loaded successfully.\n")

        return model

    # -----------------------------------------------------------------
    # Safe Mean Pooling
    # -----------------------------------------------------------------
    def safe_pool(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)
        mask_sum = mask.sum(dim=1).clamp(min=1e-6)
        pooled = (hidden * mask).sum(dim=1) / mask_sum
        return pooled

    # -----------------------------------------------------------------
    # Mizan normalization
    # -----------------------------------------------------------------
    def scale_stabilize(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        return x / (norm ** self.alpha)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.safe_pool(out.last_hidden_state, attention_mask)
        projected = self.proj(pooled)
        stabilized = self.scale_stabilize(projected)
        return stabilized


# =====================================================================
# TESTING UTILITIES
# =====================================================================

def test_single_sentence(model, tokenizer):
    print("\n==============================")
    print("🧪 TESTING SINGLE SENTENCE")
    print("==============================")

    sentences = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "A",
        ""
    ]

    for text in sentences:
        print(f"\n📌 Input: '{text}'")
        if text.strip() == "":
            text = " "

        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            emb = model(**enc)

        print("Embedding shape:", emb.shape)
        print("Norm:", torch.norm(emb, dim=-1).item())
        print("NaN:", torch.isnan(emb).any().item())


def test_batch(model, tokenizer):
    print("\n==============================")
    print("🧪 TESTING BATCH")
    print("==============================")

    batch = [
        "This is sentence one.",
        "Here is sentence two.",
        "Another sentence in the batch."
    ]

    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        emb = model(**enc)

    print("Output shape:", emb.shape)
    print("NaN:", torch.isnan(emb).any().item())


# =====================================================================
# MAIN SCRIPT
# =====================================================================

if __name__ == "__main__":
    ckpt = "checkpoints/mizan_singlefile"

    print("\n==============================")
    print("🔬 LOADING TOKENIZER")
    print("==============================")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    print("\n==============================")
    print("🔬 LOADING MODEL")
    print("==============================")

    model = DebugMizanEncoder.load_finetuned(ckpt)

    # ---- Tests ----
    test_single_sentence(model, tokenizer)
    test_batch(model, tokenizer)

    print("\n🎯 DEBUGGING COMPLETE\n")
