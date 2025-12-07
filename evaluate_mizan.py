import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ============================================================
#                    LOAD MIZAN ENCODER
# ============================================================

class MizanEvalEncoder(nn.Module):
    def __init__(self, backbone, proj_dim=384, alpha=0.2):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(backbone)
        hid = self.transformer.config.hidden_size

        self.proj = nn.Linear(hid, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.alpha = alpha

    @classmethod
    def load_finetuned(cls, ckpt_dir):
        """Load saved model + config."""
        cfg_path = os.path.join(ckpt_dir, "config.json")
        weights_path = os.path.join(ckpt_dir, "mizan_encoder.pt")

        cfg = json.load(open(cfg_path))
        print("Loaded config:", cfg)

        model = cls(
            backbone=cfg["backbone"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"],
        )

        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        print("✔ Encoder loaded.\n")

        return model

    def safe_pool(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (hidden * mask).sum(dim=1) / denom

    def scale_stabilize(self, x):
        n = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6
        return x / (n ** self.alpha)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.safe_pool(out.last_hidden_state, attention_mask)
        h = self.norm(self.proj(pooled))
        return self.scale_stabilize(h)


# ============================================================
#           COSINE + MIZAN SIMILARITY (Article #10)
# ============================================================

def cosine_sim(e1, e2):
    return torch.nn.functional.cosine_similarity(e1, e2).item()


def mizan_sim(e1, e2):
    """
    Mizan Similarity from Article #10:
    sim = 1 - ||x - y|| / (||x|| + ||y||)
    Output range: (-inf, 1], but usually 0–1 for trained positives.
    """
    num = torch.norm(e1 - e2, p=2)
    den = torch.norm(e1, p=2) + torch.norm(e2, p=2) + 1e-6
    return (1 - (num / den)).item()


# ============================================================
#               SENTENCE-BASED EVALUATION
# ============================================================

def sentence_compare(model, tokenizer, device):

    test_pairs = [
        ("A cat sits on the mat.", "A dog sits on the rug."),
        ("I love pizza.", "The moon is blue."),
        ("AI will change the world.", "Artificial intelligence will transform society."),
        ("Quantum physics is difficult.", "I enjoy eating apples."),
        ("A man is playing guitar.", "A person is performing music.")
    ]

    print("\n==============================")
    print("🔬 SENTENCE-BASED COMPARISON")
    print("==============================\n")

    for s1, s2 in test_pairs:

        t1 = tokenizer(s1, return_tensors="pt").to(device)
        t2 = tokenizer(s2, return_tensors="pt").to(device)

        with torch.no_grad():
            e1 = model(**t1)
            e2 = model(**t2)

        cos = cosine_sim(e1, e2)
        miz = mizan_sim(e1, e2)

        print("---------------------------------")
        print(f"📝 S1: {s1}")
        print(f"📝 S2: {s2}")
        print(f"Cosine similarity : {cos:.4f}")
        print(f"Mizan similarity  : {miz:.4f}")

        # Interpretation
        if miz > cos:
            explanation = "Mizan detects stronger semantic closeness than cosine."
        elif miz < cos:
            explanation = "Cosine shows stronger alignment; Mizan adds scale penalty."
        else:
            explanation = "Both metrics agree equally."

        print(f"Explanation       : {explanation}")
        print(f"Δ (Mizan - Cosine): {miz - cos:+.4f}\n")


# ============================================================
#                        MAIN
# ============================================================

if __name__ == "__main__":
    ckpt = "checkpoints/mizan_v10"   # <-- your trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer + model...\n")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = MizanEvalEncoder.load_finetuned(ckpt).to(device)
    model.eval()

    sentence_compare(model, tokenizer, device)

    print("\n🎯 Evaluation Complete")
