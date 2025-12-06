import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from scipy.stats import pearsonr, spearmanr

# ============================================================
#                  MIZAN ENCODER LOADER
# ============================================================

class MizanEvalEncoder(nn.Module):
    def __init__(self, backbone, proj_dim=384, alpha=0.15):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone)
        hidden = self.backbone.config.hidden_size

        self.proj = nn.Linear(hidden, proj_dim)
        self.alpha = alpha

    @classmethod
    def load_finetuned(cls, ckpt_dir):
        cfg_path = os.path.join(ckpt_dir, "config.json")
        weights_path = os.path.join(ckpt_dir, "mizan_encoder.pt")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError("Missing config.json")

        if not os.path.exists(weights_path):
            raise FileNotFoundError("Missing mizan_encoder.pt")

        cfg = json.load(open(cfg_path))
        print("Loaded config:", cfg)

        model = cls(
            backbone=cfg["backbone"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"]
        )

        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:   ", missing)
        print("Unexpected keys:", unexpected)
        print("✔ Loaded model weights successfully")

        return model

    # -----------------------
    # Pooling & Scaling
    # -----------------------
    def safe_pool(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (hidden * mask).sum(dim=1) / denom

    def scale_stabilize(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        return x / (n ** self.alpha)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.safe_pool(out.last_hidden_state, attention_mask)
        h = self.proj(pooled)
        return self.scale_stabilize(h)


# ============================================================
#              MIZAN & COSINE SIMILARITY FUNCTIONS
# ============================================================

def cosine_sim(e1, e2):
    return torch.nn.functional.cosine_similarity(e1, e2).cpu().item()


def mizan_sim(e1, e2, alpha=0.15):
    dot = (e1 * e2).sum(dim=-1)

    n1 = e1.norm(dim=-1).clamp(min=1e-6)
    n2 = e2.norm(dim=-1).clamp(min=1e-6)

    denom = (n1 ** alpha) * (n2 ** alpha)
    return (dot / denom).cpu().item()


# ============================================================
#                   STS-B DATA LOADER
# ============================================================

def load_sts_test(path):
    """Load STS-B test set: returns (s1, s2, score)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            try:
                score = float(parts[4])
            except:
                continue
            s1, s2 = parts[5], parts[6]
            score = score  # keep original 0–5 scale
            pairs.append((s1, s2, score))
    return pairs


# ============================================================
#                 FULL STS-B EVALUATION
# ============================================================

def evaluate_sts(model, tokenizer, sts_pairs, alpha=0.15, device="cpu"):
    cos_preds = []
    mizan_preds = []
    gold_scores = []

    for s1, s2, gold in sts_pairs:
        enc1 = tokenizer(s1, return_tensors="pt", truncation=True, padding=True).to(device)
        enc2 = tokenizer(s2, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            e1 = model(**enc1)
            e2 = model(**enc2)

        cos = cosine_sim(e1, e2)
        miz = mizan_sim(e1, e2, alpha=alpha)

        cos_preds.append(cos)
        mizan_preds.append(miz)
        gold_scores.append(gold)

    # Pearson & Spearman correlations
    cos_p = pearsonr(cos_preds, gold_scores)[0]
    cos_s = spearmanr(cos_preds, gold_scores)[0]

    miz_p = pearsonr(mizan_preds, gold_scores)[0]
    miz_s = spearmanr(mizan_preds, gold_scores)[0]

    print("\n==============================")
    print("📊 STS-B EVALUATION RESULTS")
    print("==============================")

    print("\n🔵 COSINE SIMILARITY")
    print(f"Pearson : {cos_p:.4f}")
    print(f"Spearman: {cos_s:.4f}")

    print("\n🟣 MIZAN SIMILARITY")
    print(f"Pearson : {miz_p:.4f}")
    print(f"Spearman: {miz_s:.4f}")

    print("\nΔ Improvement (Mizan − Cosine)")
    print(f"Pearson : {miz_p - cos_p:+.4f}")
    print(f"Spearman: {miz_s - cos_s:+.4f}")

    return {
        "cosine_pearson": cos_p,
        "cosine_spearman": cos_s,
        "mizan_pearson": miz_p,
        "mizan_spearman": miz_s
    }


# ============================================================
#                   MANUAL PAIR TESTING
# ============================================================

def test_similarity_pairs(model, tokenizer, alpha=0.15, device="cpu"):
    pairs = [
        ("A cat sits on the mat.", "A dog sits on the rug."),
        ("The stock market crashed today.", "Financial markets fell."),
        ("I love pizza.", "The moon is blue."),
        ("AI will change the world.", "Artificial intelligence will transform society.")
    ]

    print("\n==============================")
    print("🔎 SEMANTIC SIMILARITY PAIR TESTS")
    print("==============================")

    for s1, s2 in pairs:
        e1 = model(**tokenizer(s1, return_tensors="pt").to(device))
        e2 = model(**tokenizer(s2, return_tensors="pt").to(device))

        cos = cosine_sim(e1, e2)
        miz = mizan_sim(e1, e2, alpha)

        print("\n---------------------------------")
        print(f"📝 {s1}")
        print(f"📝 {s2}")
        print(f"Cosine similarity: {cos:.4f}")
        print(f"Mizan similarity : {miz:.4f}")
        print(f"Δ = {miz - cos:+.4f}")


# ============================================================
#                       MAIN
# ============================================================

if __name__ == "__main__":
    ckpt = "checkpoints/mizan_properloss"   # <-- update if needed
    sts_test_path = "scripts/data/sts_raw/STS-B/dev.tsv"  # or test.tsv

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading tokenizer + model…")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = MizanEvalEncoder.load_finetuned(ckpt).to(device)
    model.eval()

    # Load STS-B dataset
    pairs = load_sts_test(sts_test_path)

    # Run similarity tests
    test_similarity_pairs(model, tokenizer, alpha=0.15, device=device)

    # Run STS-B evaluation
    evaluate_sts(model, tokenizer, pairs, alpha=0.15, device=device)

    print("\n🎯 Evaluation complete.")
