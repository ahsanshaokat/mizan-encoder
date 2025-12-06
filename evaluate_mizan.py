import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from scipy.stats import pearsonr, spearmanr

# ============================================================
#                  MIZAN ENCODER FOR EVAL
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
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        model = cls(
            backbone=cfg["backbone"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"]
        )

        state = torch.load(os.path.join(ckpt_dir, "mizan_encoder.pt"), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)

        print("Loaded config:", cfg)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        print("✔ Model loaded.\n")

        return model

    def safe_pool(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return (hidden * mask).sum(dim=1) / denom

    def scale_stabilize(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        return x / (n ** self.alpha)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(input_ids, attention_mask, token_type_ids)
        pooled = self.safe_pool(out.last_hidden_state, attention_mask)
        h = self.proj(pooled)
        return self.scale_stabilize(h)


# ============================================================
#               TRUE MIZAN SIMILARITY (training-consistent)
# ============================================================

def mizan_similarity(e1, e2, p=2, eps=1e-6):
    num = torch.norm(e1 - e2, p=p, dim=-1)
    den = torch.norm(e1, p=p, dim=-1) + torch.norm(e2, p=p, dim=-1) + eps
    return (1 - num / den).cpu().item()


def cosine_sim(e1, e2):
    return torch.nn.functional.cosine_similarity(e1, e2).cpu().item()


# ============================================================
#             SENTENCE-BASED COMPARISON (NEW)
# ============================================================

def test_sentence_comparisons(model, tokenizer, alpha=0.15, device="cpu"):
    print("\n==============================")
    print("🔬 SENTENCE-BASED COMPARISON")
    print("==============================")

    tests = [
        ("A cat sits on the mat.", "A dog sits on the rug."),
        ("The stock market crashed today.", "Financial markets fell sharply."),
        ("I love pizza.", "The moon is blue."),
        ("AI will change the world.", "Artificial intelligence will transform society."),
        ("A man is playing a guitar.", "A person is performing music."),
        ("Quantum physics is difficult.", "I enjoy eating apples.")
    ]

    for s1, s2 in tests:

        e1 = model(**tokenizer(s1, return_tensors="pt").to(device))
        e2 = model(**tokenizer(s2, return_tensors="pt").to(device))

        cos = cosine_sim(e1, e2)
        miz = mizan_similarity(e1, e2)

        print("\n---------------------------------")
        print(f"📝 S1: {s1}")
        print(f"📝 S2: {s2}")
        print(f"Cosine similarity : {cos:.4f}")
        print(f"Mizan similarity  : {miz:.4f}")
        print(f"Δ (Mizan - Cosine): {miz - cos:+.4f}")


# ============================================================
#                   STS-B LOADER & EVAL
# ============================================================

def load_sts_test(path):
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
            pairs.append((s1, s2, score))
    return pairs


def evaluate_sts(model, tokenizer, sts_pairs, alpha=0.15, device="cpu"):
    cos_preds, miz_preds, golds = [], [], []

    for s1, s2, gold in sts_pairs:

        e1 = model(**tokenizer(s1, return_tensors="pt").to(device))
        e2 = model(**tokenizer(s2, return_tensors="pt").to(device))

        cos_preds.append(cosine_sim(e1, e2))
        miz_preds.append(mizan_similarity(e1, e2))
        golds.append(gold)

    c_p = pearsonr(cos_preds, golds)[0]
    c_s = spearmanr(cos_preds, golds)[0]

    m_p = pearsonr(miz_preds, golds)[0]
    m_s = spearmanr(miz_preds, golds)[0]

    print("\n==============================")
    print("📊 STS-B EVALUATION RESULTS")
    print("==============================")

    print("\n🔵 COSINE SIMILARITY")
    print(f"Pearson : {c_p:.4f}")
    print(f"Spearman: {c_s:.4f}")

    print("\n🟣 MIZAN SIMILARITY")
    print(f"Pearson : {m_p:.4f}")
    print(f"Spearman: {m_s:.4f}")

    print("\nΔ Improvement (Mizan − Cosine)")
    print(f"Pearson : {m_p - c_p:+.4f}")
    print(f"Spearman: {m_s - c_s:+.4f}")

    return {
        "cosine_pearson": c_p,
        "cosine_spearman": c_s,
        "mizan_pearson": m_p,
        "mizan_spearman": m_s
    }


# ============================================================
#                       MAIN
# ============================================================

if __name__ == "__main__":
    ckpt = "checkpoints/mizan_proper"
    sts_test_path = "scripts/data/sts_raw/STS-B/dev.tsv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading tokenizer + model…")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = MizanEvalEncoder.load_finetuned(ckpt).to(device)
    model.eval()

    # New: Sentence-level comparison
    test_sentence_comparisons(model, tokenizer, device=device)

    # STS-B
    sts_pairs = load_sts_test(sts_test_path)
    evaluate_sts(model, tokenizer, sts_pairs, device=device)

    print("\n🎯 Evaluation complete.\n")
