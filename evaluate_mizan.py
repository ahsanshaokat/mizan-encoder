import os
import json
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModel, AutoTokenizer

# ============================================================
#          SAME ARCHITECTURE AS TRAINING (IMPORTANT)
# ============================================================

class MizanEvalEncoder(nn.Module):
    def __init__(self, backbone, proj_dim, alpha):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hid = self.backbone.config.hidden_size

        self.proj = nn.Linear(hid, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.alpha = alpha

    @classmethod
    def load_finetuned(cls, ckpt):
        cfg = json.load(open(os.path.join(ckpt, "config.json")))
        model = cls(
            backbone=cfg["backbone"],
            proj_dim=cfg["proj_dim"],
            alpha=cfg["alpha"]
        )
        state = torch.load(os.path.join(ckpt, "mizan_encoder.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    # --------------------------
    # Same pooling as trainer
    # --------------------------
    def pool(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def scale(self, x):
        n = torch.norm(x, 2, dim=-1, keepdim=True) + 1e-6
        return x / (n ** self.alpha)

    # --------------------------
    # FIXED FORWARD SIGNATURE
    # --------------------------
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.pool(out.last_hidden_state, attention_mask)
        h = self.norm(self.proj(pooled))
        return self.scale(h)



# ============================================================
#             COSINE & MIZAN SIMILARITY
# ============================================================

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b).item()

def mizan_sim(e1, e2, alpha=0.15):
    dot = (e1 * e2).sum(dim=-1)
    n1 = e1.norm(dim=-1).clamp(min=1e-6)
    n2 = e2.norm(dim=-1).clamp(min=1e-6)

    raw = dot / ((n1**alpha) * (n2**alpha))
    return (raw / (1 + raw.abs())).item()  # → range (-1, +1)



# ============================================================
#         SENTENCE–BASED DIAGNOSTIC COMPARISON
# ============================================================

def sentence_compare(model, tok, device):
    tests = [
        ("A cat sits on the mat.", "A dog sits on the rug."),
        ("I love pizza.", "The moon is blue."),
        ("AI will change the world.", "Artificial intelligence will transform society."),
        ("Quantum physics is difficult.", "I enjoy eating apples.")
    ]

    print("\n==============================")
    print("🔬 SENTENCE-BASED COMPARISON")
    print("==============================")

    for s1, s2 in tests:
        e1 = model(**tok(s1, return_tensors="pt").to(device))
        e2 = model(**tok(s2, return_tensors="pt").to(device))

        cos = cosine_sim(e1, e2)
        miz = mizan_sim(e1, e2)

        print("\n---------------------------------")
        print("📝 S1:", s1)
        print("📝 S2:", s2)
        print(f"Cosine similarity : {cos:.4f}")
        print(f"Mizan similarity  : {miz:.4f}")
        print(f"Δ (Mizan - Cosine): {miz - cos:+.4f}")


# ============================================================
#                 STS-B EVALUATION
# ============================================================

def load_sts(path):
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
            pairs.append((parts[5], parts[6], score))
    return pairs


def evaluate_sts(model, tok, pairs, device):

    cos_preds, miz_preds, golds = [], [], []

    for s1, s2, g in pairs:
        e1 = model(**tok(s1, return_tensors="pt").to(device))
        e2 = model(**tok(s2, return_tensors="pt").to(device))

        cos_preds.append(cosine_sim(e1, e2))
        miz_preds.append(mizan_sim(e1, e2))
        golds.append(g)

    # correlations
    cp = pearsonr(cos_preds, golds)[0]
    cs = spearmanr(cos_preds, golds)[0]
    mp = pearsonr(miz_preds, golds)[0]
    ms = spearmanr(miz_preds, golds)[0]

    print("\n==============================")
    print("📊 STS-B EVALUATION")
    print("==============================")

    print("\nCosine:")
    print(" Pearson :", cp)
    print(" Spearman:", cs)

    print("\nMizan:")
    print(" Pearson :", mp)
    print(" Spearman:", ms)

    print("\nΔ:")
    print(" Pearson Δ:", mp - cp)
    print(" Spearman Δ:", ms - cs)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    ckpt = "checkpoints/mizan_v10"
    sts_dev = "scripts/data/sts_raw/STS-B/dev.tsv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = MizanEvalEncoder.load_finetuned(ckpt).to(device)
    model.eval()

    sentence_compare(model, tok, device)

    pairs = load_sts(sts_dev)
    evaluate_sts(model, tok, pairs, device)

    print("\n🎯 Evaluation Complete")
