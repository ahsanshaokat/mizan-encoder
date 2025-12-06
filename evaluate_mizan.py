import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from scipy.stats import pearsonr, spearmanr

# Same normalized Mizan similarity
def mizan_sim_norm(e1, e2, alpha=0.15):
    dot = (e1 * e2).sum(dim=-1)
    n1 = torch.norm(e1, dim=-1).clamp(min=1e-6)
    n2 = torch.norm(e2, dim=-1).clamp(min=1e-6)
    miz_raw = dot / ((n1**alpha) * (n2**alpha))
    return miz_raw / (1 + miz_raw.abs())  # (-1,1)


# -------------------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------------------

class MizanEncoder(nn.Module):
    def __init__(self, backbone, proj_dim=384, alpha=0.15):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hid = self.backbone.config.hidden_size

        self.pool = lambda h, m: (h*m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        self.proj = nn.Linear(hid, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.alpha = alpha

    def mizan_scale(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        return x / (n ** self.alpha)

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        pooled = self.pool(out.last_hidden_state, mask)
        h = self.norm(self.proj(pooled))
        return self.mizan_scale(h)


def load_model(ckpt):
    cfg = json.load(open(f"{ckpt}/config.json"))
    model = MizanEncoder(
        backbone=cfg["backbone"],
        proj_dim=cfg["proj_dim"],
        alpha=cfg["alpha"]
    )
    state = torch.load(f"{ckpt}/mizan_encoder.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    return model, tokenizer, cfg


# -------------------------------------------------------------
# STS-B EVALUATION
# -------------------------------------------------------------

def evaluate_stsb(model, tokenizer, cfg):
    path = cfg["sts_path"]
    pairs = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 7: continue
            try: score = float(p[4]) / 5.0
            except: continue
            pairs.append((p[5], p[6], score))

    cos_preds = []
    miz_preds = []
    gold = []

    for s1, s2, y in pairs:
        t1 = tokenizer(s1, return_tensors="pt")
        t2 = tokenizer(s2, return_tensors="pt")

        with torch.no_grad():
            e1 = model(t1["input_ids"], t1["attention_mask"])
            e2 = model(t2["input_ids"], t2["attention_mask"])

        cos = torch.nn.functional.cosine_similarity(e1, e2).item()
        miz = mizan_sim_norm(e1, e2).item()

        cos_preds.append(cos)
        miz_preds.append(miz)
        gold.append(y)

    print("\n===== STS-B Evaluation =====")
    print("COSINE Pearson :", pearsonr(cos_preds, gold)[0])
    print("COSINE Spearman:", spearmanr(cos_preds, gold)[0])

    print("\nMIZAN Pearson  :", pearsonr(miz_preds, gold)[0])
    print("MIZAN Spearman :", spearmanr(miz_preds, gold)[0])


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    ckpt = "checkpoints/mizan_proper"
    model, tokenizer, cfg = load_model(ckpt)

    evaluate_stsb(model, tokenizer, cfg)
