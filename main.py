import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# ============================================================
#              BALANCED MEAN POOLING (LOG INCLUDED)
# ============================================================

class BalancedMeanPooling(nn.Module):
    def forward(self, hidden, mask):
        mask_exp = mask.unsqueeze(-1).float()
        summed = (hidden * mask_exp).sum(dim=1)
        denom = mask_exp.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom

        print("\n[POOL DEBUG]")
        print("  mask_sum:", denom[:3].tolist())
        print("  pooled_norm:", torch.norm(pooled, dim=-1)[:3].tolist())

        return pooled


# ============================================================
#                MIZAN MAPPED ENCODER (NO TRAINING)
# ============================================================

class MizanMappedEncoder(nn.Module):
    """
    Maps HF encoder → BalancedPooling → Linear → LayerNorm → ScaleStabilizer.
    NO TRAINING. Pure deterministic mapping.
    """

    def __init__(self, backbone="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=384, alpha=0.2):
        super().__init__()

        print(f"\n🔵 Loading Transformer Backbone: {backbone}")
        self.transformer = AutoModel.from_pretrained(backbone)

        self.hidden = self.transformer.config.hidden_size
        self.pool = BalancedMeanPooling()
        self.proj = nn.Linear(self.hidden, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

        self.alpha = alpha

        # Init projection
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        print("✔ Projection Layer Initialized")
        print(f"✔ Scale Stabilizer Alpha = {alpha}")

    def scale_stabilize(self, x):
        n = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6
        stabilized = x / (n ** self.alpha)

        print("\n[SCALE DEBUG]")
        print("  raw_norms:", n.squeeze()[:3].tolist())
        print("  stabilized_norms:", torch.norm(stabilized, dim=-1)[:3].tolist())

        return stabilized

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self.pool(out.last_hidden_state, attention_mask)
        projected = self.proj(pooled)
        normalized = self.norm(projected)

        print("\n[PROJ DEBUG]")
        print("  projected_norm:", torch.norm(projected, dim=-1)[:3].tolist())

        encoded = self.scale_stabilize(normalized)

        print("[ENCODER DEBUG] final_embedding_norm:", torch.norm(encoded, dim=-1)[:3].tolist())
        return encoded


# ============================================================
#            COSINE + MIZAN SIM (WITH RAW DEBUG)
# ============================================================

def cosine_sim(e1, e2):
    cos = torch.nn.functional.cosine_similarity(e1, e2).item()
    print("\n[COSINE DEBUG] ->", cos)
    return cos


def mizan_sim(e1, e2):
    """
    Article #10 definition:
    M = 1 - ||e1 - e2|| / (||e1|| + ||e2||)
    """
    num = torch.norm(e1 - e2, p=2)
    den = torch.norm(e1, p=2) + torch.norm(e2, p=2) + 1e-6
    miz = (1 - (num / den)).item()

    print("\n[MIZAN DEBUG]")
    print("  ||e1||:", torch.norm(e1).item())
    print("  ||e2||:", torch.norm(e2).item())
    print("  ||e1-e2||:", num.item())
    print("  Mizan:", miz)

    return miz


# ============================================================
#                 SENTENCE-LEVEL EVALUATION
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
        print("\n---------------------------------")
        print("📝 S1:", s1)
        print("📝 S2:", s2)

        t1 = tokenizer(s1, return_tensors="pt").to(device)
        t2 = tokenizer(s2, return_tensors="pt").to(device)

        with torch.no_grad():
            e1 = model(**t1)
            e2 = model(**t2)

        cos = cosine_sim(e1, e2)
        miz = mizan_sim(e1, e2)

        print("\nRESULTS:")
        print("  Cosine similarity :", f"{cos:.4f}")
        print("  Mizan similarity  :", f"{miz:.4f}")
        print("  Δ (Mizan - Cosine):", f"{miz - cos:+.4f}")


# ============================================================
#               SAVE + LOAD MIZAN-MAPPED MODEL
# ============================================================

def save_mizan(backbone, proj_dim, alpha, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n🔧 Building MizanMappedEncoder (NO TRAINING)...")
    model = MizanMappedEncoder(backbone, proj_dim, alpha).to(device)
    tok = AutoTokenizer.from_pretrained(backbone)

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n💾 Saving mapped encoder → {out_dir}")
    torch.save(model.state_dict(), f"{out_dir}/mizan_encoder.pt")
    tok.save_pretrained(out_dir)

    cfg = {
        "backbone": backbone,
        "proj_dim": proj_dim,
        "alpha": alpha
    }
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=2)

    print("✔ Model + tokenizer saved.")
    return model, tok


def load_mizan(ckpt):
    cfg = json.load(open(f"{ckpt}/config.json"))
    print("\nLoaded config:", cfg)

    model = MizanMappedEncoder(
        cfg["backbone"], cfg["proj_dim"], cfg["alpha"]
    )
    state = torch.load(f"{ckpt}/mizan_encoder.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)

    tok = AutoTokenizer.from_pretrained(ckpt)
    print("✔ Loaded model + tokenizer\n")
    return model, tok


# ============================================================
#                         MAIN
# ============================================================

if __name__ == "__main__":
    OUT = "checkpoints/mizan_mapped_v1"

    print("\n==============================")
    print("🚀 MIZAN MAPPED ENCODER PIPELINE")
    print("==============================")

    # Step 1: Build & Save
    model, tokenizer = save_mizan(
        backbone="sentence-transformers/all-MiniLM-L6-v2",
        proj_dim=384,
        alpha=0.2,
        out_dir=OUT,
    )

    # Step 2: Reload for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_mizan(OUT)
    model.to(device)
    model.eval()

    # Step 3: Compare sentences
    sentence_compare(model, tokenizer, device)

    print("\n🎯 COMPLETE — MizanMappedEncoder works with full logs.")
