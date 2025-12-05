from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


# -------------------------
# LOAD MODEL
# -------------------------
tok = AutoTokenizer.from_pretrained("checkpoints/mizan_encoder_small")
model = AutoModel.from_pretrained("checkpoints/mizan_encoder_small")


# -------------------------
# ENCODER
# -------------------------
def encode(text):
    x = tok(text, return_tensors="pt", truncation=True, padding=True)
    out = model(**x)
    emb = out.last_hidden_state.mean(dim=1)
    return emb


# -------------------------
# MIZAN METRICS
# -------------------------
def mizan_metrics(a, b, eps=1e-8):
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)

    cos = F.cosine_similarity(a, b)

    ratio = torch.minimum(na, nb) / (torch.maximum(na, nb) + eps)

    pos = torch.clamp(cos, min=0.0) * ratio
    neg = torch.clamp(cos, max=0.0) * ratio

    mizan = cos * ratio

    return {
        "norm_a": na.item(),
        "norm_b": nb.item(),
        "cosine": cos.item(),
        "ratio": ratio.item(),
        "pos": pos.item(),
        "neg": neg.item(),
        "mizan": mizan.item()
    }


# -------------------------
# TEST CASES
# -------------------------

pairs = [
    (
        "Dracula is a vampire novel written by Bram Stoker.",
        "The book Dracula tells the story of Count Dracula."
    ),
    (
        "The sun is a plasma sphere.",
        "Quantum mechanics studies probability waves."
    ),
    (
        "Dracula is a vampire novel.",
        "The sun is a plasma sphere."
    )
]

print("\n==================== MIZAN DIAGNOSTICS ====================\n")

for s1, s2 in pairs:
    a = encode(s1)
    b = encode(s2)
    result = mizan_metrics(a, b)

    print(f"TEXT 1: {s1}")
    print(f"TEXT 2: {s2}")
    print("------ MIZAN SCORES ------")
    print(f"Norm A:  {result['norm_a']:.4f}")
    print(f"Norm B:  {result['norm_b']:.4f}")
    print(f"Cosine:  {result['cosine']:.4f}")
    print(f"Ratio:   {result['ratio']:.4f}")
    print(f"POS:     {result['pos']:.4f}")
    print(f"NEG:     {result['neg']:.4f}")
    print(f"MIZAN:   {result['mizan']:.4f}")
    print("------------------------------------------------------------\n")
