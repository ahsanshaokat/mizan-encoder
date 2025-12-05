import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. LOAD MODEL
# =====================================================================
tok = AutoTokenizer.from_pretrained("checkpoints/mizan_encoder_small")
model = AutoModel.from_pretrained("checkpoints/mizan_encoder_small")


# =====================================================================
# 2. EMBEDDING FUNCTION
# =====================================================================
def encode(text):
    x = tok(text, return_tensors="pt", truncation=True, padding=True)
    out = model(**x)
    return out.last_hidden_state.mean(dim=1)


# =====================================================================
# 3. SIMILARITY FUNCTIONS
# =====================================================================
def cosine(a, b):
    return F.cosine_similarity(a, b).item()

def euclidean(a, b):
    return torch.norm(a - b, dim=-1).item()

def mizan(a, b, eps=1e-8):
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    cos = F.cosine_similarity(a, b)
    ratio = torch.minimum(na, nb) / (torch.maximum(na, nb) + eps)
    return (cos * ratio).item()


# =====================================================================
# 4. FULL MIZAN METRICS
# =====================================================================
def mizan_full(a, b, eps=1e-8):
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    cos = F.cosine_similarity(a, b)
    ratio = torch.minimum(na, nb) / (torch.maximum(na, nb) + eps)
    pos = torch.clamp(cos, min=0.0) * ratio
    neg = torch.clamp(cos, max=0.0) * ratio
    miz = cos * ratio
    return {
        "norm_a": na.item(),
        "norm_b": nb.item(),
        "cosine": cos.item(),
        "ratio": ratio.item(),
        "pos": pos.item(),
        "neg": neg.item(),
        "mizan": miz.item(),
        "euclidean": torch.norm(a - b, dim=-1).item()
    }


# =====================================================================
# 5. BIG TEST DATASET (RAG + Semantic + Noise + Scale)
# =====================================================================
TEST_PAIRS = [
    # POSITIVE SEMANTIC
    ("What is AI?", "Artificial intelligence is the study of intelligent agents.", "positive"),
    ("Who wrote Dracula?", "Bram Stoker wrote the novel Dracula.", "positive"),
    ("Machine learning explanation", "Training algorithms on data to learn patterns.", "positive"),

    # NEGATIVE SEMANTIC
    ("Cats are mammals.", "Quantum field theory describes particles.", "negative"),
    ("Paris is the capital of France.", "Bananas are yellow fruits.", "negative"),

    # CONTRADICTIONS
    ("The sky is blue.", "The sky is green.", "contradiction"),
    ("Water boils at 100C.", "Water does not boil at 100C.", "contradiction"),

    # NOISE & MISSPELLINGS
    ("AI future???!!!", "The field of artificial intelligence is evolving.", "noise"),
    ("Wh0 wr0te dr@cula???", "Bram Stoker wrote Dracula.", "noise"),

    # SCALE IMBALANCE
    ("AI transforms industries.",
     "Artificial intelligence is a rapidly evolving field combining neural networks, algorithms, optimization, symbolic reasoning, decision systems, and large-scale data models.",
     "scale"),

    ("Dracula is a book.", "Dracula", "scale"),

    # RAG TARGETS
    ("Explain the plot of Dracula.",
     "Dracula tells the story of a vampire count traveling to England to spread the undead curse.",
     "rag_relevant"),

    ("Explain the plot of Dracula.",
     "The sun contains plasma undergoing nuclear fusion.",
     "rag_distractor")
]


# =====================================================================
# 6. RUN BENCHMARK
# =====================================================================
def benchmark():
    print("\n==================== BIG MIZAN / COSINE / EUCLIDEAN BENCHMARK ====================\n")

    results = []

    for t1, t2, label in TEST_PAIRS:

        a = encode(t1)
        b = encode(t2)

        cos = cosine(a, b)
        euc = euclidean(a, b)
        miz = mizan(a, b)
        full = mizan_full(a, b)

        results.append((label, cos, euc, miz))

        print(f"CASE: {label}")
        print(f"TEXT 1: {t1}")
        print(f"TEXT 2: {t2}")
        print("---------------------------")
        print(f"Cosine:        {cos:.4f}")
        print(f"Euclidean:     {euc:.4f}")
        print(f"Mizan:         {miz:.4f}")
        print(f"Ratio:         {full['ratio']:.4f}")
        print(f"POS:           {full['pos']:.4f}")
        print(f"NEG:           {full['neg']:.4f}")
        print(f"Norm A:        {full['norm_a']:.4f}")
        print(f"Norm B:        {full['norm_b']:.4f}")
        print(f"Δ (Mizan-Cos): {miz - cos:+.4f}")
        print("----------------------------------------------------------------\n")

    return results


# =====================================================================
# 7. SUMMARY STATISTICS
# =====================================================================
def summary(results):
    print("\n==================== SUMMARY ANALYSIS ====================\n")

    pos_cos, pos_miz = [], []
    neg_cos, neg_miz = [], []

    for label, cos, euc, miz in results:

        if label in ["positive", "rag_relevant"]:
            pos_cos.append(cos)
            pos_miz.append(miz)

        if label in ["negative", "contradiction", "rag_distractor"]:
            neg_cos.append(cos)
            neg_miz.append(miz)

    print("POSITIVE PAIRS (should be HIGH)")
    print(f"Avg Cosine: {np.mean(pos_cos):.4f}")
    print(f"Avg Mizan : {np.mean(pos_miz):.4f}")

    print("\nNEGATIVE/CONFLICT PAIRS (should be LOW/NEGATIVE)")
    print(f"Avg Cosine: {np.mean(neg_cos):.4f}")
    print(f"Avg Mizan : {np.mean(neg_miz):.4f}")

    improvement = (np.mean(pos_miz) - np.mean(pos_cos)) + (np.mean(neg_cos) - np.mean(neg_miz))
    print("\nTOTAL IMPROVEMENT SCORE (Mizan > Cosine):", round(improvement, 4))

    print("\n====================================================================\n")


# =====================================================================
# 8. VISUALIZATION
# =====================================================================
def visualize(results):
    labels = [r[0] for r in results]
    cos_vals = [r[1] for r in results]
    euc_vals = [r[2] for r in results]
    miz_vals = [r[3] for r in results]

    pos_indices = [i for i, l in enumerate(labels) if l in ["positive", "rag_relevant"]]
    neg_indices = [i for i, l in enumerate(labels) if l in ["negative", "contradiction", "rag_distractor"]]

    # 1. Mizan vs Cosine
    plt.figure(figsize=(8, 6))
    plt.scatter(cos_vals, miz_vals, c='blue')
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.title("Mizan Similarity vs Cosine Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Mizan Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("viz_mizan_vs_cosine.png")
    print("Saved: viz_mizan_vs_cosine.png")

    # 2. Euclidean vs Mizan
    plt.figure(figsize=(8, 6))
    plt.scatter(euc_vals, miz_vals, c='green')
    plt.title("Euclidean Distance vs Mizan Similarity")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Mizan Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("viz_euclidean_vs_mizan.png")
    print("Saved: viz_euclidean_vs_mizan.png")

    # 3. Positive vs Negative Separation
    plt.figure(figsize=(10, 5))
    plt.scatter([cos_vals[i] for i in pos_indices],
                [miz_vals[i] for i in pos_indices], c='blue', label="Positive Pairs", s=80)
    plt.scatter([cos_vals[i] for i in neg_indices],
                [miz_vals[i] for i in neg_indices], c='red', label="Negative Pairs", s=80)

    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')

    plt.title("Separation of Positive vs Negative Pairs (Cosine → Mizan)")
    plt.xlabel("Cosine Score")
    plt.ylabel("Mizan Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("viz_positive_negative_separation.png")
    print("Saved: viz_positive_negative_separation.png")

    print("\nAll visualizations saved in working directory.\n")



# =====================================================================
# EXECUTE EVERYTHING
# =====================================================================
if __name__ == "__main__":
    results = benchmark()
    summary(results)
    visualize(results)
