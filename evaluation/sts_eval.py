import json
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder
from evaluation.utils import embed_texts, cosine_sim
from mizan_vector.metrics import mizan_similarity
import torch

def evaluate_sts(model_path, data_path="data/sts.jsonl", backbone="distilbert-base-uncased"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MizanTextEncoder.from_pretrained(model_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(backbone)

    sents1, sents2, labels = [], [], []

    for line in open(data_path, encoding="utf-8"):
        obj = json.loads(line)
        sents1.append(obj["text1"])
        sents2.append(obj["text2"])
        labels.append(float(obj["label"]))

    emb1 = embed_texts(model, tokenizer, sents1, device)
    emb2 = embed_texts(model, tokenizer, sents2, device)

    # Compute similarities
    cos_scores = [cosine_sim(a, b) for a, b in zip(emb1, emb2)]
    mizan_scores = [mizan_similarity(a, b) for a, b in zip(emb1, emb2)]

    cos_corr = spearmanr(cos_scores, labels).correlation
    miz_corr = spearmanr(mizan_scores, labels).correlation

    print("\n=== STS-B Evaluation ===")
    print(f"Cosine Spearman: {cos_corr:.4f}")
    print(f"Mizan Spearman:  {miz_corr:.4f}")

    return cos_corr, miz_corr


if __name__ == "__main__":
    evaluate_sts("saved/mizan_encoder_v1")
