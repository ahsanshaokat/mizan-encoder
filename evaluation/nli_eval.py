import json
import numpy as np
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder
from evaluation.utils import embed_texts
from mizan_vector.metrics import mizan_similarity
import torch

def evaluate_nli(model_path, data_path="data/snli.jsonl", backbone="distilbert-base-uncased", threshold=0.55):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MizanTextEncoder.from_pretrained(model_path).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    s1, s2, labels = [], [], []

    for line in open(data_path, encoding="utf-8"):
        obj = json.loads(line)
        s1.append(obj["text1"])
        s2.append(obj["text2"])
        labels.append(int(obj["label"]))

    emb1 = embed_texts(model, tokenizer, s1, device)
    emb2 = embed_texts(model, tokenizer, s2, device)

    preds = []

    for a, b in zip(emb1, emb2):
        sim = mizan_similarity(a, b)
        preds.append(1 if sim >= threshold else 0)

    acc = (np.array(preds) == np.array(labels)).mean()

    print("\n=== NLI Evaluation ===")
    print(f"Accuracy (Mizan): {acc:.4f}")
    return acc


if __name__ == "__main__":
    evaluate_nli("saved/mizan_encoder_v1")
