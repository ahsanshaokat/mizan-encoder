import torch
import numpy as np
from scipy.stats import spearmanr
from mizan_vector.metrics import mizan_similarity


def embed_texts(model, tokenizer, texts, device="cpu", max_len=256):
    """
    Tokenize and embed a list of texts using the MizanTextEncoder.
    """
    embeddings = []

    for t in texts:
        out = tokenizer(
            t,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            emb = model(
                out["input_ids"].to(device),
                out["attention_mask"].to(device)
            )
        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def cosine_sim(a, b):
    """
    Compute cosine similarity between two numpy vectors.
    """
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))
