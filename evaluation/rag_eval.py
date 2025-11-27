import numpy as np
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder
from mizan_vector.metrics import mizan_similarity
from evaluation.utils import embed_texts, cosine_sim
import json
import torch

def evaluate_rag(model_path, corpus_path="data/rag_corpus.jsonl",
                 queries_path="data/rag_queries.jsonl",
                 backbone="distilbert-base-uncased"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model…")
    model = MizanTextEncoder.from_pretrained(model_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # Load corpus
    corpus = []
    ids = []
    for line in open(corpus_path, encoding="utf-8"):
        obj = json.loads(line)
        ids.append(obj["id"])
        corpus.append(obj["text"])

    # Embed corpus once
    print("Embedding corpus…")
    corpus_emb = embed_texts(model, tokenizer, corpus, device)

    # Evaluate queries
    print("Evaluating queries…")

    results = []

    for line in open(queries_path, encoding="utf-8"):
        q = json.loads(line)

        q_text = q["query"]
        true_id = q["gold_id"]

        q_emb = embed_texts(model, tokenizer, [q_text], device)[0]

        # Compute similarities
        cos_sims = [cosine_sim(q_emb, d) for d in corpus_emb]
        miz_sims = [mizan_similarity(q_emb, d) for d in corpus_emb]

        # Top result
        top_cos = ids[int(np.argmax(cos_sims))]
        top_miz = ids[int(np.argmax(miz_sims))]

        results.append({
            "query": q_text,
            "true": true_id,
            "cosine_top": top_cos,
            "mizan_top": top_miz
        })

    # Accuracy
    cos_acc = np.mean([r["cosine_top"] == r["true"] for r in results])
    miz_acc = np.mean([r["mizan_top"] == r["true"] for r in results])

    print("\n=== RAG Retrieval Evaluation ===")
    print(f"Cosine Accuracy: {cos_acc:.4f}")
    print(f"Mizan Accuracy:  {miz_acc:.4f}")

    return results, cos_acc, miz_acc


if __name__ == "__main__":
    evaluate_rag("saved/mizan_encoder_v1")
