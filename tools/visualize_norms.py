import argparse
import numpy as np
import matplotlib.pyplot as plt
from inference.batch_embedder import MizanBatchEmbedder

def visualize_norms(text_file, output="norms.png"):
    texts = [l.strip() for l in open(text_file) if l.strip()]
    embedder = MizanBatchEmbedder()

    print("→ Embedding texts...")
    emb = embedder.encode(texts)
    norms = np.linalg.norm(emb, axis=1)

    plt.figure(figsize=(8,6))
    plt.hist(norms, bins=40, alpha=0.75)
    plt.title("Embedding Norm Distribution (Mizan Encoder)")
    plt.xlabel("Norm")
    plt.ylabel("Count")
    plt.savefig(output)
    print(f"✔ Saved plot: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--output", default="norms.png")
    args = parser.parse_args()

    visualize_norms(args.text_file, args.output)
