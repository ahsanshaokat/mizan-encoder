import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from inference.batch_embedder import MizanBatchEmbedder

def visualize_space(text_file, output="space.png"):
    texts = [l.strip() for l in open(text_file) if l.strip()]
    embedder = MizanBatchEmbedder()

    print("→ Embedding texts...")
    emb = embedder.encode(texts)

    print("→ Running PCA...")
    reduced = PCA(n_components=2).fit_transform(emb)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], alpha=0.6, s=12)
    plt.title("2D Embedding Space (Mizan Encoder)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(output)

    print(f"✔ Saved: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--output", default="space.png")
    args = parser.parse_args()

    visualize_space(args.text_file, args.output)
