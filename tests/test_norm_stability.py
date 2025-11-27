import numpy as np
from inference.batch_embedder import MizanBatchEmbedder

def test_embedding_norm_stability():
    embedder = MizanBatchEmbedder()
    texts = [f"random text {i}" for i in range(50)]

    emb = embedder.encode(texts)
    norms = np.linalg.norm(emb, axis=1)

    assert norms.mean() > 0.1, "Norm collapse detected"
    assert norms.std() < 0.8, "Norm instability detected"
