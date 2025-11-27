from inference.batch_embedder import MizanBatchEmbedder

def test_full_inference_pipeline():
    embedder = MizanBatchEmbedder()

    a = embedder.encode_one("The sky is blue")
    b = embedder.encode_one("Blue sky during the day")

    assert a.shape == b.shape, "Embeddings must have same dimension"

    # MizanSimilarity should rank similar sentences high
    from mizan_vector.metrics import mizan_similarity
    score = mizan_similarity(a, b)

    assert score > 0.4, "Meaningfully similar text must yield reasonable similarity"
