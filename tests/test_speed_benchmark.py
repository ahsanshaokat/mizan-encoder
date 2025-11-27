import time
from inference.batch_embedder import MizanBatchEmbedder

def test_inference_speed():
    embedder = MizanBatchEmbedder()

    texts = ["This is some text to benchmark inference speed."] * 32
    start = time.time()
    _ = embedder.encode(texts)
    dur = time.time() - start

    assert dur < 3.0, f"Inference too slow: {dur:.2f}s"
