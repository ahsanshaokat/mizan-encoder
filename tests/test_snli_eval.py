import json
from inference.batch_embedder import MizanBatchEmbedder
from mizan_vector.metrics import mizan_similarity

def test_snli_classification_accuracy():
    path = "tests/data/snli_sample.jsonl"
    embedder = MizanBatchEmbedder()

    correct = 0
    total = 0

    for l in open(path):
        obj = json.loads(l)
        e1 = embedder.encode_one(obj["premise"])
        e2 = embedder.encode_one(obj["hypothesis"])
        sim = mizan_similarity(e1, e2)

        pred = 1 if sim > 0.4 else 0

        if pred == obj["label"]:
            correct += 1
        total += 1

    acc = correct / total
    assert acc > 0.6, f"SNLI Accuracy too low: {acc}"
