import json
import numpy as np
from scipy.stats import spearmanr
from inference.batch_embedder import MizanBatchEmbedder
from mizan_vector.metrics import mizan_similarity

def test_stsb_spearman():
    path = "tests/data/sts_sample.jsonl"
    embedder = MizanBatchEmbedder()

    sims = []
    gold = []

    for line in open(path):
        obj = json.loads(line)
        e1 = embedder.encode_one(obj["s1"])
        e2 = embedder.encode_one(obj["s2"])
        sims.append(mizan_similarity(e1, e2))
        gold.append(obj["score"])

    rho, _ = spearmanr(sims, gold)
    assert rho > 0.4, f"STS-B Spearman too low: {rho}"
