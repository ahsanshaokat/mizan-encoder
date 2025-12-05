"""
Dataset loaders for Mizan Encoder training.
Supports:
✔ STS-B (semantic textual similarity)
✔ SNLI (entailment vs contradiction)
"""

import csv
import json
import random
import torch
from torch.utils.data import Dataset


# =====================================================================
# STS-B Loader (TSV → pairs)
# =====================================================================
def load_sts_tsv(path, sample_size=None):
    import csv

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter="\t")
        next(tsv)  # skip header

        for row in tsv:
            if len(row) < 7:
                continue

            s1 = row[5]
            s2 = row[6]
            score = float(row[4]) / 5.0  # normalize 0-1

            pairs.append((s1, s2, score))

    if sample_size:
        pairs = pairs[:sample_size]

    print(f"Loaded STS pairs: {len(pairs)}")
    return pairs



# =====================================================================
# SNLI Loader (JSONL → pairs)
# =====================================================================
def load_snli_jsonl(path, sample_size=None):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if obj["gold_label"] not in ["entailment", "contradiction", "neutral"]:
                continue

            s1 = obj["sentence1"]
            s2 = obj["sentence2"]

            if obj["gold_label"] == "entailment":
                label = 1.0
            elif obj["gold_label"] == "contradiction":
                label = 0.0
            else:
                label = 0.5

            pairs.append((s1, s2, label))

    if sample_size:
        pairs = pairs[:sample_size]

    print(f"Loaded SNLI pairs: {len(pairs)}")
    return pairs


# =====================================================================
# PyTorch Dataset Wrapper
# =====================================================================
class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1, s2, lbl = self.pairs[idx]
        return s1, s2, torch.tensor(lbl, dtype=torch.float32)
