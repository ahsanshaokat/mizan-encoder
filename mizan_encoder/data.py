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
    import json
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:      # ← FIX: skip empty lines
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            label = obj.get("gold_label", None)
            if label not in ["entailment", "contradiction", "neutral"]:
                continue

            s1 = obj.get("sentence1", "").strip()
            s2 = obj.get("sentence2", "").strip()
            if not s1 or not s2:
                continue

            if label == "entailment":
                y = 1.0
            elif label == "contradiction":
                y = 0.0
            else:
                y = 0.5

            pairs.append((s1, s2, y))

    if sample_size:
        pairs = pairs[:sample_size]

    print(f"Loaded SNLI pairs: {len(pairs)} from {path}")
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
