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
def load_sts_tsv(path: str, sample_size=None):
    """
    Loads STS-B (train/dev/test) from TSV.
    Output: [(s1, s2, label)]
    label: 1 if score >= 3.0 else 0
    """
    pairs = []

    with open(path, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header

        for row in reader:
            if len(row) < 7:
                continue

            score = float(row[4])
            s1 = row[5]
            s2 = row[6]

            label = 1 if score >= 3 else 0
            pairs.append((s1, s2, label))

    if sample_size:
        random.shuffle(pairs)
        pairs = pairs[:sample_size]

    return pairs


# =====================================================================
# SNLI Loader (JSONL → pairs)
# =====================================================================
def load_snli_jsonl(path: str, sample_size=None):
    """
    Loads SNLI .jsonl files.
    Converts:
        entailment → 1
        contradiction → 0
    Skips "neutral".
    """
    pairs = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)

            label_txt = obj.get("gold_label", "")
            if label_txt == "entailment":
                label = 1
            elif label_txt == "contradiction":
                label = 0
            else:
                continue  # skip neutral / invalid

            s1 = obj["sentence1"]
            s2 = obj["sentence2"]

            pairs.append((s1, s2, label))

    if sample_size:
        random.shuffle(pairs)
        pairs = pairs[:sample_size]

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
