import json
import torch
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=256, subset=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # ---- SAFE LOAD JSONL ----
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip empty or whitespace lines
                if not line:
                    continue

                # Skip corrupted JSON lines
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.data.append(obj)

        # ---- SUBSET MODE ----
        if subset is not None:
            print(f"[INFO] CPU-fast mode: Using subset {subset}/{len(self.data)}")
            self.data = self.data[:subset]

        print(f"[INFO] Loaded {len(self.data)} clean samples.")

    # --------------------------------------------
    def encode(self, text):
        out = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

    # --------------------------------------------
    def __getitem__(self, idx):
        item = self.data[idx]
        ids1, mask1 = self.encode(item["text1"])
        ids2, mask2 = self.encode(item["text2"])

        return {
            "ids1": ids1,
            "mask1": mask1,
            "ids2": ids2,
            "mask2": mask2,
            "label": torch.tensor(item["label"])
        }

    def __len__(self):
        return len(self.data)
