import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# ============================================================
#                      DATA LOADING
# ============================================================

def load_sts(path, sample_size=None):
    """Load STS-B data (tab-separated):
       sentence1 | sentence2 | score
       Skips header rows safely.
    """
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            # Skip malformed or header lines
            if len(parts) < 7:
                continue

            # Try parsing the score column
            try:
                score = float(parts[4])
            except ValueError:
                # This happens for header "old_index" or similar → SKIP
                continue

            s1, s2 = parts[5], parts[6]
            score = score / 5.0  # normalize 0–1

            pairs.append((s1, s2, score))

    if sample_size:
        pairs = random.sample(pairs, min(sample_size, len(pairs)))

    return pairs



def load_snli(path, sample_size=None):
    """Load SNLI JSONL: convert labels to similarity scores."""
    import json

    label_to_score = {
        "entailment": 1.0,
        "neutral": 0.5,
        "contradiction": 0.0
    }

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["gold_label"] not in label_to_score:
                continue
            s1 = obj["sentence1"]
            s2 = obj["sentence2"]
            score = label_to_score[obj["gold_label"]]
            pairs.append((s1, s2, score))

    if sample_size:
        pairs = random.sample(pairs, min(sample_size, len(pairs)))

    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)


def collate_batch(batch):
    t1 = [b[0] for b in batch]
    t2 = [b[1] for b in batch]
    y = torch.tensor([b[2] for b in batch], dtype=torch.float)
    return t1, t2, y


# ============================================================
#                MIZAN ENCODER IMPLEMENTATION
# ============================================================

class BalancedMeanPooling(nn.Module):
    """Mean pooling that avoids NaNs & empty masks."""

    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        weighted = hidden * mask
        summed = weighted.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class MizanEncoder(nn.Module):
    """Minimal Mizan encoder:
    HF backbone → pooling → projection → Mizan scaling
    """

    def __init__(self, backbone_name="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=384, alpha=0.15):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hid = self.backbone.config.hidden_size

        self.pool = BalancedMeanPooling()
        self.proj = nn.Linear(hid, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

        self.alpha = alpha

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def scale_stabilize(self, x):
        """Mizan power-normalization."""
        norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        denom = norm.pow(self.alpha)
        return x / denom

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled = self.pool(out.last_hidden_state, attention_mask)

        h = self.proj(pooled)
        h = self.norm(h)
        h = self.scale_stabilize(h)

        return h


# ============================================================
#                 MIZAN SIMILARITY LOSS (STABLE)
# ============================================================

class MizanLoss(nn.Module):
    """Stable similarity loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, labels):
        cos = torch.nn.functional.cosine_similarity(e1, e2)

        pos = labels * (1 - cos)                # positives push toward 1
        neg = (1 - labels) * torch.relu(cos - self.margin)

        return (pos + neg).mean()


# ============================================================
#                     TRAINING LOOP
# ============================================================

def train_mizan(config):

    print("\n=========== CONFIG ===========")
    print(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
    model = MizanEncoder(
        backbone_name=config["backbone"],
        proj_dim=config["proj_dim"],
        alpha=config["alpha"]
    ).to(device)

    print("\nLoading datasets...")

    sts_pairs = load_sts(config["sts_path"], sample_size=config["sts_samples"])
    nli_pairs = load_snli(config["nli_path"], sample_size=config["nli_samples"])

    all_pairs = sts_pairs + nli_pairs
    dataset = PairDataset(all_pairs)

    loader = DataLoader(
        dataset, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_batch
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    total_steps = len(loader) * config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = MizanLoss(margin=0.3)

    # -------- TRAINING --------

    print("\n=========== TRAINING START ===========")

    model.train()

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        for step, (t1, t2, y) in enumerate(loader):

            batch = tokenizer(
                t1 + t2,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            # split encoded batch
            bs = len(t1)
            input_ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            emb1 = model(input_ids[:bs], att[:bs])
            emb2 = model(input_ids[bs:], att[bs:])

            y = y.to(device)

            loss = loss_fn(emb1, emb2, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Step {step} | Loss {loss.item():.4f}")

    # -------- SAVE --------

    os.makedirs(config["output_dir"], exist_ok=True)

    print("\nSaving model...")
    torch.save(model.state_dict(), os.path.join(config["output_dir"], "mizan_encoder.pt"))
    tokenizer.save_pretrained(config["output_dir"])

    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("✔ Training complete!")


# ============================================================
#                         MAIN ENTRY
# ============================================================

if __name__ == "__main__":

    config = {
        "backbone": "sentence-transformers/all-MiniLM-L6-v2",

        "proj_dim": 384,
        "alpha": 0.15,

        "batch_size": 16,
        "epochs": 1,

        "lr": 1e-5,

        "sts_path": "scripts/data/sts_raw/STS-B/train.tsv",
        "nli_path": "scripts/data/snli_1.0/snli_1.0_train.jsonl",

        "sts_samples": 2000,
        "nli_samples": 8000,

        "output_dir": "checkpoints/mizan_singlefile"
    }

    train_mizan(config)
