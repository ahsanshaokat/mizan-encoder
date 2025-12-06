import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# ============================================================
#                    DATA LOADING
# ============================================================

def load_sts(path, sample_size=None):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7: 
                continue
            try:
                score = float(parts[4])
            except:
                continue
            s1, s2 = parts[5], parts[6]
            score = score / 5.0          # normalize 0–1
            pairs.append((s1, s2, score))

    if sample_size:
        pairs = random.sample(pairs, min(sample_size, len(pairs)))
    return pairs


def load_snli(path, sample_size=None):
    import json
    mapping = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            y = mapping.get(obj["gold_label"], None)
            if y is None:
                continue
            pairs.append((obj["sentence1"], obj["sentence2"], y))
    if sample_size:
        pairs = random.sample(pairs, min(sample_size, len(pairs)))
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch):
    t1 = [b[0] for b in batch]
    t2 = [b[1] for b in batch]
    y  = torch.tensor([b[2] for b in batch], dtype=torch.float)
    return t1, t2, y


# ============================================================
#                    MIZAN ENCODER
# ============================================================

class BalancedMeanPool(nn.Module):
    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class MizanEncoder(nn.Module):
    def __init__(self, backbone, proj_dim=384, alpha=0.15):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        hid = self.backbone.config.hidden_size

        self.pool = BalancedMeanPool()
        self.proj = nn.Linear(hid, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.alpha = alpha

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    # Mizan Scale Stabilization
    def mizan_scale(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        return x / (n ** self.alpha)

    def forward(self, ids, mask, type_ids=None):
        out = self.backbone(input_ids=ids, attention_mask=mask, token_type_ids=type_ids)
        pooled = self.pool(out.last_hidden_state, mask)
        h = self.norm(self.proj(pooled))
        return self.mizan_scale(h)


# ============================================================
#               NORMALIZED MIZAN SIMILARITY
# ============================================================

def mizan_sim_norm(e1, e2, alpha=0.15):
    dot = (e1 * e2).sum(dim=-1)
    n1 = torch.norm(e1, dim=-1).clamp(min=1e-6)
    n2 = torch.norm(e2, dim=-1).clamp(min=1e-6)
    miz_raw = dot / ((n1 ** alpha) * (n2 ** alpha))

    miz_norm = miz_raw / (1 + miz_raw.abs())
    return miz_norm  # bounded (-1, +1)


# ============================================================
#                  NORMALIZED MIZAN LOSS
# ============================================================

class MizanLoss(nn.Module):
    """
    True supervised loss for Mizan similarity.
    Uses normalized similarity in (-1, +1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, e1, e2, labels):
        pred = mizan_sim_norm(e1, e2)

        # BCE-like behavior on similarity
        loss = (pred - (labels * 2 - 1)) ** 2  
        return loss.mean()


# ============================================================
#                    TRAINING LOOP
# ============================================================

def train(config):

    print("===== CONFIG =====")
    print(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
    model = MizanEncoder(
        backbone=config["backbone"],
        proj_dim=config["proj_dim"],
        alpha=config["alpha"]
    ).to(device)

    # Load datasets
    sts  = load_sts(config["sts_path"], config["sts_samples"])
    nli  = load_snli(config["nli_path"], config["nli_samples"])
    data = PairDataset(sts + nli)

    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    total_steps = len(loader) * config["epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = MizanLoss()
    print("===== TRAINING START =====")

    model.train()
    for ep in range(config["epochs"]):
        for step, (t1, t2, y) in enumerate(loader):

            batch = tokenizer(t1 + t2, return_tensors="pt", padding=True, truncation=True, max_length=128)
            bs = len(t1)

            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            e1 = model(ids[:bs],  mask[:bs])
            e2 = model(ids[bs:], mask[bs:])

            labels = y.to(device)
            loss = loss_fn(e1, e2, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Ep {ep+1} Step {step} | Loss {loss.item():.4f}")

    # Save model
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), f"{config['output_dir']}/mizan_encoder.pt")
    tokenizer.save_pretrained(config["output_dir"])
    with open(f"{config['output_dir']}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✔ Training complete!")


# ============================================================
# MAIN
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

        "output_dir": "checkpoints/mizan_proper"
    }

    train(config)
