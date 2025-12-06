import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# ============================================================
#                    DATA LOADING
# ============================================================

def load_sts(path, n=None):
    """Load STS-B train split."""
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
            label = score / 5.0          # 0–1
            pairs.append((s1, s2, label))
    if n:
        pairs = random.sample(pairs, min(len(pairs), n))
    return pairs


def load_snli(path, n=None):
    """Load SNLI JSONL."""
    import json
    m = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            y = m.get(obj["gold_label"], None)
            if y is None:
                continue
            pairs.append((obj["sentence1"], obj["sentence2"], y))
    if n:
        pairs = random.sample(pairs, min(len(pairs), n))
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
#                 MIZAN ENCODER (Article #10)
# ============================================================

class BalancedMeanPooling(nn.Module):
    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class MizanEncoder(nn.Module):
    """Transformer → Balanced Mean Pooling → Linear → LayerNorm → Scale-Stabilizer"""
    def __init__(self, backbone, proj_dim=384, alpha=0.2):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(backbone)
        hid = self.transformer.config.hidden_size

        self.pool = BalancedMeanPooling()
        self.proj = nn.Linear(hid, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

        self.alpha = alpha

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def scale_stabilize(self, x):
        n = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6
        return x / (n ** self.alpha)

    def forward(self, ids, mask):
        out = self.transformer(input_ids=ids, attention_mask=mask)
        pooled = self.pool(out.last_hidden_state, mask)
        h = self.norm(self.proj(pooled))
        return self.scale_stabilize(h)


# ============================================================
#              TRUE MIZAN CONTRASTIVE LOSS (Article #10)
# ============================================================

class MizanContrastiveLoss(nn.Module):
    """
    The official MizanContrastiveLoss from Article #10.
    No collapse, positive + negative contrastive forces.
    """
    def __init__(self, margin=0.5, p=2, eps=1e-6):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps

    def mizan_sim(self, x, y):
        num = torch.norm(x - y, p=self.p, dim=-1)
        den = torch.norm(x, p=self.p, dim=-1) + torch.norm(y, p=self.p, dim=-1) + self.eps
        return 1 - (num / den)

    def forward(self, e1, e2, label):
        sim = self.mizan_sim(e1, e2)

        pos = 1 - sim                        # positive want sim → 1
        neg = torch.relu(self.margin - sim)  # negatives must be below margin

        return torch.where(label == 1, pos, neg).mean()


# ============================================================
#                    TRAINING LOOP
# ============================================================

def train(config):

    print("===== CONFIG =====")
    print(json.dumps(config, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
    model = MizanEncoder(
        backbone=config["backbone"],
        proj_dim=config["proj_dim"],
        alpha=config["alpha"],
    ).to(device)

    # load & mix datasets
    sts  = load_sts(config["sts_path"], config["sts_samples"])
    snli = load_snli(config["nli_path"], config["nli_samples"])
    dataset = PairDataset(sts + snli)

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    total_steps = len(loader) * config["epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = MizanContrastiveLoss(margin=0.5)

    print("\n===== TRAINING START =====")

    model.train()
    for ep in range(config["epochs"]):
        for step, (s1, s2, labels) in enumerate(loader):

            batch = tokenizer(
                s1 + s2,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            bs = len(s1)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            e1 = model(ids[:bs],  mask[:bs])
            e2 = model(ids[bs:], mask[bs:])

            labels = labels.to(device)
            loss = loss_fn(e1, e2, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Ep {ep+1} Step {step} | Loss {loss.item():.4f}")

    # 保存
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), f"{config['output_dir']}/mizan_encoder.pt")
    tokenizer.save_pretrained(config["output_dir"])
    with open(f"{config['output_dir']}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✔ Training complete!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    config = {
        "backbone": "sentence-transformers/all-MiniLM-L6-v2",
        "proj_dim": 384,
        "alpha": 0.2,

        "batch_size": 16,
        "epochs": 1,
        "lr": 1e-5,

        "sts_path": "scripts/data/sts_raw/STS-B/train.tsv",
        "nli_path": "scripts/data/snli_1.0/snli_1.0_train.jsonl",
        "sts_samples": 2000,
        "nli_samples": 8000,

        "output_dir": "checkpoints/mizan_v10"
    }

    train(config)
