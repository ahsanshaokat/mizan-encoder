import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# ============================================================
#                 DATA LOADING (Binary Labels Only)
# ============================================================

def load_sts(path, n=None):
    """Load STS-B but keep ONLY strong positives and negatives."""
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
            label = score / 5.0

            # --------- Option 1 rules ---------
            if label >= 0.8:
                pairs.append((s1, s2, 1))
            elif label <= 0.3:
                pairs.append((s1, s2, 0))
            # else ignore mid-range
    if n:
        pairs = random.sample(pairs, min(len(pairs), n))
    return pairs


def load_snli(path, n=None):
    """Load SNLI but SKIP neutral cases."""
    import json
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lbl = obj["gold_label"]

            if lbl == "entailment":
                y = 1
            elif lbl == "contradiction":
                y = 0
            else:
                continue  # skip neutral

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
#                    MIZAN ENCODER
# ============================================================

class BalancedMeanPooling(nn.Module):
    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class MizanEncoder(nn.Module):
    """Transformer → Balanced Pool → Projection → LayerNorm → Scale Stabilizer"""
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
#                MIZAN CONTRASTIVE LOSS (Article #10)
# ============================================================

class MizanContrastiveLoss(nn.Module):
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

        pos = 1 - sim
        neg = torch.relu(self.margin - sim)

        return torch.where(label == 1, pos, neg), sim  # return sim for logging


# ============================================================
#                    TRAINING LOOP (With Logs)
# ============================================================

def train(config):

    print("===== CONFIG =====")
    print(json.dumps(config, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------
    # Load tokenizer + model
    # ----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
    model = MizanEncoder(
        backbone=config["backbone"],
        proj_dim=config["proj_dim"],
        alpha=config["alpha"],
    ).to(device)

    # ----------------------------------------
    # Dataset
    # ----------------------------------------
    sts  = load_sts(config["sts_path"], config["sts_samples"])
    snli = load_snli(config["nli_path"], config["nli_samples"])

    dataset = PairDataset(sts + snli)
    print(f"\nLoaded dataset: {len(dataset)} pairs")
    print(f"Positives: {sum([1 for x in dataset.pairs if x[2]==1])}")
    print(f"Negatives: {sum([1 for x in dataset.pairs if x[2]==0])}")

    loader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=True, collate_fn=collate)

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

            # --------------------------
            # Tokenize and forward pass
            # --------------------------
            batch = tokenizer(s1 + s2, return_tensors="pt",
                              padding=True, truncation=True, max_length=128)

            bs = len(s1)
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            e1 = model(ids[:bs], mask[:bs])
            e2 = model(ids[bs:], mask[bs:])

            labels = labels.to(device)

            # --------------------------
            # Compute loss + similarity
            # --------------------------
            loss_each, sim_vals = loss_fn(e1, e2, labels)
            loss = loss_each.mean()

            # --------------------------
            # Logging DEBUG INFO
            # --------------------------
            if step % 50 == 0:
                print(f"\n--- DEBUG Step {step} ---")
                print(f"Label batch example: {labels[:5].tolist()}")
                print(f"Mizan similarity example: {sim_vals[:5].tolist()}")
                print(f"Loss components example: {loss_each[:5].tolist()}")
                print(f"Embedding norm e1: {torch.norm(e1,dim=-1)[:3].tolist()}")
                print(f"Embedding norm e2: {torch.norm(e2,dim=-1)[:3].tolist()}")

            # --------------------------
            # Backprop
            # --------------------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Ep {ep+1} Step {step} | Loss {loss.item():.4f}")

    # ----------------------------------------
    # Save model
    # ----------------------------------------
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), f"{config['output_dir']}/mizan_encoder.pt")
    tokenizer.save_pretrained(config["output_dir"])

    with open(f"{config['output_dir']}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✔ Training complete!")
