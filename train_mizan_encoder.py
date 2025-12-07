"""
train_mizan_encoder.py

Mizan Encoder – Hybrid Training (Option C)
Backbone: BAAI/bge-base-en-v1.5
Projection Dim: 512
Training Stages:
  Stage 1 – Train projection + LN only
  Stage 2 – Unfreeze last 2 transformer layers
  Stage 3 – Full fine-tuning with low LR
"""

import os
import math
import json
import random
import logging
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MizanTrainer")


# ============================================================
#                 Balanced Mean Pooling
# ============================================================

class BalancedMeanPooling(nn.Module):
    """
    Custom mean-pooling ensuring stable length normalization.
    """

    def forward(self, hidden, mask):
        mask = mask.unsqueeze(-1).float()  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / count
        return pooled


# ============================================================
#                    Mizan Encoder Model
# ============================================================

class MizanEncoder(nn.Module):
    """
    Transformer Backbone → BalancedPool → Projection → LayerNorm → Scale Stabilizer
    """

    def __init__(self, backbone="BAAI/bge-base-en-v1.5", proj_dim=512, alpha=0.2):
        super().__init__()
        self.backbone_name = backbone
        self.alpha = alpha

        self.transformer = AutoModel.from_pretrained(backbone)
        hidden_size = self.transformer.config.hidden_size  # 768 for BGE-base

        self.pool = BalancedMeanPooling()
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def scale_stabilize(self, x):
        """
        Mizan proportional scaling:
            x' = x / ||x||^alpha
        """
        norms = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        return x / (norms ** self.alpha)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = self.pool(out.last_hidden_state, attention_mask)
        projected = self.proj(pooled)
        normalized = self.ln(projected)
        stabilized = self.scale_stabilize(normalized)
        return stabilized


# ============================================================
#                    Mizan Similarity
# ============================================================

def mizan_similarity(e1, e2):
    """
    M = 1 - ||e1 - e2|| / (||e1|| + ||e2||)
    """
    num = torch.norm(e1 - e2, p=2, dim=-1)
    den = torch.norm(e1, p=2, dim=-1) + torch.norm(e2, p=2, dim=-1) + 1e-6
    return 1 - (num / den)


# ============================================================
#                    Training Loss Functions
# ============================================================

class MizanLoss(nn.Module):
    """
    Combined training objective:
    - Mizan similarity regression loss
    - Cosine auxiliary loss
    - Contrastive margin for negative samples
    """

    def __init__(self, w_mizan=1.0, w_cos=0.3, w_margin=0.2, margin=0.4):
        super().__init__()
        self.w_mizan = w_mizan
        self.w_cos = w_cos
        self.w_margin = w_margin
        self.margin = margin

        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, e1, e2, labels):
        """
        labels: similarity score (0 to 1)
        """

        # ---- Mizan similarity regression ----
        miz = mizan_similarity(e1, e2)
        miz_loss = self.mse(miz, labels)

        # ---- Cosine auxiliary loss ----
        cos_sim = self.cos(e1, e2)
        cos_loss = self.mse(cos_sim, labels)

        # ---- Contrastive margin loss ----
        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5

        margin_loss = 0.0
        if neg_mask.any():
            neg_pairs = 1 - mizan_similarity(e1[neg_mask], e2[neg_mask])
            margin_loss = torch.clamp(self.margin - neg_pairs, min=0).mean()

        total = (
            self.w_mizan * miz_loss +
            self.w_cos * cos_loss +
            self.w_margin * margin_loss
        )

        return total, {
            "mizan_loss": miz_loss.item(),
            "cos_loss": cos_loss.item(),
            "margin_loss": margin_loss if isinstance(margin_loss, float) else margin_loss.item(),
        }


# ============================================================
#                     Dataset Wrapper
# ============================================================

@dataclass
class PairData:
    sentence1: str
    sentence2: str
    score: float


class PairDataset(Dataset):
    """
    Generic dataset handling for STS/SICK/PAWS/etc.
    """

    def __init__(self, pairs: List[PairData], tokenizer, max_length=128):
        self.data = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        t1 = self.tokenizer(
            item.sentence1,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        t2 = self.tokenizer(
            item.sentence2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids_1": t1["input_ids"].squeeze(0),
            "mask_1": t1["attention_mask"].squeeze(0),
            "input_ids_2": t2["input_ids"].squeeze(0),
            "mask_2": t2["attention_mask"].squeeze(0),
            "label": torch.tensor(item.score, dtype=torch.float)
        }

# ============================================================
#                  Load + Normalize Datasets
# ============================================================

def normalize_score(score):
    """
    Normalize various dataset scoring schemes into 0..1
    STS → 0..5 → /5
    SICK → 1..5 → (score-1)/4
    PAWS → binary (0 or 1)
    NLI → entailment=1, neutral=0.5, contradiction=0
    """
    if score is None:
        return 0.0
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        if score <= 1:
            return float(score)
        if score <= 5:
            return float(score) / 5.0
    return 0.0


# ------------------------------------------------------------
#                STS Benchmark (train + dev)
# ------------------------------------------------------------

def load_sts(tokenizer):
    ds = load_dataset("stsb_multi_mt", "en")

    pairs = []
    # stsb_multi_mt has: train, test (no validation)
    for split in ["train", "test"]:
        for row in ds[split]:
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            score = normalize_score(row["similarity_score"])
            pairs.append(PairData(s1, s2, score))

    logger.info(f"[STS] Loaded {len(pairs)} pairs")
    return pairs



# ------------------------------------------------------------
#                          PAWS-X
# ------------------------------------------------------------

def load_paws(tokenizer):
    ds = load_dataset("paws-x", "en")
    pairs = []

    for split in ["train", "validation"]:
        for row in ds[split]:
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            label = 1.0 if row["label"] == 1 else 0.0
            pairs.append(PairData(s1, s2, label))

    logger.info(f"[PAWS-X] Loaded {len(pairs)} pairs")
    return pairs



# ------------------------------------------------------------
#                          SNLI + MNLI
# ------------------------------------------------------------

def label_nli(row):
    if row["label"] == 0:   # entailment
        return 1.0
    if row["label"] == 1:   # neutral
        return 0.5
    if row["label"] == 2:   # contradiction
        return 0.0
    return 0.0

def load_nli(tokenizer, max_samples=60000):
    pairs = []

    # SNLI
    snli = load_dataset("snli")
    for split in ["train", "validation"]:
        for row in snli[split]:
            if row["label"] == -1:
                continue
            s1 = row["premise"]
            s2 = row["hypothesis"]
            score = label_nli(row)
            pairs.append(PairData(s1, s2, score))
            if len(pairs) >= max_samples:
                break

    # MNLI
    mnli = load_dataset("multi_nli")
    for split in ["train", "validation_matched"]:
        for row in mnli[split]:
            if row["label"] == -1:
                continue
            s1 = row["premise"]
            s2 = row["hypothesis"]
            score = label_nli(row)
            pairs.append(PairData(s1, s2, score))
            if len(pairs) >= max_samples * 2:
                break

    logger.info(f"[NLI] Loaded {len(pairs)} pairs")
    return pairs


# ------------------------------------------------------------
#                     Hard Negative Generation
# ------------------------------------------------------------

def generate_hard_negatives(pairs, num_negs=30000):
    """
    Hard negatives:
    - Sample random mismatched sentence pairs
    - Assign similarity = 0
    """
    all_sentences = []
    for p in pairs:
        all_sentences.append(p.sentence1)
        all_sentences.append(p.sentence2)

    negs = []
    for _ in range(num_negs):
        s1 = random.choice(all_sentences)
        s2 = random.choice(all_sentences)
        if s1 != s2:
            negs.append(PairData(s1, s2, 0.0))

    logger.info(f"[Negatives] Generated {len(negs)} hard negative samples")
    return negs


# ============================================================
#             Combine All Datasets (Weighted)
# ============================================================

def build_full_dataset(tokenizer):
    sts = load_sts(tokenizer)
    paws = load_paws(tokenizer)
    nli = load_nli(tokenizer, max_samples=40000)

    # Combine + Weight
    full = []
    full.extend(sts)     # 1.0x
    full.extend(paws)    # 1.0x
    full.extend(nli)     # reduced by limiting max_samples

    # Hard negatives
    hard_negs = generate_hard_negatives(full, num_negs=30000)
    full.extend(hard_negs)

    random.shuffle(full)
    logger.info(f"[Dataset] Final combined pairs: {len(full)}")

    return full


# ============================================================
#                Collate Function for Dataloader
# ============================================================

def collate_fn(batch):
    input_ids_1 = torch.stack([item["input_ids_1"] for item in batch])
    mask_1 = torch.stack([item["mask_1"] for item in batch])

    input_ids_2 = torch.stack([item["input_ids_2"] for item in batch])
    mask_2 = torch.stack([item["mask_2"] for item in batch])

    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids_1": input_ids_1,
        "mask_1": mask_1,
        "input_ids_2": input_ids_2,
        "mask_2": mask_2,
        "labels": labels
    }


# ============================================================
#                      Evaluation (STS-B Dev)
# ============================================================

import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaluate_stsb(model, tokenizer, device):
    """
    Evaluate on STS-B dev using Mizan similarity.
    """
    ds = load_dataset("stsb_multi_mt", "en")["validation"]

    gold_scores = []
    pred_scores = []

    model.eval()
    with torch.no_grad():
        for row in ds:
            s1 = row["sentence1"]
            s2 = row["sentence2"]

            inputs1 = tokenizer(
                s1, return_tensors="pt", max_length=128,
                truncation=True, padding="max_length"
            ).to(device)

            inputs2 = tokenizer(
                s2, return_tensors="pt", max_length=128,
                truncation=True, padding="max_length"
            ).to(device)

            e1 = model(inputs1["input_ids"], inputs1["attention_mask"])
            e2 = model(inputs2["input_ids"], inputs2["attention_mask"])

            miz = mizan_similarity(e1, e2).item()
            pred_scores.append(miz)

            gold = normalize_score(row["similarity_score"])
            gold_scores.append(gold)

    pear = pearsonr(pred_scores, gold_scores)[0]
    spear = spearmanr(pred_scores, gold_scores)[0]

    logger.info(f"[Eval] STS-B Pearson: {pear:.4f}, Spearman: {spear:.4f}")
    return pear, spear


# ============================================================
#                    One Training Step
# ============================================================

def train_step(model, batch, loss_fn, optimizer, device):
    model.train()

    e1 = model(batch["input_ids_1"].to(device), batch["mask_1"].to(device))
    e2 = model(batch["input_ids_2"].to(device), batch["mask_2"].to(device))

    labels = batch["labels"].to(device)

    loss, logs = loss_fn(e1, e2, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), logs


# ============================================================
#                    Training Stage Function
# ============================================================

def run_stage(
        model, dataloader, device, stage_name,
        lr, num_epochs, loss_fn, unfreeze_layers=None):
    """
    Generic training loop for stage 1/2/3.
    """

    logger.info(f"=============== {stage_name} ===============")

    # Freeze or unfreeze layers
    for name, param in model.transformer.named_parameters():
        param.requires_grad = False

    if unfreeze_layers is not None:
        for name, param in model.transformer.named_parameters():
            for layer_id in unfreeze_layers:
                if f"layer.{layer_id}." in name:
                    param.requires_grad = True

    # Projection + LN must always be trainable
    for param in model.proj.parameters():
        param.requires_grad = True
    for param in model.ln.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    total_steps = len(dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            loss_value, logs = train_step(model, batch, loss_fn, optimizer, device)
            scheduler.step()
            total_loss += loss_value

        avg_loss = total_loss / len(dataloader)
        logger.info(f"[{stage_name}] Epoch {epoch+1}/{num_epochs} - Loss={avg_loss:.4f}")

    logger.info(f"=============== END {stage_name} ===============")


# ============================================================
#                  Full Hybrid Training Pipeline
# ============================================================

def hybrid_train(model, tokenizer, train_pairs, device, batch_size=32):
    dataset = PairDataset(train_pairs, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )

    loss_fn = MizanLoss()

    # ---------------- Stage 1 ---------------- #
    run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        stage_name="Stage 1: Train projection + LN",
        lr=1e-3,
        num_epochs=1,
        loss_fn=loss_fn,
        unfreeze_layers=[]  # None; transformer fully frozen
    )

    # ---------------- Stage 2 ---------------- #
    last_layers = [10, 11]  # unfreeze last 2 layers of encoder
    run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        stage_name="Stage 2: Unfreeze last 2 transformer layers",
        lr=2e-5,
        num_epochs=1,
        loss_fn=loss_fn,
        unfreeze_layers=last_layers
    )

    # ---------------- Stage 3 ---------------- #
    all_layers = list(range(12))  # unfreeze all encoder layers
    run_stage(
        model=model,
        dataloader=dataloader,
        device=device,
        stage_name="Stage 3: Full fine-tuning",
        lr=5e-6,
        num_epochs=1,
        loss_fn=loss_fn,
        unfreeze_layers=all_layers
    )


# ============================================================
#                       SAVE MODEL
# ============================================================

def save_mizan(model, tokenizer, out_dir="mizan_trained_encoder"):
    os.makedirs(out_dir, exist_ok=True)

    # Save transformer + projection + LN
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    # Save config
    config = {
        "backbone": model.backbone_name,
        "proj_dim": model.proj.out_features,
        "alpha": model.alpha
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))

    logger.info(f"[SAVE] Model saved to {out_dir}")


# ============================================================
#                     MAIN
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] Using {device}")

    backbone = "BAAI/bge-base-en-v1.5"
    proj_dim = 512

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = MizanEncoder(backbone=backbone, proj_dim=proj_dim).to(device)

    # -------- Load datasets -------- #
    train_pairs = build_full_dataset(tokenizer)

    # -------- Train model -------- #
    hybrid_train(model, tokenizer, train_pairs, device)

    # -------- Evaluate on STS-B -------- #
    evaluate_stsb(model, tokenizer, device)

    # -------- Save model -------- #
    save_mizan(model, tokenizer, out_dir="mizan_trained_encoder")


if __name__ == "__main__":
    main()
