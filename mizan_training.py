import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from mizan_encoder.hf_model import MizanEncoderHF
from mizan_encoder.loss import MizanLoss
from mizan_encoder.data import load_sts_tsv, load_snli_jsonl, PairDataset


# ------------------------------------------------------------
# Load config
# ------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Collate function (batching strings)
# ------------------------------------------------------------
def collate_pairs(batch):
    t1 = [x[0] for x in batch]
    t2 = [x[1] for x in batch]
    lab = torch.tensor([x[2] for x in batch], dtype=torch.float)
    return t1, t2, lab


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def train(config_path):

    cfg = load_config(config_path)
    print("📄 Loaded config:", cfg)

    # -------------------------------
    # Tokenizer + Model
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])

    model = MizanEncoderHF.from_pretrained(
        cfg["backbone"],
        pooling="balanced-mean",
        proj_dim=cfg["proj_dim"]
    )

    # -------------------------------
    # Datasets
    # -------------------------------
    print("📘 Loading STS...")
    sts = load_sts_tsv(cfg["sts_path"], sample_size=cfg["sts_samples"])

    print("📘 Loading SNLI...")
    nli = load_snli_jsonl(cfg["nli_path"], sample_size=cfg["nli_samples"])

    all_pairs = sts + nli
    dataset = PairDataset(all_pairs)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_pairs
    )

    # -------------------------------
    # Prepare training
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Training on:", device)

    model = model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
        {"params": model.proj.parameters(), "lr": cfg["lr_proj"]},
    ], weight_decay=0.01)

    total_steps = len(loader) * cfg["epochs"]
    warmup = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_steps
    )

    loss_fn = MizanLoss(alpha=cfg["alpha"])

    # -------------------------------
    # Training loop
    # -------------------------------
    model.train()

    for epoch in range(cfg["epochs"]):
        for step, (t1, t2, labels) in enumerate(loader):

            t1 = tokenizer(t1, return_tensors="pt", padding=True, truncation=True).to(device)
            t2 = tokenizer(t2, return_tensors="pt", padding=True, truncation=True).to(device)

            labels = labels.to(device)

            emb1 = model(**t1)
            emb2 = model(**t2)

            loss = loss_fn(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()

            # Prevent NaNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if step % 200 == 0:
                print(f"Epoch {epoch+1}/{cfg['epochs']} Step {step} | Loss = {loss.item():.4f}")

    # -------------------------------
    # Save final model
    # -------------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("\n💾 Saving model:", cfg["output_dir"])

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    print("✅ Training complete. No NaNs should exist.")


if __name__ == "__main__":
    train("configs/small.json")
