import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mizan_encoder.hf_model import MizanEncoderHF
from mizan_encoder.loss import MizanLoss
from mizan_encoder.data import load_sts_tsv, load_snli_jsonl, PairDataset


# =========================================================
# LOAD CONFIG
# =========================================================
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# COLLATE FN (NEEDED FOR STRING BATCHING)
# =========================================================
def collate_pairs(batch):
    # batch is a list of tuples: (text1, text2, label)
    t1 = [x[0] for x in batch]
    t2 = [x[1] for x in batch]
    labels = torch.tensor([x[2] for x in batch], dtype=torch.float)
    return t1, t2, labels


# =========================================================
# TRAINING FUNCTION
# =========================================================
def train(config_path):

    # 1) LOAD CONFIG
    config = load_config(config_path)
    print("📄 Loaded config:", config_path)

    # 2) TOKENIZER + MODEL
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])

    model = MizanEncoderHF.from_pretrained(
        config["backbone"],
        pooling=config["pooling"],
        proj_dim=config["proj_dim"],
        alpha=config["alpha"]
    )

    # 3) LOAD DATASETS (FIXED ORDER)
    print("📚 Loading STS-B dataset...")
    sts = load_sts_tsv(config["sts_path"], sample_size=config["sts_samples"])

    print("📚 Loading SNLI dataset...")
    nli = load_snli_jsonl(config["nli_path"], sample_size=config["nli_samples"])

    all_pairs = sts + nli
    dataset = PairDataset(all_pairs)

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_pairs   # FIXED
    )

    # 4) TRAIN SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Training on:", device)

    model = model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["lr_backbone"]},
        {"params": model.proj.parameters(), "lr": config["lr_proj"]},
    ])

    loss_fn = MizanLoss(alpha=config["alpha"])

    # 5) TRAIN LOOP
    for epoch in range(config["epochs"]):
        for text1, text2, labels in loader:

            # TOKENIZE FIRST BATCH
            t1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(device)
            t2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to(device)

            # MODEL FORWARD (FIXED)
            emb1 = model(input_ids=t1["input_ids"], attention_mask=t1["attention_mask"])
            emb2 = model(input_ids=t2["input_ids"], attention_mask=t2["attention_mask"])

            labels = labels.to(device)

            # LOSS
            loss = loss_fn(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{config['epochs']} | Loss = {loss.item():.4f}")

    # 6) SAVE MODEL
    os.makedirs(config["output_dir"], exist_ok=True)
    print("💾 Saving model to:", config["output_dir"])

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print("✅ Training complete.")


# RUN
train("configs/small.json")
