import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mizan_encoder.hf_model import MizanEncoderHF
from mizan_encoder.loss import MizanLoss
from mizan_encoder.data import load_sts_tsv, load_snli_jsonl, PairDataset


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def train(config_path):

    # =====================================================
    # 1) LOAD CONFIG
    # =====================================================
    config = load_config(config_path)
    print("📄 Loaded config:", config_path)

    # =====================================================
    # 2) TOKENIZER & MODEL
    # =====================================================
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])

    model = MizanEncoderHF.from_pretrained(
        config["backbone"],
        pooling=config["pooling"],
        proj_dim=config["proj_dim"],
        alpha=config["alpha"]
    )

    # =====================================================
    # 3) LOAD DATASETS
    # =====================================================
    print("📚 Loading STS-B dataset...")
    nli = load_snli_jsonl(config["nli_path"], sample_size=config["nli_samples"])

    print("📚 Loading SNLI dataset...")
    sts = load_sts_tsv(config["sts_path"], sample_size=config["sts_samples"])


    all_pairs = sts + nli
    dataset = PairDataset(all_pairs)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    # =====================================================
    # 4) TRAIN SETUP
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Training on:", device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = MizanLoss(alpha=config["alpha"])

    # =====================================================
    # 5) TRAIN LOOP
    # =====================================================
    for epoch in range(config["epochs"]):
        for batch in loader:
            text1, text2, labels = batch

            # Tokenize
            t1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            t2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

            # Forward pass WITH gradients
            emb1 = model(t1["input_ids"], t1["attention_mask"])
            emb2 = model(t2["input_ids"], t2["attention_mask"])

            # Compute loss
            labels = labels.to(device).float()
            loss = loss_fn(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Loss = {loss.item():.4f}")




    # =====================================================
    # 6) SAVE MODEL
    # =====================================================
    os.makedirs(config["output_dir"], exist_ok=True)
    print("💾 Saving model to:", config["output_dir"])

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print("✅ Training complete.")

train("configs/small.json")