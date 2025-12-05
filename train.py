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
    sts = load_sts_tsv(config["sts_path"], sample_size=config["sts_samples"])

    print("📚 Loading SNLI dataset...")
    nli = load_snli_jsonl(config["nli_path"], sample_size=config["nli_samples"])

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
        total_loss = 0

        for text1, text2, labels in loader:
            labels = labels.to(device)

            emb1 = model.encode(text1, tokenizer, device=device)
            emb2 = model.encode(text2, tokenizer, device=device)

            loss = loss_fn(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss = {avg:.4f}")

    # =====================================================
    # 6) SAVE MODEL
    # =====================================================
    os.makedirs(config["output_dir"], exist_ok=True)
    print("💾 Saving model to:", config["output_dir"])

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print("✅ Training complete.")
