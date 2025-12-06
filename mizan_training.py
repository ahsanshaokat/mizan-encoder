import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from mizan_encoder.hf_model import MizanEncoderHF
from mizan_encoder.loss import MizanContrastiveLoss, StableMizanLoss
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
# Training loop - FIXED VERSION
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
        proj_dim=cfg["proj_dim"],
        alpha=cfg["alpha"]  # CRITICAL: Pass alpha
    )

    # -------------------------------
    # Datasets
    # -------------------------------
    print("📘 Loading STS...")
    sts = load_sts_tsv(cfg["sts_path"], sample_size=cfg["sts_samples"])
    print(f"Loaded STS pairs: {len(sts)}")

    print("📘 Loading SNLI...")
    nli = load_snli_jsonl(cfg["nli_path"], sample_size=cfg["nli_samples"])
    print(f"Loaded SNLI pairs: {len(nli)}")

    all_pairs = sts + nli
    print(f"Total pairs: {len(all_pairs)}")
    
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

    # Use separate learning rates for backbone and projection
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

    # Use StableMizanLoss for robustness
    loss_fn = StableMizanLoss(margin=0.3)
    # Alternatively: loss_fn = MizanContrastiveLoss(margin=0.3)

    # -------------------------------
    # Debug helper
    # -------------------------------
    def debug_step(emb1, emb2, labels, step, loss):
        if step % 100 == 0:
            print(f"\n--- Step {step} Debug ---")
            print(f"Loss: {loss.item():.6f}")
            
            # Check embeddings
            norm1 = torch.norm(emb1, dim=-1)
            norm2 = torch.norm(emb2, dim=-1)
            print(f"Embedding norms - min: {norm1.min().item():.4f}, max: {norm1.max().item():.4f}")
            print(f"Labels range: {labels.min().item():.2f} to {labels.max().item():.2f}")
            
            # Check for NaN
            if torch.isnan(emb1).any() or torch.isnan(emb2).any():
                print("⚠️ WARNING: NaN in embeddings!")
            if torch.isnan(loss):
                print("⚠️ WARNING: NaN loss!")

    # -------------------------------
    # Training loop
    # -------------------------------
    model.train()
    
    for epoch in range(cfg["epochs"]):
        print(f"\n🚀 Starting Epoch {epoch+1}/{cfg['epochs']}")
        
        for step, (t1, t2, labels) in enumerate(loader):
            
            # Tokenize
            t1_enc = tokenizer(t1, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            t2_enc = tokenizer(t2, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            labels = labels.to(device)
            
            # Forward pass
            emb1 = model(**t1_enc)
            emb2 = model(**t2_enc)
            
            # Check for NaN before loss
            if torch.isnan(emb1).any() or torch.isnan(emb2).any():
                print(f"⚠️ NaN in embeddings at step {step}, skipping batch")
                continue
            
            # Compute loss
            loss = loss_fn(emb1, emb2, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️ NaN loss at step {step}, skipping")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping - CRITICAL for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Debug and logging
            debug_step(emb1, emb2, labels, step, loss)
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{cfg['epochs']} Step {step} | Loss = {loss.item():.4f}")

    # -------------------------------
    # Save final model
    # -------------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"\n💾 Saving model to: {cfg['output_dir']}")

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    
    # Save training config
    with open(os.path.join(cfg["output_dir"], "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("✅ Training complete successfully!")


if __name__ == "__main__":
    train("configs/small.json")