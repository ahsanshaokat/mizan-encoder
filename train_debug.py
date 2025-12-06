import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# Custom safe pooling
class EmergencyPooling(nn.Module):
    def forward(self, hidden, mask):
        # Simple mean pooling with extreme safety
        mask = mask.float()
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Use a loop to avoid NaN propagation
        batch_size, seq_len, hidden_dim = hidden.shape
        result = torch.zeros(batch_size, hidden_dim, device=hidden.device)
        
        for i in range(batch_size):
            valid_indices = mask[i] > 0
            if valid_indices.any():
                result[i] = hidden[i][valid_indices].mean(dim=0)
            else:
                result[i] = hidden[i].mean(dim=0)
        
        return result

# Simple encoder for debugging
class DebugEncoder(nn.Module):
    def __init__(self, backbone="sentence-transformers/all-MiniLM-L6-v2", proj_dim=384):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.pooler = EmergencyPooling()
        self.proj = nn.Linear(self.backbone.config.hidden_size, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Basic safety checks
        if torch.isnan(input_ids).any():
            print("❌ ERROR: NaN in input_ids!")
            raise ValueError("NaN in inputs")
        
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Check for NaN
        if torch.isnan(out.last_hidden_state).any():
            print("❌ ERROR: NaN in transformer output!")
            # Try to recover
            out.last_hidden_state = torch.nan_to_num(out.last_hidden_state, nan=0.0)
        
        pooled = self.pooler(out.last_hidden_state, attention_mask)
        
        if torch.isnan(pooled).any():
            print("❌ ERROR: NaN after pooling!")
            pooled = torch.zeros_like(pooled)
        
        projected = self.proj(pooled)
        normalized = self.layer_norm(projected)
        
        # Final L2 normalization
        normalized = torch.nn.functional.normalize(normalized, dim=-1)
        
        return normalized

# Simple contrastive loss
class SimpleContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, labels):
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        
        # Simple contrastive loss
        loss = torch.where(
            labels > 0.5,
            1 - cos_sim,  # Positive pairs
            torch.clamp(cos_sim - self.margin, min=0)  # Negative pairs
        )
        
        return loss.mean()

# Debug training
def debug_train():
    print("🔍 DEBUG MODE: Testing with minimal setup")
    
    # Simple config
    config = {
        "backbone": "sentence-transformers/all-MiniLM-L6-v2",
        "proj_dim": 384,
        "batch_size": 2,  # Tiny batch
        "epochs": 1,
        "lr": 1e-5,
        "test_sentences": [
            ("Hello world", "Hi there", 0.8),
            ("The cat sat", "A dog ran", 0.2),
        ]
    }
    
    # Initialize
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"])
    model = DebugEncoder(config["backbone"], config["proj_dim"]).to(device)
    loss_fn = SimpleContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Single training step
    print("\n🧪 Testing single forward pass...")
    
    for i, (text1, text2, label) in enumerate(config["test_sentences"]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text1: {text1}")
        print(f"Text2: {text2}")
        print(f"Label: {label}")
        
        # Tokenize
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        label_tensor = torch.tensor([label], device=device)
        
        # Forward pass with debug
        print("\n📊 Forward pass debug:")
        try:
            with torch.no_grad():
                emb1 = model(**inputs1)
                emb2 = model(**inputs2)
            
            print(f"Embedding 1 shape: {emb1.shape}")
            print(f"Embedding 1 norm: {torch.norm(emb1, dim=-1).item():.4f}")
            print(f"Embedding 2 shape: {emb2.shape}")
            print(f"Embedding 2 norm: {torch.norm(emb2, dim=-1).item():.4f}")
            
            # Check for NaN
            if torch.isnan(emb1).any() or torch.isnan(emb2).any():
                print("❌ ERROR: NaN in embeddings!")
                continue
            
            # Loss
            loss = loss_fn(emb1, emb2, label_tensor)
            print(f"Loss: {loss.item():.6f}")
            
            # Backward test
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if torch.isnan(param.grad).any():
                        print(f"⚠️ NaN gradient in {name}")
            
            print(f"Total gradient norm: {total_grad_norm:.6f}")
            optimizer.step()
            
            print("✅ Step completed successfully!")
            
        except Exception as e:
            print(f"❌ ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_train()