import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------
# Debug Model with NaN tracing
# ------------------------------------------------------------
class DebugMizanEncoder(nn.Module):
    def __init__(self, backbone="sentence-transformers/all-MiniLM-L6-v2", proj_dim=384, alpha=0.15):
        super().__init__()
        
        print(f"🔧 Initializing with backbone: {backbone}")
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(backbone)
        hidden_size = self.backbone.config.hidden_size
        print(f"📐 Hidden size: {hidden_size}")
        
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.alpha = alpha
        
        # Initialize projection layer safely
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def safe_pooling(self, hidden, mask):
        """Simple safe mean pooling"""
        print(f"   🤖 Pooling - Input shape: {hidden.shape}, Mask shape: {mask.shape}")
        
        # Convert mask
        mask = mask.float().unsqueeze(-1)  # [batch, seq, 1]
        
        # Check for all-zero masks
        mask_sum = mask.sum(dim=1)
        if (mask_sum == 0).any():
            print("   ⚠️ WARNING: Some masks are all zeros!")
            mask = mask + 1e-8  # Add tiny value
        
        # Weighted average
        weighted = hidden * mask
        sum_weighted = weighted.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-6)
        
        pooled = sum_weighted / sum_mask
        
        print(f"   📊 Pooled shape: {pooled.shape}")
        
        # Check for NaN
        if torch.isnan(pooled).any():
            print("   ❌ ERROR: NaN in pooled output!")
            # Replace NaN with zeros
            pooled = torch.nan_to_num(pooled, nan=0.0)
        
        return pooled
    
    def scale_stabilize(self, x):
        """Safe Mizan normalization with debugging"""
        print(f"   🎯 Scale stabilize - Input shape: {x.shape}")
        print(f"   📏 Input stats - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")
        
        # Check input
        if torch.isnan(x).any():
            print("   ❌ ERROR: NaN input to scale_stabilize!")
            x = torch.nan_to_num(x, nan=0.0)
        
        if torch.isinf(x).any():
            print("   ❌ ERROR: Inf input to scale_stabilize!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calculate norm
        norm = torch.norm(x, dim=-1, keepdim=True)
        print(f"   📐 Norm stats - min: {norm.min().item():.6f}, max: {norm.max().item():.6f}")
        
        # Clamp norm
        norm = torch.clamp(norm, min=1e-6, max=1e6)
        
        # Apply alpha
        denominator = norm ** self.alpha
        print(f"   α = {self.alpha}, Denominator stats - min: {denominator.min().item():.6f}, max: {denominator.max().item():.6f}")
        
        # Divide
        result = x / (denominator + 1e-8)
        
        # Check result
        if torch.isnan(result).any():
            print("   ❌ ERROR: NaN in scale_stabilize result!")
            result = torch.nan_to_num(result, nan=0.0)
        
        print(f"   ✅ Output stats - min: {result.min().item():.6f}, max: {result.max().item():.6f}, norm: {torch.norm(result, dim=-1).mean().item():.6f}")
        
        return result
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        print("\n" + "="*60)
        print("🚀 FORWARD PASS DEBUG")
        print("="*60)
        
        # 1. Check inputs
        print(f"📥 Input IDs shape: {input_ids.shape}")
        print(f"🎭 Attention mask shape: {attention_mask.shape}")
        
        if torch.isnan(input_ids).any():
            print("❌ CRITICAL: NaN in input_ids!")
            return torch.zeros((input_ids.size(0), self.proj.out_features), 
                             device=input_ids.device)
        
        # 2. Backbone forward
        print("\n🔍 Step 1: Backbone forward...")
        try:
            backbone_output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True  # Get all hidden states
            )
            
            last_hidden = backbone_output.last_hidden_state
            print(f"   📦 Last hidden state shape: {last_hidden.shape}")
            
            # Check backbone output
            if torch.isnan(last_hidden).any():
                print("   ❌ ERROR: NaN in backbone output!")
                
                # Check each layer
                for i, hidden_state in enumerate(backbone_output.hidden_states):
                    if torch.isnan(hidden_state).any():
                        print(f"   🎯 NaN found in layer {i}!")
                        break
                
                last_hidden = torch.nan_to_num(last_hidden, nan=0.0)
            
            # Stats
            print(f"   📊 Hidden state stats - min: {last_hidden.min().item():.6f}, max: {last_hidden.max().item():.6f}")
            
        except Exception as e:
            print(f"   ❌ ERROR in backbone: {e}")
            return torch.zeros((input_ids.size(0), self.proj.out_features), 
                             device=input_ids.device)
        
        # 3. Pooling
        print("\n🔍 Step 2: Pooling...")
        pooled = self.safe_pooling(last_hidden, attention_mask)
        
        # 4. Projection
        print("\n🔍 Step 3: Projection...")
        print(f"   🔄 Projection: {self.proj.in_features} -> {self.proj.out_features}")
        
        # Check weights
        if torch.isnan(self.proj.weight).any():
            print("   ❌ ERROR: NaN in projection weights!")
            self.proj.weight.data = torch.nan_to_num(self.proj.weight.data, nan=0.0)
        
        projected = self.proj(pooled)
        print(f"   📐 Projected shape: {projected.shape}")
        print(f"   📊 Projected stats - min: {projected.min().item():.6f}, max: {projected.max().item():.6f}")
        
        if torch.isnan(projected).any():
            print("   ❌ ERROR: NaN after projection!")
            projected = torch.nan_to_num(projected, nan=0.0)
        
        # 5. Scale stabilize
        print("\n🔍 Step 4: Scale stabilize...")
        output = self.scale_stabilize(projected)
        
        print("\n" + "="*60)
        print(f"✅ Forward pass complete. Output shape: {output.shape}")
        print("="*60 + "\n")
        
        return output

# ------------------------------------------------------------
# Test Function
# ------------------------------------------------------------
def test_single_sentence():
    print("🧪 Testing with single sentence...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model and tokenizer
    model = DebugMizanEncoder(
        backbone="sentence-transformers/all-MiniLM-L6-v2",
        proj_dim=384,
        alpha=0.15
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test sentences
    test_sentences = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "A",  # Very short sentence
        "",   # Empty string (edge case)
    ]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n{'#'*50}")
        print(f"Test {i+1}: '{sentence}'")
        print(f"{'#'*50}")
        
        if not sentence:
            sentence = " "  # Handle empty
        
        # Tokenize
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        print(f"Tokenized: {inputs['input_ids'].shape}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        # Forward pass
        with torch.no_grad():
            try:
                output = model(**inputs)
                
                # Check final output
                if torch.isnan(output).any():
                    print("❌ FINAL OUTPUT HAS NaN!")
                else:
                    print(f"✅ Success! Output shape: {output.shape}")
                    print(f"   Output norm: {torch.norm(output, dim=-1).item():.6f}")
                    
            except Exception as e:
                print(f"❌ Exception: {e}")
                import traceback
                traceback.print_exc()

def test_batch():
    print("\n🧪 Testing with batch of sentences...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DebugMizanEncoder(
        backbone="sentence-transformers/all-MiniLM-L6-v2",
        proj_dim=384,
        alpha=0.15
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test batch
    sentences = [
        "This is the first sentence.",
        "Here is another one.",
        "And a third for good measure.",
    ]
    
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    print(f"Batch size: {len(sentences)}")
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(**inputs)
        
        if torch.isnan(output).any():
            print("❌ Batch output has NaN!")
        else:
            print(f"✅ Batch test successful!")
            print(f"   Output shape: {output.shape}")

# ------------------------------------------------------------
# Check Parameter Initialization
# ------------------------------------------------------------
def check_parameters():
    print("\n🔍 Checking model parameters...")
    
    model = DebugMizanEncoder(
        backbone="sentence-transformers/all-MiniLM-L6-v2",
        proj_dim=384,
        alpha=0.15
    )
    
    print("\nParameter statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60} shape: {str(list(param.shape)):20} "
                  f"mean: {param.data.mean().item():.6f} "
                  f"std: {param.data.std().item():.6f} "
                  f"NaN: {torch.isnan(param.data).any().item()}")
    
    # Check specific layers
    print("\n🔬 Detailed checks:")
    
    # Check projection layer
    proj_weight = model.proj.weight
    print(f"Projection weight - min: {proj_weight.min().item():.6f}, "
          f"max: {proj_weight.max().item():.6f}")
    
    # Check if any parameters are all zeros
    for name, param in model.named_parameters():
        if param.requires_grad and (param.data == 0).all():
            print(f"⚠️ WARNING: {name} is all zeros!")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🔬 MIZAN ENCODER NaN DEBUGGER")
    print("="*60)
    
    # 1. Check parameters
    check_parameters()
    
    # 2. Test single sentences
    test_single_sentence()
    
    # 3. Test batch
    test_batch()
    
    print("\n" + "="*60)
    print("🎯 DEBUGGING COMPLETE")
    print("="*60)