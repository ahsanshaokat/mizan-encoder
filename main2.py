from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

tok = AutoTokenizer.from_pretrained("checkpoints/mizan_encoder_small")
model = AutoModel.from_pretrained("checkpoints/mizan_encoder_small")

def encode(text):
    x = tok(text, return_tensors="pt")
    out = model(**x)
    return out.last_hidden_state.mean(dim=1)   # basic pooling

a = encode("Dracula is a vampire novel.")
b = encode("The sun is a plasma sphere.")

print(a.shape, b.shape)
print("sim:", F.cosine_similarity(a, b))
