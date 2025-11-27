import torch
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder

def test_encoder_forward_pass():
    model = MizanTextEncoder()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    out = tok("hello world", return_tensors="pt")
    emb = model(out["input_ids"], out["attention_mask"])

    assert emb.shape == (1, 384)
    assert not torch.isnan(emb).any(), "Embedding cannot contain NaNs"
