import torch
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder

class MizanEmbedder:
    """
    Simple inference wrapper for single-text embedding.
    """

    def __init__(self, model_path="saved/mizan_encoder_v1",
                 backbone="distilbert-base-uncased",
                 device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MizanTextEncoder.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)

    def encode(self, text, max_len=256):
        tokens = self.tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            emb = self.model(
                tokens["input_ids"].to(self.device),
                tokens["attention_mask"].to(self.device)
            )

        return emb.squeeze(0).cpu().numpy()
