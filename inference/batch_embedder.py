import torch
import numpy as np
from transformers import AutoTokenizer
from mizan_encoder import MizanTextEncoder

class MizanBatchEmbedder:
    """
    Batch embed many texts efficiently.
    """

    def __init__(self, model_path="saved/mizan_encoder_v1",
                 backbone="distilbert-base-uncased",
                 device=None,
                 batch_size=16):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model = MizanTextEncoder.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)

    def encode(self, texts, max_len=256):
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]

            tokens = self.tokenizer(
                batch,
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

            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)
