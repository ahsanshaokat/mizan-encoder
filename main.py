from mizan_encoder.encoder import MizanTextEncoder
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = MizanTextEncoder.from_pretrained("saved/mizan_encoder_v1").eval()

text = "The balance of justice is essential."
batch = tok(text, return_tensors="pt")

emb = model(batch["input_ids"], batch["attention_mask"])
print(emb.shape)
