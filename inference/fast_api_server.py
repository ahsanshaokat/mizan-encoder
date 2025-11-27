from fastapi import FastAPI
from pydantic import BaseModel
from inference.embed import MizanEmbedder
from inference.scoring import cosine, hybrid
from mizan_vector.metrics import mizan_similarity

app = FastAPI(title="Mizan Encoder API", version="1.0")

embedder = MizanEmbedder()

class EmbedRequest(BaseModel):
    text: str

class ScoreRequest(BaseModel):
    text1: str
    text2: str
    mode: str = "mizan"   # cosine / mizan / hybrid

@app.post("/embed")
def embed(req: EmbedRequest):
    return {"embedding": embedder.encode(req.text).tolist()}

@app.post("/score")
def score(req: ScoreRequest):
    emb1 = embedder.encode(req.text1)
    emb2 = embedder.encode(req.text2)

    if req.mode == "cosine":
        s = cosine(emb1, emb2)
    elif req.mode == "hybrid":
        s = hybrid(emb1, emb2)
    else:
        s = mizan_similarity(emb1, emb2)

    return {"score": float(s)}

# Run with:
# uvicorn inference.fast_api_server:app --reload
