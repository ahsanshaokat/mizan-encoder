import torch


def mizan_similarity(x, y, p=2, eps=1e-6):
    """Numerically stable mizan similarity"""
    num = torch.norm(x - y, p=p, dim=-1)
    den = torch.norm(x, p=p, dim=-1) + torch.norm(y, p=p, dim=-1) + eps
    
    # Clamp to prevent NaN
    ratio = num / den
    ratio = torch.clamp(ratio, 0.0, 2.0)
    
    return 1 - ratio


def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


def mizan_distance(x, y):
    return 1 - mizan_similarity(x, y)


def check_embeddings_stable(emb, name=""):
    """Check if embeddings have numerical issues"""
    if torch.isnan(emb).any():
        print(f"⚠️ NaN in {name} embeddings!")
        return False
    
    norm = torch.norm(emb, dim=-1)
    if (norm < 1e-6).any():
        print(f"⚠️ Very small norms in {name} embeddings!")
        return False
    
    return True