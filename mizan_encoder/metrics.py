import torch

def mizan_similarity(x, y, p=2, eps=1e-6):
    num = torch.norm(x - y, p=p, dim=-1)
    den = torch.norm(x, p=p, dim=-1) + torch.norm(y, p=p, dim=-1) + eps
    return 1 - (num / den)


def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


def mizan_distance(x, y):
    return 1 - mizan_similarity(x, y)
