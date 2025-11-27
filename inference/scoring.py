import numpy as np
from mizan_vector.metrics import mizan_similarity

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))

def hybrid(a, b, w_cos=0.5, w_miz=0.5):
    """
    Weighted mixture of cosine + Mizan similarity.
    """
    cos = cosine(a, b)
    miz = mizan_similarity(a, b)
    return w_cos * cos + w_miz * miz
