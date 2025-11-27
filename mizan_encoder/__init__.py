"""
Mizan Encoder Package
---------------------

This package provides:

- MizanTextEncoder (Transformer-based embedding model)
- BalancedMeanPooling (scale-aware pooling layer)
- MizanContrastiveLoss (metric learning for MizanSimilarity)
- Utility modules (logging, seeds, checkpoints)
- Metrics (MizanSimilarity, Cosine, Distance functions)

Optimized for:
- Multi-scale embeddings
- Long/short chunk imbalance
- Proportional similarity
- Stable embedding space geometry
"""

from .config import MizanConfig
from .encoder import MizanTextEncoder
from .loss import MizanContrastiveLoss
from .pooling import BalancedMeanPooling
from .metrics import mizan_similarity
