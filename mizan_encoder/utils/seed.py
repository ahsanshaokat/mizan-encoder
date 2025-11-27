"""
Reproducibility helper.
Sets seeds for:
- Python
- NumPy
- PyTorch (CPU/GPU)
"""

import torch
import numpy as np
import random

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔒 Global seed set to {seed}")
