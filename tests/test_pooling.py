import torch
from mizan_encoder.pooling import BalancedMeanPooling

def test_balanced_mean_pooling_masking():
    pool = BalancedMeanPooling()

    tokens = torch.tensor([
        [[1., 1.], [3., 3.], [5., 5.]],
    ])
    mask = torch.tensor([[1, 1, 0]])

    out = pool(tokens, mask)

    assert torch.allclose(out, torch.tensor([[2., 2.]])), \
        "Mask must exclude padded tokens"

def test_pooling_single_token():
    pool = BalancedMeanPooling()

    tokens = torch.tensor([[[4., 8.]]])
    mask = torch.tensor([[1]])

    out = pool(tokens, mask)
    assert torch.equal(out, torch.tensor([[4., 8.]]))
