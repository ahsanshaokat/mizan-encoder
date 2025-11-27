import torch
from mizan_encoder.loss import MizanContrastiveLoss

def test_mizan_similarity_basic():
    loss_fn = MizanContrastiveLoss()
    x = torch.tensor([[1., 2., 3.]])
    y = torch.tensor([[1., 2., 3.]])
    sim = loss_fn.mizan_sim(x, y)
    assert sim > 0.999, "Identical vectors should have similarity ≈ 1"

def test_mizan_similarity_scale_penalty():
    loss_fn = MizanContrastiveLoss()
    x = torch.tensor([[1., 1., 1.]])
    y = torch.tensor([[10., 10., 10.]])
    sim = loss_fn.mizan_sim(x, y)
    assert sim < 0.5, "Vectors with same direction but different scale must be penalized"

def test_contrastive_loss_positive():
    loss_fn = MizanContrastiveLoss()
    x = torch.randn(4, 384)
    y = x.clone()
    label = torch.ones(4)
    loss = loss_fn(x, y, label)
    assert loss < 0.1, "Positive pairs must produce low loss"

def test_contrastive_loss_negative():
    loss_fn = MizanContrastiveLoss()
    x = torch.randn(4, 384)
    y = torch.randn(4, 384)
    label = torch.zeros(4)
    loss = loss_fn(x, y, label)
    assert loss > 0.2, "Negative pairs must produce high loss"
