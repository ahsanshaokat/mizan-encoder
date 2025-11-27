import torch

def collate_batch(batch):
    """
    Custom collate function to merge batch items.

    We avoid default PyTorch collation to maintain
    correct tensor shapes for text pairs.
    """

    ids1 = torch.stack([item["ids1"] for item in batch])
    mask1 = torch.stack([item["mask1"] for item in batch])
    ids2 = torch.stack([item["ids2"] for item in batch])
    mask2 = torch.stack([item["mask2"] for item in batch])
    labels = torch.stack([item["label"] for item in batch]).long()

    return {
        "ids1": ids1,
        "mask1": mask1,
        "ids2": ids2,
        "mask2": mask2,
        "label": labels
    }
