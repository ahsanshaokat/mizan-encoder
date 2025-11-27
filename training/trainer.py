import torch
from tqdm import tqdm

class MizanTrainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch}")

        for batch in loop:
            ids1 = batch["ids1"].to(self.device)
            mask1 = batch["mask1"].to(self.device)
            ids2 = batch["ids2"].to(self.device)
            mask2 = batch["mask2"].to(self.device)
            labels = batch["label"].to(self.device)

            emb1 = self.model(ids1, mask1)
            emb2 = self.model(ids2, mask2)

            loss = self.criterion(emb1, emb2, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(loader)

    def save(self, path):
        self.model.save_pretrained(path)
