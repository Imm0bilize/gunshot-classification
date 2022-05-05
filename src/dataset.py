import torch
from torch.utils.data import Dataset


class GunShotDataset(Dataset):
    def __init__(self):
        self.dataset = torch.rand((512, 3, 256, 256))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.dataset[idx]
