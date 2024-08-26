import torch
from torch.utils.data import DataLoader


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader
):
    """Runs inference"""
    model.eval()
