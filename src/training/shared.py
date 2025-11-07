from torch.optim.adamw import AdamW


def set_lr(optimizer: AdamW, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
