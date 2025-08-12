import logging

import torch

logger = logging.Logger(__file__)


def standard_scale(x: torch.Tensor):
    logger.info("Scaling...")
    return (x - x.mean(dim=0)) / x.std(dim=0)
