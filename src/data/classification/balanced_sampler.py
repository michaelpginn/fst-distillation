import random
from typing import Iterator

from torch.utils.data import Sampler


class BalancedResampledSampler(Sampler[int]):
    """
    Each epoch, randomly sample a balanced set of positive and negative indices.
    - If one class has more items, it is downsampled to the size of the smaller class.
    - Returns a fresh random selection each epoch.
    - Optionally interleaves pos/neg for per-batch balance.
    """

    def __init__(
        self,
        pos_indices: list[int],
        neg_indices: list[int],
    ) -> None:
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.rng = random.Random()

    def __iter__(self) -> Iterator[int]:
        # Downsample to the smaller class count to keep the epoch balanced
        m = min(len(self.pos_indices), len(self.neg_indices))
        # Sample without replacement; will vary each epoch
        pos_sample = self.rng.sample(self.pos_indices, m)
        neg_sample = self.rng.sample(self.neg_indices, m)
        mixed = pos_sample + neg_sample
        self.rng.shuffle(mixed)
        return iter(mixed)

    def __len__(self) -> int:
        return 2 * min(len(self.pos_indices), len(self.neg_indices))
