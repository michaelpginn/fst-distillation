import torch
from torch.utils.data import DataLoader

from src.data.classification.balanced_sampler import (
    BalancedResampledSampler,
)
from src.modeling.input_processing import pad_batch_collate_fn

from .dataset import AlignedInflectionDataset


def create_dataloader(
    dataset: AlignedInflectionDataset,
    batch_size: int,
):
    """Notably, this dataloader will include (organic) positive and (synthetic) negative examples.

    Arguments:
        data_path: str, path to an aligned inflection file
        batch_size: int
        pretrained_tokenizer: AlignedInflectionTokenizer | None, if provided, uses a pretrained tokenizer rather than training a new one
    """
    tokenizer = dataset.tokenizer

    def collate_fn(batch: list[dict[str, int | list[int]]]):
        collated_batch = pad_batch_collate_fn(batch, tokenizer.pad_token_id)
        collated_batch["seq_lengths"] = torch.tensor(
            [len(ex["input_ids"]) for ex in batch]  # type:ignore
        )
        return collated_batch

    positive_indices = list(range(dataset.num_positives))
    negative_indices = list(range(dataset.num_positives, len(dataset)))
    sampler = BalancedResampledSampler(
        pos_indices=positive_indices, neg_indices=negative_indices
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn
    )
    return dataloader
