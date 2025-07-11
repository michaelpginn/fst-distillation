import torch
from torch.utils.data import DataLoader

from src.modeling.input_processing import pad_batch_collate_fn

from .dataset import AlignedInflectionDataset
from .tokenizer import AlignedInflectionTokenizer


def create_dataloader(
    aligned_data_path: str,
    batch_size: int,
    syncretic_example_lookup: dict[str, list[tuple]],
    pretrained_tokenizer: AlignedInflectionTokenizer | None = None,
):
    """Notably, this dataloader will include (organic) positive and (synthetic) negative examples.

    Arguments:
        data_path: str, path to an aligned inflection file
        batch_size: int
        pretrained_tokenizer: AlignedInflectionTokenizer | None, if provided, uses a pretrained tokenizer rather than training a new one
    """
    dataset = AlignedInflectionDataset(
        path=aligned_data_path,
        tokenizer=pretrained_tokenizer,
        syncretic_example_lookup=syncretic_example_lookup,
    )
    tokenizer = dataset.tokenizer

    def collate_fn(batch: list[dict[str, int | list[int]]]):
        collated_batch = pad_batch_collate_fn(batch, tokenizer.pad_token_id)
        collated_batch["seq_lengths"] = torch.tensor(
            [len(ex["input_ids"]) for ex in batch]  # type:ignore
        )
        return collated_batch

    train_dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return train_dataloader, tokenizer
