import torch
from torch.utils.data import DataLoader

from src.modeling.input_processing import pad_batch_collate_fn

from .dataset import AlignedLanguageModelingDataset


def create_dataloader(
    dataset: AlignedLanguageModelingDataset,
    batch_size: int,
):
    """Dataloader for aligned LM task (next symbol prediction)

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

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader
