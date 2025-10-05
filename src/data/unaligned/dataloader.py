from torch.utils.data import DataLoader

from src.modeling.input_processing import create_pad_mask, pad_batch_collate_fn

from .dataset import String2StringDataset
from .tokenizer import String2StringTokenizer


def create_dataloader(
    dataset: String2StringDataset,
    batch_size: int,
    pretrained_tokenizer: String2StringTokenizer | None = None,
):
    """
    Arguments:
        data_path: str, path to a ST-style data file
        batch_size: int
        pretrained_tokenizer: Tokenizer | None, if provided, uses a pretrained tokenizer rather than training a new one
    """
    tokenizer = dataset.tokenizer
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch: list[dict[str, int | list[int]]]):
        collated_batch = pad_batch_collate_fn(batch, tokenizer.pad_token_id)

        # Create masks
        collated_batch["source_pad_mask"] = create_pad_mask(
            collated_batch["source_input_ids"], pad_token_id
        )
        collated_batch["target_pad_mask"] = create_pad_mask(
            collated_batch["target_input_ids"], pad_token_id
        )
        return collated_batch

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return dataloader, tokenizer
