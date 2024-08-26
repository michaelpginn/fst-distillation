import torch
from torch.utils.data import DataLoader

from ..utils import create_pad_mask, pad_batch_collate_fn
from .dataset import SharedTaskInflectionDataset


def create_dataloaders(
    train_path: str,
    eval_path: str,
    batch_size: int
):
    train_dataset = SharedTaskInflectionDataset(path=train_path, tokenizer=None)
    tokenizer = train_dataset.tokenizer
    eval_dataset =  SharedTaskInflectionDataset(path=eval_path, tokenizer=tokenizer)
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch: list[dict[str, list[int]]]):
        collated_batch = pad_batch_collate_fn(batch, tokenizer.pad_token_id)

        # Create masks
        collated_batch['source_pad_mask'] = create_pad_mask(collated_batch['source_input_ids'], pad_token_id)
        collated_batch['target_pad_mask'] = create_pad_mask(collated_batch['target_input_ids'], pad_token_id)
        return collated_batch

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_dataloader, eval_dataloader, tokenizer
