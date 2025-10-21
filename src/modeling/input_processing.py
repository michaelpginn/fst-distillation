import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def pad_batch_collate_fn(batch: list[dict[str, int | list[int]]], pad_token_id: int):
    """Collates a batch where each item is a dict[str, list[int]]. Pads sequences to longest and creates tensors."""
    return {
        key: (
            pad_sequence(
                [torch.as_tensor(example[key], dtype=torch.long) for example in batch],
                batch_first=True,
                padding_value=pad_token_id,
            )
            .clone()
            .contiguous()
            if isinstance(batch[0][key], list)
            else torch.as_tensor([ex[key] for ex in batch])
        )
        for key in batch[0].keys()
    }


def create_causal_mask(seq_length: int) -> Tensor:
    """Creates causal mask (upper triangular True), where `True` positions are masked"""
    mask = torch.tril(torch.ones(seq_length, seq_length) == 1)
    return ~mask


def create_pad_mask(tensor: Tensor, pad_token: int) -> Tensor:
    return tensor == pad_token
