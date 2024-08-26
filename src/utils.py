from jaxtyping import Bool
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

def pad_batch_collate_fn(batch: list[dict[str, list[int]]], pad_token_id: int):
    """Collates a batch where each item is a dict[str, list[int]]. Pads sequences to longest and creates tensors."""

    keys = batch[0].keys()
    grouped_sequences = {key: [] for key in keys}
    for example in batch:
        for key in keys:
            grouped_sequences[key].append(torch.tensor(example[key], dtype=torch.long))

    padded_tensors = {key: pad_sequence(grouped_sequences[key], batch_first=True) for key in keys}
    return padded_tensors

def create_causal_mask(seq_length: int) -> Tensor:
    """Creates causal mask (upper triangular True)"""
    mask = torch.tril(torch.ones(seq_length, seq_length) == 1)
    return ~mask

def create_pad_mask(tensor: Tensor, pad_token: int) -> Tensor:
    return (tensor == pad_token)
