from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modeling.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "mps"


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    max_length: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Runs inference. Returns a list of predicted token IDs in the same order as the inputs, and the label IDs."""
    model.eval()

    predicted_ids: List[List[int]] = []
    label_ids: List[List[int]] = []

    for batch in tqdm(dataloader, "Predicting"):
        batch_size = batch["source_input_ids"].size(0)

        # Start with just column of <BOS> tokens
        output_sequence = torch.full(
            (batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )

        # Since sequences may end at different points, we use dynamic stopping
        # We track when each sequence has reached an <eos> token,
        # after which we ignore future predictions and add <pad>
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            source_input_ids = batch["source_input_ids"].to(device)
            source_pad_mask = batch["source_pad_mask"].to(device)

            # Generate logits [batch_size, seq_length, vocab_size]
            out = model(
                src=source_input_ids,
                tgt=output_sequence,
                src_pad_mask=source_pad_mask,
            )
            # We only care about the last column (the next predicted tokens)
            next_token_logits = out[:, -1, :]  # [batch_size, vocab_size]

            # Greedy decoding, may want to change
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            output_sequence = torch.cat(
                [output_sequence, next_tokens.unsqueeze(1)], dim=1
            )

            finished |= next_tokens == tokenizer.eos_token_id
            if finished.all():
                break

        # We have generated seqs for this batch. Let's add to a full list of preds.
        predicted_ids += output_sequence.tolist()
        label_ids += batch["target_input_ids"].tolist()
    return predicted_ids, label_ids
