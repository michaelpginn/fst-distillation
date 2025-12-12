import logging
import math
import tempfile
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from src.modeling.birnn import BiRNN
from src.modeling.rnn import RNNModel
from src.modeling.spectral_penalty import spectral_penalty
from src.modeling.tokenizer import Tokenizer
from src.training.shared import set_lr

device = "cuda" if torch.cuda.is_available() else "mps"

logger = logging.getLogger(__file__)


def compute_loss(model: RNNModel | BiRNN, batch, spectral_norm_weight: float | None):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    next_input_ids = batch["next_input_ids"].to(device, non_blocking=True)
    next_output_ids = batch["next_output_ids"].to(device, non_blocking=True)
    seq_lengths = batch["seq_lengths"].to(device, non_blocking=True)
    out = model(
        input_ids=input_ids, seq_lengths=seq_lengths, next_input_ids=next_input_ids
    )
    if isinstance(out, tuple):
        out = out[0]
    loss = torch.nn.functional.cross_entropy(
        out.permute(0, 2, 1),
        next_output_ids,
        ignore_index=model.tokenizer.pad_token_id,
        label_smoothing=0.1,
    )
    preds = out.argmax(dim=-1)
    mask = next_output_ids != model.tokenizer.pad_token_id
    accuracy = (preds == next_output_ids)[mask].float().mean().item()

    # Occasionally log a few decoded predictions for debugging/monitoring
    try:
        if torch.rand(1).item() < 0.02:  # ~2% of calls
            num_examples = min(8, input_ids.size(0))
            for i in range(num_examples):
                inp = " ".join(
                    model.tokenizer.decode(input_ids[i].tolist(), return_as="list")
                )
                tgt = " ".join(
                    model.tokenizer.decode(
                        next_output_ids[i].tolist(), return_as="list"
                    )
                )
                preds_masked = preds[i].masked_fill(
                    next_output_ids[i] == model.tokenizer.pad_token_id,
                    model.tokenizer.pad_token_id,
                )
                prd = " ".join(
                    model.tokenizer.decode(preds_masked.tolist(), return_as="list")
                )
                logger.info(f"\ninput: {inp}\ntargt: {tgt}\npreds: {prd}")
    except Exception as e:
        logger.debug(f"Skipping sample logging due to error: {e}")

    if spectral_norm_weight is not None:
        spec_loss = 0.0
        for m in model.W_h:
            spec_penalty, spec_norm = spectral_penalty(m.weight)
            spec_loss += spec_penalty
        spec_loss = spectral_norm_weight * spec_loss
    else:
        spec_loss = None

    return loss, spec_loss, accuracy


def train(
    model: RNNModel | BiRNN,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    tokenizer: Tokenizer,
    epochs: int,
    learning_rate: float = 0.0001,
    min_learning_rate: float = 1e-5,
    warmup_steps: int | Literal["auto"] | None = "auto",
    spectral_norm_weight: float | None = 0.1,
    seed: int = 0,
):
    """Trains the model with the specified parameters."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    if torch.cuda.is_available():
        model.compile(dynamic=True, options={"triton.cudagraphs": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type:ignore
    total_steps = epochs * len(train_dataloader)
    if warmup_steps == "auto":
        # Set to 3% of total steps
        warmup_steps = int(0.03 * total_steps)
    elif warmup_steps is None:
        warmup_steps = 0
    wandb.config.update({"warmup_steps": warmup_steps})

    logger.info("Training...")
    step = 0
    for epoch in trange(epochs):
        logger.info(("-" * 25) + f"Epoch {epoch}" + ("-" * 25))
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            loss, spec_loss, _ = compute_loss(
                model,
                batch,
                spectral_norm_weight=spectral_norm_weight,
            )
            (loss + spec_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if warmup_steps and step < warmup_steps:
                # Linear warmup
                new_lr = learning_rate * step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                new_lr = (
                    min_learning_rate
                    + (learning_rate - min_learning_rate) * cosine_decay
                )
            wandb.log({"train.lr": new_lr}, step=step)
            set_lr(optimizer, new_lr)
            optimizer.step()
            step += 1
            epoch_loss += loss.detach().item()
        train_loss = epoch_loss / len(train_dataloader)

        model.eval()
        epoch_loss = 0
        epoch_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                loss, _, accuracy = compute_loss(
                    model,
                    batch,
                    spectral_norm_weight=spectral_norm_weight,
                )
                epoch_loss += loss.detach().item()
                epoch_accuracy += accuracy
        eval_loss = epoch_loss / len(eval_dataloader)
        eval_accuracy = epoch_accuracy / len(eval_dataloader)

        total_spec_norm = 0
        for m in model.W_h:
            _, spec_norm = spectral_penalty(m.weight)
            total_spec_norm += spec_norm

        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info(f"Validation loss: {eval_loss:.4f}")
        logger.info(f"Validation accuracy: {eval_accuracy:.2f}")

        wandb.log(
            {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "validation": {
                    "loss": eval_loss,
                    "accuracy": eval_accuracy,
                },
                "spectral_norm": total_spec_norm / len(model.W_h),
            },
            step=step,
        )

    # Upload checkpoint to wandb
    temp_file = tempfile.NamedTemporaryFile(suffix="ckpt.pth")
    torch.save(model.state_dict(), temp_file)
    artifact = wandb.Artifact("final_checkpoint", type="model")
    artifact.add_file(temp_file.name)
    wandb.log_artifact(artifact)

    return eval_loss  # type:ignore
