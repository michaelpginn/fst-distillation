import logging
import tempfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.modeling.rnn import RNNModel
from src.modeling.spectral_penalty import spectral_penalty
from src.modeling.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "mps"

logger = logging.getLogger(__file__)


def compute_loss(
    model: RNNModel, batch, epoch: int, spectral_norm_weight: float | None
):
    input_ids = batch["input_ids"].to(device)
    seq_lengths = batch["seq_lengths"].to(device)
    labels = batch["labels"].to(device)
    out = model(
        input_ids=input_ids,
        seq_lengths=seq_lengths,
    )
    loss = torch.nn.functional.cross_entropy(out.permute(0, 2, 1), labels)

    if spectral_norm_weight is not None:
        spec_loss = 0.0
        for m in model.W_h:
            spec_penalty, spec_norm = spectral_penalty(m.weight)
            spec_loss += spec_penalty
        loss += spectral_norm_weight * spec_loss

    return loss


def train(
    model: RNNModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    tokenizer: Tokenizer,
    epochs: int,
    learning_rate: float = 0.0001,
    spectral_norm_weight: float | None = 0.1,
    seed: int = 0,
):
    """Trains the model with the specified parameters."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type:ignore

    model = model.to(device)

    logger.info("Training...")
    for epoch in range(epochs):
        logger.info(("-" * 25) + f"Epoch {epoch}" + ("-" * 25))

        model.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader, "Training"):
            optimizer.zero_grad()
            loss = compute_loss(
                model,
                batch,
                epoch=epoch,
                spectral_norm_weight=spectral_norm_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.detach().item()
        train_loss = epoch_loss / len(train_dataloader)

        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, "Evaluating"):
                loss = compute_loss(
                    model,
                    batch,
                    epoch=epoch,
                    spectral_norm_weight=spectral_norm_weight,
                )
                epoch_loss += loss.detach().item()
        eval_loss = epoch_loss / len(eval_dataloader)

        total_spec_norm = 0
        for m in model.W_h:
            _, spec_norm = spectral_penalty(m.weight)
            total_spec_norm += spec_norm

        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info(f"Validation loss: {eval_loss:.4f}")
        wandb.log(
            {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "validation": {"loss": eval_loss},
                "spectral_norm": total_spec_norm / len(model.W_h),
            },
            step=epoch,
        )

    # Upload checkpoint to wandb
    temp_file = tempfile.NamedTemporaryFile(suffix="ckpt.pth")
    torch.save(model.state_dict(), temp_file)
    artifact = wandb.Artifact("final_checkpoint", type="model")
    artifact.add_file(temp_file.name)
    wandb.log_artifact(artifact)

    return eval_loss  # type:ignore
