import logging
import tempfile

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import trange

import wandb
from src.modeling.rnn import RNNModel
from src.modeling.spectral_penalty import spectral_penalty
from src.modeling.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "mps"

logger = logging.getLogger(__file__)


def compute_loss(
    model: Module, batch, epoch: int, tokenizer, spectral_norm_weight: float | None
):
    """Compute loss and stats for a single batch"""
    input_ids = batch["input_ids"].to(device)
    seq_lengths = batch["seq_lengths"].to(device)
    labels = batch["label"].float().to(device)
    out = model(
        input_ids=input_ids,
        seq_lengths=seq_lengths,
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)

    if spectral_norm_weight is not None:
        spec_loss = 0.0
        for m in model.W_h:
            spec_penalty, spec_norm = spectral_penalty(m.weight)
            spec_loss += spec_penalty
        loss += spectral_norm_weight * spec_loss

    # Stats
    preds = torch.nn.functional.sigmoid(out) > 0.5
    correct = preds == labels.bool()
    incorrect_inputs = [
        tokenizer.decode(ids.tolist()) for ids in input_ids[~correct].detach()
    ]
    incorrect_labels = labels[~correct].bool().tolist()
    incorrect_inputs = [
        f"[{label}] {char_string}"
        for char_string, label in zip(incorrect_inputs, incorrect_labels)
    ]
    return {
        "loss": loss,
        "num_pos": sum(labels),
        "num_pred_pos": sum(preds),
        "num_true_pos": sum(correct & labels.bool()),
        "incorrect_inputs": incorrect_inputs,
    }


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
    model = model.to(device)
    if torch.cuda.is_available():
        model.compile(dynamic=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type:ignore

    logger.info("Training...")
    for epoch in trange(epochs):
        logger.info(("-" * 25) + f"Epoch {epoch}" + ("-" * 25))
        model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            stats = compute_loss(
                model,
                batch,
                epoch=epoch,
                tokenizer=tokenizer,
                spectral_norm_weight=spectral_norm_weight,
            )
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += stats["loss"].detach().item()
        train_loss = epoch_loss / len(train_dataloader)

        model.eval()
        epoch_loss = 0
        epoch_pos = 0  # Number of positive examples
        epoch_pred_pos = 0  # Number of predicted positives
        epoch_true_pos = 0  # Number of true positives
        incorrect = []
        with torch.no_grad():
            for batch in eval_dataloader:
                stats = compute_loss(
                    model,
                    batch,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    spectral_norm_weight=spectral_norm_weight,
                )
                epoch_loss += stats["loss"].detach().item()
                epoch_pos += stats["num_pos"]
                epoch_pred_pos += stats["num_pred_pos"]
                epoch_true_pos += stats["num_true_pos"]
                incorrect.extend(stats["incorrect_inputs"])
        precision = epoch_true_pos / epoch_pred_pos
        recall = epoch_true_pos / epoch_pos
        f1 = 2 * precision * recall / (precision + recall)
        eval_stats = {"f1": f1, "precision": precision, "recall": recall}
        eval_loss = epoch_loss / len(eval_dataloader)

        total_spec_norm = 0
        for m in model.W_h:
            _, spec_norm = spectral_penalty(m.weight)
            total_spec_norm += spec_norm

        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info(f"Validation loss: {eval_loss:.4f}")
        logger.info(f"Validation F1: {eval_stats['f1']:.1%}")
        logger.info("Some incorrect val instances:\n" + "\n".join(incorrect[:5]))
        wandb.log(
            {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "validation": {"loss": eval_loss, **eval_stats},
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
