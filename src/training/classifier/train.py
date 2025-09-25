import logging
import tempfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.modeling.tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "mps"

logger = logging.getLogger(__file__)


def training_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,  # type:ignore
):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, "Training"):
        input_ids = batch["input_ids"].to(device)
        seq_lengths = batch["seq_lengths"]
        labels = batch["label"].float().to(device)
        optimizer.zero_grad()
        out = model(
            input_ids=input_ids,
            seq_lengths=seq_lengths,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.detach().item()
    return epoch_loss / len(dataloader)


def validation_loop(
    model: torch.nn.Module, dataloader: DataLoader, tokenizer: Tokenizer
):
    model.eval()
    epoch_loss = 0
    epoch_pos = 0  # Number of positive examples
    epoch_pred_pos = 0  # Number of predicted positives
    epoch_true_pos = 0  # Number of true positives
    all_incorrect: list[str] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, "Evaluating"):
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            seq_lengths = batch["seq_lengths"]
            labels: torch.Tensor = batch["label"].float().to(device)
            out = model(
                input_ids=input_ids,
                seq_lengths=seq_lengths,
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
            epoch_loss += loss.detach().item()

            # Compute F1 on the fly
            epoch_pos += sum(labels)
            preds = torch.nn.functional.sigmoid(out) > 0.5
            correct = preds == labels.bool()
            epoch_pred_pos += sum(preds)
            epoch_true_pos += sum(correct & labels.bool())

            # Log incorrect predictions
            incorrect_inputs = [
                tokenizer.decode(ids.tolist()) for ids in input_ids[~correct].detach()
            ]
            incorrect_labels = labels[~correct].bool().tolist()
            all_incorrect.extend(
                [
                    f"[{label}] {char_string}"
                    for char_string, label in zip(incorrect_inputs, incorrect_labels)
                ]
            )
    precision = epoch_true_pos / epoch_pred_pos
    recall = epoch_true_pos / epoch_pos
    f1 = 2 * precision * recall / (precision + recall)
    stats = {"f1": f1, "precision": precision, "recall": recall}
    return (
        epoch_loss / len(dataloader),
        stats,
        all_incorrect,
    )


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    tokenizer: Tokenizer,
    epochs: int,
    learning_rate: float = 0.0001,
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
        train_loss = training_loop(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
        )
        validation_loss, validation_stats, incorrect_examples = validation_loop(
            model=model, dataloader=eval_dataloader, tokenizer=tokenizer
        )
        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info(f"Validation loss: {validation_loss:.4f}")
        logger.info(f"Validation F1: {validation_stats['f1']:.1%}")
        logger.info(
            "Some incorrect val instances:\n" + "\n".join(incorrect_examples[:5])
        )
        wandb.log(
            {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "validation": {"loss": validation_loss, **validation_stats},
            }
        )

    # Upload checkpoint to wandb
    temp_file = tempfile.NamedTemporaryFile(suffix="ckpt.pth")
    torch.save(model.state_dict(), temp_file)
    artifact = wandb.Artifact("final_checkpoint", type="model")
    artifact.add_file(temp_file.name)
    wandb.log_artifact(artifact)
