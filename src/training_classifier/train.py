import tempfile
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.optional_wandb import wandb

device = "cuda" if torch.cuda.is_available() else "mps"


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


def validation_loop(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, "Evaluating"):
            input_ids = batch["input_ids"].to(device)
            seq_lengths = batch["seq_lengths"]
            labels = batch["label"].float().to(device)
            out = model(
                input_ids=input_ids,
                seq_lengths=seq_lengths,
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels)
            epoch_loss += loss.detach().item()
    return epoch_loss / len(dataloader)


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    epochs: int,
    learning_rate: float = 0.0001,
    seed: int = 0,
):
    """Trains the model with the specified parameters."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type:ignore

    model = model.to(device)

    print("Training...")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch}", "-" * 25)
        train_loss = training_loop(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
        )
        validation_loss = validation_loop(model=model, dataloader=eval_dataloader)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

    # Upload checkpoint to wandb
    temp_file = tempfile.NamedTemporaryFile(suffix="ckpt.pth")
    torch.save(model.state_dict(), temp_file)
    artifact = wandb.Artifact("final_checkpoint", type="model")
    artifact.add_file(temp_file.name)
    wandb.log_artifact(artifact)
