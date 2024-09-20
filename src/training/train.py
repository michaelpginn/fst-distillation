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
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, "Training"):
        source_input_ids = batch["source_input_ids"].to(device)
        source_pad_mask = batch["source_pad_mask"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_pad_mask = batch["target_pad_mask"].to(device)
        target_labels = batch["target_label_ids"].to(device)

        optimizer.zero_grad()
        out = model(
            src=source_input_ids,
            tgt=target_input_ids,
            src_pad_mask=source_pad_mask,
            tgt_pad_mask=target_pad_mask,
        )
        loss = loss_fn(out.permute(0, 2, 1), target_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.detach().item()

    return epoch_loss / len(dataloader)


def validation_loop(model: torch.nn.Module, dataloader: DataLoader, loss_fn: Callable):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, "Evaluating"):
            source_input_ids = batch["source_input_ids"].to(device)
            source_pad_mask = batch["source_pad_mask"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_pad_mask = batch["target_pad_mask"].to(device)
            target_labels = batch["target_label_ids"].to(device)

            out = model(
                src=source_input_ids,
                tgt=target_input_ids,
                src_pad_mask=source_pad_mask,
                tgt_pad_mask=target_pad_mask,
            )
            loss = loss_fn(out.permute(0, 2, 1), target_labels)
            epoch_loss += loss.detach().item()
    return epoch_loss / len(dataloader)


def train(
    project_name: str,
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

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model = model.to(device)

    print("Training...")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch}", "-" * 25)
        train_loss = training_loop(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        validation_loss = validation_loop(
            model=model, dataloader=train_dataloader, loss_fn=loss_fn
        )
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
