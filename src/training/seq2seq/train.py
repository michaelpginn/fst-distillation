import math
import tempfile
from contextlib import nullcontext
from typing import Callable, Literal

import torch
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ..shared import set_lr

device = "cuda" if torch.cuda.is_available() else "mps"


def batch_forward(batch, model: torch.nn.Module, loss_fn: Callable):
    """Run forward and get loss for a single batch"""
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
    return loss


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    epochs: int,
    learning_rate: float = 0.0001,
    min_learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    warmup_steps: int | Literal["auto"] | None = "auto",
    # early_stopping_patience: int = 100,
    seed: int = 0,
):
    """Trains the model with the specified parameters."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=0.1, ignore_index=model.tokenizer.pad_token_id
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * len(train_dataloader)
    if warmup_steps == "auto":
        # Set to 3% of total steps
        warmup_steps = int(0.03 * total_steps)
    elif warmup_steps is None:
        warmup_steps = 0
    wandb.config.update({"warmup_steps": warmup_steps})

    model = model.to(device)
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if device in ("cuda", "cpu")
        else nullcontext()
    )

    print("Training...")
    step = 0
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch}", "-" * 25)

        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, "Training"):
            with autocast_ctx:
                loss = batch_forward(batch, model, loss_fn)
                optimizer.zero_grad()
                loss.backward()
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
                wandb.log({"train.lr": new_lr})
                set_lr(optimizer, new_lr)
                optimizer.step()
            step += 1
            train_loss += loss.detach().item()
        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        validation_loss: float = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, "Training"):
                with autocast_ctx:
                    loss = batch_forward(batch, model, loss_fn)
                    validation_loss += loss.detach().item()
        validation_loss /= len(eval_dataloader)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        wandb.log(
            {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "validation": {"loss": validation_loss},
            }
        )
        # Early stopping
        # if not best_val_loss or validation_loss < best_val_loss[0]:
        #     best_val_loss = (validation_loss, epoch)
        # elif best_val_loss and best_val_loss[1] < epoch - early_stopping_patience:
        #     print(
        #         f"Early stop! Eval loss hasn't improved in {early_stopping_patience} epochs"
        #     )
        #     break

    # Upload checkpoint to wandb
    temp_file = tempfile.NamedTemporaryFile(suffix="ckpt.pth")
    torch.save(model.state_dict(), temp_file)
    # artifact = wandb.Artifact("final_checkpoint", type="model")
    # artifact.add_file(temp_file.name)
    # wandb.log_artifact(artifact)
