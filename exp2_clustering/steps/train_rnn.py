"""Usage: python -m exp2-clustering.train_rnn

Trains an RNN acceptor to do binary classification on aligned inflection data. Saves model checkpoint to the `checkpoints/` directory.

You must run `run_alignment.py` first to produce aligned data files.
"""

import pathlib
from argparse import ArgumentParser

import torch

from src.modeling import RNNModel
from src.optional_wandb import wandb
from src.tasks.inflection_classification import create_dataloader
from src.tasks.inflection_classification.dataset import (
    AlignedInflectionDataset,
    load_examples_from_file,
)
from src.training_classifier import train


def train_rnn(
    language: str,
    batch_size: int,
    epochs: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    seed=0,
):
    hyperparams = locals()
    wandb.init(
        entity="lecs-general",
        project="fst-distillation.exp2.rnn_classifier",
        config={**hyperparams},
        save_code=True,
        group=language,
    )

    # In order to create negative examples, we need to pre-load all of the examples so
    # we don't accidentally create negative examples that are valid
    train_examples = load_examples_from_file(
        pathlib.Path(__file__).parent.parent / f"aligned_data/{language}.trn.aligned"
    )
    eval_examples = load_examples_from_file(
        pathlib.Path(__file__).parent.parent / f"aligned_data/{language}.dev.aligned"
    )
    train_dataset = AlignedInflectionDataset(
        positive_examples=train_examples,
        all_positive_examples=train_examples + eval_examples,
        tokenizer=None,
    )
    tokenizer = train_dataset.tokenizer
    eval_dataset = AlignedInflectionDataset(
        positive_examples=eval_examples,
        all_positive_examples=train_examples + eval_examples,
        tokenizer=tokenizer,
    )
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
    eval_dataloader = create_dataloader(eval_dataset, batch_size=batch_size)
    model = RNNModel(
        tokenizer=tokenizer, d_model=d_model, num_layers=num_layers, dropout=dropout
    )
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    checkpoint_path = (
        pathlib.Path(__file__).parent.parent / f"runs/{wandb.run.name}/model.pt"  # type:ignore
    )
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_dict": model.config_dict,
            "tokenizer_dict": model.tokenizer.state_dict,
        },
        checkpoint_path,
    )
    run_name = wandb.run.name  # type:ignore
    wandb.finish()
    return run_name


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--language", default="swe")
    parser.add_argument("--batch-size", default=1024)
    parser.add_argument("--epochs", default=150)
    parser.add_argument("--hidden-dim", default=64)
    parser.add_argument("--num-layers", default=2)
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--learning-rate", default=0.0002)

    args = parser.parse_args()
    train_rnn(
        language=args.language,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        d_model=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
    )
