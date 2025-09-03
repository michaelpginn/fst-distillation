"""Usage: python -m exp2-clustering.train_rnn

Trains an RNN acceptor to do binary classification on aligned inflection data. Saves model checkpoint to the `checkpoints/` directory.

You must run `run_alignment.py` first to produce aligned data files.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict

import torch

from src.modeling import RNNModel
from src.optional_wandb import wandb
from src.tasks.inflection_classification import create_dataloader
from src.tasks.inflection_classification.dataset import load_examples_from_file
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

    # In order to create negative examples, we need to pre-load all of the examples so
    # we don't accidentally create negative examples that are valid
    aligned_train_path = (
        pathlib.Path(__file__).parent / f"aligned_data/{language}.trn.aligned.jsonl"
    )
    aligned_eval_path = (
        pathlib.Path(__file__).parent / f"aligned_data/{language}.dev.aligned.jsonl"
    )
    all_examples = load_examples_from_file(
        aligned_train_path
    ) + load_examples_from_file(aligned_eval_path)
    syncretic_example_lookup: DefaultDict[str, list[tuple]] = defaultdict(lambda: [])
    for ex in all_examples:
        syncretic_example_lookup["".join(ex.aligned_chars_as_strs)].append(
            tuple(ex.features)
        )

    # Create dataloaders
    train_dataloader, tokenizer = create_dataloader(
        aligned_data_path=aligned_train_path,
        batch_size=batch_size,
        syncretic_example_lookup=syncretic_example_lookup,
    )
    eval_dataloader, _ = create_dataloader(
        aligned_data_path=aligned_eval_path,
        batch_size=batch_size,
        pretrained_tokenizer=tokenizer,
        syncretic_example_lookup=syncretic_example_lookup,
    )
    model = RNNModel(
        tokenizer=tokenizer, d_model=d_model, num_layers=num_layers, dropout=dropout
    )
    wandb.init(
        entity="lecs-general",
        project="fst-distillation.exp2",
        config={**hyperparams},
        save_code=True,
        group=language,
    )
    # wandb.watch(models=model, log_freq=1)
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    checkpoint_path = pathlib.Path(__file__).parent / f"runs/{wandb.run.name}/model.pt"  # type:ignore
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_dict": model.config_dict,
            "tokenizer_dict": model.tokenizer.state_dict,
        },
        checkpoint_path,
    )
    return wandb.run.name  # type:ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--language", default="swe")
    parser.add_argument("--batch-size", default=1024)
    parser.add_argument("--epochs", default=150)
    parser.add_argument("--hidden-dim", default=64)
    parser.add_argument("--num-layers", default=2)
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--learning-rate", default=0.0001)

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
