"""Usage: python -m exp2-clustering.train_rnn

Trains an RNN acceptor to do binary classification on aligned inflection data. Saves model checkpoint to the `checkpoints/` directory.

You must run `run_alignment.py` first to produce aligned data files.
"""

import pathlib
from collections import defaultdict
from typing import DefaultDict

import torch

from src.modeling import RNNModel
from src.optional_wandb import wandb
from src.tasks.inflection_classification import create_dataloader
from src.tasks.inflection_classification.dataset import load_examples_from_file
from src.training_classifier import train


def train_rnn(
    aligned_train_path: str,
    aligned_eval_path: str,
    test_path: str,
    batch_size=256,
    epochs=100,
    learning_rate=0.0001,
    d_model=512,
    num_layers=4,
    dropout=0.1,
    seed=0,
):
    language = aligned_train_path.split("/")[-1].split(".")[0]
    hyperparams = locals()

    # In order to create negative examples, we need to pre-load all of the examples so
    # we don't accidentally create negative examples that are valid
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
    wandb.watch(models=model, log_freq=1)
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    checkpoint_path = pathlib.Path(__file__).parent / f"checkpoints/{wandb.run.name}.pt"  # type:ignore
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_dict": model.config_dict,
            "tokenizer_dict": model.tokenizer.state_dict,
        },
        checkpoint_path,
    )


if __name__ == "__main__":
    train_rnn(
        aligned_train_path="exp2-clustering/aligned_data/hil.trn.aligned.jsonl",
        aligned_eval_path="./exp2-clustering/aligned_data/hil.dev.aligned.jsonl",
        test_path="./task0-data/GOLD-TEST/hil.tst",
        epochs=100,
        dropout=0.5,
    )
