"""Usage: python -m experiments.clustering.train_rnn

Trains an RNN acceptor to do binary classification on aligned inflection data. Saves model checkpoint to the `checkpoints/` directory.

You must run `run_alignment.py` first to produce aligned data files.
"""

import pathlib
from argparse import ArgumentParser
from logging import getLogger
from typing import Literal

import torch

import wandb
from experiments.shared import (
    DataFiles,
    add_task_parser,
    get_data_files,
    get_identifier,
)
from src.data.aligned.example import load_examples_from_file
from src.modeling import RNNModel

logger = getLogger(__name__)


def train_rnn(
    identifier: str,
    data_files: DataFiles,
    objective: Literal["classification", "lm"],
    batch_size: int,
    epochs: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    no_epsilon_inputs: bool,
    activation: Literal["relu", "gelu", "tanh"],
    spectral_norm_weight: float | None,
    seed=0,
) -> tuple[wandb.Run | None, float]:
    logger.info(f"Training on {identifier}")
    hyperparams = locals()
    project_name = f"fst-distillation.clustering.rnn_{objective}"

    # Check if this run is a duplicate
    try:
        filters = {
            f"config.{key}": value
            for key, value in hyperparams.items()
            if isinstance(value, (str, float, int))
        }
        runs = wandb.Api().runs(
            path=f"lecs-general/{project_name}",
            filters=filters,
        )
        if len(runs) > 0 and any(
            r._state == "finished" or r._state == "running" for r in runs
        ):
            logger.info("Skipping run, identical run already found!!")
            eval_loss = (
                runs[0]
                .history(keys=["validation.loss"])["validation.loss"]
                .iloc[-1]
                .item()
            )
            return runs[0], eval_loss
    except:
        print("Project does not exist yet")

    wandb.init(
        entity="lecs-general",
        project=project_name,
        config={**hyperparams},
        save_code=True,
    )

    # In order to create negative examples (for classification), we need to pre-load all of the examples so
    # we don't accidentally create negative examples that are valid
    train_examples = load_examples_from_file(
        data_files["train_aligned"], remove_epsilons=no_epsilon_inputs
    )
    eval_examples = load_examples_from_file(
        data_files["eval_aligned"], remove_epsilons=no_epsilon_inputs
    )
    wandb.log({"train_size": len(train_examples)})

    if objective == "classification":
        from src.data.aligned.classification.dataloader import create_dataloader
        from src.data.aligned.classification.dataset import AlignedClassificationDataset
        from src.training.classifier.train import train

        train_dataset = AlignedClassificationDataset(
            positive_examples=train_examples,
            all_positive_examples=train_examples + eval_examples,
            tokenizer=None,
        )
        tokenizer = train_dataset.tokenizer
        eval_dataset = AlignedClassificationDataset(
            positive_examples=eval_examples,
            all_positive_examples=train_examples + eval_examples,
            tokenizer=tokenizer,
        )
    elif objective == "lm":
        from src.data.aligned.language_modeling.dataloader import create_dataloader
        from src.data.aligned.language_modeling.dataset import (
            AlignedLanguageModelingDataset,
        )
        from src.training.language_modeling.train import train

        train_dataset = AlignedLanguageModelingDataset(
            examples=train_examples,
            tokenizer=None,
        )
        tokenizer = train_dataset.tokenizer
        eval_dataset = AlignedLanguageModelingDataset(
            examples=eval_examples,
            tokenizer=tokenizer,
        )

    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)  # type:ignore
    eval_dataloader = create_dataloader(eval_dataset, batch_size=batch_size)  # type:ignore
    model = RNNModel(
        tokenizer=tokenizer,
        output_head=objective,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
    )
    last_eval_loss: float = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        spectral_norm_weight=spectral_norm_weight,
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
    run = wandb.run
    wandb.finish()
    return run, last_eval_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    add_task_parser(parser)
    parser.add_argument("--objective", choices=["classification", "lm"])
    parser.add_argument("--batch-size", default=2048)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--hidden-dim", default=64)
    parser.add_argument("--num-layers", default=1)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--learning-rate", default=0.001)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--spec-weight", default=0.1)
    args = parser.parse_args()
    train_rnn(
        identifier=get_identifier(args),
        data_files=get_data_files(args),
        objective=args.objective,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        d_model=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        no_epsilon_inputs=False,
        activation=args.activation,
        spectral_norm_weight=args.spec_weight,
    )
