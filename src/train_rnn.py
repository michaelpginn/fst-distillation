"""Usage: python -m src.training.train_rnn

Trains an RNN on aligned data. Saves model checkpoint to the `checkpoints/` directory.

You must run `run_alignment.py` first to produce aligned data files.
"""

from logging import getLogger
from typing import Literal

import torch

import wandb

from .data.aligned.example import load_examples_from_file
from .modeling import RNNModel
from .paths import Paths, create_arg_parser, create_paths_from_args

logger = getLogger(__name__)


def train_rnn(
    paths: Paths,
    objective: Literal["classification", "lm"],
    batch_size: int,
    epochs: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    use_many_to_many_transitions: bool,
    activation: Literal["relu", "gelu", "tanh"],
    spectral_norm_weight: float | None,
    wandb_run: wandb.Run | None = None,
    seed=0,
) -> tuple[wandb.Run | None, float]:
    logger.info(f"Training on {paths['identifier']}")
    hyperparams = locals()
    project_name = f"fst-distillation.clustering.rnn_{objective}"

    if wandb_run is None:
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

        wandb_run = wandb.init(
            entity="lecs-general",
            project=project_name,
            config={**hyperparams},
            save_code=True,
        )
    else:
        wandb_run.config.update(hyperparams)

    # In order to create negative examples (for classification), we need to pre-load all of the examples so
    # we don't accidentally create negative examples that are valid
    train_examples = load_examples_from_file(
        paths["train_aligned"], remove_epsilons=use_many_to_many_transitions
    )
    eval_examples = load_examples_from_file(
        paths["eval_aligned"], remove_epsilons=use_many_to_many_transitions
    )
    wandb_run.log({"train_size": len(train_examples)}, commit=False)

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
        paths["models_folder"] / f"{wandb_run.name}/model.pt"  # type:ignore
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
    wandb_run.finish()
    return wandb_run, last_eval_loss


if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument("--objective", choices=["classification", "lm"], required=True)
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
        paths=create_paths_from_args(args),
        objective=args.objective,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        d_model=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        use_many_to_many_transitions=False,
        activation=args.activation,
        spectral_norm_weight=args.spec_weight,
    )
