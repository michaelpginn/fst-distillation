"""Runs the full experiment for a given language"""

import itertools
import logging
from dataclasses import asdict
from math import ceil

from tqdm import tqdm

import wandb
from src.data.aligned.example import load_examples_from_file
from src.paths import create_arg_parser, create_paths_from_args
from src.train_alignment_predictor import train_alignment_predictor

from .extract_fst import ExtractionHyperparameters, extract_fst
from .train_rnn import train_rnn

logger = logging.getLogger(__name__)

parser = create_arg_parser()
parser.add_argument("--objective", choices=["lm", "classification"])
parser.add_argument("--batch-size", default=2048)
parser.add_argument("--epochs", default=200)
args = parser.parse_args()
paths = create_paths_from_args(args)

# 1. Check if we've already aligned the language. If not, run alignment on raw files.
if (not paths["train_aligned"].exists()) or (not paths["eval_aligned"].exists()):
    from .run_alignment import run_alignment

    logger.info("Couldn't find aligned data, running alignment!")
    run_alignment(paths)
    # The aligned paths should exist now, we can just use the paths from before

    # 1b. Train alignment model
    train_size = len(load_examples_from_file(paths["train_aligned"]))
    # Scale based on the params I've found work pretty well
    lr = 0.001
    batch_size = ceil(train_size * 128 / 3000)
    num_batches = ceil(train_size / batch_size)
    epochs = ceil((500 * num_batches / 20))
    train_alignment_predictor(paths, batch_size, epochs=epochs, learning_rate=lr)

# 2. Train model and pick best by eval loss
best_model = None
training_hyperparam_options: list[tuple[str, list]] = [
    ("d_model", [16, 32, 64]),
    ("dropout", [0, 0.1]),
    ("learning_rate", [2e-3, 1e-2]),
]
all_combos = itertools.product(*[opts for _, opts in training_hyperparam_options])
for combo in tqdm(all_combos, desc="Sweeping"):
    logger.info(f"Training with params: {combo}")
    d_model, dropout, learning_rate = combo
    run, last_eval_loss = train_rnn(
        paths=paths,
        objective=args.objective,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        d_model=d_model,
        num_layers=1,
        dropout=dropout,
        learning_rate=learning_rate,
        no_epsilon_inputs=False,
        activation="tanh",
        spectral_norm_weight=0.1,
    )
    assert run is not None
    if best_model is None or last_eval_loss < best_model[0]:
        logger.info(
            f"Run {run} is better than prior ({last_eval_loss} < {best_model[0] if best_model else 'N/A'})"
        )
        best_model = (last_eval_loss, run)

# 3. Using the best model, run extraction sweep
assert best_model is not None
model_eval_loss, best_run = best_model

clustering_hyperparam_options: list[tuple[str, list]] = [
    ("state_split_classifier", ["svm", "logistic"]),
    ("minimum_transition_count", [None, 10, 25, 50]),
    ("kmeans_num_clusters", [50, 100, 250, 500, 1000, 1500]),
]
clustering_hyperparam_options.extend([])
all_extraction_combos = itertools.product(
    *[opts for _, opts in clustering_hyperparam_options]
)
for combo in tqdm(all_extraction_combos, desc="Sweeping"):
    (
        state_split_classifier,
        minimum_transition_count,
        kmeans_num_clusters,
    ) = combo
    hyperparams = ExtractionHyperparameters(
        model_shortname=best_run.name,  # type:ignore
        clustering_method="kmeans",
        state_split_classifier=state_split_classifier,
        n_components=None,
        minimum_transition_count=minimum_transition_count,
        kmeans_num_clusters=kmeans_num_clusters,
    )
    wandb.init(
        entity="lecs-general",
        project="fst-distillation.clustering.extraction",
        config={
            **asdict(hyperparams),
            "rnn": {
                **dict(best_run.config),
                "eval.loss": model_eval_loss,
                "name": best_run.name,
            },
            "identifier": paths["identifier"],
        },
    )
    wandb.run.summary["training_run"] = best_run.url  # type:ignore
    try:
        results, fst = extract_fst(
            hyperparams=hyperparams,
            paths=paths,
        )
        wandb.log(results)
        fst.save
    except Exception as e:
        logger.warning(e)
    wandb.finish()
