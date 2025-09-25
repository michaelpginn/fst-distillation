"""Runs the full experiment for a given language"""

import argparse
import itertools
import logging
from dataclasses import asdict

import wandb

from ..shared import add_task_parser, get_data_files
from .steps.extract_fst import ExtractionHyperparameters, extract_fst
from .steps.train_rnn import train_rnn

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
add_task_parser(parser)
args = parser.parse_args()
data_files = get_data_files(args)

# Check if we've already aligned the language. If not, run alignment on raw files.
if (not data_files["train_aligned"].exists()) or (
    not data_files["eval_aligned"].exists()
):
    from .steps.align_data import run_alignment

    logger.info("Couldn't find aligned data, running alignment!")
    run_alignment(
        [data_files["train"], data_files["eval"]],
        has_features=data_files["has_features"],
    )
    # The aligned paths should exist now, we can just use the paths from before

# Train model
training_hyperparam_options: list[tuple[str, list]] = [
    ("num_layers", [1, 2, 4]),
    ("d_model", [32, 64, 128]),
    ("dropout", [0, 0.1, 0.5]),
    ("learning_rate", [1e-4, 2e-4, 1e-3]),
]
all_combos = itertools.product(*[opts for _, opts in training_hyperparam_options])
for combo in all_combos:
    logger.info(f"Training with params: {combo}")
    num_layers, d_model, dropout, learning_rate = combo
    rnn_config = {
        "rnn.num_layers": num_layers,
        "rnn.d_model": d_model,
        "rnn.dropout": dropout,
        "rnn.learning_rate": learning_rate,
    }
    run_name = train_rnn(
        data_files=data_files,
        batch_size=2048,
        epochs=200,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    assert run_name is not None
    for clustering_method in ["kmeans"]:
        clustering_hyperparam_options: list[tuple[str, list]] = [
            ("state_split_classifier", ["svm", "logistic"]),
            ("pca_components", [None, 32, 16]),
            ("minimum_transition_count", [None, 50, 100, 1000]),
        ]
        if clustering_method == "kmeans":
            clustering_hyperparam_options.extend(
                [("kmeans_num_clusters", [500, 1000, 1500])]
            )
            all_extraction_combos = itertools.product(
                *[opts for _, opts in clustering_hyperparam_options]
            )
            for combo in all_extraction_combos:
                (
                    state_split_classifier,
                    pca_components,
                    minimum_transition_count,
                    kmeans_num_clusters,
                ) = combo
                hyperparams = ExtractionHyperparameters(
                    clustering_method="kmeans",
                    state_split_classifier=state_split_classifier,
                    n_components=pca_components,
                    minimum_transition_count=minimum_transition_count,
                    kmeans_num_clusters=kmeans_num_clusters,
                )
                wandb.init(
                    entity="lecs-general",
                    project="fst-distillation.clustering.extraction",
                    config={
                        **asdict(hyperparams),
                        **rnn_config,
                        "language": args.language,
                        "model_name": run_name,
                    },
                )
                results = extract_fst(
                    hyperparams=hyperparams,
                    data_files=data_files,
                    model_id=run_name,
                )
                wandb.log(results)
                wandb.finish()
        elif clustering_method == "dbscan":
            clustering_hyperparam_options.extend(
                [("eps", [1, 5, 10]), ("min_samples", [100, 500, 1000])]
            )
            all_extraction_combos = itertools.product(
                *[opts for _, opts in clustering_hyperparam_options]
            )
            for combo in all_extraction_combos:
                (
                    state_split_classifier,
                    pca_components,
                    minimum_transition_count,
                    eps,
                    min_samples,
                ) = combo
                hyperparams = ExtractionHyperparameters(
                    clustering_method="dbscan",
                    state_split_classifier=state_split_classifier,
                    n_components=pca_components,
                    minimum_transition_count=minimum_transition_count,
                    eps=eps,
                    min_samples=min_samples,
                )
                wandb.init(
                    entity="lecs-general",
                    project="fst-distillation.clustering.extraction",
                    config={
                        **asdict(hyperparams),
                        **rnn_config,
                        "language": args.language,
                        "model_name": run_name,
                    },
                )
                results = extract_fst(
                    hyperparams=hyperparams,
                    data_files=data_files,
                    model_id=run_name,
                )
                wandb.log(results)
                wandb.finish()
        else:
            raise ValueError()
