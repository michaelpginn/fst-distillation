"""Runs the full experiment for a given language"""

import argparse
import itertools
import logging
from dataclasses import asdict
from pathlib import Path

import wandb
from exp2_clustering.steps.align_data import run_alignment
from exp2_clustering.steps.extract_fst import ExtractionHyperparameters, extract_fst
from exp2_clustering.steps.train_rnn import train_rnn
from exp2_clustering.util import find_data_file

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("language", help="Isocode for the language")
args = parser.parse_args()

# Check if we've already aligned the language. If not, run alignment on raw files.
raw_train_path = find_data_file(f"{args.language}.trn")
raw_dev_path = find_data_file(f"{args.language}.dev")
raw_test_path = find_data_file(f"{args.language}.tst")

aligned_data_folder = Path(__file__).parent / "aligned_data"
train_path = aligned_data_folder / f"{args.language}.trn.aligned"
dev_path = aligned_data_folder / f"{args.language}.dev.aligned"

if (not train_path.exists()) or (not dev_path.exists()):
    logger.info("Couldn't find aligned data, running alignment!")
    run_alignment([raw_train_path, raw_dev_path])
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
        language=args.language,
        batch_size=2048,
        epochs=150,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    for clustering_method in ["kmeans", "dbscan"]:
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
                    pca_components=pca_components,
                    minimum_transition_count=minimum_transition_count,
                    kmeans_num_clusters=kmeans_num_clusters,
                )
                wandb.init(
                    entity="lecs-general",
                    project="fst-distillation.exp2.extraction",
                    config={
                        **asdict(hyperparams),
                        **rnn_config,
                        "language": args.language,
                        "model_name": run_name,
                    },
                )
                eval_results, test_results = extract_fst(
                    hyperparams=hyperparams,
                    aligned_train_path=train_path,
                    eval_path=raw_dev_path,
                    test_path=raw_test_path,
                    model_id=args.model_id,
                )
                wandb.log({"eval": eval_results, "test": test_results})
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
                    pca_components=pca_components,
                    minimum_transition_count=minimum_transition_count,
                    eps=eps,
                    min_samples=min_samples,
                )
                wandb.init(
                    entity="lecs-general",
                    project="fst-distillation.exp2.extraction",
                    config={
                        **asdict(hyperparams),
                        **rnn_config,
                        "language": args.language,
                        "model_name": run_name,
                    },
                )
                eval_results, test_results = extract_fst(
                    hyperparams=hyperparams,
                    aligned_train_path=train_path,
                    eval_path=raw_dev_path,
                    test_path=raw_test_path,
                    model_id=args.model_id,
                )
                wandb.log({"eval": eval_results, "test": test_results})
                wandb.finish()
        else:
            raise ValueError()
