"""Runs the extraction script as a hyperparameter sweep"""

import argparse
from dataclasses import asdict

import wandb
from exp2_clustering.extract_fst import ExtractionHyperparameters, extract_fst

parser = argparse.ArgumentParser()
parser.add_argument("model_id", help="WandB ID for the training run")
parser.add_argument("--language", default="swe", help="Isocode for the language")
args = parser.parse_args()

num_clusters_options = [
    500,
    1000,
    1500,
]

top_k_options = range(2, 7)

for num_clusters in num_clusters_options:
    for top_k in top_k_options:
        params = ExtractionHyperparameters(
            num_initial_clusters=num_clusters,
            pca_components=64,  # model dim, aka no pca
            transitions_top_k=top_k,
            transitions_top_p=None,
            transitions_min_n=None,
        )
        wandb.init(
            entity="lecs-general",
            project="fst-distillation.exp2.extraction-sweep",
            config={**asdict(params), "model_id": args.model_id},
            group=args.language,
        )
        results = extract_fst(
            hyperparams=params,
            language=args.language,
            model_id=args.model_id,
        )
        wandb.log(results)
        wandb.finish()
