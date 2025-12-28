import logging

import numpy as np

import wandb
from src.extract_fst import ExtractionHyperparameters, compute_activations, extract_fst
from src.paths import create_arg_parser, create_paths_from_args

parser = create_arg_parser()
parser.add_argument("ablation_name")
parser.add_argument("model_id")
parser.add_argument(
    "--mode",
    choices=["sample", "search"],
    help="If sample, will train an alignment predictor to sample from the domain. If search, will use n-grams with BFS to collect activations",
    required=True,
)
parser.add_argument("--no-full-domain", action="store_true")
args = parser.parse_args()
paths = create_paths_from_args(args)

num_extract_runs = 100
logger = logging.getLogger(__name__)

activations, transition_labels = compute_activations(
    ExtractionHyperparameters(
        model_shortname=args.model_id,  # type:ignore
        clustering_method="kmeans",
        kmeans_num_clusters=1,
        n_components=None,
        full_domain=not args.no_full_domain,
        full_domain_mode=args.mode,
        full_domain_search_n=3,
    ),
    paths,
)
max_clusters = len(np.unique(activations, axis=0))

sweep_configuration = {
    "name": paths["identifier"],
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "eval.f1"},
    "parameters": {
        "state_split_classifier": {"values": ["svm", "logistic"]},
        "minimum_transition_count": {
            "values": [None, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]
        },
        "kmeans_num_clusters": {"min": 50, "max": max_clusters},
        **(
            {"full_domain_search_n": {"values": [3, 4]}}
            if args.mode == "search" and not args.no_full_domain
            else {}
        ),
    },
}


def _run_extraction():
    with wandb.init(
        entity="lecs-general",
        project=f"fst-distillation.extraction.v2.{args.ablation_name}",
        config={
            "rnn": {
                # "eval.loss": best_run_loss,
                "name": args.model_id,  # type:ignore
            },
            "identifier": paths["identifier"],
        },
        # dir=args.wandb_dir,
    ) as run:
        full_domain_search_n = run.config.get("full_domain_search_n", 3)
        hyperparams = ExtractionHyperparameters(
            model_shortname=args.model_id,  # type:ignore
            clustering_method="kmeans",
            use_faiss=False,
            state_split_classifier=run.config["state_split_classifier"],
            n_components=None,
            minimum_transition_count=run.config["minimum_transition_count"],
            kmeans_num_clusters=run.config["kmeans_num_clusters"],
            full_domain=not args.no_full_domain,
            full_domain_mode=args.mode,
            full_domain_search_n=full_domain_search_n,
        )
        results, _ = extract_fst(
            hparams=hyperparams,
            paths=paths,
            precomputed_activations=(activations, transition_labels)  # type:ignore
            if full_domain_search_n == 3
            else None,
        )
        run.log(results)
        # run.summary["training_run"] = best_run.url  # type:ignore


logger.info("Creating new sweep")
fst_sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    entity="lecs-general",
    project=f"fst-distillation.extraction.v2.{args.ablation_name}",
)

wandb.agent(fst_sweep_id, function=_run_extraction, count=num_extract_runs)
sweep = wandb.Api().sweep(
    f"lecs-general/fst-distillation.extraction.v2.{args.ablation_name}/sweeps/{fst_sweep_id}"
)
best_run = sweep.best_run()
print(f"Best run: {best_run.url}")
