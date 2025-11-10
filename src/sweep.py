"""Runs the full experiment for a given language"""

import ast
import logging
from pprint import pformat

import wandb
from src.data.aligned.example import load_examples_from_file
from src.paths import create_arg_parser, create_paths_from_args
from src.train_alignment_predictor import predict_full_domain, train_alignment_predictor

from .extract_fst import ExtractionHyperparameters, extract_fst
from .train_rnn import train_rnn

logger = logging.getLogger(__name__)

parser = create_arg_parser()
parser.add_argument("--objective", choices=["lm", "classification"], required=True)
parser.add_argument("--override-alignment", action="store_true")
args = parser.parse_args()
paths = create_paths_from_args(args)

# =========================================
# 1. CRP ALIGNMENT
# =========================================
if (
    args.override_alignment
    or (not paths["train_aligned"].exists())
    or (not paths["eval_aligned"].exists())
):
    from .run_alignment import run_alignment

    logger.info("Couldn't find aligned data, running alignment!")
    run_alignment(paths, iterations=100)
train_size = len(load_examples_from_file(paths["train_aligned"]))
max_batch_size = train_size // 10

# =========================================
# 2. ALIGNMENT PREDICTOR TRAINING
# =========================================
if args.override_alignment or not paths["full_domain_aligned"].exists():

    def single_run_train_alignment():
        with wandb.init(
            entity="lecs-general",
            project="fst-distillation.clustering.alignment_prediction",
        ) as run:
            logger.info(f"Training with params: {pformat(run.config)}")
            train_alignment_predictor(
                paths,
                run.config["batch_size"],
                epochs=run.config["epochs"],
                learning_rate=run.config["learning_rate"],
                weight_decay=run.config["weight_decay"],
                d_model=run.config["d_model"],
                num_layers=run.config["num_layers"],
                num_heads=8,
                dropout=run.config["dropout"],
                wandb_run=run,
            )

    sweep_configuration = {
        "name": paths["identifier"],
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "validation.loss"},
        "parameters": {
            "learning_rate": {"values": [2e-4, 1e-3, 2e-3]},
            "weight_decay": {"values": [0.1, 0.2, 0.3]},
            "d_model": {"values": [16, 32, 64]},
            "num_layers": {"values": [2, 3, 4]},
            "dropout": {"values": [0.1, 0.2, 0.3]},
            "batch_size": {
                "values": [
                    b for b in [2, 4, 8, 16, 32, 64, 128] if b <= max_batch_size
                ][-3:]
            },
            "epochs": {"values": [200, 400, 600, 800]},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 100,  # minimum epochs before possible pruning
            "max_iter": 800,  # maximum epochs (full resource)
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity="lecs-general",
        project="fst-distillation.clustering.alignment_prediction",
    )
    wandb.agent(sweep_id, function=single_run_train_alignment, count=50)
    sweep = wandb.Api().sweep(
        f"lecs-general/fst-distillation.clustering.alignment_prediction/sweeps/{sweep_id}"
    )
    best_run = sweep.best_run()
    if isinstance(best_run.summary_metrics, str):
        alignment_pred_loss = ast.literal_eval(best_run.summary_metrics)["validation"][
            "loss"
        ]
    else:
        alignment_pred_loss = best_run.summary_metrics["validation"]["loss"]
    predict_full_domain(paths, best_run.name, best_run.config["batch_size"])


# =========================================
# 3. RNN TRAINING
# =========================================
rnn_project_name = f"fst-distillation.clustering.rnn_{args.objective}"


def single_run_train_rnn():
    with wandb.init(entity="lecs-general", project=rnn_project_name) as run:
        logger.info(f"Training with params: {pformat(run.config)}")
        train_rnn(
            paths=paths,
            objective=args.objective,
            batch_size=run.config["batch_size"],
            epochs=run.config["epochs"],
            d_model=run.config["d_model"],
            num_layers=1,
            dropout=run.config["dropout"],
            learning_rate=run.config["learning_rate"],
            use_many_to_many_transitions=False,
            activation="tanh",
            spectral_norm_weight=0.1,
            wandb_run=run,
        )


sweep_configuration = {
    "name": paths["identifier"],
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "validation.loss"},
    "parameters": {
        "d_model": {"values": [16, 32, 64]},
        "dropout": {"values": [0, 0.1, 0.3]},
        "learning_rate": {"values": [2e-4, 1e-3, 2e-3, 1e-2]},
        "batch_size": {
            "values": [b for b in [2, 4, 8, 16, 32, 64, 128] if b <= max_batch_size][
                -3:
            ]
        },
        "epochs": {"values": [200, 600, 1000]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 75,  # minimum epochs before possible pruning
        "max_iter": 1000,  # maximum epochs (full resource)
    },
}
sweep_id = wandb.sweep(
    sweep=sweep_configuration, entity="lecs-general", project=rnn_project_name
)
wandb.agent(sweep_id, function=single_run_train_rnn, count=50)
sweep = wandb.Api().sweep(f"lecs-general/{rnn_project_name}/sweeps/{sweep_id}")
best_run = sweep.best_run()
if isinstance(best_run.summary_metrics, str):
    best_run_loss = ast.literal_eval(best_run.summary_metrics)["validation"]["loss"]
else:
    best_run_loss = best_run.summary_metrics["validation"]["loss"]

# =========================================
# 4. FST EXTRACTION
# =========================================


def single_run_extract_fst():
    with wandb.init(
        entity="lecs-general",
        project="fst-distillation.clustering.extraction",
        config={
            "rnn": {
                # **dict(best_run.config),
                "eval.loss": best_run_loss,
                "name": best_run.name,
            },
            "identifier": paths["identifier"],
        },
    ) as run:
        hyperparams = ExtractionHyperparameters(
            model_shortname=best_run.name,  # type:ignore
            clustering_method="kmeans",
            state_split_classifier=run.config["state_split_classifier"],
            n_components=None,
            minimum_transition_count=run.config["minimum_transition_count"],
            kmeans_num_clusters=run.config["kmeans_num_clusters"],
        )
        results, _ = extract_fst(
            hyperparams=hyperparams,
            paths=paths,
        )
        run.log(results)
        run.summary["training_run"] = best_run.url


sweep_configuration = {
    "name": paths["identifier"],
    "method": "grid",
    "metric": {"goal": "maximize", "name": "eval.f1"},
    "parameters": {
        "state_split_classifier": {"values": ["svm", "logistic"]},
        "minimum_transition_count": {"values": [None, 10, 25, 50]},
        "kmeans_num_clusters": {"values": [50, 100, 250, 500, 1000, 1500]},
    },
}

fst_sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    entity="lecs-general",
    project="fst-distillation.clustering.extraction",
)
wandb.agent(fst_sweep_id, function=single_run_extract_fst)
sweep = wandb.Api().sweep(
    f"lecs-general/fst-distillation.clustering.extraction/sweeps/{fst_sweep_id}"
)
best_run = sweep.best_run()
print(f"Best run: {best_run.url}")
