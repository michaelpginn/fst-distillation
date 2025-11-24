"""Runs the full experiment for a given language"""

import ast
import logging
import math
import multiprocessing
import os
from pprint import pformat

import numpy as np

import wandb
from src.data.aligned.example import load_examples_from_file
from src.paths import Paths, create_arg_parser, create_paths_from_args
from src.train_alignment_predictor import predict_full_domain, train_alignment_predictor

from .extract_fst import ExtractionHyperparameters, compute_activations, extract_fst
from .train_rnn import train_rnn

logger = logging.getLogger(__name__)

WANDB_DIRECTORY = "/scratch/alpine/migi8081/fst-distillation/wandb/"
os.environ["WANDB_DIR"] = os.path.abspath(WANDB_DIRECTORY)


def main():
    parser = create_arg_parser()
    parser.add_argument(
        "--objective", choices=["lm", "classification", "transduction"], required=True
    )
    parser.add_argument("--override-alignment", action="store_true")
    args = parser.parse_args()
    paths = create_paths_from_args(args)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

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
    train_examples = load_examples_from_file(paths["train_aligned"])
    train_size = len(train_examples)
    max_batch_size = train_size // 5

    # =========================================
    # 2. ALIGNMENT PREDICTOR TRAINING
    # =========================================
    if args.override_alignment or not paths["full_domain_aligned"].exists():
        logger.info("Running alignment predictor sweep")

        def single_run_train_alignment():
            with wandb.init(
                entity="lecs-general",
                project="fst-distillation.alignment_prediction",
                dir=WANDB_DIRECTORY,
            ) as run:
                run.config.update({"slurm_job_id": slurm_job_id})
                logger.info(f"Training with params: {pformat(run.config)}")
                train_alignment_predictor(
                    paths,
                    run.config["batch_size"],
                    epochs=run.config["epochs"],
                    learning_rate=run.config["learning_rate"],
                    weight_decay=run.config["weight_decay"],
                    d_model=run.config["d_model"],
                    num_layers=run.config["num_layers"],
                    num_heads=2,
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
                        b for b in [2, 4, 8, 16, 32, 64, 96] if b <= max_batch_size
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
            project="fst-distillation.alignment_prediction",
        )
        wandb.agent(sweep_id, function=single_run_train_alignment, count=100)
        sweep = wandb.Api().sweep(
            f"lecs-general/fst-distillation.alignment_prediction/sweeps/{sweep_id}"
        )
        best_run = sweep.best_run()
        predict_full_domain(paths, best_run.name, best_run.config["batch_size"])

    else:
        # Load the best run
        best_run = None
        for sweep in (
            wandb.Api()
            .project(
                name="fst-distillation.alignment_prediction",
                entity="lecs-general",
            )
            .sweeps()
        ):
            if sweep.name == paths["identifier"]:
                logger.info(
                    f"Found existing alignment predictor sweep {paths['identifier']}"
                )
                best_run = sweep.best_run()
                break

    assert best_run is not None
    if isinstance(best_run.summary_metrics, str):
        alignment_pred_loss = ast.literal_eval(best_run.summary_metrics)["validation"][
            "loss"
        ]
    else:
        alignment_pred_loss = best_run.summary_metrics["validation"]["loss"]

    # =========================================
    # 3. RNN TRAINING
    # =========================================
    rnn_project_name = f"fst-distillation.rnn_{args.objective}"

    # Load the best run
    best_run = None
    try:
        for sweep in (
            wandb.Api().project(name=rnn_project_name, entity="lecs-general").sweeps()
        ):
            if sweep.name == paths["identifier"]:
                if (
                    any(r.state != "finished" for r in sweep.runs)
                    or len(sweep.runs) < 100
                ):
                    raise ValueError(
                        f"Found sweep for {paths['identifier']}, but sweep is not finished or crashed! Delete and try again."
                    )
                logger.info(
                    f"Found existing finished sweep {paths['identifier']}, reusing best run instead of running."
                )
                best_run = sweep.best_run()
                break
    except ValueError:
        logger.warning(f"Didn't find project {rnn_project_name}, creating new!")

    # If not, run the sweep
    if best_run is None:
        logger.info(f"No sweep was found for {paths['identifier']}. Running now.")

        def single_run_train_rnn():
            with wandb.init(
                entity="lecs-general", project=rnn_project_name, dir=WANDB_DIRECTORY
            ) as run:
                run.config.update({"slurm_job_id": slurm_job_id})
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
                "d_model": {"values": [16, 32, 64, 128]},
                "dropout": {"values": [0, 0.1, 0.3]},
                "learning_rate": {"values": [2e-4, 1e-3, 2e-3, 1e-2]},
                "batch_size": {
                    "values": [
                        b
                        for b in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                        if b <= max_batch_size
                    ][-4:]
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
        wandb.agent(sweep_id, function=single_run_train_rnn, count=100)
        sweep = wandb.Api().sweep(f"lecs-general/{rnn_project_name}/sweeps/{sweep_id}")
        best_run = sweep.best_run()
    if isinstance(best_run.summary_metrics, str):
        best_run_loss = ast.literal_eval(best_run.summary_metrics)["validation"]["loss"]
    else:
        best_run_loss = best_run.summary_metrics["validation"]["loss"]
    assert best_run is not None

    # =========================================
    # 4. FST EXTRACTION
    # =========================================

    # Pre-compute activations since they shouldn't change across the sweep
    # the hparams don't matter except for the model name
    activations, transition_labels = compute_activations(
        ExtractionHyperparameters(
            model_shortname=best_run.name,  # type:ignore
            clustering_method="kmeans",
            kmeans_num_clusters=1,
            n_components=None,
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
        },
    }
    fst_sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity="lecs-general",
        project="fst-distillation.extraction",
    )
    cpus = 10  # TODO: Don't hardcode this
    procs = []
    logger.info(f"Running extraction sweep in parallel with {cpus} agents")
    for _ in range(cpus):
        p = multiprocessing.Process(
            target=start_extract_fst_agent,
            kwargs={
                "sweep_id": fst_sweep_id,
                "count": math.ceil(500 / cpus),
                "paths": paths,
                "precomputed_activations": (activations, transition_labels),
                "alignment_pred_loss": alignment_pred_loss,
                "best_run_fields": (best_run_loss, best_run.name, best_run.url),
                "slurm_job_id": slurm_job_id,
            },
        )
        p.start()
        procs.append(p)

    # Block until all finished
    for p in procs:
        p.join()

    sweep = wandb.Api().sweep(
        f"lecs-general/fst-distillation.extraction/sweeps/{fst_sweep_id}"
    )
    best_run = sweep.best_run()
    print(f"Best run: {best_run.url}")


# Has to be module-level since we use it with multiproc
def start_extract_fst_agent(
    sweep_id: str,
    count: int,
    paths: Paths,
    precomputed_activations: tuple[np.ndarray, list[list[str]]],
    alignment_pred_loss,
    best_run_fields: tuple,
    slurm_job_id,
):
    best_run_loss, best_run_name, best_run_url = best_run_fields

    def _run_extraction():
        with wandb.init(
            entity="lecs-general",
            project="fst-distillation.extraction",
            config={
                "rnn": {
                    "eval.loss": best_run_loss,
                    "name": best_run_name,  # type:ignore
                },
                "alignment_predictor": {
                    "eval.loss": alignment_pred_loss,
                },
                "identifier": paths["identifier"],
            },
            dir=WANDB_DIRECTORY,
        ) as run:
            run.config.update({"slurm_job_id": slurm_job_id})
            hyperparams = ExtractionHyperparameters(
                model_shortname=best_run_name,  # type:ignore
                clustering_method="kmeans",
                state_split_classifier=run.config["state_split_classifier"],
                n_components=None,
                minimum_transition_count=run.config["minimum_transition_count"],
                kmeans_num_clusters=run.config["kmeans_num_clusters"],
            )
            results, _ = extract_fst(
                hparams=hyperparams,
                paths=paths,
                precomputed_activations=precomputed_activations,
            )
            run.log(results)
            run.summary["training_run"] = best_run_url  # type:ignore

    wandb.agent(sweep_id, function=_run_extraction, count=count)


if __name__ == "__main__":
    main()
