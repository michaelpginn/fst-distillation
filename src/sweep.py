"""Runs the full experiment for a given language"""

import ast
import logging
import os
from pathlib import Path
from pprint import pformat

import numpy as np

import wandb
from src.data.aligned.example import load_examples_from_file
from src.extract_bimachine import extract_bimachine
from src.paths import create_arg_parser, create_paths_from_args
from src.train_alignment_predictor import predict_full_domain, train_alignment_predictor

from .extract_fst import ExtractionHyperparameters, extract_fst
from .train_rnn import train_rnn

logger = logging.getLogger(__name__)


def main():
    parser = create_arg_parser()
    parser.add_argument(
        "--objective",
        choices=["lm", "classification", "transduction", "bimachine"],
        required=True,
    )
    parser.add_argument("--override-alignment", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["sample", "search"],
        help="If sample, will train an alignment predictor to sample from the domain. If search, will use n-grams with BFS to collect activations",
        required=True,
    )
    parser.add_argument("--wandb-dir")
    args = parser.parse_args()
    paths = create_paths_from_args(args)
    paths_strs = {k: str(v) if isinstance(v, Path) else v for k, v in paths.items()}
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if args.wandb_dir:
        os.environ["WANDB_DIR"] = os.path.abspath(args.wandb_dir)

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
    train_examples, _ = load_examples_from_file(
        paths["train_aligned"], paths["merge_outputs"]
    )
    train_size = len(train_examples)
    max_batch_size = train_size // 5
    if train_size > 5000:
        num_neural_runs = 25
        num_extract_runs = 50
    else:
        num_neural_runs = 50
        num_extract_runs = 300

    # =========================================
    # 2. ALIGNMENT PREDICTOR TRAINING
    # =========================================
    if args.mode == "search" or paths["merge_outputs"] != "none":
        alignment_pred_loss = None
    else:
        best_run = None
        if not args.override_alignment and paths["full_domain_aligned"].exists():
            # Load the best run
            for sweep in (
                wandb.Api()
                .project(
                    name="fst-distillation.alignment_prediction.v2",
                    entity="lecs-general",
                )
                .sweeps()
            ):
                if sweep.name == paths["identifier"]:
                    logger.info(
                        f"Found existing alignment predictor sweep {paths['identifier']}"
                    )
                    # If we didn't use identical paths, do a new run
                    if sweep.best_run().config["paths"] == paths_strs:  # type:ignore
                        best_run = sweep.best_run()
                        break

        if best_run is None:
            logger.info("Running alignment predictor sweep")

            def single_run_train_alignment():
                with wandb.init(
                    entity="lecs-general",
                    project="fst-distillation.alignment_prediction.v2",
                    dir=args.wandb_dir,
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
                project="fst-distillation.alignment_prediction.v2",
            )
            wandb.agent(
                sweep_id, function=single_run_train_alignment, count=num_neural_runs
            )
            sweep = wandb.Api().sweep(
                f"lecs-general/fst-distillation.alignment_prediction.v2/sweeps/{sweep_id}"
            )
            best_run = sweep.best_run()
            predict_full_domain(paths, best_run.name, best_run.config["batch_size"])

        assert best_run is not None
        if isinstance(best_run.summary_metrics, str):
            alignment_pred_loss = ast.literal_eval(best_run.summary_metrics)[
                "validation"
            ]["loss"]
        else:
            alignment_pred_loss = best_run.summary_metrics["validation"]["loss"]

    # =========================================
    # 3. RNN TRAINING
    # =========================================
    rnn_project_name = f"fst-distillation.rnn_{args.objective}.v2"

    # Load the best run
    best_run = None
    existing_sweep = None
    try:
        for sweep in (
            wandb.Api().project(name=rnn_project_name, entity="lecs-general").sweeps()
        ):
            if sweep.name == paths["identifier"]:
                if (
                    len([r for r in sweep.runs if r.state == "finished"])
                    < num_neural_runs
                ):
                    existing_sweep = sweep
                    raise ValueError(
                        f"Found sweep for {paths['identifier']}, but sweep is not finished or crashed! Resuming..."
                    )
                if sweep.best_run().config["paths"] == paths_strs:  # type:ignore
                    logger.info(
                        f"Found existing finished sweep {paths['identifier']}, reusing best run instead of running."
                    )
                    best_run = sweep.best_run()
                    break
    except ValueError as e:
        logger.warning(e)

    # If not, run the sweep
    if best_run is None:
        logger.info(
            f"No (finished) sweep was found for {paths['identifier']}. Running now."
        )

        def single_run_train_rnn():
            with wandb.init(
                entity="lecs-general", project=rnn_project_name, dir=args.wandb_dir
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
                    merge_outputs=paths["merge_outputs"],
                    activation="tanh",
                    spectral_norm_weight=0.1,
                    label_smoothing=0.1,
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
                        for b in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
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

        if existing_sweep is None:
            logger.info("Creating new sweep")
            sweep_id = wandb.sweep(
                sweep=sweep_configuration,
                entity="lecs-general",
                project=rnn_project_name,
            )
            remaining_runs = num_neural_runs
        else:
            logger.info(f"Reusing sweep {existing_sweep.id}")
            sweep_id = f"lecs-general/{rnn_project_name}/{existing_sweep.id}"
            remaining_runs = num_neural_runs - len(
                [r for r in existing_sweep.runs if r.state == "finished"]
            )
        wandb.agent(sweep_id, function=single_run_train_rnn, count=remaining_runs)
        sweep = wandb.Api().sweep(f"lecs-general/{rnn_project_name}/sweeps/{sweep_id}")
        best_run = sweep.best_run()

        # Push model
        with wandb.init(id=best_run.id, resume="must"):
            wandb.log_artifact(
                paths["models_folder"] / f"{best_run.name}/model.pt",
                name="checkpoint",
                type="model",
            )
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
    if args.objective == "bimachine":
        from .extract_bimachine import compute_activations

        activations, transition_labels = compute_activations(
            ExtractionHyperparameters(
                model_shortname=best_run.name,  # type:ignore
                clustering_method="kmeans",
                kmeans_num_clusters=1,
                n_components=None,
                full_domain_mode=args.mode,
                full_domain_search_n=3,
            ),
            paths,
        )
        max_clusters = min(
            len(np.unique(activations["forward"], axis=0)),
            len(np.unique(activations["backward"], axis=0)),
        )
    else:
        from .extract_fst import compute_activations

        activations, transition_labels = compute_activations(
            ExtractionHyperparameters(
                model_shortname=best_run.name,  # type:ignore
                clustering_method="kmeans",
                kmeans_num_clusters=1,
                n_components=None,
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
                if args.mode == "search"
                else {}
            ),
        },
    }
    fst_sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity="lecs-general",
        project="fst-distillation.extraction.v2",
    )

    def _run_extraction():
        with wandb.init(
            entity="lecs-general",
            project="fst-distillation.extraction.v2",
            config={
                "rnn": {
                    "eval.loss": best_run_loss,
                    "name": best_run.name,  # type:ignore
                },
                "alignment_predictor": {
                    "eval.loss": alignment_pred_loss,
                },
                "identifier": paths["identifier"],
            },
            dir=args.wandb_dir,
        ) as run:
            run.config.update({"slurm_job_id": slurm_job_id})
            full_domain_search_n = run.config.get("full_domain_search_n", 2)
            hyperparams = ExtractionHyperparameters(
                model_shortname=best_run.name,  # type:ignore
                clustering_method="kmeans",
                use_faiss=True,
                state_split_classifier=run.config["state_split_classifier"],
                n_components=None,
                minimum_transition_count=run.config["minimum_transition_count"],
                kmeans_num_clusters=run.config["kmeans_num_clusters"],
                full_domain_mode=args.mode,
                full_domain_search_n=full_domain_search_n,
            )
            if args.objective == "bimachine":
                results, _ = extract_bimachine(
                    hparams=hyperparams,
                    paths=paths,
                    precomputed_activations=(activations, transition_labels)
                    if full_domain_search_n == 3
                    else None,  # type:ignore
                )
            else:
                results, _ = extract_fst(
                    hparams=hyperparams,
                    paths=paths,
                    precomputed_activations=(activations, transition_labels)  # type:ignore
                    if full_domain_search_n == 3
                    else None,
                )
            run.log(results)
            run.summary["training_run"] = best_run.url  # type:ignore

    wandb.agent(fst_sweep_id, function=_run_extraction, count=num_extract_runs)
    sweep = wandb.Api().sweep(
        f"lecs-general/fst-distillation.extraction.v2/sweeps/{fst_sweep_id}"
    )
    best_run = sweep.best_run()
    print(f"Best run: {best_run.url}")


if __name__ == "__main__":
    main()
