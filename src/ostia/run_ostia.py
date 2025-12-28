import os
import pprint
import sys
from logging import getLogger
from typing import Literal

from pyfoma.algorithms import ostia
from pyfoma.fst import FST

import wandb

from ..data.unaligned.example import load_examples_from_file
from ..evaluate import evaluate_all
from ..paths import Paths, create_arg_parser, create_paths_from_args

logger = getLogger(__name__)
sys.setrecursionlimit(10000)


def run_ostia(paths: Paths, order: Literal["lex", "dd"]):
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    wandb.init(
        entity="lecs-general",
        project="fst-distillation.ostia",
        config={**locals(), "slurm_job_id": slurm_job_id},
    )

    train_examples = load_examples_from_file(
        paths["train"], paths["has_features"], paths["output_split_into_chars"]
    )
    eval_examples = load_examples_from_file(
        paths["eval"], paths["has_features"], paths["output_split_into_chars"]
    )
    test_examples = load_examples_from_file(
        paths["test"], paths["has_features"], paths["output_split_into_chars"]
    )

    samples: list[tuple[str | list[str], str | list[str]]] = []
    for ex in train_examples:
        input_string = ["<sep>"] + list(ex.input_string)
        if ex.features is not None:
            input_string = [f"[{f}]" for f in ex.features] + input_string
        assert ex.output_string is not None
        samples.append((input_string, ex.output_string))

    fst = ostia(samples, order, use_cython=True)
    fst.save(f"{paths['identifier']}.fst")
    logger.info(f"Created FST with {len(fst.states)} states")
    fst = FST.re(".* '<sink>':'#'") @ fst

    # train_metrics = evaluate_all(fst, train_examples, output_raw_string=True)
    # logger.info(f"Train metrics: {pprint.pformat(train_metrics)}")
    eval_metrics = evaluate_all(fst, eval_examples, output_raw_string=True, log=True)
    logger.info(f"Eval metrics: {pprint.pformat(eval_metrics)}")
    test_metrics = evaluate_all(fst, test_examples, output_raw_string=True)
    logger.info(f"Test metrics: {pprint.pformat(test_metrics)}")

    wandb.log(
        {
            "num_states": len(fst.states),
            "train": train_metrics,
            "eval": eval_metrics,
            "test": test_metrics,
        }
    )


if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument(
        "--order",
        choices=["lex", "dd"],
        required=True,
        help="What order to merge states in? 'lex' is original OSTIA, 'dd' is DD-OSTIA",
    )
    args = parser.parse_args()
    paths = create_paths_from_args(args)
    run_ostia(paths, args.order)
