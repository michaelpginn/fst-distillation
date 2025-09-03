"""Runs the full experiment for a given language"""

import argparse
import itertools
import logging
from pathlib import Path

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
    ("num_layers", [1, 2, 4, 8]),
    ("d_model", [16, 32, 64, 128, 256]),
    ("dropout", [0, 0.1, 0.2, 0.5, 0.75, 0.9]),
    ("learning_rate", [2e-5, 1e-4, 1e-3, 1e-2]),
]
all_combos = itertools.product(*[opts for _, opts in training_hyperparam_options])
for combo in all_combos:
    logger.info(f"Training with params: {combo}")
    num_layers, d_model, dropout, learning_rate = combo
    run_name = train_rnn(
        language=args.language,
        batch_size=2048,
        epochs=150,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    for num_clusters in [100, 500, 750, 1000, 1250]:
        params = ExtractionHyperparameters(
            num_initial_clusters=num_clusters,
            pca_components=64,  # model dim, aka no pca
            transitions_top_k=1,
            transitions_top_p=None,
            transitions_min_n=None,
        )
        eval_result, test_results = extract_fst(
            hyperparams=params,
            aligned_train_path=train_path,
            eval_path=raw_dev_path,
            test_path=raw_test_path,
            model_id=args.model_id,
        )
        # TODO: Log only the best F1 score
        logger.info(f"Extracted with {num_clusters=}, eval f1 = {eval_result['f1']}")
