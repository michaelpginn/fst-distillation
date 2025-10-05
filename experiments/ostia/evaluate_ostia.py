import argparse
import pprint
from logging import getLogger

from pyfoma.fst import FST

from experiments.ostia.ostia import ostia
from experiments.shared import add_task_parser, get_data_files
from src.data.unaligned.example import load_examples_from_file
from src.evaluate import evaluate_all

logger = getLogger(__name__)

parser = argparse.ArgumentParser()
add_task_parser(parser)
args = parser.parse_args()
data_files = get_data_files(args)

train_examples = load_examples_from_file(
    data_files["train"], data_files["has_features"]
)
eval_examples = load_examples_from_file(data_files["eval"], data_files["has_features"])

samples: list[tuple[str | list[str], str | list[str]]] = []
for ex in train_examples:
    input_string = list(ex.input_string)
    if ex.features is not None:
        input_string = (
            [f"[{f}]" for f in ex.features] + ["<sep>"] + input_string + ["<sink>"]
        )
    assert ex.output_string is not None
    samples.append((input_string, ex.output_string))

fst = ostia(samples)
fst = FST.re(".* '<sink>':'#'") @ fst

train_metrics = evaluate_all(fst, train_examples, output_raw_string=True)
logger.info(f"Train metrics: {pprint.pformat(train_metrics)}")
eval_metrics = evaluate_all(fst, eval_examples, output_raw_string=True)
logger.info(f"Eval metrics: {pprint.pformat(eval_metrics)}")
