import pprint
from logging import getLogger

from pyfoma.fst import FST

from ..data.unaligned.example import load_examples_from_file
from ..evaluate import evaluate_all
from ..paths import create_arg_parser, create_paths_from_args
from .ostia import ostia

logger = getLogger(__name__)

parser = create_arg_parser()
args = parser.parse_args()
paths = create_paths_from_args(args)

train_examples = load_examples_from_file(paths["train"], paths["has_features"])
eval_examples = load_examples_from_file(paths["eval"], paths["has_features"])
test_examples = load_examples_from_file(paths["test"], paths["has_features"])

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
eval_metrics = evaluate_all(fst, test_examples, output_raw_string=True)
logger.info(f"Test metrics: {pprint.pformat(eval_metrics)}")
