"""Baseline that simply predicts the same string back"""

from .data.unaligned.example import load_examples_from_file
from .paths import create_arg_parser, create_paths_from_args

parser = create_arg_parser()
args = parser.parse_args()
paths = create_paths_from_args(args)
test_examples = load_examples_from_file(
    paths["test"], paths["has_features"], paths["output_split_into_chars"]
)

accuracy = len(
    [ex for ex in test_examples if ex.input_string == ex.output_string]
) / len(test_examples)

print(f"Score: {accuracy}")
