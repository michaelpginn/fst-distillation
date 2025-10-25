"""Usage: python -m src.alignment.run_alignment <task> <dataset>
Given one or more files in the shared task format, runs Hulden alignment over all examples (from any file), excluding tags.

Given the inputs:

cat cat V;Sing
cat cats    V;PL
dog dogs    V;PL

This script will produce the output file:

(c,c)(a,a)(t,t) V;Sing
(c,c)(a,a)(t,t)( ,s)    V;PL
(d,d)(o,o)(g,g)( ,s)    V:PL
"""

import logging
import pathlib

from .crpalign import Aligner
from .data.unaligned.example import load_examples_from_file
from .paths import Paths, create_arg_parser, create_paths_from_args

logger = logging.getLogger(__name__)


def run_alignment(paths: Paths):
    logger.info("Running alignment")

    output_folder = paths["aligned_folder"]
    output_folder.mkdir(exist_ok=True)
    file_paths = [paths["train"], paths["eval"]]
    examples_per_file = [
        load_examples_from_file(path, has_features=paths["has_features"])
        for path in file_paths
    ]
    all_examples = [ex for file_examples in examples_per_file for ex in file_examples]

    wordpairs: list[tuple[list[str], list[str]]] = []
    for ex in all_examples:
        assert ex.output_string is not None
        wordpairs.append(
            (tokenize_to_chars(ex.input_string), tokenize_to_chars(ex.output_string))
        )

    aligner = Aligner(
        wordpairs=wordpairs,
        iterations=100,
        align_symbol=paths["alignment_symbol"],
    )
    alignments: list[tuple[str, str]] = aligner.alignedpairs

    logger.info(f"Writing outputs to {output_folder}")
    current_offset = 0
    for file_index in range(len(file_paths)):
        num_examples = len(examples_per_file[file_index])
        file_name = pathlib.Path(file_paths[file_index]).name + ".aligned"
        with open(output_folder / file_name, "w") as f:
            for example_index in range(current_offset, current_offset + num_examples):
                in_str, out_str = alignments[example_index]
                aligned_tuples = "".join(
                    [
                        f"({in_char},{out_char})"
                        for in_char, out_char in zip(
                            tokenize_to_chars(in_str), tokenize_to_chars(out_str)
                        )
                    ]
                )
                if (features := all_examples[example_index].features) is not None:
                    features_string = ";".join(features)
                    f.write(f"{aligned_tuples}\t{features_string}\n")
                else:
                    f.write(f"{aligned_tuples}\n")

        current_offset += num_examples


def tokenize_to_chars(s: str, post_diacritics: set[str] = {"¹", "²"}):
    """Splits a string into chars.
    Any symbols in `post_diacritics` will be attached to the previous character as a single symbol."""
    chars = list(s)
    new_chars: list[str] = []
    for c in chars:
        if c in post_diacritics:
            new_chars[-1] += c
        else:
            new_chars.append(c)
    return new_chars


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    run_alignment(
        paths=create_paths_from_args(args),
    )
