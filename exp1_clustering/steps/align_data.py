"""Usage: python -m exp1_clustering.run_alignment <train_file> <eval_file> ...
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

import argparse
import logging
import pathlib
from os import PathLike

from src.hulden_alignment import Aligner
from src.tasks.inflection_classification.example import ALIGNMENT_SYMBOL
from src.tasks.inflection_seq2seq.dataset import load_examples_from_file

logger = logging.getLogger(__name__)


def run_alignment(file_paths: list[PathLike], alignment_character=ALIGNMENT_SYMBOL):
    output_folder = pathlib.Path(__file__).parent.parent / "aligned_data"
    output_folder.mkdir(exist_ok=True)

    logger.info("Running alignment")
    examples_per_file = [load_examples_from_file(path) for path in file_paths]
    all_examples = [ex for file_examples in examples_per_file for ex in file_examples]
    assert not any(ex.target is None for ex in all_examples)
    aligner = Aligner(
        wordpairs=[(ex.lemma, ex.target) for ex in all_examples],  # type:ignore
        iterations=100,
        align_symbol=alignment_character,
    )
    alignments: list[tuple[str, str]] = aligner.alignedpairs

    logger.info(f"Writing outputs to {output_folder}")
    current_offset = 0
    for file_index in range(len(file_paths)):
        num_examples = len(examples_per_file[file_index])
        file_name = pathlib.Path(file_paths[file_index]).name + ".aligned"
        with open(output_folder / file_name, "w") as f:
            for example_index in range(current_offset, current_offset + num_examples):
                aligned_strings = alignments[example_index]
                aligned_tuples = "".join(
                    [
                        f"({in_char},{out_char})"
                        for in_char, out_char in zip(*aligned_strings)
                    ]
                )
                features_string = ";".join(all_examples[example_index].features)
                f.write(f"{aligned_tuples}\t{features_string}\n")
        current_offset += num_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    run_alignment(args.files)
