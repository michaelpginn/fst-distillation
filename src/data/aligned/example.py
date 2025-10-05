import re
from dataclasses import dataclass
from os import PathLike

ALIGNMENT_SYMBOL = "~"


@dataclass
class AlignedStringExample:
    aligned_chars: list[tuple[str, str]]
    features: list[str] | None
    label: bool

    @property
    def aligned_chars_as_strs(self):
        return [f"({p[0]},{p[1]})" for p in self.aligned_chars]

    @property
    def input_string(self):
        return "".join(
            in_char for in_char, _ in self.aligned_chars if in_char != ALIGNMENT_SYMBOL
        )

    @property
    def output_string(self):
        return "".join(
            out_char
            for _, out_char in self.aligned_chars
            if out_char != ALIGNMENT_SYMBOL
        )


def load_examples_from_file(path: str | PathLike, remove_epsilons=False):
    """Loads `AlignedStringExample` instances from a TSV file.
    If the file includes two columns, treat the second column as a
    feature column (e.g. for inflection)."""
    examples: list[AlignedStringExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            if len(row) == 1:
                [chars] = row
                features = None
            elif len(row) == 2:
                [chars, features] = row
                features = [f"[{f}]" for f in features.split(";")]
            else:
                raise ValueError("File must be TSV with 1-2 columns")
            char_pairs: list[tuple[str, str]] = re.findall(r"\((.*?),(.*?)\)", chars)
            examples.append(AlignedStringExample(char_pairs, features, label=True))

    if remove_epsilons:
        return remove_epsilon_inputs(examples)
    else:
        return examples


def remove_epsilon_inputs(examples: list[AlignedStringExample]):
    """Removes epsilon input pairs by delaying outputs.

    For example, (~,r)(~,e)(d,d)(~,x)(o,o) would become (<sep>:re)(d,d)(o,xo)
    """

    def process_example(example: AlignedStringExample):
        new_aligned_chars: list[tuple[str, str]] = []

        out_char_buffer = []
        seen_first_input = False
        for in_char, out_char in example.aligned_chars:
            if in_char == ALIGNMENT_SYMBOL:
                out_char_buffer.append(out_char)
            elif len(out_char_buffer) > 0:
                if seen_first_input:
                    new_aligned_chars.append(
                        (in_char, "".join(out_char_buffer) + out_char)
                    )
                else:
                    # For a prefix before any input chars, attach to sep
                    new_aligned_chars.append(
                        ("<sep>", "".join(out_char_buffer) + "<sep>")
                    )
                out_char_buffer = []
            else:
                new_aligned_chars.append((in_char, out_char))

        if len(out_char_buffer) > 0:
            new_aligned_chars.append(("<sink>", "".join(out_char_buffer) + "<sink>"))
        else:
            new_aligned_chars.append(("<sink>", "<sink>"))

        example.aligned_chars = new_aligned_chars
        return example

    return [process_example(ex) for ex in examples]
