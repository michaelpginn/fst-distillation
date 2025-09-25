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


def load_examples_from_file(path: str | PathLike):
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
    return examples
