from dataclasses import dataclass
from os import PathLike


@dataclass
class String2StringExample:
    input_string: str | list[str]
    features: list[str] | None
    output_string: str | list[str] | None


def load_examples_from_file(
    path: PathLike, has_features: bool, output_split_into_chars: bool
):
    """Loads `String2StringExample` instances from a TSV file"""
    examples: list[String2StringExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            output_string = None
            features = None
            if has_features:
                try:
                    if len(row) == 2:
                        # Test data, no target forms
                        [input_string, features] = row
                    else:
                        [input_string, output_string, features] = row
                    features = features.split(";")
                except ValueError:
                    raise ValueError(
                        "Wrong number of columns, you probably should remove --features"
                    )
            else:
                try:
                    if len(row) == 1:
                        [input_string] = row
                    else:
                        [input_string, output_string] = row
                except ValueError:
                    raise ValueError(
                        "Wrong number of columns, you probably want --features"
                    )
            if len(input_string.strip()) == 0:
                continue
            if output_split_into_chars and output_string:
                output_string = output_string.split()
            examples.append(String2StringExample(input_string, features, output_string))

    return examples
