from dataclasses import dataclass
from os import PathLike


@dataclass
class String2StringExample:
    input_string: str
    features: list[str] | None
    output_string: str | None


def load_examples_from_file(path: PathLike, has_features: bool):
    """Loads `String2StringExample` instances from a TSV file"""
    examples: list[String2StringExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            output_string = None
            features = None
            if has_features:
                if len(row) == 2:
                    # Test data, no target forms
                    [input_string, features] = row
                else:
                    [input_string, output_string, features] = row
                features = features.split(";")
            else:
                if len(row) == 1:
                    [input_string] = row
                else:
                    [input_string, output_string] = row
            examples.append(String2StringExample(input_string, features, output_string))

    return examples
