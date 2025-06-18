from torch.utils.data import Dataset

from .example import InflectionExample
from .tokenizer import SharedTaskInflectionTokenizer


def load_examples_from_file(path: str):
    """Loads `InflectionExample` instances from a TSV file"""
    examples: list[InflectionExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            if len(row) == 2:
                # Test data, no target forms
                [lemma, features] = row
                target = None
            else:
                [lemma, target, features] = row
            features = features.split(";")
            examples.append(InflectionExample(lemma, features, target))
    return examples


class SharedTaskInflectionDataset(Dataset):
    """Represents data from the SIGMORPHON shared tasks. Data should be provided in a TSV file without headers."""

    def __init__(self, path: str, tokenizer: SharedTaskInflectionTokenizer | None):
        """Initialize a dataset. If a pretrained `tokenizer` is provided, will use to tokenize, otherwise, a new one will be created."""
        self.raw_examples = load_examples_from_file(path)
        print(f"Loaded {len(self.raw_examples)} rows.")

        # Tokenize
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SharedTaskInflectionTokenizer()
            self.tokenizer.learn_vocab(self.raw_examples)
        self.examples = [self.tokenizer.tokenize(ex) for ex in self.raw_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
