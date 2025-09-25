from os import PathLike

from torch.utils.data import Dataset

from .example import load_examples_from_file
from .tokenizer import SharedTaskInflectionTokenizer


class SharedTaskInflectionDataset(Dataset):
    """Represents data from the SIGMORPHON shared tasks. Data should be provided in a TSV file without headers."""

    def __init__(
        self,
        path: PathLike,
        tokenizer: SharedTaskInflectionTokenizer | None,
        has_features: bool,
    ):
        """Initialize a dataset. If a pretrained `tokenizer` is provided, will use to tokenize, otherwise, a new one will be created."""
        self.raw_examples = load_examples_from_file(path, has_features)
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
