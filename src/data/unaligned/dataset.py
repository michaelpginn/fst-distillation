from torch.utils.data import Dataset

from .example import String2StringExample
from .tokenizer import String2StringTokenizer


class String2StringDataset(Dataset):
    """Represents string to string data."""

    def __init__(
        self,
        examples: list[String2StringExample],
        tokenizer: String2StringTokenizer | None,
        has_features: bool,
    ):
        """Initialize a dataset. If a pretrained `tokenizer` is provided, will use to tokenize, otherwise, a new one will be created."""
        print(f"Loaded {len(examples)} rows.")
        self.raw_examples = examples

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = String2StringTokenizer()
            self.tokenizer.learn_vocab(examples)
        self.examples = [self.tokenizer.tokenize(ex) for ex in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
