import logging
import re
from os import PathLike

from torch.utils.data import Dataset

from src.tasks.inflection_classification.negative_examples import (
    create_negative_examples,
)

from .example import AlignedInflectionExample
from .tokenizer import AlignedInflectionTokenizer

logger = logging.getLogger(__file__)


def load_examples_from_file(path: str | PathLike):
    """Loads `AlignedInflectionExample` instances from a TSV file"""
    examples: list[AlignedInflectionExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            if len(row) != 2:
                raise ValueError("File must be TSV with two columns")
            [chars, features] = row
            char_pairs: list[tuple[str, str]] = re.findall(r"\((.*?),(.*?)\)", chars)
            features = [f"[{f}]" for f in features.split(";")]
            examples.append(AlignedInflectionExample(char_pairs, features, label=True))
    return examples


class AlignedInflectionDataset(Dataset):
    """
    Represents pre-aligned inflection data. Data should be provided in a TSV file without headers.

    The dataset will include positive and negative examples, IN THAT ORDER. Use `num_positives` to determine where the negative examples start.
    """

    def __init__(
        self,
        positive_examples: list[AlignedInflectionExample],
        all_positive_examples: list[AlignedInflectionExample],
        tokenizer: AlignedInflectionTokenizer | None,
    ):
        """
        Initialize a dataset. If a pretrained `tokenizer` is provided, will use to
        tokenize, otherwise, a new one will be created.

        Args:
            positive_examples: The examples to use for *this dataset*
            all_positive_examples: All seen positive examples across datasets
        """
        logger.info(f"Loaded {len(positive_examples)} rows.")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AlignedInflectionTokenizer()
            self.tokenizer.learn_vocab(positive_examples)

        logger.info("Creating negative examples")
        negative_examples = create_negative_examples(
            positive_examples,
            all_positive_examples=all_positive_examples,
            num_tag_swaps_per_ex=5,
            num_random_perturbs_per_ex=10,
            num_insertions_per_ex=10,
        )
        logger.info(f"Created {len(negative_examples)} negative examples")
        self.examples = [
            self.tokenizer.tokenize(ex) for ex in positive_examples + negative_examples
        ]
        self.num_positives = len(positive_examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
