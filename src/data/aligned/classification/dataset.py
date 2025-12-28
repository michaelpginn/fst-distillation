import logging

from torch.utils.data import Dataset

from ..example import AlignedStringExample
from .negative_examples import (
    create_negative_examples,
)
from .tokenizer import AlignedClassificationTokenizer

logger = logging.getLogger(__file__)


class AlignedClassificationDataset(Dataset):
    """
    Represents pre-aligned inflection data.

    The dataset will include positive and negative examples, IN THAT ORDER. Use `num_positives` to determine where the negative examples start.
    """

    def __init__(
        self,
        positive_examples: list[AlignedStringExample],
        tokenizer: AlignedClassificationTokenizer | None,
    ):
        """
        Initialize a dataset. If a pretrained `tokenizer` is provided, will use to
        tokenize, otherwise, a new one will be created.

        Args:
            positive_examples: The examples to use for *this dataset*
        """
        logger.info(f"Loaded {len(positive_examples)} rows.")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AlignedClassificationTokenizer()
            self.tokenizer.learn_vocab(positive_examples)

        logger.info("Creating negative examples")
        negative_examples = create_negative_examples(
            positive_examples, tokenizer=self.tokenizer, num_negs_per_ex=3
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
