import logging

from torch.utils.data import Dataset

from ..example import AlignedStringExample
from .tokenizer import AlignedTransductionTokenizer

logger = logging.getLogger(__file__)


class AlignedTransductionDataset(Dataset):
    """
    Represents pre-aligned inflection data.
    """

    def __init__(
        self,
        examples: list[AlignedStringExample],
        tokenizer: AlignedTransductionTokenizer | None,
        is_bidirect: bool = False,
    ):
        """
        Initialize a dataset. If a pretrained `tokenizer` is provided, will use to
        tokenize, otherwise, a new one will be created.
        """
        logger.info(f"Loaded {len(examples)} rows.")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AlignedTransductionTokenizer(is_bidirect=is_bidirect)
            self.tokenizer.learn_vocab(examples)

        self.examples = [self.tokenizer.tokenize(ex) for ex in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
