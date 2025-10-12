import logging

from torch.utils.data import Dataset

from src.data.aligned.alignment_prediction.example import AlignmentPredictionExample

from .tokenizer import AlignmentPredictionTokenizer

logger = logging.getLogger(__file__)


class AlignmentPredictionDataset(Dataset):
    """Represents data for alignment prediction."""

    def __init__(
        self,
        examples: list[AlignmentPredictionExample],
        tokenizer: AlignmentPredictionTokenizer | None,
    ):
        """
        Initialize a dataset. If a pretrained `tokenizer` is provided, will use to
        tokenize, otherwise, a new one will be created.
        """
        logger.info(f"Loaded {len(examples)} rows.")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AlignmentPredictionTokenizer()
            self.tokenizer.learn_vocab(examples)

        self.examples = [self.tokenizer.tokenize(ex) for ex in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
