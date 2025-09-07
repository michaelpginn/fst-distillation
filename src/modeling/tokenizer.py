import abc
import logging
from typing import Any, List, Literal

logger = logging.getLogger(__file__)


class Tokenizer(metaclass=abc.ABCMeta):
    """Base class for tokenizers. Concrete subclasses must override `tokenize` and `create_vocab`."""

    token_to_id: dict[str, int] | None = None
    id_to_token: dict[int, str] | None = None

    special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>", "<sep>"]
    unk_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3
    sep_token_id = 4

    def learn_vocab(self, examples: list[Any]):
        """Fit the tokenizer to a training dataset. Fills `token_to_id` and `id_to_token`."""
        vocabulary = self.special_tokens + self.create_vocab(examples)
        self.token_to_id = {
            token: id for id, token in zip(range(len(vocabulary)), vocabulary)
        }
        self.id_to_token = {
            id: token for id, token in zip(range(len(vocabulary)), vocabulary)
        }
        logger.info("Created vocabulary.")

    @abc.abstractmethod
    def create_vocab(self, examples: list[Any]) -> list[str]:
        """Collect examples and create a list of the tokens in the vocabulary. Must return a list of strings."""
        pass

    @abc.abstractmethod
    def tokenize(self, example: Any) -> dict[str, int | list[int] | None]:
        """Tokenize a single example, returning a dict of inputs used by the model."""
        pass

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens=True,
        return_as: Literal["str", "list"] = "str",
    ):
        if self.id_to_token is None:
            raise Exception("Need to learn vocab with `create_vocab` first!")
        decoded = str("") if return_as == "str" else list[str]()

        for id in token_ids:
            if skip_special_tokens and id <= len(self.special_tokens):
                continue
            if isinstance(decoded, str):
                decoded += self.id_to_token[id]
            else:
                decoded.append(self.id_to_token[id])

        return decoded

    def __len__(self):
        if self.token_to_id is None:
            logger.warning("Length of empty tokenizer is 0.")
            return 0
        return len(self.token_to_id)

    @property
    def state_dict(self):
        if self.id_to_token is None:
            raise Exception("Need to learn vocab with `create_vocab` first!")
        return self.id_to_token

    @classmethod
    def from_state_dict(cls, state_dict: dict[int, str]):
        tokenizer = cls()
        tokenizer.id_to_token = state_dict
        tokenizer.token_to_id = {token: id for id, token in state_dict.items()}
        return tokenizer
