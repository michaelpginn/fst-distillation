import abc
from typing import Any, List


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
        print("Created vocabulary.")

    @abc.abstractmethod
    def create_vocab(self, examples: list[Any]) -> list[str]:
        """Collect examples and create a list of the tokens in the vocabulary. Must return a list of strings."""
        pass

    @abc.abstractmethod
    def tokenize(self, example: Any) -> dict[str, list[int] | None]:
        """Tokenize a single example. Should return a Dict with the following fields:
        - source_input_ids
        - target_input_ids
        - target_label_ids
        """
        pass

    def decode(self, token_ids: List[int]) -> str:
        if self.id_to_token is None:
            raise Exception("Need to learn vocab with `create_vocab` first!")
        decoded_string = ""
        for id in token_ids:
            if id == self.eos_token_id or id == self.pad_token_id:
                break
            if id == self.bos_token_id:
                continue
            decoded_string += self.id_to_token[id]

        return decoded_string

    def __len__(self):
        if self.token_to_id is None:
            print("Warning: length of empty tokenizer is 0.")
            return 0
        return len(self.token_to_id)
