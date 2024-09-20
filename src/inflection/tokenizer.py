from ..tokenizer import Tokenizer
from .example import InflectionExample


class SharedTaskInflectionTokenizer(Tokenizer):
    def create_vocab(self, examples: list[InflectionExample]) -> list[str]:
        vocab: set[str] = set()
        for example in examples:
            vocab.update(set(example.lemma.lower()))
            vocab.update(set(example.features))
            if example.target is not None:
                vocab.update(set(example.target.lower()))
        return sorted(vocab)

    def tokenize(self, example: InflectionExample) -> dict[str, list[int] | None]:
        if self.token_to_id is None or self.id_to_token is None:
            raise ValueError(
                "Your tokenizer has no vocabulary! Call `create_vocab` or `load_from_file` to train your tokenizer."
            )

        source_input_ids = [self.bos_token_id]
        target_input_ids = None
        target_label_ids = None

        # Source should be `<bos> features <sep> lemma`
        source_input_ids += [
            self.token_to_id.get(feature, self.unk_token_id)
            for feature in example.features
        ]
        source_input_ids.append(self.sep_token_id)
        source_input_ids += [
            self.token_to_id.get(char, self.unk_token_id) for char in example.lemma
        ]

        # Target should be `<bos> target`
        if example.target is not None:
            target_input_ids = [self.bos_token_id]
            target_input_ids += [
                self.token_to_id.get(char, self.unk_token_id) for char in example.target
            ]

            # Labels are the same, omitting the <bos> and adding an <eos>
            target_label_ids = target_input_ids[1:] + [self.eos_token_id]

        return {
            "source_input_ids": source_input_ids,
            "target_input_ids": target_input_ids,
            "target_label_ids": target_label_ids,
        }
