from src.modeling.tokenizer import Tokenizer

from .example import AlignedStringExample


class AlignedInflectionTokenizer(Tokenizer):
    def create_vocab(self, examples: list[AlignedStringExample]) -> list[str]:
        vocab: set[str] = set()
        for example in examples:
            vocab.update(set(example.aligned_chars_as_strs))
            if example.features is not None:
                vocab.update(set(example.features))
        return sorted(vocab)

    def tokenize(
        self, example: AlignedStringExample
    ) -> dict[str, int | list[int] | None]:
        if self.token_to_id is None or self.id_to_token is None:
            raise ValueError(
                "Your tokenizer has no vocabulary! Call `create_vocab` or `load_from_file` to train your tokenizer."
            )

        input_ids = [self.bos_token_id]

        # Source should be `<bos> features <sep> aligned chars`
        if example.features is not None:
            input_ids += [
                self.token_to_id.get(feature, self.unk_token_id)
                for feature in example.features
            ]
            input_ids.append(self.sep_token_id)
        input_ids += [
            self.token_to_id.get(pair, self.unk_token_id)
            for pair in example.aligned_chars_as_strs
        ]
        return {"input_ids": input_ids, "label": int(example.label)}
