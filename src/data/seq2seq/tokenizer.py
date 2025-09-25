from src.modeling.tokenizer import Tokenizer

from .example import String2StringExample


class SharedTaskInflectionTokenizer(Tokenizer):
    def create_vocab(self, examples: list[String2StringExample]) -> list[str]:
        vocab: set[str] = set()
        for example in examples:
            vocab.update(set(example.input_string.lower()))
            if example.features is not None:
                vocab.update(set(example.features))
            if example.output_string is not None:
                vocab.update(set(example.output_string.lower()))
        return sorted(vocab)

    def tokenize(
        self, example: String2StringExample
    ) -> dict[str, int | list[int] | None]:
        if self.token_to_id is None or self.id_to_token is None:
            raise ValueError(
                "Your tokenizer has no vocabulary! Call `create_vocab` or `load_from_file` to train your tokenizer."
            )

        source_input_ids = [self.bos_token_id]
        target_input_ids = None
        target_label_ids = None

        # Source should be `<bos> features <sep> lemma`
        if example.features is not None:
            source_input_ids += [
                self.token_to_id.get(feature, self.unk_token_id)
                for feature in example.features
            ]
            source_input_ids.append(self.sep_token_id)
        source_input_ids += [
            self.token_to_id.get(char, self.unk_token_id)
            for char in example.input_string
        ]

        # Target should be `<bos> target`
        if example.output_string is not None:
            target_input_ids = [self.bos_token_id]
            target_input_ids += [
                self.token_to_id.get(char, self.unk_token_id)
                for char in example.output_string
            ]

            # Labels are the same, omitting the <bos> and adding an <eos>
            target_label_ids = target_input_ids[1:] + [self.eos_token_id]

        return {
            "source_input_ids": source_input_ids,
            "target_input_ids": target_input_ids,
            "target_label_ids": target_label_ids,
        }
