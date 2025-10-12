from src.data.aligned.alignment_prediction.example import AlignmentPredictionExample
from src.modeling.tokenizer import Tokenizer


class AlignmentPredictionTokenizer(Tokenizer):
    """Tokenizer for alignment prediction, such as "[3P] run" -> "[3P] run~"

    Outputs are:

    {
        source_input_ids (list[int]): Source input ids, in the format `<bos> features <sep> chars`
        target_input_ids (list[int]): Target input ids, in the format `<bos> features <sep> chars`
        target_labels (list[int]): Target labels `features <sep> chars <eos>`
    }
    """

    def create_vocab(self, examples: list[AlignmentPredictionExample]) -> list[str]:
        vocab: set[str] = set()
        for example in examples:
            assert example.aligned is not None
            vocab.update(set(in_char for in_char in example.aligned))
            if example.features is not None:
                vocab.update(set(example.features))
        return sorted(vocab)

    def tokenize(
        self, example: AlignmentPredictionExample
    ) -> dict[str, int | list[int] | None]:
        if self.token_to_id is None or self.id_to_token is None:
            raise ValueError(
                "Your tokenizer has no vocabulary! Call `create_vocab` or `load_from_file` to train your tokenizer."
            )

        source_input_ids = [self.bos_token_id]

        # Source should be `<bos> features <sep> aligned chars <sink>`
        if example.features is not None:
            source_input_ids += [
                self.token_to_id.get(feature, self.unk_token_id)
                for feature in example.features
            ] + [self.sep_token_id]

        if example.aligned is not None:
            target_input_ids = source_input_ids.copy()
            for aligned_char in example.aligned:
                target_input_ids.append(
                    self.token_to_id.get(aligned_char, self.unk_token_id)
                )
            target_labels = target_input_ids[1:] + [self.eos_token_id]
        else:
            target_input_ids = None
            target_labels = None

        for in_char in example.unaligned:
            source_input_ids.append(self.token_to_id.get(in_char, self.unk_token_id))

        if target_input_ids is not None and target_labels is not None:
            return {
                "source_input_ids": source_input_ids,
                "target_input_ids": target_input_ids,
                "target_label_ids": target_labels,
            }
        return {
            "source_input_ids": source_input_ids,
        }
