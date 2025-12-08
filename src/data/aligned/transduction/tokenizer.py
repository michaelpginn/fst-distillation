from src.modeling.tokenizer import Tokenizer

from ..example import AlignedStringExample


class AlignedTransductionTokenizer(Tokenizer):
    """Tokenizer for language modeling. Outputs are:

    {
        input_ids (list[str]): Input ids, in the format `<bos> features <sep> input_chars <sink>`
        next_input_ids (list[str]): Next input ids, `features <sep> input_chars <sink> <eos>`
        next_output_ids (list[str]): Next output ids, `features <sep> output_chars <sink> <eos>`
    }
    """

    def __init__(self, is_bidirect: bool = False):
        self.is_bidirect = is_bidirect

    def create_vocab(self, examples: list[AlignedStringExample]) -> list[str]:
        vocab: set[str] = set()
        for example in examples:
            vocab.update({c for pair in example.aligned_chars for c in pair})
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
        output_ids = []

        # Source should be `<bos> features <sep> aligned chars <sink>`
        if example.features is not None:
            feature_ids = [
                self.token_to_id.get(feature, self.unk_token_id)
                for feature in example.features
            ]
            input_ids += feature_ids
            output_ids += feature_ids
        if len(example.aligned_chars) == 0 or example.aligned_chars[0][0] != "<sep>":
            input_ids.append(self.sep_token_id)
            output_ids.append(self.sep_token_id)

        for i, o in example.aligned_chars:
            input_ids.append(self.token_to_id.get(i, self.unk_token_id))
            output_ids.append(self.token_to_id.get(o, self.unk_token_id))

        if len(example.aligned_chars) == 0 or example.aligned_chars[-1][0] != "<sink>":
            input_ids.append(self.sink_token_id)
            output_ids.append(self.sink_token_id)

        if self.is_bidirect:
            # For bidirectional, we need the input to have one extra on both end (bos and eos)
            # compared to the input ids
            input_ids += [self.eos_token_id]
            next_input_ids = input_ids[1:-1]
            next_output_ids = output_ids
        else:
            next_input_ids = input_ids[1:] + [self.eos_token_id]
            next_output_ids = output_ids + [self.eos_token_id]
        return {
            "input_ids": input_ids,
            "next_input_ids": next_input_ids,
            "next_output_ids": next_output_ids,
        }
