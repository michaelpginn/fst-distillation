import random
import re


from src.data.aligned.classification.tokenizer import AlignedClassificationTokenizer

from ..example import AlignedStringExample


def create_negative_examples(
    positive_examples: list[AlignedStringExample],
    tokenizer: AlignedClassificationTokenizer,
    num_negs_per_ex: int,
    seed=13,
):
    random.seed(seed)
    assert tokenizer.id_to_token

    pair_lookup: dict[str, set[str]] = {}

    negatives: list[AlignedStringExample] = []
    for ex in positive_examples:
        for _ in range(num_negs_per_ex):
            k = random.randint(1, len(ex.aligned_chars))
            indices_to_replace = random.sample(list(range(len(ex.aligned_chars))), k=k)
            new_aligned_chars = ex.aligned_chars.copy()
            for idx in indices_to_replace:
                in_char, out_char = ex.aligned_chars[idx]

                # Cache result if needed
                if in_char not in pair_lookup:
                    pair_lookup[in_char] = set()
                    for other_tok_id in tokenizer.token_ids_matching_input(in_char):
                        other_tok = tokenizer.id_to_token[other_tok_id]
                        other_out_char = re.match(r"\((.*),(.*)\)", other_tok).group(2)  # type:ignore
                        pair_lookup[in_char].add(other_out_char)

                # Pick a new out char
                new_out_char = random.choice(
                    [c for c in pair_lookup[in_char] if c != out_char]
                )
                new_aligned_chars[idx] = (in_char, new_out_char)
            negatives.append(
                AlignedStringExample(
                    new_aligned_chars, features=ex.features, label=False
                )
            )

    return negatives
