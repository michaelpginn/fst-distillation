import random
from collections import defaultdict

from tqdm import tqdm

from ..example import ALIGNMENT_SYMBOL, AlignedStringExample


def create_negative_examples(
    positive_examples: list[AlignedStringExample],
    all_positive_examples: list[AlignedStringExample],
    num_tag_swaps_per_ex=5,
    num_random_perturbs_per_ex=5,
    num_insertions_per_ex=5,
    num_deletions_per_ex=3,
    seed=13,
):
    random.seed(seed)

    lemma_lookup: dict[str, list[AlignedStringExample]] = defaultdict(lambda: [])
    for ex in all_positive_examples:
        lemma_lookup[ex.input_string].append(ex)

    negative_examples: list[AlignedStringExample] = []

    for ex in tqdm(positive_examples, "Creating negatives through tag swapping"):
        invalid_features = []

        # Select features that we have defined on this lemma (unless syncretic)
        invalid_features = [
            ex2.features
            for ex2 in lemma_lookup[ex.input_string]
            if ex2.aligned_chars != ex.aligned_chars and ex2.features is not None
        ]
        if len(invalid_features) > num_tag_swaps_per_ex:
            invalid_features = random.sample(invalid_features, k=num_tag_swaps_per_ex)
        for feat in invalid_features:
            negative_examples.append(
                AlignedStringExample(
                    aligned_chars=ex.aligned_chars, features=list(feat), label=False
                )
            )

    # 2. Random perturb characters
    all_symbols = set(
        char for ex in positive_examples for pair in ex.aligned_chars for char in pair
    )
    for ex in tqdm(positive_examples, "Creating negatives through perturbation"):
        for _ in range(num_random_perturbs_per_ex):
            synthetic_example = AlignedStringExample(
                aligned_chars=random_perturb(ex.aligned_chars, all_symbols=all_symbols),
                features=ex.features,
                label=False,
            )
            negative_examples.append(synthetic_example)

        for _ in range(num_insertions_per_ex):
            synthetic_example = AlignedStringExample(
                aligned_chars=random_insert(ex.aligned_chars, all_symbols=all_symbols),
                features=ex.features,
                label=False,
            )
            negative_examples.append(synthetic_example)

        num_alignment_symbols_on_input = sum(1 for c in ex.aligned_chars if c[0] == "~")
        if num_alignment_symbols_on_input > 0:
            for _ in range(min(num_deletions_per_ex, num_alignment_symbols_on_input)):
                if (perturbed_chars := random_delete(ex.aligned_chars)) is not None:
                    synthetic_example = AlignedStringExample(
                        aligned_chars=perturbed_chars,
                        features=ex.features,
                        label=False,
                    )
                    negative_examples.append(synthetic_example)

    return negative_examples


def random_perturb(pairs: list[tuple[str, str]], all_symbols: set[str]):
    """Randomly replace some number of output characters"""
    k = random.randint(1, len(pairs))
    perturb_indices = random.sample(range(len(pairs)), k=k)
    perturbed_pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs):
        if index not in perturb_indices:
            perturbed_pairs.append(pair)
        else:
            in_char, out_char = pair
            out_char = random.choice(list(all_symbols - set([out_char])))
            perturbed_pairs.append((in_char, out_char))
    return perturbed_pairs


def random_insert(pairs: list[tuple[str, str]], all_symbols: set[str]):
    """Randomly insert ("~", character) anywhere in the string"""
    k = random.randint(1, len(pairs) + 1)
    insert_indices = random.sample(range(len(pairs) + 1), k=k)
    for insert_index in sorted(insert_indices, reverse=True):
        new_pair = (
            ALIGNMENT_SYMBOL,
            random.choice(list(all_symbols - set([ALIGNMENT_SYMBOL]))),
        )
        pairs = pairs[:insert_index] + [new_pair] + pairs[insert_index:]
    return pairs


def random_delete(pairs: list[tuple[str, str]]):
    """Randomly delete (~, character) pairs"""
    alignment_char_indices = [
        i for i, c in enumerate(pairs) if c[0] == ALIGNMENT_SYMBOL
    ]
    if len(alignment_char_indices) == 0:
        return None
    k = random.randint(1, len(alignment_char_indices))
    delete_indices = random.sample(alignment_char_indices, k)
    for delete_index in sorted(delete_indices, reverse=True):
        pairs = pairs[:delete_index] + pairs[delete_index + 1 :]
    return pairs
