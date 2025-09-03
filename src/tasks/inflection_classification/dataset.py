import random
import re
from os import PathLike

from torch.utils.data import Dataset

from .example import AlignedInflectionExample
from .tokenizer import AlignedInflectionTokenizer


def load_examples_from_file(path: str | PathLike):
    """Loads `AlignedInflectionExample` instances from a TSV file"""
    examples: list[AlignedInflectionExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            if len(row) != 2:
                raise ValueError("File must be TSV with two columns")
            [chars, features] = row
            char_pairs: list[tuple[str, str]] = re.findall(r"\((.*?),(.*?)\)", chars)
            features = [f"[{f}]" for f in features.split(";")]
            examples.append(AlignedInflectionExample(char_pairs, features, label=True))
    return examples


def create_negative_examples(
    positive_examples: list[AlignedInflectionExample],
    syncretic_example_lookup: dict[str, list[tuple]],
    num_tag_swaps_per_ex=5,
    num_random_perturbs_per_ex=5,
    num_insertions_per_ex=5,
    seed=13,
):
    random.seed(13)
    all_examples: list[AlignedInflectionExample] = []

    # 1. Right input/output pair, wrong tag
    all_features = set(tuple(ex.features) for ex in positive_examples)
    for ex in positive_examples:
        valid_features = set(
            syncretic_example_lookup["".join(ex.aligned_chars_as_strs)]
        )
        invalid_features = random.sample(
            list(all_features - valid_features), k=num_tag_swaps_per_ex
        )
        for feat in invalid_features:
            all_examples.append(
                AlignedInflectionExample(
                    aligned_chars=ex.aligned_chars, features=list(feat), label=False
                )
            )

    # 2. Random perturb characters
    all_symbols = set(
        char for ex in positive_examples for pair in ex.aligned_chars for char in pair
    )

    def random_perturb(pairs: list[tuple[str, str]]):
        k = random.randint(1, len(pairs))
        perturb_indices = random.sample(range(len(pairs)), k=k)
        perturbed_pairs: list[tuple[str, str]] = []
        for index, pair in enumerate(pairs):
            if index not in perturb_indices:
                perturbed_pairs.append(pair)
            else:
                in_char, out_char = pair
                if random.random() > 0.5:
                    in_char = random.choice(list(all_symbols - set([in_char])))
                if random.random() > 0.5:
                    out_char = random.choice(list(all_symbols - set([out_char])))
                perturbed_pairs.append((in_char, out_char))
        return perturbed_pairs

    def random_insert(pairs: list[tuple[str, str]]):
        """Randomly insert (" ", character) tuples to the start or end"""
        k = random.randint(1, len(pairs))
        new_pairs = [(" ", random.choice(list(all_symbols))) for _ in range(k)]
        if random.random() > 0.5:
            return pairs + new_pairs
        else:
            return new_pairs + pairs

    def is_valid(example: AlignedInflectionExample):
        return (
            tuple(synthetic_example.features)
            in syncretic_example_lookup[
                "".join(synthetic_example.aligned_chars_as_strs)
            ]
        )

    for ex in positive_examples:
        for _ in range(num_random_perturbs_per_ex):
            synthetic_example = AlignedInflectionExample(
                aligned_chars=random_perturb(ex.aligned_chars),
                features=ex.features,
                label=False,
            )
            if is_valid(synthetic_example):
                continue
            all_examples.append(synthetic_example)

        for _ in range(num_insertions_per_ex):
            synthetic_example = AlignedInflectionExample(
                aligned_chars=random_insert(ex.aligned_chars),
                features=ex.features,
                label=False,
            )
            if is_valid(synthetic_example):
                continue
            all_examples.append(synthetic_example)

    return all_examples


class AlignedInflectionDataset(Dataset):
    """Represents pre-aligned inflection data. Data should be provided in a TSV file without headers."""

    def __init__(
        self,
        path: PathLike,
        tokenizer: AlignedInflectionTokenizer | None,
        syncretic_example_lookup: dict[str, list[tuple]],
    ):
        """Initialize a dataset. If a pretrained `tokenizer` is provided, will use to tokenize, otherwise, a new one will be created."""
        positive_examples = load_examples_from_file(path)
        print(f"Loaded {len(positive_examples)} rows.")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AlignedInflectionTokenizer()
            self.tokenizer.learn_vocab(positive_examples)

        negative_examples = create_negative_examples(
            positive_examples,
            syncretic_example_lookup,
            num_tag_swaps_per_ex=5,
            num_random_perturbs_per_ex=10,
            num_insertions_per_ex=10,
        )
        print(f"Created {len(negative_examples)} negative examples")
        self.examples = [
            self.tokenizer.tokenize(ex) for ex in positive_examples + negative_examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
