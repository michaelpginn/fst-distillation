import logging
import math
import random
from collections import defaultdict
from typing import Counter

from src.data.aligned.example import AlignedStringExample

from .example import AlignmentPredictionExample

logger = logging.getLogger(__name__)


def domain_cover(
    train_examples: list[AlignmentPredictionExample],
):
    """Create a list that covers the whole domain based on some sample, EXCLUDING THE TRAIN EXAMPLES.
    - For inputs with features, the list will include all combinations of inputs and features
    - For inputs without features, TODO
    """
    if train_examples[0].features is not None:
        return _domain_cover_features(train_examples)
    else:
        return _domain_cover_no_features(train_examples)


def _domain_cover_features(train_examples: list[AlignmentPredictionExample]):
    """Swap features between lemmas that share at least one other feature bundle.

    E.g. Given the train set has:

        c a t   N;Sg
        d o g   N;Sg
        c a t   N;Pl

    We can create the example:

        d o g   N;Pl

    as well, bc "cat" and "dog" share a feature bundle.
    """
    lemma_to_features: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(
        lambda: set()
    )
    all_train_inputs = set()
    for ex in train_examples:
        lemma = tuple(ex.unaligned)
        lemma_to_features[lemma].add(tuple(ex.features))  # type:ignore
        all_train_inputs.add((lemma, tuple(ex.features)))  # type:ignore

    generated: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    lemmas = list(lemma_to_features.keys())
    for i, lem1 in enumerate(lemmas):
        for lem2 in lemmas[i + 1 :]:
            # Check if overlap
            if lemma_to_features[lem1] & lemma_to_features[lem2]:
                for f in lemma_to_features[lem2]:
                    if (lem1, f) not in all_train_inputs:
                        generated.add((lem1, f))
                for f in lemma_to_features[lem1]:
                    if (lem2, f) not in all_train_inputs:
                        generated.add((lem2, f))
    return [
        AlignmentPredictionExample(list(lem), None, list(feat))
        for (lem, feat) in generated
    ]


def _domain_cover_no_features(
    train_examples: list[AlignmentPredictionExample], n=2, max_samples=3000
):
    """Since it's infeasible to cover the *whole* domain, we will approximate it using
    an n-gram language model.
    """

    # Collect counts for all n-grams
    # Wow this is like a homework assignment
    def _ngrams(s: list[str]):
        for i in range(0, len(s) - n + 1):
            yield tuple(s[i : i + n])

    ngram_counts: Counter[tuple[str, ...]] = Counter()
    lengths: Counter[int] = Counter()
    for ex in train_examples:
        s = ["#"] + ex.unaligned + ["#"]
        ngram_counts.update(list(_ngrams(s)))
        lengths.update([len(s)])

    # Compute Gaussian distribution for length
    sum_of_numbers = sum(number * count for number, count in lengths.items())
    count = sum(count for n, count in lengths.items())
    mean = sum_of_numbers / count
    total_squares = sum(number * number * count for number, count in lengths.items())
    mean_of_squares = total_squares / count
    variance = mean_of_squares - mean * mean
    std_dev = math.sqrt(variance)

    # Sampling time
    new_lemmas: set[tuple[str, ...]] = set()
    while len(new_lemmas) < max_samples:
        chosen_length = round(random.gauss(mu=mean, sigma=std_dev))
        start_grams = [gram for gram in ngram_counts if gram[0] == "#"]
        new_lemma = list(
            random.choices(
                start_grams, weights=[ngram_counts[g] for g in start_grams], k=1
            )[0]
        )

        try:
            while len(new_lemma) < chosen_length - 1:
                next_gram_candidates = [
                    gram
                    for gram in ngram_counts
                    if list(gram[:-1]) == new_lemma[-(n - 1) :] and gram[-1] != "#"
                ]
                if len(next_gram_candidates) == 0:
                    raise Exception()
                next_gram = random.choices(
                    next_gram_candidates,
                    weights=[ngram_counts[g] for g in next_gram_candidates],
                    k=1,
                )[0]
                new_lemma.append(next_gram[-1])

        except Exception:
            logger.info("Skipping, no matching n-gram")
            continue
        last_gram_candidates = [
            gram
            for gram in ngram_counts
            if list(gram[:-1]) == new_lemma[-(n - 1) :] and gram[-1] == "#"
        ]
        if len(last_gram_candidates) == 0:
            logger.info("Skipping, no match end n-gram")
            continue
        new_lemma.append("#")
        logger.info(f"New lemma: {new_lemma}")
        new_lemmas.add(tuple(new_lemma))

    return [
        AlignmentPredictionExample(list(lem[1:-1]), None, None) for lem in new_lemmas
    ]


def ngram_bfs(train_examples: list[AlignedStringExample], n=3, max_length=7):
    # Performs *search*, returning all possible input strings (except existing train strings) in increasing length order
    logger.info(f"BFS using {n}-grams up to {max_length=}")

    def _ngrams(s: list[str]):
        for i in range(0, len(s) - n + 1):
            yield tuple(s[i : i + n])

    ngrams: dict[tuple[str, ...], set[str]] = defaultdict(set)
    train_strings: set[tuple[str, ...]] = set()
    for ex in train_examples:
        if ex.features is not None:
            raise ValueError("Cannot use search with features")
        s = [in_char for in_char, _ in ex.aligned_chars]
        train_strings.add(tuple(s))
        for gram in _ngrams(s):
            ngrams[gram[:-1]].add(gram[-1])

    logger.info(f"Generated {len(ngrams)} {n}-grams")

    all_strings: list[list[str]] = []
    queue: list[tuple[str, ...]] = []
    # Start with any start ngrams
    for key, chars in ngrams.items():
        if key[0] == "<sep>":
            for c in chars:
                queue.append((*key, c))
    # BFS
    added = 0
    while len(queue) > 0:
        next_string = queue.pop(0)
        if next_string[-1] == "<sink>":
            if next_string not in train_strings:
                all_strings.append(list(next_string))
                added += 1
                if added % 1000 == 0:
                    logger.info("Generated %d strings" % added)
            continue
        if len(next_string) >= max_length:
            # Throw away this string
            continue
        # Look for matching ngrams
        for c in ngrams[tuple(next_string[-(n - 1) :])]:
            queue.append((*next_string, c))
    return all_strings
