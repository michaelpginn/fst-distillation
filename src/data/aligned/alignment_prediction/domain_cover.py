from collections import defaultdict

from .example import AlignmentPredictionExample


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
    lemma_to_features: dict[str, set[tuple[str, ...]]] = defaultdict(lambda: set())
    all_train_inputs = set()
    for ex in train_examples:
        lemma = "".join(ex.unaligned)
        lemma_to_features[lemma].add(tuple(ex.features))  # type:ignore
        all_train_inputs.add(("".join(ex.unaligned), tuple(ex.features)))  # type:ignore

    generated: set[tuple[str, tuple[str, ...]]] = set()
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


def _domain_cover_no_features(train_examples: list[AlignmentPredictionExample]):
    raise NotImplementedError()
