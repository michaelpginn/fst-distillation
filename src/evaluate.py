from typing import List


def accuracy(predictions: List[str], labels: List[str]) -> float:
    if len(predictions) != len(labels):
        raise ValueError()
    return sum(1 for pr, la in zip(predictions, labels) if pr == la) / len(predictions)


def levenshtein(predictions: List[str], labels: List[str]) -> float:
    """Computes the unweighted edit distance between top prediction and correct form"""
    if len(predictions) != len(labels):
        raise ValueError()

    def _compute_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                    )
            distances = distances_
        return distances[-1]

    return sum(
        [_compute_distance(pr, la) for pr, la in zip(predictions, labels)]
    ) / len(predictions)
