from dataclasses import dataclass

from src.data.aligned.example import ALIGNMENT_SYMBOL, AlignedStringExample


@dataclass
class AlignmentPredictionExample:
    unaligned: list[str]
    aligned: list[str] | None
    features: list[str] | None

    @classmethod
    def from_aligned(
        cls, example: AlignedStringExample, alignment_symbol=ALIGNMENT_SYMBOL
    ):
        in_chars = [c for c, _ in example.aligned_chars]
        return AlignmentPredictionExample(
            unaligned=[c for c in in_chars if c != alignment_symbol],
            aligned=in_chars,
            features=example.features,
        )
