from dataclasses import dataclass

ALIGNMENT_SYMBOL = "~"


@dataclass
class AlignedInflectionExample:
    aligned_chars: list[tuple[str, str]]
    features: list[str]
    label: bool

    @property
    def aligned_chars_as_strs(self):
        return [f"({p[0]},{p[1]})" for p in self.aligned_chars]

    @property
    def lemma(self):
        return "".join(
            in_char for in_char, _ in self.aligned_chars if in_char != ALIGNMENT_SYMBOL
        )

    @property
    def inflected(self):
        return "".join(
            out_char
            for _, out_char in self.aligned_chars
            if out_char != ALIGNMENT_SYMBOL
        )
