from dataclasses import dataclass


@dataclass
class AlignedInflectionExample:
    aligned_chars: list[tuple[str, str]]
    features: list[str]
    label: bool

    @property
    def aligned_chars_as_strs(self):
        return [f"({p[0]},{p[1]})" for p in self.aligned_chars]
