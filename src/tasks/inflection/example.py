from dataclasses import dataclass

@dataclass
class InflectionExample:
    lemma: str
    features: list[str]
    target: str | None
