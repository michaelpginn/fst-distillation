import weakref
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Microtransition:
    input_symbol: str
    output_symbol: str
    source: weakref.ReferenceType["Microstate"]
    target: weakref.ReferenceType["Microstate"]


@dataclass(eq=False)
class Microstate:
    position: np.ndarray
    is_final: bool
    incoming: Microtransition | None = None
    outgoing: weakref.ReferenceType[Microtransition] | None = None
    macrostate: weakref.ReferenceType["Macrostate"] | None = None

    def __hash__(self):
        return id(self)


@dataclass(eq=False)
class Macrotransition:
    input_symbol: str
    output_symbol: str
    source: weakref.ReferenceType["Macrostate"]
    target: weakref.ReferenceType["Macrostate"]

    def __hash__(self):
        return id(self)


@dataclass
class Macrostate:
    label: str
    microstates: set[Microstate] = field(default_factory=set)
    incoming: set[weakref.ReferenceType[Macrotransition]] = field(default_factory=set)
    outgoing: set[Macrotransition] = field(default_factory=set)

    def compute_outgoing_distributions(self):
        """Creates a dictionary

        {input symbol: {(output symbol, target state): probability}}

        that records the count for each possible (output symbol, target state) for a given input symbol
        """
        distribution: dict[str, Counter[tuple[str, str]]] = defaultdict(
            lambda: Counter()
        )
        for microstate in self.microstates:
            if (
                microstate.outgoing is not None
                and (outgoing := microstate.outgoing()) is not None
                and (target_state := outgoing.target()) is not None
                and target_state.macrostate is not None
                and (target_macrostate := target_state.macrostate())
            ):
                distribution[outgoing.input_symbol][
                    (outgoing.output_symbol, target_macrostate.label)
                ] += 1

        return distribution

        # probabilities: dict[str, dict[tuple[str, str], float]] = {}
        # for symbol in distribution:
        #     total_count = sum(distribution[symbol].values())
        #     probabilities[symbol] = {
        #         k: float(v / total_count) for k, v in distribution[symbol].items()
        #     }
        # return probabilities
