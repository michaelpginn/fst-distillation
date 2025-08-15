from dataclasses import dataclass
from typing import Self
from weakref import ReferenceType

import numpy as np

# We use weak refs in order to prevent circular references

@dataclass(frozen=True)
class TransitionLabel:
    input_symbol: str
    output_symbol: str
    
@dataclass
class Microstate:
    activation: np.ndarray
    outgoing_transition: tuple[TransitionLabel, ReferenceType[Self]] | None
    incoming_transition: tuple[TransitionLabel, ReferenceType[Self]] | None
    macrostate: ReferenceType['Macrostate']

@dataclass
class Macrostate:
    label: str
    microstates: set[Microstate]
    is_final: bool = False
