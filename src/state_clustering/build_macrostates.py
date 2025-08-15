import re
import weakref
import numpy as np

from src.state_clustering.types import Macrostate, Microstate, TransitionLabel


def split_transition_label(label: str) -> TransitionLabel:
    """Splits a transition label into an input and output label (if possible) """
    if match := re.match(r"\((.*),(.*)\)", label):
        input_symbol = match.group(1)
        output_symbol = match.group(2)
    else:
        input_symbol = label
        output_symbol = label
    return TransitionLabel(input_symbol, output_symbol)


def build_macrostates(
    activations: np.ndarray,
    cluster_labels: np.ndarray[tuple[int], np.dtype[np.str_]],
    tokens: list[list[str]]
) -> tuple[set[Macrostate], Macrostate]:
    """Builds macrostates from the outputs of activation clustering. 
    
    n: Total number of examples
    m: Total number of symbols across examples (ie number of microstates)
    d: Dimension of microstate space

    Args:
        activations: (m, d) Raw activation values for all symbols, including <bos> tokens.
        cluster_labels: (m) Cluster labels produced by clustering algorithm for each activation
        tokens: (n, ?) 2-D list of symbols, split by example. Each example may be a different length.
    """
    assert len(activations) == len(cluster_labels)
    assert len(activations) == sum(len(t) for t in tokens)

    macrostates = {
        label: Macrostate(label=label, microstates=set()) for label in set(cluster_labels)
    }
    initial_macrostate: Macrostate | None = None

    offset = 0
    for example in tokens:
        previous_microstate: Microstate | None = None

        for index, token in enumerate(example):
            activation = activations[offset + index]
            cluster_label = cluster_labels[offset + index]
            transition_label = split_transition_label(token)

            if index == 0:
                microstate = Microstate(
                    activation=activation,
                    outgoing_transition=None,
                    incoming_transition=None,
                    macrostate=weakref.ref(macrostates[cluster_label]),
                )
                previous_microstate = microstate
            else:
                # If we aren't at the first microstate, create an incoming transition (and the symmetric outgoing transition)
                assert previous_microstate is not None
                microstate = Microstate(
                    activation=activation,
                    outgoing_transition=None,
                    incoming_transition=(transition_label, weakref.ref(previous_microstate)),
                    macrostate=weakref.ref(macrostates[cluster_label]),
                )
                previous_microstate.outgoing_transition = (transition_label, weakref.ref(microstate))
            
            # Update the macrostate
            macrostates[cluster_label].microstates.add(microstate)
            if initial_macrostate is None:
                initial_macrostate = macrostates[cluster_label]
            if index == len(example) - 1:
                macrostates[cluster_label].is_final = True

    assert initial_macrostate is not None
    return set(macrostates.values()), initial_macrostate