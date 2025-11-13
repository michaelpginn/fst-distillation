import re
import weakref
from dataclasses import dataclass, field
from typing import Literal

from src.state_clustering.types import Macrostate, Macrotransition


def generalize_transitions(macrostates: dict[str, Macrostate]):
    """Given a dict of macrostates, performs *generalization* to recover
    from inaccessible transitions caused by splitting.

    Specifically:

    1. We look for states that have been split in the format 'state-i-j-...' and
       builds a split history tree.

    2. Starting at the leaves, for each input symbol x...
        a. If only one node in the tree has a transition for x, copy that transition to all other nodes
        b. If multiple nodes have identical transitions for x, copy that transition to all other nodes
        c. Else (conflicting transitions), leave as is.

    3. After resolving all children of a node, assign the combined transition dict to the parent.

    4. Move up a level and repeat until reaching the root.
    """

    # 1. Build trees
    nodes: dict[str, StateNode] = {}
    root_node_labels: set[str] = set()
    for m_state_label in macrostates:
        prior_label = None
        for match in re.finditer(r"\-\d+", m_state_label):
            label = m_state_label[: match.end()]
            if label not in nodes:
                # Create a node
                if label == m_state_label:
                    # Leaf
                    nodes[label] = StateNode(
                        is_macrosate=True,
                        label=label,
                        transitions={
                            t.input_symbol: (t.output_symbol, t.target().label)  # type:ignore
                            for t in macrostates[m_state_label].outgoing
                        },
                    )
                else:
                    # Internal
                    nodes[label] = StateNode(
                        is_macrosate=False, label=label, transitions={}
                    )
                if prior_label is not None:
                    nodes[prior_label].children.append(nodes[label])
                else:
                    root_node_labels.add(label)
            prior_label = label

    # 2. Recursively process nodes
    def _generalize_transitions(node: StateNode):
        if len(node.children) == 0:
            return
        transitions: TransitionsDict = {}

        # Aggregate transitions from children
        for child in node.children:
            _generalize_transitions(child)

            for t_in, t_out in child.transitions.items():
                if t_in in transitions:
                    if transitions[t_in] != t_out:
                        # Conflicting transitions between children
                        transitions[t_in] = "mixed"
                else:
                    transitions[t_in] = t_out

        node.transitions = transitions

        # For non-mixed transitions, copy to all children that are leaves
        # (at this point we don't care about the child StateNode, since we will never use it thereafter)
        for t_in, t_out in transitions.items():
            if t_out == "mixed":
                continue
            for child in node.children:
                if not child.is_macrosate or t_in in child.transitions:
                    continue
                (output_symbol, target_label) = t_out
                new_transition = Macrotransition(
                    input_symbol=t_in,
                    output_symbol=output_symbol,
                    source=weakref.ref(macrostates[child.label]),
                    target=weakref.ref(macrostates[target_label]),
                )
                macrostates[child.label].outgoing.add(new_transition)
                macrostates[target_label].incoming.add(weakref.ref(new_transition))

    for label in root_node_labels:
        _generalize_transitions(nodes[label])


TransitionsDict = dict[str, tuple[str, str] | Literal["mixed"]]


@dataclass
class StateNode:
    is_macrosate: bool
    label: str
    transitions: TransitionsDict  # in: (out, target)
    children: list["StateNode"] = field(default_factory=list)
