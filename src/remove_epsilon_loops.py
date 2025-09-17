import logging

from pyfoma._private.states import State
from pyfoma.fst import FST

logger = logging.getLogger(__file__)


def remove_epsilon_loops(fst: FST):
    """Removes loops whose inputs consist solely of epsilons."""
    queue = [fst.initialstate]
    visited: set[State] = set()
    while len(queue) > 0:
        state = queue.pop(0)
        visited.add(state)
        next_states = process_state(fst, state)
        next_states -= visited
        queue.extend(list(next_states))


def process_state(fst: FST, state: State):
    next_states: set[State] = set()
    for label, transition_set in list(state.transitions.items()):
        for transition in list(transition_set):
            # Single-state loop (ε:whatever)*
            if label[0] == "" and transition.targetstate is state:
                # Add a new state reached by (ε:whatever) with all of the same outgoing transitions
                new_state = State()
                fst.states.add(new_state)
                transition.targetstate = new_state
                _copy_all_transitions(from_state=state, to_state=new_state)
                if state in fst.finalstates:
                    fst.finalstates.add(new_state)
            elif label[0] == "":
                # Other epsilon transition, which might be the start of a loop
                detect_epsilon_loop(fst, transition.targetstate, epsilon_stack=[state])

            # Recursive DFS
            next_states.add(transition.targetstate)
    return next_states


def detect_epsilon_loop(fst: FST, state: State, epsilon_stack: list[State]):
    logger.debug(
        f"Processing state: {state.name}. Stack has {len(epsilon_stack)} items."
    )
    if state in epsilon_stack:
        # We have a loop!
        new_state = State()
        fst.states.add(new_state)
        _copy_all_transitions(from_state=state, to_state=new_state)
        if state in fst.finalstates:
            fst.finalstates.add(new_state)
        # Connect the most recent state in the stack -> new state
        for label, transition_set in epsilon_stack[-1].transitions.items():
            if label[0] == "":
                for transition in transition_set:
                    if transition.targetstate == state:
                        transition.targetstate = new_state
        return

    for label, transition_set in list(state.transitions.items()):
        if label[0] != "":
            continue
        for transition in list(transition_set):
            detect_epsilon_loop(
                fst, transition.targetstate, epsilon_stack=epsilon_stack + [state]
            )


def _copy_all_transitions(from_state: State, to_state: State):
    for label, transition_set in from_state.transitions.items():
        # Don't add epsilon transitions to the new sink state
        if label[0] == "":
            continue
        for transition in transition_set:
            to_state.add_transition(
                other=transition.targetstate, label=label, weight=transition.weight
            )


if __name__ == "__main__":
    fst = FST()
    s2 = State()
    s3 = State()
    fst.states.add(s2)
    fst.states.add(s3)
    fst.initialstate.add_transition(s2, ("y",))
    s2.add_transition(s3, ("", "x"))
    s3.add_transition(s2, ("", "z"))
    remove_epsilon_loops(fst)
    fst.render()
