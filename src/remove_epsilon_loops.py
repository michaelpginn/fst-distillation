from pyfoma._private.states import State
from pyfoma.fst import FST


def remove_epsilon_loops(fst: FST):
    """Removes loops whose inputs consist solely of epsilons."""
    process_state(fst.initialstate, visited=set())


def process_state(state: State, visited: set[State]):
    if state in visited:
        return

    visited.add(state)

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
            process_state(transition.targetstate, visited=visited)


def detect_epsilon_loop(fst: FST, state: State, epsilon_stack: list[State]):
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
                    breakpoint()
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
