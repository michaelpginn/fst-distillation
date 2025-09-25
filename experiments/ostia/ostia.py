from collections import defaultdict

from pyfoma.atomic import State, Transition
from pyfoma.fst import FST


def prefix(word: str):
    return [word[:i] for i in range(len(word) + 1)]


def lcp(strs: list[str]):
    """Longest common prefix"""
    assert len(strs) > 0
    prefix = ""
    for i in range(min(len(s) for s in strs)):
        if len(set(s[i] for s in strs)) == 1:
            prefix += strs[0][i]
        else:
            break
    return prefix


def build_prefix_tree(samples: list[tuple[str, str]]):
    """Builds a tree sequential transducer (prefix tree),
    where outputs are delayed until the word-final symbol (#)."""

    # Create states for each prefix in the inputs
    state_labels: set[str] = set()
    alphabet = set(["#"])
    for sample in samples:
        if "#" in sample[0]:
            raise ValueError("Inputs should not contain reserved `#` symbol")
        state_labels.update(prefix(sample[0] + "#"))
        alphabet.update(sample[0] + sample[1])
    states = {label: State(name=label) for label in state_labels}

    # Create transitions to form prefix tree
    for label, state in states.items():
        if len(label) >= 1:
            prior_state = states[label[:-1]]
            # Determine the output: empty string unless final state
            if label.endswith("#"):
                state.finalweight = 0
                outputs = [output for input, output in samples if input == label[:-1]]
                if len(outputs) > 1:
                    raise ValueError(
                        f"Provided samples are ambiguous for input {label[:-1]}!!"
                    )
                transition_output = outputs[0]
            else:
                transition_output = ""
            prior_state.add_transition(state, label=(label[-1], transition_output))

    fst = FST(alphabet=alphabet)
    fst.states = set(states.values())
    fst.initialstate = states[""]
    fst.finalstates = {s for label, s in states.items() if label.endswith("#")}
    return fst


def convert_to_otst(fst: FST):
    """Converts a tree sequential transducer into an onward tree sequential transducer (OTST)."""

    def process_state(state: State):
        if len(state.transitions) == 0:
            return ""

        new_transitions: list[Transition] = []
        for transitions in state.transitions.values():
            for transition in transitions:
                in_label, out_label = transition.label
                downstream_prefix = process_state(transition.targetstate)
                new_out_label = out_label + downstream_prefix
                transition.label = (in_label, new_out_label)
                new_transitions.append(transition)

        # Find and remove the common prefix
        transition_outputs: set[str] = {
            transition.label[1] for transition in new_transitions
        }
        common_prefix = lcp(list(transition_outputs))
        updated_transition_dict = defaultdict(lambda: set())
        for transition in new_transitions:
            new_label = (
                transition.label[0],
                transition.label[1].removeprefix(common_prefix),
            )
            transition.label = new_label
            updated_transition_dict[new_label].add(transition)
        state.transitions = updated_transition_dict
        return common_prefix

    process_state(fst.initialstate)
    return fst


def dedupe_transitions(state: State) -> State:
    new_transitions_dict = defaultdict(set)
    for label, transitions in state.transitions.items():
        for transition in transitions:
            if not any(
                t.targetstate == transition.targetstate
                for t in new_transitions_dict[label]
            ):
                new_transitions_dict[label].add(transition)
    state.transitions = new_transitions_dict
    return state


def merge(fst: FST, p: State, q: State) -> FST:
    """Merges state q into state p"""
    # Any incoming edges to q will now go to p
    needs_deduping = set()
    for state in fst.states:
        for _, transition in state.all_transitions():
            if transition.targetstate == q:
                transition.targetstate = p
                needs_deduping.add(state)
    for state in needs_deduping:
        dedupe_transitions(state)

    # Copy all outgoing from q to p
    for _, transition in q.all_transitions():
        p.add_transition(transition.targetstate, transition.label)
    p = dedupe_transitions(p)

    # Clean up
    if q is fst.initialstate:
        fst.initialstate = p
    if q in fst.finalstates:
        fst.finalstates.add(p)
        fst.finalstates.remove(q)
    fst.states.remove(q)
    del q
    return fst


def subseq_violations(fst: FST) -> tuple[State, tuple[Transition, Transition]] | None:
    """Returns None if the transducer is subsequential, and a tuple of (source state, two edges) that violate the determinism condition if it is not subsequential"""
    for state in fst.states:
        state._transitionsin = None
        for _, transitions in state.transitionsin.items():
            if len(transitions) > 1:
                violating_transitions = list(t[1] for t in transitions)
                violating_transitions = sorted(
                    violating_transitions, key=lambda t: t.targetstate.name
                )
                return (state, tuple(violating_transitions[:2]))
    return None


def push_back(fst: FST, suffix: str, incoming: Transition, source: State) -> FST:
    """Removes a (output-side) suffix from the incoming edge and
    preprends it to all outgoing edges"""
    old_label = incoming.label
    new_label = (incoming.label[0], incoming.label[1].removesuffix(suffix))
    incoming.label = new_label
    source.transitions[old_label].remove(incoming)
    source.transitions[new_label].add(incoming)
    dedupe_transitions(source)

    new_transition_dict = defaultdict(set)
    for label, transition in incoming.targetstate.all_transitions():
        new_label = (label[0], suffix + label[1])
        transition.label = new_label
        new_transition_dict[new_label].add(transition)
    incoming.targetstate.transitions = new_transition_dict
    incoming.targetstate = dedupe_transitions(incoming.targetstate)
    return fst


def ostia(samples: list[tuple[str, str]]):
    T = build_prefix_tree(samples)
    T = convert_to_otst(T)

    def next_state(fst: FST, state: State):
        states_sorted = sorted(fst.states, key=lambda s: s.name or "")
        for s in states_sorted:
            if state.name < s.name:  # type:ignore
                return s
        return None

    def first_state(fst: FST):
        return sorted(fst.states, key=lambda s: s.name or "")[0]

    def last_state(fst: FST):
        return sorted(fst.states, key=lambda s: s.name or "")[-1]

    # Take states in lexicographic order
    q = first_state(T)
    while (q.name or "") < (last_state(T).name or ""):
        q = next_state(T, q)
        if not q:
            break

        # Find p < q where q can merge into p
        p = first_state(T)
        while (p.name or "") < (q.name or ""):
            if p is None:
                break
            T_bar = T.__copy__()
            print(f"==============Trying merge '{q.name}' -> '{p.name}'==============")
            T = merge(T, p, q)
            # breakpoint()
            while (violations := subseq_violations(T)) is not None:
                source_state, violating_edges = violations
                a, v = violating_edges[0].label
                s = violating_edges[0].targetstate
                w = violating_edges[1].label[1]
                t = violating_edges[1].targetstate
                if ((v != w) and (a == "#")) or (
                    (s.name or "") < (q.name or "") and v not in prefix(w)
                ):
                    break
                u = lcp([v, w])
                T = push_back(T, v.removeprefix(u), violating_edges[0], source_state)
                T = push_back(T, w.removeprefix(u), violating_edges[1], source_state)
                T = merge(T, s, t)
            # If T is subsequent, we're good to go to the next merge
            # If not, revert
            if subseq_violations(T) is None:
                print("Merged")
                break
            else:
                print("Aborting merge")
                T = T_bar
                q = [s for s in T.states if s.name == q.name][0]
                p = next_state(T, p)
    return T


ostia(
    [("", ""), ("a", "bb"), ("aa", "bbc"), ("aaa", "bbbb"), ("aaaa", "bbbbc")]
).render()
