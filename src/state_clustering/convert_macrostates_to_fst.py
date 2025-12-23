import logging
import weakref
from typing import Literal

import numpy as np
from pyfoma.atomic import State
from pyfoma.fst import FST
from sklearn import linear_model, svm

from src.state_clustering.types import Macrostate, Macrotransition, Microstate

logger = logging.getLogger(__file__)


def convert_macrostates_to_fst(
    initial_macrostate: Macrostate,
    macrostates: dict[str, Macrostate],
    state_splitting_classifier: Literal["svm", "logistic"],
    minimum_transition_count: int | None,
    do_merge: bool = True,
    do_minimize: bool = True,
) -> FST:
    """Processes an initial grouping of macrostates,
    adding transitions and splitting states if necessary. Returns an FST."""
    if do_merge:
        initial_macrostate = merge(initial_macrostate, macrostates)

    queue: list[Macrostate] = [initial_macrostate]
    visited_labels: set[str] = set()
    unsplittable_state_labels: set[str] = set()
    while len(queue) > 0:
        current_macrostate = queue.pop(0)
        if current_macrostate.label in visited_labels:
            continue
        visited_labels.add(current_macrostate.label)
        outgoing_distributions = current_macrostate.compute_outgoing_distributions()
        logger.debug(f"Processing state: {current_macrostate.label}")
        chosen_transitions: set[Macrotransition] = set()
        nondeterministic_input_symbols: set[str] = set()

        for input_symbol, distribution in outgoing_distributions.items():
            outputs_sorted = sorted(
                distribution.items(), key=lambda x: x[1], reverse=True
            )
            outputs_over_threshold = [
                output
                for output in outputs_sorted
                if minimum_transition_count and output[1] >= minimum_transition_count
            ]
            if (
                len(outputs_over_threshold) <= 1
                or current_macrostate.label in unsplittable_state_labels
            ):
                # If we don't have any over threshold, use the most common
                output_symbol, target_state_label = outputs_sorted[0][0]
                chosen_transitions.add(
                    Macrotransition(
                        input_symbol=input_symbol,
                        output_symbol=output_symbol,
                        source=weakref.ref(current_macrostate),
                        target=weakref.ref(macrostates[target_state_label]),
                    )
                )
            else:
                nondeterministic_input_symbols.add(input_symbol)

        if len(nondeterministic_input_symbols) == 0:
            # No problems! Assign macrotransitions...
            logger.debug(f"Assigning transitions to m-state {current_macrostate.label}")
            current_macrostate.outgoing = chosen_transitions
            for transition in chosen_transitions:
                target = transition.target()
                if target:
                    target.incoming.add(weakref.ref(transition))
                    # ...and continue to target states
                    if target.label not in visited_labels:
                        queue.append(target)
                else:
                    breakpoint()
        else:
            # Problems! We have to split
            logger.debug(f"Splitting state: {current_macrostate.label}")
            logger.debug(f"Bad symbols: {outgoing_distributions}")
            # logger.info(f"Distribution: {pprint.pformat(outgoing_distributions)}")
            try:
                new_macrostates, macrostates_to_recheck = split_state(
                    current_macrostate,
                    offending_input_symbols=nondeterministic_input_symbols,
                    state_splitting_classifier=state_splitting_classifier,
                    minimum_transition_count=minimum_transition_count,
                )
                del macrostates[current_macrostate.label]
                for m in new_macrostates:
                    assert m.label not in macrostates
                    macrostates[m.label] = m
                queue = macrostates_to_recheck + new_macrostates + queue
                for m in macrostates_to_recheck:
                    if m.label in visited_labels:
                        visited_labels.remove(m.label)
            except Exception:
                # Splitting failed, just re-add to queeu
                unsplittable_state_labels.add(current_macrostate.label)
                visited_labels.remove(current_macrostate.label)
                queue = [current_macrostate] + queue

    # generalize_transitions(macrostates)

    # Finally, build the actual FST
    fst_states: dict[str, State] = dict()
    final_states: set[State] = set()
    for macrostate in macrostates.values():
        new_state = State(name=macrostate.label)
        fst_states[macrostate.label] = new_state
        # TODO: May want to make this a threshold
        if any(m.is_final for m in macrostate.microstates):
            new_state.finalweight = 0
            final_states.add(new_state)
    alphabet: set[str] = set()
    for macrostate in macrostates.values():
        for transition in macrostate.outgoing:
            if (target := transition.target()) is not None:
                if transition.input_symbol == transition.output_symbol:
                    label = (transition.input_symbol,)
                else:
                    label = (transition.input_symbol, transition.output_symbol)
                fst_states[macrostate.label].add_transition(
                    other=fst_states[target.label],
                    label=label,
                    weight=1,
                )
            alphabet.update([transition.input_symbol, transition.output_symbol])
    fst = FST()
    fst.states = set(fst_states.values())
    fst.initialstate = fst_states[initial_macrostate.label]
    fst.finalstates = final_states
    fst = fst.filter_accessible()
    logger.info(f"After splitting, FST has {len(fst.states)} states")
    logger.info("Minimizing and determinizing")
    # TODO: Add back epsilon loop thing
    if do_minimize:
        fst = fst.minimize()
    logger.info(f"Created FST with {len(fst.states)} states")
    return fst


def split_state(
    macrostate: Macrostate,
    offending_input_symbols: set[str],
    state_splitting_classifier: Literal["svm", "logistic"],
    minimum_transition_count: int | None,
):
    """Splits a state, possibly recursively.

    Args:
        macrostate: The Macrostate to consider for splitting.
        offending_input_symbols: Set of input symbols whose outgoing
            distributions from this macrostate are considered nondeterministic.
        minimum_transition_count: Minimum count for a transition to be included

    Returns `tuple[list[Macrostates], list[Macrostates]]`, consisting of
        1. The new Macrostates created by splitting
        2. Macrostates that need to re-compute transitions
    """
    outgoing_distributions = macrostate.compute_outgoing_distributions()

    # 1. Find worst input symbol (by entropy)
    most_over_threshold_input_symbol = (0, None, None)
    for symbol in offending_input_symbols:
        outputs_sorted = sorted(
            outgoing_distributions[symbol].items(), key=lambda x: x[1], reverse=True
        )
        outputs_over_threshold = [
            output[0]
            for output in outputs_sorted
            if minimum_transition_count and output[1] >= minimum_transition_count
        ]
        if len(outputs_over_threshold) > most_over_threshold_input_symbol[0]:
            most_over_threshold_input_symbol = (
                len(outputs_over_threshold),
                symbol,
                outputs_over_threshold,
            )
    logger.debug(f"Splitting on {most_over_threshold_input_symbol}")
    _, input_symbol_to_split, outputs_to_split = most_over_threshold_input_symbol
    assert input_symbol_to_split is not None
    assert outputs_to_split is not None

    # 2. Find microstates corresponding to each selected output
    states_to_split: list[np.ndarray] = []
    labels = []
    for microstate in macrostate.microstates:
        if (
            microstate.outgoing is not None
            and (outgoing := microstate.outgoing()) is not None
            and (target_state := outgoing.target()) is not None
            and target_state.macrostate is not None
            and (target_macrostate := target_state.macrostate())
            and outgoing.input_symbol == most_over_threshold_input_symbol[1]
            and (output_key := (outgoing.output_symbol, target_macrostate.label))
            in outputs_to_split
        ):
            states_to_split.append(microstate.position)
            label = outputs_to_split.index(output_key)
            labels.append(label)

    # 3. SVM
    # print(f"Labels to split: {Counter(labels)}")
    if state_splitting_classifier == "svm":
        clf = svm.LinearSVC(class_weight="balanced")
    else:
        clf = linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)

    clf = clf.fit(np.stack(states_to_split), labels)
    preds = clf.predict(
        np.stack([microstate.position for microstate in macrostate.microstates])
    )
    # print(f"Scores: {clf.score(states_to_split, labels)}")
    # print(f"Preds: {Counter(preds)}")
    if len(set(preds)) == 1:
        logger.warning("State is unsplittable")
        raise Exception("Unsplittable state")
        # knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
        # knn.fit(np.stack(states_to_split), labels)
        # preds = knn.predict(
        #     np.stack([microstate.position for microstate in macrostate.microstates])
        # )
        # if len(set(preds)) == 1:
        #     logger.warning("Unsplittable even with knn")
        #     raise Exception("Unsplittable state")

    # 4. Create n new macrostates for points
    new_macrostates = [
        Macrostate(label=macrostate.label + f"-{index}")
        for index in range(len(outputs_to_split))
    ]
    for microstate, predicted_label in zip(macrostate.microstates, preds):
        new_macrostates[predicted_label].microstates.add(microstate)
        microstate.macrostate = weakref.ref(new_macrostates[predicted_label])

    logger.debug(
        f"New macrostates: {[m.label + ': ' + str(len(m.microstates)) + ' μ-states' for m in new_macrostates]}"
    )

    # 5. Remove all outgoing from this macrostate, so that we don't accidentally reference it later
    for outgoing in macrostate.outgoing:
        if (target := outgoing.target()) is not None:
            target.incoming = {
                transition
                for transition in target.incoming
                if (t := transition()) and t.source() != macrostate
            }

    # 6. For all previous states to current state, check recursively if needs split
    macrostates_to_recheck: list[Macrostate] = []
    for incoming in macrostate.incoming:
        if (incoming := incoming()) is not None and (
            source := incoming.source()
        ) is not None:
            source.outgoing = {
                transition
                for transition in source.outgoing
                if transition.target() != macrostate
            }
            if not any(m.label == source.label for m in macrostates_to_recheck):
                macrostates_to_recheck.append(source)
    return new_macrostates, macrostates_to_recheck


def merge(initial_macrostate: Macrostate, macrostates: dict[str, Macrostate]):
    """Tries to merge states to resolve non-onward transitions"""
    queue: list[Macrostate] = [initial_macrostate]
    visited_labels: set[str] = set()
    new_initial_macrostate = initial_macrostate
    while len(queue) > 0:
        current_macrostate = queue.pop(0)
        if current_macrostate.label in visited_labels:
            continue
        visited_labels.add(current_macrostate.label)
        for input_symbol in current_macrostate.compute_outgoing_distributions().keys():
            # Recompute each time since it may have changed
            outputs = current_macrostate.compute_outgoing_distributions()[
                input_symbol
            ].items()

            # If we have multiple output symbols, we can't merge, so skip instead
            output_symbols = set([o[0][0] for o in outputs])
            if len(outputs) <= 1 or len(output_symbols) > 1:
                for (_, target), _ in outputs:
                    if target not in visited_labels:
                        queue.append(macrostates[target])
                continue

            # If not, we can try merging the downstream states
            downstream_states = [macrostates[o[0][1]] for o in outputs]
            merged_state = perform_merge(downstream_states)
            macrostates[merged_state.label] = merged_state
            for s in downstream_states:
                if s.label == new_initial_macrostate.label:
                    new_initial_macrostate = merged_state
                if s in queue:
                    queue.remove(s)
                del macrostates[s.label]
            queue.append(merged_state)

    return new_initial_macrostate


def perform_merge(states: list[Macrostate]) -> Macrostate:
    assert len(states) > 1
    label = "[" + "+".join([s.label for s in states]) + "]"
    logger.debug(f"Merging {[s.label for s in states]}")

    microstates: set[Microstate] = set()
    incoming: set[weakref.ReferenceType[Macrotransition]] = set()
    outgoing: set[Macrotransition] = set()

    for s in states:
        microstates.update(s.microstates)
        incoming.update(s.incoming)
        outgoing.update(s.outgoing)

    new_macrostate = Macrostate(
        label=label,
        microstates=microstates,
        incoming=incoming,
        outgoing=outgoing,
    )
    for m in new_macrostate.microstates:
        m.macrostate = weakref.ref(new_macrostate)
    for t in new_macrostate.incoming:
        if (t := t()) is not None:
            t.target = weakref.ref(new_macrostate)
    for t in new_macrostate.outgoing:
        t.source = weakref.ref(new_macrostate)

    return new_macrostate
