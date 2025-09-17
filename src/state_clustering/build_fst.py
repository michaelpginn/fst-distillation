import logging
import weakref
from typing import Literal

import numpy as np
from pyfoma._private.states import State
from pyfoma.fst import FST
from sklearn import linear_model, svm
from sklearn.neighbors import KNeighborsClassifier

from src.state_clustering.types import Macrostate, Macrotransition

logger = logging.getLogger(__file__)


def build_fst(
    initial_macrostate: Macrostate,
    macrostates: dict[str, Macrostate],
    state_splitting_classifier: Literal["svm", "logistic"],
    minimum_transition_count: int | None,
) -> FST:
    """Processes an initial grouping of macrostates,
    adding transitions and splitting states if necessary. Returns an FST."""
    queue: list[Macrostate] = [initial_macrostate]
    visited_labels: set[str] = set()
    while len(queue) > 0:
        print(f"\rItems in queue: {len(queue)}", end="", flush=True)
        # logger.info(f"Queue: {[m.label for m in queue]}")
        current_macrostate = queue.pop(0)
        if current_macrostate.label in visited_labels:
            continue
        visited_labels.add(current_macrostate.label)
        outgoing_distributions = current_macrostate.compute_outgoing_distributions()
        # logger.info(f"Processing state: {current_macrostate.label}")
        # logger.info(f"Outgoing: {outgoing_distributions}")
        chosen_transitions: set[Macrotransition] = set()
        nondeterministic_input_symbols: set[str] = set()
        for input_symbol, distribution in outgoing_distributions.items():
            outputs_sorted = sorted(
                distribution.items(), key=lambda x: x[1], reverse=True
            )
            outputs_over_threshold = [
                output
                for output in outputs_sorted
                if minimum_transition_count and output[1] > minimum_transition_count
            ]
            if len(outputs_over_threshold) <= 1:
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
            current_macrostate.outgoing = chosen_transitions
            for transition in chosen_transitions:
                target = transition.target()
                if target:
                    target.incoming.add(weakref.ref(transition))
                    # ...and continue to target states
                    if target.label not in visited_labels:
                        queue.append(target)
        else:
            # Problems! We have to split
            logger.debug(f"Splitting state: {current_macrostate.label}")
            logger.debug(f"Bad symbols: {outgoing_distributions}")
            # logger.info(f"Distribution: {pprint.pformat(outgoing_distributions)}")
            new_macrostates, macrostates_to_recheck = split_state(
                current_macrostate,
                offending_input_symbols=nondeterministic_input_symbols,
                state_splitting_classifier=state_splitting_classifier,
                minimum_transition_count=minimum_transition_count,
            )
            # logger.info(
            #     f"Split into new macrostates: {[m.label for m in new_macrostates]}"
            # )
            # logger.info(
            #     f"Re-adding to queue: {[m.label for m in macrostates_to_recheck]}"
            # )
            del macrostates[current_macrostate.label]
            for m in new_macrostates:
                assert m.label not in macrostates
                macrostates[m.label] = m
            queue = macrostates_to_recheck + queue
            for m in macrostates_to_recheck:
                if m.label in visited_labels:
                    visited_labels.remove(m.label)

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
                fst_states[macrostate.label].add_transition(
                    other=fst_states[target.label],
                    label=(transition.input_symbol, transition.output_symbol),
                )
            alphabet.update([transition.input_symbol, transition.output_symbol])
    fst = FST()
    fst.states = set(fst_states.values())
    fst.initialstate = fst_states[initial_macrostate.label]
    fst.finalstates = final_states
    fst = fst.filter_accessible()
    logger.info(f"After splitting, FST has {len(fst.states)} states")
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
            if minimum_transition_count and output[1] > minimum_transition_count
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
        logger.warning(
            f"Splitting state {macrostate.label} failed. Falling back to k-NN."
        )
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=3)
        knn.fit(np.stack(states_to_split), labels)
        preds = knn.predict(
            np.stack([microstate.position for microstate in macrostate.microstates])
        )

    # 4. Create n new macrostates for points
    new_macrostates = [
        Macrostate(label=macrostate.label + f"-{index}")
        for index in range(len(outputs_to_split))
    ]
    for microstate, predicted_label in zip(macrostate.microstates, preds):
        new_macrostates[predicted_label].microstates.add(microstate)
        microstate.macrostate = weakref.ref(new_macrostates[predicted_label])

    logger.debug(
        f"New macrostates: {[m.label + ': ' + str(len(m.microstates)) + 'μ-states' for m in new_macrostates]}"
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
