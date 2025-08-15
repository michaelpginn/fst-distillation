"""Usage: python -m exp2-clustering.extract_fst <checkpoint path> <train dataset path>

Runs the Giles (1991) clustering algorithm to produce an FST from a trained RNN.

You should run `train_rnn.py` first to train a model and produce a checkpoint.
"""

import argparse
import logging
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
import torch
from pyfoma._private.states import State
from pyfoma.fst import FST
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.learn import standard_scale
from src.modeling.rnn import RNNModel
from src.state_clustering.build_macrostates import build_macrostates
from src.state_clustering.types import Macrostate, Microstate
from src.tasks.inflection_classification.dataset import load_examples_from_file
from src.tasks.inflection_classification.tokenizer import AlignedInflectionTokenizer
from src.training_classifier.train import device

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ExtractionHyperparameters:
    num_initial_clusters: int
    pca_components: int = 30

    # Only one of the following should be set
    transitions_top_k: int | None = None
    """For each (start state, input symbol), take the top k most common transitions"""
    transitions_top_p: float | None = None
    """For each (start state, input symbol), take transitions within the top p% of transitions"""
    transitions_min_n: int | None = None
    """For each (start state, input symbol), take transitions if they occur more than n times"""

    def __post_init__(self):
        modes_set = (
            (1 if self.transitions_top_k is not None else 0)
            + (1 if self.transitions_top_p is not None else 0)
            + (1 if self.transitions_min_n is not None else 0)
        )
        if modes_set != 1:
            raise ValueError("Must set exactly one transition mode!")




def extract_fst(
    hyperparams: ExtractionHyperparameters,
    language: str,
    model_id: str,
    visualize: bool = False,
):
    # Make paths
    model_path = Path(__file__).parent / f"runs/{model_id}/model.pt"
    train_path = (
        pathlib.Path(__file__).parent / f"aligned_data/{language}.trn.aligned.jsonl"
    )
    eval_path = (
        pathlib.Path(__file__).parent / f"aligned_data/{language}.dev.aligned.jsonl"
    )

    # Load stuff
    checkpoint_dict = torch.load(model_path, weights_only=True)
    tokenizer = AlignedInflectionTokenizer.from_state_dict(
        checkpoint_dict["tokenizer_dict"]
    )
    model = RNNModel.load(checkpoint_dict, tokenizer)
    model.to(device)
    model.eval()
    train_examples = load_examples_from_file(train_path)
    eval_examples = load_examples_from_file(eval_path)

    # 1. For each training example, collect activations
    activations = []
    for example in tqdm(train_examples, "Computing hidden states"):
        inputs = model.tokenizer.tokenize(example)
        hidden_states, _ = model.rnn(
            model.embedding(torch.tensor(inputs["input_ids"]).to(device))
        )
        activations.append(hidden_states)
    activations = torch.concat(activations)

    # 2. Perform standardization -> dim reduction -> clustering
    # activations = StandardScaler().fit_transform(activations)
    activations = standard_scale(activations).cpu().detach().numpy()
    if hyperparams.pca_components < activations.shape[-1]:
        pca = PCA(n_components=hyperparams.pca_components, whiten=True)
        activations = pca.fit_transform(activations)
    _, labels, _ = k_means(activations, n_clusters=hyperparams.num_initial_clusters)  # type:ignore
    assert labels is not None

    if visualize:
        pca_data = pandas.DataFrame(activations[:, :2], columns=["PC1", "PC2"])  # type:ignore
        pca_data["cluster"] = pandas.Categorical(labels)
        seaborn.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
        plt.show()

    # 3. Collect macrostates and transitions
    tokens = [
        cast(
            list[str], 
            model.tokenizer.decode(
                model.tokenizer.tokenize(example)["input_ids"],  # type:ignore
                skip_special_tokens=False,
                return_as="list",
            )
        )
        for example in train_examples
    ]
    macrostates, initial_state = build_macrostates(
        activations=activations,
        cluster_labels=labels,
        tokens=tokens,
    )



    # fst = FST()
    # state_lookup = {
    #     f"cluster-{label}": State(name=f"cluster-{label}") for label in set(labels)
    # }
    # fst.states = set(state_lookup.values())
    # fst.initialstate = state_lookup[
    #     f"cluster-{labels[0]}"
    # ]  # Use the label of the <bos> cluster



    # 4. Use the original inputs to produce a counter of transitions between each pair of states
    #
    # Notably, we will reduce so that each state has some number n of transitions associated with a given input label
    # That means that the resulting FST is guaranteed deterministic (which it should be realistically)
    @dataclass(frozen=True)
    class _TransitionKey:
        start_state_label: str
        input_symbol: str

    @dataclass(frozen=True)
    class _TransitionValue:
        end_state_label: str
        output_symbol: str

    transition_counts: defaultdict[_TransitionKey, Counter[_TransitionValue]] = (
        defaultdict(lambda: Counter())
    )
    final_state_labels: set[str] = set()
    offset = 0
    for example in tqdm(train_examples, "Collecting transitions"):
        # Either a feature label ("[PST]"), special char, or (input:output) pair
        transition_labels_as_list: list[str] = model.tokenizer.decode(
            model.tokenizer.tokenize(example)["input_ids"],  # type:ignore
            skip_special_tokens=False,
            return_as="list",
        )
        for symbol_index in range(len(transition_labels_as_list) - 1):
            start_state_label = f"cluster-{labels[offset + symbol_index]}"
            end_state_label = f"cluster-{labels[offset + symbol_index + 1]}"
            transition_label = transition_labels_as_list[symbol_index + 1]

            if match := re.match(r"\((.*),(.*)\)", transition_label):
                input_symbol = match.group(1)
                output_symbol = match.group(2)
            else:
                input_symbol = transition_label
                output_symbol = transition_label

            transition_counts[
                _TransitionKey(
                    start_state_label=start_state_label, input_symbol=input_symbol
                )
            ].update(
                [
                    _TransitionValue(
                        end_state_label=end_state_label, output_symbol=output_symbol
                    )
                ]
            )
        final_state_labels.add(
            f"cluster-{labels[offset + len(transition_labels_as_list) - 1]}"
        )
        offset += len(transition_labels_as_list)

    # 5. Use the transition reduction heuristic to reduce the number of transitions and thus produce the final FST
    #
    # FIXME: for now, I use "all transitions"
    # FIXME: Probably want pick based on unique input symbols, not pairs
    # We should implement "k most common", "threshold", etc
    for transition_key, counter in tqdm(
        transition_counts.items(), "Creating transitions"
    ):
        if top_k := hyperparams.transitions_top_k:
            chosen_transitions = [k for k, v in counter.most_common(top_k)]
        elif top_p := hyperparams.transitions_top_p:
            # Compute the highest prob transitions s.t. cumprob > top_p
            total_count = sum(counter.values())
            probs = [(k, v / total_count) for k, v in counter.items()]
            probs_sorted = sorted(probs, key=lambda t: t[1], reverse=True)
            cum_prob = 0
            chosen_transitions: list[_TransitionValue] = []
            while cum_prob < top_p:
                next_transition, prob = probs_sorted.pop(0)
                chosen_transitions.append(next_transition)
                cum_prob += prob
        elif min_n := hyperparams.transitions_min_n:
            chosen_transitions = [k for k, v in counter.items() if v >= min_n]
        else:
            raise ValueError()

        for transition_value in chosen_transitions:
            fst.alphabet.update(
                {transition_key.input_symbol, transition_value.output_symbol}
            )
            if transition_key.input_symbol == transition_value.output_symbol:
                label = (transition_key.input_symbol,)
            else:
                label = (transition_key.input_symbol, transition_value.output_symbol)
            start_state = state_lookup[transition_key.start_state_label]
            end_state = state_lookup[transition_value.end_state_label]
            start_state.add_transition(end_state, label=label, weight=0.0)

    fst.finalstates = {state_lookup[label] for label in final_state_labels}
    for state in fst.states:
        state.finalweight = 0.0

    # 6. Compose with space insertion FST for alignment
    # FIXME: We don't need to insert spaces by tags
    logger.info("Composing with space inserter")
    space_inserter = FST.regex("('':' ')*(. ('':' ')*)*")
    # fst = space_inserter @ fst

    logger.info("Minimizing and determinizing")
    fst = fst.filter_accessible().determinize().minimize()

    logger.info(f"Created FST with {len(fst.states)} states")

    # 7. Evaluate on the dev set
    accepted_input = 0  # Proportion where the transducer accepts the input
    correct_count = 0  # Number of examples where correct output is produced
    average_num_generations = 0  # Number of total generations per input
    matched_prefix_length = 0  # Average length of matched prefix
    for example in tqdm(eval_examples, "Evaluating"):
        input_string = (
            example.features + ["<sep>"] + [c[0] for c in example.aligned_chars]
        )
        correct_output = (
            example.features + ["<sep>"] + [c[1] for c in example.aligned_chars]
        )
        correct_output = "".join(correct_output)
        generated_outputs = list(fst.generate(input_string))

        if len(generated_outputs) > 0:
            accepted_input += 1
            if correct_output in generated_outputs:
                correct_count += 1
            average_num_generations += len(generated_outputs)
            # TODO: Why are we getting multiple outputs? Shouldn't it be deterministic?

    accepted = accepted_input / len(eval_examples)
    correct = correct_count / len(eval_examples)
    average_gens = average_num_generations / accepted_input if accepted_input > 0 else 0
    logger.info(f"""Stats:
Accepted: {accepted:.2%}
Correct: {correct:.2%}
Average num generations: {average_gens:.2}/input""")

    if visualize:
        logger.info("Rendering")
        fst.render(view=True, filename="fst")

    return {"accepted": accepted, "correct": correct, "average_num_gens": average_gens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="WandB ID for the training run")
    parser.add_argument("--language", default="swe", help="Isocode for the language")
    args = parser.parse_args()
    extract_fst(
        hyperparams=ExtractionHyperparameters(
            num_initial_clusters=1000,
            transitions_top_k=None,
            transitions_top_p=0.1,
        ),
        language=args.language,
        model_id=args.model_id,
    )
