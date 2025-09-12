"""Usage: python -m exp2-clustering.extract_fst <checkpoint path> <train dataset path>

Runs the Giles (1991) clustering algorithm to produce an FST from a trained RNN.

You should run `train_rnn.py` first to train a model and produce a checkpoint.
"""

import argparse
import logging
import pprint
import re
import warnings
import weakref
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas
import seaborn
import torch
from hdbscan import HDBSCAN
from pyfoma.fst import FST
from sklearn.cluster import OPTICS, k_means
from sklearn.decomposition import PCA
from tqdm import tqdm

from exp2_clustering.util import find_data_file
from src.learn import standard_scale
from src.modeling.rnn import RNNModel
from src.remove_epsilon_loops import remove_epsilon_loops
from src.state_clustering.build_fst import build_fst
from src.state_clustering.hopkins import hopkins
from src.state_clustering.types import (
    Macrostate,
    Microstate,
    Microtransition,
)
from src.tasks.inflection_classification.dataset import load_examples_from_file
from src.tasks.inflection_classification.tokenizer import AlignedInflectionTokenizer
from src.tasks.inflection_seq2seq.dataset import (
    load_examples_from_file as load_unaligned,
)
from src.tasks.inflection_seq2seq.example import InflectionExample
from src.training_classifier.train import device

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class ExtractionHyperparameters:
    clustering_method: Literal["kmeans", "optics", "hdbscan"]
    kmeans_num_clusters: int | None = None
    min_samples: int | None = None

    pca_components: int | None = None

    # Only one of the following should be set
    transitions_top_k: int | None = None
    """For each (start state, input symbol), take the top k most common transitions"""
    transitions_top_p: float | None = None
    """For each (start state, input symbol), take transitions within the top p% of transitions"""
    transitions_min_n: int | None = None
    """For each (start state, input symbol), take transitions if they occur more than n times"""

    minimum_transition_count: int = 100
    """Minimum count for a given transition to be guaranteed to be included"""

    generations_top_k: int = 1
    """How many generations to pick, in increasing length order"""

    def __post_init__(self):
        modes_set = (
            (1 if self.transitions_top_k is not None else 0)
            + (1 if self.transitions_top_p is not None else 0)
            + (1 if self.transitions_min_n is not None else 0)
        )
        if modes_set != 1:
            raise ValueError("Must set exactly one transition mode!")
        if self.clustering_method == "kmeans" and self.kmeans_num_clusters is None:
            raise ValueError("Must set `kmeans_num_clusters` when using k-means!")
        if self.clustering_method == "optics" and (self.min_samples is None):
            raise ValueError("Must set `optics_min_pts` when using OPTICS!")


def extract_fst(
    hyperparams: ExtractionHyperparameters,
    aligned_train_path: PathLike,
    eval_path: PathLike,
    test_path: PathLike,
    model_id: str,
    visualize: bool = False,
):
    model_path = Path(__file__).parent.parent / f"runs/{model_id}/model.pt"
    checkpoint_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    tokenizer = AlignedInflectionTokenizer.from_state_dict(
        checkpoint_dict["tokenizer_dict"]
    )
    model = RNNModel.load(checkpoint_dict, tokenizer)
    model.to(device)
    model.eval()
    train_examples = load_examples_from_file(aligned_train_path)
    eval_examples = load_unaligned(eval_path)
    test_examples = load_unaligned(test_path)

    # 1. For each training example, collect activations
    activations = []
    for example in tqdm(train_examples, "Computing hidden states"):
        inputs = model.tokenizer.tokenize(example)
        hidden_states, _ = model.rnn(
            model.embedding(torch.tensor(inputs["input_ids"]).to(device))
        )
        activations.append(hidden_states.cpu().detach())
    activations = torch.concat(activations)

    # 2. Perform standardization -> dim reduction -> clustering
    logger.info("Standardizing...")
    activations = standard_scale(activations).numpy()
    if (
        hyperparams.pca_components
        and hyperparams.pca_components < activations.shape[-1]
    ):
        logger.info(f"Reducing dimensionality (PC = {hyperparams.pca_components})...")
        pca = PCA(n_components=hyperparams.pca_components, whiten=True)
        activations = pca.fit_transform(activations)

    hopkins_stat = hopkins(activations)
    logger.info(f"Hopkins statistic: {hopkins_stat}")

    logger.info(f"Clustering with '{hyperparams.clustering_method}'")
    if hyperparams.clustering_method == "kmeans":
        _, labels, _ = k_means(activations, n_clusters=hyperparams.kmeans_num_clusters)  # type:ignore
    elif hyperparams.clustering_method == "optics":
        labels = OPTICS(
            min_samples=hyperparams.min_samples,  # type:ignore
            n_jobs=-1,
        ).fit_predict(activations)
    elif hyperparams.clustering_method == "hdbscan":
        labels = HDBSCAN(
            min_cluster_size=50,
            min_samples=hyperparams.min_samples,
            core_dist_n_jobs=-1,
        ).fit_predict(activations)
    else:
        raise ValueError()

    assert labels is not None

    if visualize:
        logger.info("Visualizing principle components")
        pca_data = pandas.DataFrame(activations[:, :2], columns=["PC1", "PC2"])  # type:ignore
        pca_data["cluster"] = pandas.Categorical(labels)
        seaborn.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
        plt.show()

    # 3. Create states
    microstates: list[Microstate] = []
    macrostates: dict[str, Macrostate] = {
        f"cluster-{label}": Macrostate(label=f"cluster-{label}")
        for label in set(labels)
    }
    initial_macrostate = macrostates[f"cluster-{labels[0]}"]
    offset = 0
    for example in tqdm(train_examples, "Collecting microstates"):
        # <bos> [TAG1] [TAG2] ... (c:c) (c:c) ...
        transition_labels_as_list: list[str] = model.tokenizer.decode(
            model.tokenizer.tokenize(example)["input_ids"],  # type:ignore
            skip_special_tokens=False,
            return_as="list",
        )
        previous_microstate: Microstate | None = None
        for symbol_index, transition_label in enumerate(transition_labels_as_list):
            microstate = Microstate(
                position=activations[offset + symbol_index],
                is_final=symbol_index == len(transition_labels_as_list) - 1,
            )
            # Add incoming transition
            if symbol_index != 0:
                assert previous_microstate is not None
                if match := re.match(r"\((.*),(.*)\)", transition_label):
                    input_symbol = match.group(1)
                    output_symbol = match.group(2)
                else:
                    input_symbol = transition_label
                    output_symbol = transition_label
                input_symbol = "" if input_symbol == "~" else input_symbol
                output_symbol = "" if output_symbol == "~" else output_symbol
                transition = Microtransition(
                    input_symbol=input_symbol,
                    output_symbol=output_symbol,
                    source=weakref.ref(previous_microstate),
                    target=weakref.ref(microstate),
                )
                microstate.incoming = transition
                previous_microstate.outgoing = weakref.ref(transition)

            microstates.append(microstate)
            assigned_macrostate = macrostates[
                f"cluster-{labels[offset + symbol_index]}"
            ]
            assigned_macrostate.microstates.add(microstate)
            microstate.macrostate = weakref.ref(assigned_macrostate)
            previous_microstate = microstate
        offset += len(transition_labels_as_list)

    # 4. Materialize states
    fst = build_fst(
        initial_macrostate,
        macrostates=macrostates,
        minimum_transition_count=hyperparams.minimum_transition_count,
    )

    # 5. Use the transition reduction heuristic to reduce the number of transitions and thus produce the final FST
    # for transition_key, counter in tqdm(
    #     transition_counts.items(), "Creating transitions"
    # ):
    #     if top_k := hyperparams.transitions_top_k:
    #         chosen_transitions = [k for k, v in counter.most_common(top_k)]
    #     elif top_p := hyperparams.transitions_top_p:
    #         # Compute the highest prob transitions s.t. cumprob > top_p
    #         total_count = sum(counter.values())
    #         probs = [(k, v / total_count) for k, v in counter.items()]
    #         probs_sorted = sorted(probs, key=lambda t: t[1], reverse=True)
    #         cum_prob = 0
    #         chosen_transitions: list[_TransitionValue] = []
    #         while cum_prob < top_p:
    #             next_transition, prob = probs_sorted.pop(0)
    #             chosen_transitions.append(next_transition)
    #             cum_prob += prob
    #     elif min_n := hyperparams.transitions_min_n:
    #         chosen_transitions = [k for k, v in counter.items() if v >= min_n]
    #     else:
    #         raise ValueError()

    #     for transition_value in chosen_transitions:
    #         fst.alphabet.update(
    #             {transition_key.input_symbol, transition_value.output_symbol}
    #         )
    #         if transition_key.input_symbol == transition_value.output_symbol:
    #             label = (transition_key.input_symbol,)
    #         else:
    #             label = (transition_key.input_symbol, transition_value.output_symbol)
    #         start_state = state_lookup[transition_key.start_state_label]
    #         end_state = state_lookup[transition_value.end_state_label]
    #         start_state.add_transition(end_state, label=label, weight=0.0)

    # fst.finalstates = {state_lookup[label] for label in final_state_labels}
    # for state in fst.states:
    #     state.finalweight = 0.0

    logger.info("Minimizing and determinizing")
    fst = fst.filter_accessible().minimize()
    remove_epsilon_loops(fst)
    logger.info(f"Created FST with {len(fst.states)} states")

    # fst.save("checkpoint.fst")
    # logger.info(f"Saved to {Path('checkpoint.fst')}")

    eval_metrics = evaluate_all(fst, eval_examples, hyperparams.generations_top_k)
    logger.info(f"Eval metrics: {pprint.pformat(eval_metrics)}")
    test_metrics = evaluate_all(fst, test_examples, hyperparams.generations_top_k)
    logger.info(f"Test metrics: {pprint.pformat(test_metrics)}")

    if visualize:
        logger.info("Rendering")
        fst.render(view=True, filename="fst")
    return eval_metrics, test_metrics


def evaluate_all(fst: FST, examples: list[InflectionExample], generations_top_k: int):
    labels: list[str] = []
    preds: list[set[str]] = []
    for example in tqdm(examples, "Evaluating"):
        features = [f"[{f}]" for f in example.features]
        input_string = features + ["<sep>"] + [c for c in example.lemma]
        assert example.target is not None
        correct_output = "".join(features + ["<sep>"] + [c for c in example.target])
        labels.append(correct_output)

        # Generate outputs by composing input acceptor with transducer
        logger.debug(f"Composing input string: {input_string}")
        input_fsa = FST.re("".join(f"'{c}'" for c in input_string))
        logger.debug("Composing input @ output")
        output_fst = input_fsa @ fst
        logger.debug("Minimizing")
        output_fst = output_fst.minimize()
        logger.debug("Removing epsilon loops")
        remove_epsilon_loops(output_fst)
        output_fst.render(view=False)
        output_fst = output_fst.project(-1)
        logger.debug("Generating top k words")
        example_preds = output_fst.words_nbest(generations_top_k)
        logger.debug(f"Words: {example_preds}")
        example_preds = ["".join(c[0] for c in chars) for _, chars in example_preds]
        preds.append(set(example_preds))
    return compute_metrics(labels, preds)


def compute_metrics(labels: list[str], predictions: list[set[str]]):
    assert len(labels) == len(predictions)
    precision_sum = 0
    recall_sum = 0

    for label, preds in zip(labels, predictions):
        if label not in preds:
            # Add 0 to both prec and recall
            continue
        precision_sum += 1 / len(preds)
        recall_sum += 1
    precision = precision_sum / len(labels)
    recall = recall_sum / len(labels)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="WandB shortname for the training run")
    parser.add_argument("--language", default="swe", help="Isocode for the language")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    train_path = (
        Path(__file__).parent.parent / "aligned_data" / f"{args.language}.trn.aligned"
    )
    eval_path = find_data_file(f"{args.language}.dev")
    test_path = find_data_file(f"{args.language}.tst")

    extract_fst(
        hyperparams=ExtractionHyperparameters(
            clustering_method="kmeans",
            kmeans_num_clusters=1000,
            min_samples=500,
            pca_components=30,
            transitions_top_k=1,
            transitions_top_p=None,
            minimum_transition_count=100,
            generations_top_k=1,
        ),
        aligned_train_path=train_path,
        eval_path=eval_path,
        test_path=test_path,
        model_id=args.model_id,
        visualize=args.visualize,
    )
