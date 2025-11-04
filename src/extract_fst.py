"""Usage: python -m experiments.clustering.extract_fst <checkpoint path> <train dataset path>

Runs the Giles (1991) clustering algorithm to produce an FST from a trained RNN.

You should run `train_rnn.py` first to train a model and produce a checkpoint.
"""

import logging
import pprint
import re
import typing
import warnings
import weakref
from dataclasses import dataclass
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn
import torch
import umap
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, OPTICS, k_means
from sklearn.decomposition import PCA
from tqdm import tqdm

from .data.aligned.classification.tokenizer import AlignedClassificationTokenizer
from .data.aligned.example import (
    ALIGNMENT_SYMBOL,
    AlignedStringExample,
    load_examples_from_file,
)
from .data.aligned.language_modeling.tokenizer import (
    AlignedLanguageModelingTokenizer,
)
from .data.unaligned.example import (
    load_examples_from_file as load_unaligned,
)
from .evaluate import evaluate_all
from .modeling.rnn import RNNModel
from .modeling.tokenizer import Tokenizer
from .paths import Paths, create_arg_parser, create_paths_from_args
from .state_clustering.convert_macrostates_to_fst import convert_macrostates_to_fst
from .state_clustering.hopkins import hopkins
from .state_clustering.types import (
    Macrostate,
    Microstate,
    Microtransition,
)
from .training.classifier.train import device

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class ExtractionHyperparameters:
    model_shortname: str
    clustering_method: Literal["kmeans", "optics", "hdbscan", "dbscan"]
    kmeans_num_clusters: int | None = None
    min_samples: int | None = None
    eps: float | None = None

    dim_reduction_method: Literal["pca", "umap", "none"] = "none"
    n_components: int | None = None
    umap_n_neighbors: int | None = None
    umap_min_distance: float | None = None

    minimum_transition_count: int | None = 100
    """Minimum count for a given transition to be guaranteed to be included"""
    state_split_classifier: Literal["svm", "logistic"] = "svm"

    generations_top_k: int = 1
    """How many generations to pick, in increasing length order"""

    visualize: bool = False

    def __post_init__(self):
        if self.clustering_method == "kmeans" and self.kmeans_num_clusters is None:
            raise ValueError("Must set `kmeans_num_clusters` when using k-means!")
        if self.clustering_method == "optics" and (self.min_samples is None):
            raise ValueError("Must set `optics_min_pts` when using OPTICS!")


def extract_fst(
    hyperparams: ExtractionHyperparameters,
    paths: Paths,
):
    model, tokenizer, task = _load_model(hyperparams, paths)

    aligned_train_examples = load_examples_from_file(paths["train_aligned"])
    raw_train_examples = load_unaligned(paths["train"], paths["has_features"])
    raw_eval_examples = load_unaligned(paths["eval"], paths["has_features"])
    raw_test_examples = load_unaligned(paths["test"], paths["has_features"])

    activations, all_transition_labels = _collect_activations(
        hyperparams, paths, aligned_train_examples, model, tokenizer, task
    )
    activations = _standardize(hyperparams, activations)
    logger.info(f"Hopkins statistic: {hopkins(activations)}")
    labels = _cluster(hyperparams, activations)
    macrostates, initial_macrostate = _collect_microstates(
        hyperparams, model, activations, all_transition_labels, labels
    )
    fst = convert_macrostates_to_fst(
        initial_macrostate,
        macrostates=macrostates,
        state_splitting_classifier=hyperparams.state_split_classifier,
        minimum_transition_count=hyperparams.minimum_transition_count,
    )
    metrics = {
        "train": evaluate_all(fst, raw_train_examples),
        "eval": evaluate_all(fst, raw_eval_examples, log=True),
        "test": evaluate_all(fst, raw_test_examples),
    }
    logger.info(pprint.pformat(metrics))
    if hyperparams.visualize:
        fst.render(view=True, filename="fst")
    return metrics, fst


def _load_model(hyperparams: ExtractionHyperparameters, paths: Paths):
    model_path = paths["models_folder"] / f"{hyperparams.model_shortname}/model.pt"
    checkpoint_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    task = checkpoint_dict["config_dict"].get("output_head", "classification")
    if task == "classification":
        tokenizer = AlignedClassificationTokenizer.from_state_dict(
            checkpoint_dict["tokenizer_dict"]
        )
    elif task == "lm":
        tokenizer = AlignedLanguageModelingTokenizer.from_state_dict(
            checkpoint_dict["tokenizer_dict"]
        )
    else:
        raise ValueError(f"Unknown model head: {task}")
    model = RNNModel.load(checkpoint_dict, tokenizer)
    model.to(device)
    model.eval()
    return model, tokenizer, task


def _collect_activations(
    hyperparams: ExtractionHyperparameters,
    paths: Paths,
    aligned_train_examples: list[AlignedStringExample],
    model: RNNModel,
    tokenizer: Tokenizer,
    task: Literal["classification", "lm"],
):
    activations: list[torch.Tensor] = []
    all_transition_labels: list[list[str]] = []

    torch.set_default_device(device)
    with torch.no_grad():
        # 1A. Collect full examples (input and outputs) from the train set
        for example in tqdm(
            aligned_train_examples, "Computing hidden states for train"
        ):
            inputs = tokenizer.tokenize(example)
            hidden_states, _ = model.compute_hidden_states(
                embeddings=model.embedding(
                    torch.tensor(inputs["input_ids"]).unsqueeze(0)
                ),
                seq_lengths=torch.tensor([len(inputs["input_ids"])]),  # type:ignore
            )
            activations.append(hidden_states.squeeze(0).cpu().detach())
            transition_labels: list[str] = model.tokenizer.decode(
                model.tokenizer.tokenize(example)["input_ids"],  # type:ignore
                skip_special_tokens=False,
                return_as="list",
            )
            all_transition_labels.append(transition_labels)

        # 1B. Also collect inputs for the whole domain
        if task == "lm" and paths["full_domain_aligned"].exists():
            full_alignment = load_examples_from_file(paths["full_domain_aligned"])
            for example in tqdm(
                full_alignment, "Computing hidden states for full domain"
            ):
                tokenizer = cast(AlignedLanguageModelingTokenizer, tokenizer)
                # First compute hidden states for start symbol and features,
                # then walk hidden space by picking the highest next symbol with the correct input side
                input_prefix = tokenizer.tokenize(
                    AlignedStringExample([], example.features, True)
                )
                transition_labels: list[str] = model.tokenizer.decode(
                    input_prefix["input_ids"][:-1],  # type:ignore
                    skip_special_tokens=False,
                    return_as="list",
                )
                hidden_states, H_t = model.compute_hidden_states(
                    embeddings=model.embedding(
                        torch.tensor(
                            input_prefix["input_ids"][:-1]  # type:ignore
                        ).unsqueeze(0)
                    ),
                    seq_lengths=torch.tensor(
                        [len(input_prefix["input_ids"]) - 1]  # type:ignore
                    ),
                )
                for char, _ in example.aligned_chars:
                    logits: torch.Tensor = model.out(hidden_states[:, -1]).squeeze(0)
                    # Mask to only allowed next symbols (which start with the input symbol)
                    possible_ids = tokenizer.token_ids_matching_input(char)
                    mask = torch.ones(logits.size(0), dtype=torch.bool)
                    mask[torch.tensor(possible_ids)] = False
                    logits[mask] = 0
                    chosen_symbol_index = torch.argmax(logits).item()
                    H_t = model.compute_timestep(
                        H_t_min1=H_t,
                        x_t=model.embedding(torch.tensor([[chosen_symbol_index]])),
                        mask=None,
                    )
                    hidden_states = torch.concat([hidden_states, H_t[:, -1]], dim=1)
                    transition_labels.append(tokenizer.decode([chosen_symbol_index]))  # type:ignore
                # Finally add the <sink>
                H_t = model.compute_timestep(
                    H_t_min1=H_t,
                    x_t=model.embedding(torch.tensor([[tokenizer.sink_token_id]])),
                    mask=None,
                )
                hidden_states = torch.concat([hidden_states, H_t[:, -1]], dim=1)
                activations.append(hidden_states.squeeze(0).cpu().detach())
                transition_labels.append(tokenizer.id_to_token[tokenizer.sink_token_id])  # type:ignore
                all_transition_labels.append(transition_labels)

    return torch.concat(activations), all_transition_labels


def _standardize(hyperparams: ExtractionHyperparameters, activations: torch.Tensor):
    logger.info("Standardizing...")
    activations = (activations - activations.mean(dim=0)) / activations.std(dim=0)
    activations_np = activations.numpy()
    if (
        hyperparams.dim_reduction_method != "none"
        and hyperparams.n_components < activations_np.shape[-1]
    ):
        logger.info(f"Reducing dimensionality (PC = {hyperparams.n_components})...")
        if hyperparams.dim_reduction_method == "pca":
            pca = PCA(n_components=hyperparams.n_components, random_state=0)
            activations_np = pca.fit_transform(activations_np)
        elif hyperparams.dim_reduction_method == "umap":
            map = umap.UMAP()
            activations_np = map.fit_transform(activations_np)
        activations_np = typing.cast(np.ndarray, activations_np)
    return activations_np


def _cluster(hyperparams: ExtractionHyperparameters, activations: np.ndarray):
    logger.info(f"Clustering with '{hyperparams.clustering_method}'")
    if hyperparams.clustering_method == "kmeans":
        if hyperparams.kmeans_num_clusters > len(activations):  # type:ignore
            raise ValueError(
                "kmeans_num_clusters is larger than the number of points, aborting!"
            )
        _, labels, _ = k_means(  # type:ignore
            activations, n_clusters=hyperparams.kmeans_num_clusters, random_state=0
        )
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
    elif hyperparams.clustering_method == "dbscan":
        assert hyperparams.eps is not None
        assert hyperparams.min_samples is not None
        labels = DBSCAN(
            eps=hyperparams.eps, min_samples=hyperparams.min_samples, n_jobs=-1
        ).fit_predict(activations)
    else:
        raise ValueError()
    assert labels is not None

    if hyperparams.visualize:
        logger.info("Visualizing principle components")
        pca_data = pandas.DataFrame(activations[:, :2], columns=["PC1", "PC2"])  # type:ignore
        pca_data["cluster"] = pandas.Categorical(labels)
        seaborn.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
        plt.show()
    return labels


def _collect_microstates(
    hyperparams: ExtractionHyperparameters,
    model: RNNModel,
    activations: np.ndarray,
    all_transition_labels: list[list[str]],
    labels: np.ndarray,
):
    microstates: list[Microstate] = []
    macrostates: dict[str, Macrostate] = {
        f"cluster-{label}": Macrostate(label=f"cluster-{label}")
        for label in set(labels)
    }
    initial_macrostate = macrostates[f"cluster-{labels[0]}"]

    offset = 0
    for ex_transition_labels in tqdm(all_transition_labels, "Collecting microstates"):
        previous_microstate: Microstate | None = None
        for symbol_index, transition_label in enumerate(ex_transition_labels):
            microstate = Microstate(
                position=activations[offset + symbol_index],
                is_final=symbol_index == len(ex_transition_labels) - 1,
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
                input_symbol = "" if input_symbol == ALIGNMENT_SYMBOL else input_symbol
                output_symbol = (
                    "" if output_symbol == ALIGNMENT_SYMBOL else output_symbol
                )
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
        offset += len(ex_transition_labels)
    return macrostates, initial_macrostate


if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument("--model-id", help="WandB shortname for the training run")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    extract_fst(
        hyperparams=ExtractionHyperparameters(
            model_shortname=args.model_id,
            dim_reduction_method="none",
            n_components=16,
            umap_n_neighbors=10,
            umap_min_distance=0.01,
            clustering_method="kmeans",
            kmeans_num_clusters=2000,
            min_samples=100,
            eps=1,
            minimum_transition_count=2,
            state_split_classifier="svm",
            generations_top_k=1,
            visualize=args.visualize,
        ),
        paths=create_paths_from_args(args),
    )
