"""Usage: python -m experiments.clustering.extract_fst <checkpoint path> <train dataset path>

Runs the Giles (1991) clustering algorithm to produce an FST from a trained RNN.

You should run `train_rnn.py` first to train a model and produce a checkpoint.
"""

import argparse
import logging
import pprint
import re
import typing
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

from experiments.shared import DataFiles, add_task_parser, get_data_files
from src.data.aligned.classification.tokenizer import AlignedClassificationTokenizer
from src.data.aligned.example import ALIGNMENT_SYMBOL, load_examples_from_file
from src.data.aligned.language_modeling.tokenizer import (
    AlignedLanguageModelingTokenizer,
)
from src.data.unaligned.example import (
    load_examples_from_file as load_unaligned,
)
from src.evaluate import evaluate_all
from src.modeling.rnn import RNNModel
from src.state_clustering.build_fst import build_fst
from src.state_clustering.hopkins import hopkins
from src.state_clustering.types import (
    Macrostate,
    Microstate,
    Microtransition,
)
from src.training.classifier.train import device

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__file__)

DEBUG = False


@dataclass(frozen=True)
class ExtractionHyperparameters:
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

    def __post_init__(self):
        if self.clustering_method == "kmeans" and self.kmeans_num_clusters is None:
            raise ValueError("Must set `kmeans_num_clusters` when using k-means!")
        if self.clustering_method == "optics" and (self.min_samples is None):
            raise ValueError("Must set `optics_min_pts` when using OPTICS!")


def extract_fst(
    hyperparams: ExtractionHyperparameters,
    data_files: DataFiles,
    model_id: str,
    visualize: bool = False,
):
    model_path = Path(__file__).parent.parent / f"runs/{model_id}/model.pt"
    checkpoint_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    if (
        task := checkpoint_dict["config_dict"].get("output_head", "classification")
    ) == "classification":
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
    aligned_train_examples = load_examples_from_file(data_files["train_aligned"])
    raw_train_examples = load_unaligned(data_files["train"], data_files["has_features"])
    raw_eval_examples = load_unaligned(data_files["eval"], data_files["has_features"])
    raw_test_examples = load_unaligned(data_files["test"], data_files["has_features"])

    # 1. For each training example, collect activations
    activations = []
    with torch.no_grad():
        for example in tqdm(aligned_train_examples, "Computing hidden states"):
            inputs = model.tokenizer.tokenize(example)
            hidden_states = model.compute_hidden_states(
                embeddings=model.embedding(
                    torch.tensor(inputs["input_ids"], device=device).unsqueeze(0)
                ),
                seq_lengths=torch.tensor([len(inputs["input_ids"])], device=device),  # type:ignore
            )
            activations.append(hidden_states.squeeze(0).cpu().detach())
    activations = torch.concat(activations)

    # 2. Perform standardization -> dim reduction -> clustering
    logger.info("Standardizing...")
    activations = (activations - activations.mean(dim=0)) / activations.std(dim=0)
    activations = activations.numpy()
    if (
        hyperparams.dim_reduction_method != "none"
        and hyperparams.n_components < activations.shape[-1]
    ):
        logger.info(f"Reducing dimensionality (PC = {hyperparams.n_components})...")
        if hyperparams.dim_reduction_method == "pca":
            pca = PCA(n_components=hyperparams.n_components, random_state=0)
            activations = pca.fit_transform(activations)
        elif hyperparams.dim_reduction_method == "umap":
            map = umap.UMAP()
            activations = map.fit_transform(activations)
        activations = typing.cast(np.ndarray, activations)

    hopkins_stat = hopkins(activations)
    logger.info(f"Hopkins statistic: {hopkins_stat}")

    logger.info(f"Clustering with '{hyperparams.clustering_method}'")
    if hyperparams.clustering_method == "kmeans":
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
    labels_to_debug_ = {}
    for example in tqdm(aligned_train_examples, "Collecting microstates"):
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
            if example.input_string == "tyngdkraft":
                labels_to_debug_[assigned_macrostate.label] = microstate
            previous_microstate = microstate
        offset += len(transition_labels_as_list)

    # 4. Materialize states
    fst = build_fst(
        initial_macrostate,
        macrostates=macrostates,
        state_splitting_classifier=hyperparams.state_split_classifier,
        minimum_transition_count=hyperparams.minimum_transition_count,
        breakpoint_on=labels_to_debug_,
    )

    logger.info("Minimizing and determinizing")
    fst = fst.filter_accessible().minimize()
    # remove_epsilon_loops(fst)
    logger.info(f"Created FST with {len(fst.states)} states")

    # fst.save("checkpoint.fst")
    # logger.info(f"Saved to {Path('checkpoint.fst')}")

    # train_metrics = evaluate_all(fst, raw_train_examples)
    # logger.info(f"Train metrics: {pprint.pformat(train_metrics)}")
    eval_metrics = evaluate_all(fst, raw_eval_examples)
    logger.info(f"Eval metrics: {pprint.pformat(eval_metrics)}")
    test_metrics = evaluate_all(fst, raw_test_examples)
    logger.info(f"Test metrics: {pprint.pformat(test_metrics)}")

    if visualize:
        logger.info("Rendering")
        fst.render(view=True, filename="fst")
    return {"eval": eval_metrics, "test": test_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_task_parser(parser)
    parser.add_argument("--model-id", help="WandB shortname for the training run")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    extract_fst(
        hyperparams=ExtractionHyperparameters(
            dim_reduction_method="none",
            n_components=16,
            umap_n_neighbors=10,
            umap_min_distance=0.01,
            clustering_method="kmeans",
            kmeans_num_clusters=1000,
            min_samples=100,
            eps=1,
            minimum_transition_count=25,
            state_split_classifier="logistic",
            generations_top_k=1,
        ),
        data_files=get_data_files(args),
        model_id=args.model_id,
        visualize=args.visualize,
    )
