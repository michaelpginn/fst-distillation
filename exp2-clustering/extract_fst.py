"""Usage: python -m exp2-clustering.extract_fst <checkpoint path> <train dataset path>

Runs the Giles (1991) clustering algorithm to produce an FST from a trained RNN.

You should run `train_rnn.py` first to train a model and produce a checkpoint.
"""

import argparse
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas
import seaborn
import torch
from pyfoma import FST, State
from pyfoma import algorithms as alg
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.modeling.rnn import RNNModel
from src.tasks.inflection_classification.dataset import load_examples_from_file
from src.tasks.inflection_classification.example import AlignedInflectionExample
from src.tasks.inflection_classification.tokenizer import AlignedInflectionTokenizer


def extract_fst(
    model: RNNModel,
    examples: list[AlignedInflectionExample],
    num_initial_clusters: int,
    visualize: bool = False,
):
    # 1. For each training example, collect activations
    activations = []
    for example in tqdm(examples, "Computing hidden states"):
        inputs = model.tokenizer.tokenize(example)
        hidden_states, _ = model.rnn(model.embedding(torch.tensor(inputs["input_ids"])))
        activations.append(hidden_states.detach())
    activations = torch.concat(activations).numpy()

    # 2. Perform clustering
    activations = StandardScaler().fit_transform(activations)
    pca = PCA(n_components=30, whiten=True)
    activations = pca.fit_transform(activations)
    _, labels, _ = k_means(activations, n_clusters=num_initial_clusters)  # type:ignore
    assert labels is not None

    if visualize:
        pca_data = pandas.DataFrame(activations[:, :2], columns=["PC1", "PC2"])  # type:ignore
        pca_data["cluster"] = pandas.Categorical(labels)
        seaborn.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_data)
        plt.show()

    # 3. Create states
    fst = FST()
    state_lookup = {
        f"cluster-{label_id}": State(name=f"cluster-{label_id}")
        for label_id in range(max(labels) + 1)
    }
    fst.states = set(state_lookup.values())
    fst.initialstate = state_lookup[f"cluster-{labels[0]}"]

    # 4. Use the original inputs to produce a counter of transitions between each pair of states
    transition_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(
        lambda: Counter()
    )
    final_state_labels: set[str] = set()
    offset = 0
    for example in tqdm(examples, "Collecting transitions"):
        input_symbols = model.tokenizer.decode(
            model.tokenizer.tokenize(example)["input_ids"],  # type:ignore
            skip_special_tokens=False,
            return_as="list",
        )
        for symbol_index in range(len(input_symbols) - 1):
            start_state_label = f"cluster-{labels[offset + symbol_index]}"
            end_state_label = f"cluster-{labels[offset + symbol_index + 1]}"
            transition_label = input_symbols[symbol_index + 1]
            transition_counts[(start_state_label, end_state_label)].update(
                [transition_label]
            )
        final_state_labels.add(f"cluster-{labels[offset + len(input_symbols) - 1]}")
        offset += len(input_symbols)

    # 5. Use the transition reduction heuristic to reduce the number of transitions and thus produce the final FST
    #
    # FIXME: for now, I use "all transitions"
    # We should implement "k most common", "threshold", etc
    for (start_state_label, end_state_label), counter in tqdm(
        transition_counts.items(), "Creating transitions"
    ):
        for label, _ in counter.most_common(1):
            if match := re.match(r"\((.*?),(.*?)\)", label):
                top_label = (match.group(1), match.group(2))
                if top_label[0] == top_label[1]:
                    top_label = (top_label[0],)
                fst.alphabet.update({match.group(1), match.group(2)})
            else:
                top_label = label.replace("<", "").replace(">", "")
                fst.alphabet.update({top_label})
                top_label = (top_label,)
            start_state = state_lookup[start_state_label]
            end_state = state_lookup[end_state_label]
            start_state.add_transition(end_state, label=top_label, weight=0.0)

    fst.finalstates = {state_lookup[label] for label in final_state_labels}
    for state in fst.states:
        state.finalweight = 0.0

    print("Minimizing and determinizing")
    fst = alg.minimized(alg.determinized(alg.filtered_accessible(fst)))
    print("Rendering")
    fst.render(view=False, filename="fst")

    l = list(fst.generate("V"))
    # l = [s for s in fst.generate("V")]
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("dataset_path")
    args = parser.parse_args()

    checkpoint_dict = torch.load(args.model_path, weights_only=True)
    tokenizer = AlignedInflectionTokenizer.from_state_dict(
        checkpoint_dict["tokenizer_dict"]
    )
    model = RNNModel.load(checkpoint_dict, tokenizer)
    examples = load_examples_from_file(args.dataset_path)
    extract_fst(model, examples, num_initial_clusters=10)
