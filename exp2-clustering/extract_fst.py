"""Usage: python -m exp2-clustering.extract_fst <checkpoint path> <train dataset path>

Runs the Giles (1991) clustering algorithm to produce an FST from a trained RNN.

You should run `train_rnn.py` first to train a model and produce a checkpoint.
"""

import argparse
import logging
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas
import seaborn
import torch
from pyfoma.fst import FST, State
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.modeling.rnn import RNNModel
from src.tasks.inflection_classification.dataset import load_examples_from_file
from src.tasks.inflection_classification.example import AlignedInflectionExample
from src.tasks.inflection_classification.tokenizer import AlignedInflectionTokenizer

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def extract_fst(
    model_path: str,
    train_path: str,
    eval_path: str,
    num_initial_clusters: int,
    visualize: bool = False,
):
    checkpoint_dict = torch.load(model_path, weights_only=True)
    tokenizer = AlignedInflectionTokenizer.from_state_dict(
        checkpoint_dict["tokenizer_dict"]
    )
    model = RNNModel.load(checkpoint_dict, tokenizer)
    train_examples = load_examples_from_file(train_path)
    eval_examples = load_examples_from_file(eval_path)

    # 1. For each training example, collect activations
    activations = []
    for example in tqdm(train_examples, "Computing hidden states"):
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
    fst.initialstate = state_lookup[f"cluster-{labels[0]}"] # Use the label of the <bos> cluster

    # 4. Use the original inputs to produce a counter of transitions between each pair of states
    transition_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(
        lambda: Counter()
    )
    final_state_labels: set[str] = set()
    offset = 0
    for example in tqdm(train_examples, "Collecting transitions"):
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
    # FIXME: Probably want pick based on unique input symbols, not pairs
    # We should implement "k most common", "threshold", etc
    for (start_state_label, end_state_label), counter in tqdm(
        transition_counts.items(), "Creating transitions"
    ):
        for label, _ in counter.most_common(5):
            if match := re.match(r"\((.*?),(.*?)\)", label):
                # Paired (input, output) transition
                top_label = (match.group(1), match.group(2))
                if top_label[0] == top_label[1]:
                    top_label = (top_label[0],)
                fst.alphabet.update({match.group(1), match.group(2)})
            else:
                # Unpaired transition (special character or tag)
                fst.alphabet.update({label})
                top_label = (label,)
            start_state = state_lookup[start_state_label]
            end_state = state_lookup[end_state_label]
            start_state.add_transition(end_state, label=top_label, weight=0.0)

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
    accepted_input = 0          # Proportion where the transducer accepts the input
    correct_count = 0            # Number of examples where correct output is produced
    average_num_generations = 0         # Number of total generations per input
    matched_prefix_length = 0   # Average length of matched prefix
    for example in tqdm(eval_examples, "Evaluating"):
        input_string = example.features + ['<sep>'] + [c[0] for c in example.aligned_chars]
        correct_output = example.features + ['<sep>'] + [c[1] for c in example.aligned_chars]
        correct_output = ''.join(correct_output)

        generated_outputs = list(fst.generate(input_string))
        if len(generated_outputs) > 0:
            accepted_input += 1
            if correct_output in generated_outputs:
                correct_count += 1
            average_num_generations += len(generated_outputs)
            # TODO: Why are we getting multiple outputs? Shouldn't it be deterministic?

    logger.info(f"""Accepted: {accepted_input/len(eval_examples):.2%}
        Correct: {correct_count/len(eval_examples):.2%}
        Average num generations: {average_num_generations/accepted_input:.2%}""")


    logger.info("Rendering")
    fst.render(view=True, filename="fst")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--train_path", "-t", help=".aligned data file")
    parser.add_argument("--eval_path", "-e", help=".aligned data file")
    args = parser.parse_args()
    extract_fst(
        model_path=args.model_path,
        train_path=args.train_path,
        eval_path=args.eval_path,
        num_initial_clusters=10
    )
