import logging
import pprint
from collections import defaultdict
from dataclasses import dataclass
from random import sample
from typing import Counter

import torch
from pyfoma.fst import FST
from tqdm import tqdm

from src.data.aligned.alignment_prediction.domain_cover import ngram_bfs
from src.data.aligned.example import (
    ALIGNMENT_SYMBOL,
    AlignedStringExample,
    load_examples_from_file,
)
from src.data.aligned.transduction.tokenizer import AlignedTransductionTokenizer
from src.data.tokenize_with_diacritics import tokenize
from src.evaluate import compute_metrics
from src.extract_fst import (
    ExtractionHyperparameters,
    _cluster,
    _collect_microstates,
    _standardize,
    convert_macrostates_to_fst,
)
from src.modeling.birnn import BiRNN
from src.modeling.tokenizer import Tokenizer
from src.paths import Paths, create_arg_parser, create_paths_from_args

from .data.unaligned.example import String2StringExample
from .data.unaligned.example import load_examples_from_file as load_unaligned
from .training.classifier.train import device

logger = logging.getLogger(__name__)


@dataclass
class Bimachine:
    forward_fst: FST
    backward_fst: FST
    output_table: dict[tuple[str, str, str], str]

    def generate(self, input: list[str]):
        assert input[0] == "<bos>", input[-1] == "<eos>"
        forward_states = []
        backward_states = []
        s = self.forward_fst.initialstate
        for c in input[1:-1]:
            trans = s.transitions_by_input[c]
            forward_states.append(s.name)
            if len(trans) != 1:
                trans = s.transitions_by_input[""]
            if len(trans) != 1:
                return None
            s = list(trans)[0][1].targetstate
        s = self.backward_fst.initialstate
        for c in input[1:-1][::-1]:
            trans = s.transitions_by_input[c]
            backward_states.append(s.name)
            if len(trans) != 1:
                trans = s.transitions_by_input[""]
            if len(trans) != 1:
                return None
            s = list(trans)[0][1].targetstate
        output = []
        for forward_state, backward_state, input_char in zip(
            forward_states, reversed(backward_states), input[1:-1]
        ):
            out_char = self.output_table.get(
                (forward_state, backward_state, input_char)
            )
            if out_char is None:
                return None
            output.append(out_char)
        return output


def compute_activations(hparams: ExtractionHyperparameters, paths: Paths):
    """Collects and standardizes activations for the train and full domain"""
    model, tokenizer = _load_model(hparams, paths)
    aligned_train_examples = load_examples_from_file(paths["train_aligned"])
    activations, all_transition_labels = _collect_activations(
        hparams, paths, aligned_train_examples, model, tokenizer
    )
    activations = {k: _standardize(hparams, a) for k, a in activations.items()}
    return activations, all_transition_labels


def extract_bimachine(hparams: ExtractionHyperparameters, paths: Paths):
    raw_train_examples = load_unaligned(
        paths["train"], paths["has_features"], paths["output_split_into_chars"]
    )
    raw_eval_examples = load_unaligned(
        paths["eval"], paths["has_features"], paths["output_split_into_chars"]
    )
    raw_test_examples = load_unaligned(
        paths["test"], paths["has_features"], paths["output_split_into_chars"]
    )

    activations, all_transition_labels = compute_activations(hparams, paths)
    # Note: for a given index, the transition label is the *incoming* transition to that state
    forward_clusters = _cluster(hparams, activations["forward"])
    backward_clusters = _cluster(hparams, activations["backward"])
    forward_macrostates, forward_initial_macrostate, forward_μstates = (
        _collect_microstates(
            activations["forward"],
            all_transition_labels["forward_in"],
            forward_clusters,
        )
    )
    backward_macrostates, backward_initial_macrostate, backward_μstates = (
        _collect_microstates(
            activations["backward"],
            all_transition_labels["backward_in"],
            backward_clusters,
        )
    )
    # We can reuse this where transtions are always identity (ie FSA)
    forward_fst = convert_macrostates_to_fst(
        forward_initial_macrostate,
        macrostates=forward_macrostates,
        state_splitting_classifier=hparams.state_split_classifier,
        minimum_transition_count=hparams.minimum_transition_count,
        do_merge=hparams.do_merge,
        do_minimize=False,
    )
    backward_fst = convert_macrostates_to_fst(
        backward_initial_macrostate,
        macrostates=backward_macrostates,
        state_splitting_classifier=hparams.state_split_classifier,
        minimum_transition_count=hparams.minimum_transition_count,
        do_merge=hparams.do_merge,
        do_minimize=False,
    )

    # Build the output table {(f-state, b-state, in symbol) : Counter of out symbols}
    output_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    offset = 0
    for i in range(len(all_transition_labels["forward_in"])):
        seq_length = len(all_transition_labels["forward_in"][i])
        base = offset
        for j in range(1, seq_length):
            forward_in = all_transition_labels["forward_in"][i][j]
            forward_out = all_transition_labels["forward_out"][i][j]
            # Within a seq, <bos> <sep> ... <sink>
            forward_microstate = forward_μstates[base + j - 1]
            # Since these are reversed per sequence, we need to unreverse them
            # Backwards states: <eos> <sink> ... <sep>
            seq_length = len(all_transition_labels["forward_in"][i])
            backward_microstate = backward_μstates[base + (seq_length - j - 1)]
            forward_label = forward_microstate.macrostate().label  # type:ignore
            backward_label = backward_microstate.macrostate().label  # type:ignore
            output_counts[(forward_label, backward_label, forward_in)].update(
                [forward_out if forward_out != ALIGNMENT_SYMBOL else ""]
            )
        offset += seq_length

    assert offset == len(forward_μstates) == len(backward_μstates)

    final_outputs: dict[tuple[str, str, str], str] = {
        key: ctr.most_common(1)[0][0] for key, ctr in output_counts.items()
    }
    bimachine = Bimachine(forward_fst, backward_fst, final_outputs)

    def evaluate(examples: list[String2StringExample], log=False):
        labels: list[str] = []
        preds: list[set[str]] = []
        indices_to_log = sample(range(len(examples)), k=10)
        for idx, ex in enumerate(examples):
            assert ex.output_string is not None
            input_string = ["<sep>"] + tokenize(ex.input_string) + ["<sink>"]
            correct_output = ["<sep>"] + tokenize(ex.output_string) + ["<sink>"]
            if ex.features is not None:
                features = [f"[{f}]" for f in ex.features]
                input_string = features + input_string
                correct_output = features + correct_output
            labels.append("".join(correct_output))
            input_string = ["<bos>"] + input_string + ["<eos>"]
            output = bimachine.generate(input_string)
            preds.append({"".join(output)} if output else set())
            if log and idx in indices_to_log:
                logger.info(f"Input:\t{''.join(input_string)}")
                logger.info(f"Gold:\t{''.join(correct_output)}")
                logger.info(f"Predicted:\t{output}")
        return compute_metrics(labels, preds)

    metrics = {
        "train": evaluate(raw_train_examples),
        "eval": evaluate(raw_eval_examples, log=True),
        "test": evaluate(raw_test_examples),
    }
    logger.info(pprint.pformat(metrics))
    return metrics, bimachine


def _load_model(hyperparams: ExtractionHyperparameters, paths: Paths):
    model_path = paths["models_folder"] / f"{hyperparams.model_shortname}/model.pt"
    checkpoint_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    tokenizer = AlignedTransductionTokenizer.from_state_dict(
        checkpoint_dict["tokenizer_dict"]
    )
    tokenizer.is_bidirect = True
    model = BiRNN.load(checkpoint_dict, tokenizer)
    model.to(device)
    model.eval()
    return model, tokenizer


def _collect_activations(
    hyperparams: ExtractionHyperparameters,
    paths: Paths,
    aligned_train_examples: list[AlignedStringExample],
    model: BiRNN,
    tokenizer: Tokenizer,
):
    activations: dict[str, list[torch.Tensor]] = {
        "forward": [],
        "backward": [],
    }
    all_transition_labels: dict[str, list[list[str]]] = {
        "forward_in": [],
        "forward_out": [],
        "backward_in": [],
        "backward_out": [],
    }

    torch.set_default_device(device)
    with torch.no_grad():
        # 1A. Collect full examples (input and outputs) from the train set
        for example in tqdm(
            aligned_train_examples, "Computing hidden states for train"
        ):
            inputs = tokenizer.tokenize(example)
            _, forward_states, backward_states = model.forward(
                torch.tensor([inputs["input_ids"]]),
                torch.tensor([len(inputs["input_ids"])]),  # type:ignore
                torch.tensor([inputs["next_input_ids"]]),
            )
            # Should be states reached after reading <bos> <sep> ... <sink> | <eos>
            # Cut off the <eos> state
            forward_states = forward_states.squeeze(0)[:-1].cpu().detach()
            # Should be states after <eos> <sink> ... <sep> | <bos>
            backward_states = backward_states.squeeze(0)[:-1].cpu().detach()
            activations["forward"].append(forward_states)
            activations["backward"].append(backward_states)
            in_labels: list[str] = model.tokenizer.decode(
                inputs["next_input_ids"],  # type:ignore
                skip_special_tokens=False,
                return_as="list",
            )
            out_labels: list[str] = model.tokenizer.decode(
                inputs["next_output_ids"],  # type:ignore
                skip_special_tokens=False,
                return_as="list",
            )
            all_transition_labels["forward_in"].append(["<bos>"] + in_labels)
            all_transition_labels["forward_out"].append(["<bos>"] + out_labels)
            all_transition_labels["backward_in"].append(
                ["<eos>"] + list(reversed(in_labels))
            )
            all_transition_labels["backward_out"].append(
                ["<eos>"] + list(reversed(out_labels))
            )

        # 1B. Also collect inputs for the whole domain

        if hyperparams.full_domain:
            if hyperparams.full_domain_mode == "sample":
                # new_activations, new_labels = sample_full_domain(
                #     paths, model, tokenizer, task
                # )
                raise NotImplementedError()
            elif hyperparams.full_domain_mode == "search":
                new_activations, new_labels = search_full_domain(
                    hyperparams, aligned_train_examples, model, tokenizer
                )
            else:
                raise ValueError()
            for key in activations:
                activations[key].extend(new_activations[key])
            for key in all_transition_labels:
                all_transition_labels[key].extend(new_labels[key])

    return {k: torch.concat(a) for k, a in activations.items()}, all_transition_labels


def search_full_domain(
    hyperparams: ExtractionHyperparameters,
    aligned_train_examples: list[AlignedStringExample],
    model: BiRNN,
    tokenizer: Tokenizer,
):
    activations: dict[str, list[torch.Tensor]] = {
        "forward": [],
        "backward": [],
    }
    all_transition_labels: dict[str, list[list[str]]] = {
        "forward_in": [],
        "forward_out": [],
        "backward_in": [],
        "backward_out": [],
    }

    # BFS through state space
    all_inputs = ngram_bfs(
        aligned_train_examples, n=hyperparams.full_domain_search_n, max_length=8
    )
    assert tokenizer.token_to_id
    for input_string in tqdm(
        all_inputs, desc="Computing hidden states for full domain"
    ):
        token_ids = [tokenizer.token_to_id[t] for t in input_string]
        token_ids = [
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
            *token_ids,
            tokenizer.sink_token_id,
            tokenizer.eos_token_id,
        ]
        input_ids = torch.tensor([token_ids], device=device)
        next_input_ids = torch.tensor([token_ids[1:-1]], device=device)
        preds, forward_states, backward_states = model.forward(
            input_ids,
            torch.tensor([len(token_ids)]),  # type:ignore
            next_input_ids,
        )
        next_output_ids = preds.squeeze(0).argmax(-1)
        forward_states = forward_states.squeeze(0)[:-1].cpu().detach()
        backward_states = backward_states.squeeze(0)[:-1].cpu().detach()
        activations["forward"].append(forward_states)
        activations["backward"].append(backward_states)
        in_labels: list[str] = model.tokenizer.decode(
            next_input_ids[0].tolist(),  # type:ignore
            skip_special_tokens=False,
            return_as="list",
        )
        out_labels: list[str] = model.tokenizer.decode(
            next_output_ids.tolist(),  # type:ignore
            skip_special_tokens=False,
            return_as="list",
        )
        all_transition_labels["forward_in"].append(["<bos>"] + in_labels)
        all_transition_labels["forward_out"].append(["<bos>"] + out_labels)
        all_transition_labels["backward_in"].append(
            ["<eos>"] + list(reversed(in_labels))
        )
        all_transition_labels["backward_out"].append(
            ["<eos>"] + list(reversed(out_labels))
        )
    return activations, all_transition_labels


if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument("--model-id", help="WandB shortname for the training run")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    extract_bimachine(
        hparams=ExtractionHyperparameters(
            model_shortname=args.model_id,
            dim_reduction_method="none",
            clustering_method="kmeans",
            kmeans_num_clusters=100,
            use_faiss=False,
            minimum_transition_count=None,
            state_split_classifier="svm",
            full_domain=True,
            full_domain_mode="search",
            full_domain_search_n=3,
            do_merge=False,
            visualize=args.visualize,
        ),
        paths=create_paths_from_args(args),
    )
