import logging
from random import sample

from pyfoma._private import algorithms
from pyfoma.fst import FST
from tqdm import tqdm

from .data.unaligned.example import String2StringExample

logger = logging.getLogger(__name__)


def evaluate_all(
    fst: FST,
    examples: list[String2StringExample],
    output_raw_string=False,
    log=False,
):
    labels: list[str] = []
    preds: list[set[str]] = []
    indices_to_log = sample(range(len(examples)), k=10)
    for idx, example in tqdm(enumerate(examples), "Evaluating"):
        input_string = [c for c in example.input_string]
        assert example.output_string is not None
        correct_output = [c for c in example.output_string]
        if example.features is not None:
            features = [f"[{f}]" for f in example.features]
            input_string = features + ["<sep>"] + input_string + ["<sink>"]
            if not output_raw_string:
                correct_output = features + ["<sep>"] + correct_output + ["<sink>"]
        labels.append("".join(correct_output))

        # Generate outputs by composing input acceptor with transducer
        logger.debug(f"Composing input string: {''.join(input_string)}")
        input_fsa = FST.re("".join(f"'{c}'" for c in input_string))
        logger.debug("Composing input @ fst")
        output_fst = input_fsa @ fst
        logger.debug("Minimizing")
        output_fst = output_fst.minimize()

        if len(output_fst.finalstates) == 0:
            logger.debug(
                f"FST has no accepting states for input {''.join(input_string)}"
            )
            preds_for_example = set()
        else:
            output_fst = output_fst.project(-1)

            # The following runs endlessly for very long inputs
            best_word = "".join(c[0] for c in algorithms.best_word(output_fst))
            preds_for_example = set([best_word])
        preds.append(preds_for_example)
        if log and idx in indices_to_log:
            logger.info(f"Input:\t{''.join(input_string)}")
            logger.info(f"Gold:\t{''.join(correct_output)}")
            logger.info(f"Predicted:\t{preds_for_example}")
    return compute_metrics(labels, preds)


def compute_metrics(labels: list[str], predictions: list[set[str]]):
    assert len(labels) == len(predictions)
    precision_sum = 0
    recall_sum = 0
    accepted_sum = 0

    for label, preds in zip(labels, predictions):
        if len(preds) > 0:
            accepted_sum += 1
        if label not in preds:
            # Add 0 to both prec and recall
            continue
        precision_sum += 1 / len(preds)
        recall_sum += 1
    precision = precision_sum / len(labels)
    recall = recall_sum / len(labels)
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    accepted_percentage = accepted_sum / len(labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accepted_percentage": accepted_percentage,
    }
