"""Usage: python -m src.alignment.train_alignment_predictor

Trains a transformer to use for predicting input strings with alignment characters.

You must run `run_alignment.py` first to produce aligned data files.
"""

from logging import getLogger

import torch

import wandb

from .data.aligned.alignment_prediction.dataloader import create_dataloader
from .data.aligned.alignment_prediction.dataset import (
    AlignmentPredictionDataset,
)
from .data.aligned.alignment_prediction.domain_cover import domain_cover
from .data.aligned.alignment_prediction.example import AlignmentPredictionExample
from .data.aligned.example import ALIGNMENT_SYMBOL, load_examples_from_file
from .modeling.transformer import TransformerModel
from .paths import Paths, create_arg_parser, create_paths_from_args
from .training.seq2seq import evaluate, predict
from .training.seq2seq.train import train

logger = getLogger(__name__)


def train_alignment_predictor(
    paths: Paths,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed=0,
):
    logger.info(f"Training on {paths['identifier']}")
    hyperparams = locals()
    wandb.init(
        entity="lecs-general",
        project="fst-distillation.clustering.alignment_prediction",
        config={**hyperparams},
        save_code=True,
    )

    # Load examples into alignment prediction format
    train_examples = load_examples_from_file(paths["train_aligned"])
    eval_examples = load_examples_from_file(paths["eval_aligned"])
    train_examples = [
        AlignmentPredictionExample.from_aligned(ex) for ex in train_examples
    ]
    eval_examples = [
        AlignmentPredictionExample.from_aligned(ex) for ex in eval_examples
    ]
    wandb.log({"train_size": len(train_examples)})
    train_dataset = AlignmentPredictionDataset(
        examples=train_examples,
        tokenizer=None,
    )
    tokenizer = train_dataset.tokenizer
    eval_dataset = AlignmentPredictionDataset(
        examples=eval_examples,
        tokenizer=tokenizer,
    )
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)  # type:ignore
    eval_dataloader = create_dataloader(eval_dataset, batch_size=batch_size)  # type:ignore

    model = TransformerModel(
        tokenizer=tokenizer,
        d_model=16,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
    )
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )

    predicted, labels = predict(
        model, eval_dataloader, tokenizer=tokenizer, max_length=1024
    )
    assert labels is not None
    predicted = [tokenizer.decode(p, skip_special_tokens=True) for p in predicted]
    labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
    accuracy = evaluate.accuracy(
        predictions=predicted,  # type:ignore
        labels=labels,  # type:ignore
    )
    print(f"incorrect: {[(p, l) for p, l in zip(predicted, labels) if p != l]}")
    wandb.log({"eval/accuracy": accuracy})

    # Run on full domain and write to a file
    logger.info("Running prediction on full domain")
    full_domain_examples = domain_cover(train_examples)
    full_domain_dataset = AlignmentPredictionDataset(full_domain_examples, tokenizer)
    full_domain_dataloader = create_dataloader(
        full_domain_dataset, batch_size=batch_size
    )
    full_domain_predictions, _ = predict(
        model, full_domain_dataloader, tokenizer=tokenizer, max_length=1024
    )
    with open(paths["full_domain_aligned"], "w") as f:
        for pred, original in zip(full_domain_predictions, full_domain_examples):
            decoded = tokenizer.decode(
                pred, skip_special_tokens=False, return_as="list"
            )
            try:
                split_index = decoded.index("<sep>")
                chars = [
                    c
                    for c in decoded[split_index + 1 :]
                    if c not in tokenizer.special_tokens
                ]
                if [c for c in chars if c != ALIGNMENT_SYMBOL] != original.unaligned:
                    # We messed up predicting the original lemma, skip
                    continue
                features = decoded[:split_index]
                f.write("".join(f"({c},?)" for c in chars))
                f.write(f"\t{';'.join(f[1:-1] for f in features)}\n")
            except:
                continue
    logger.info(
        f"Wrote {len(full_domain_predictions)} alignment preds to {paths['full_domain_aligned']}"
    )

    checkpoint_path = paths["models_folder"] / f"{wandb.run.name}/model.pt"  # type:ignore
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config_dict": model.config_dict,
            "tokenizer_dict": model.tokenizer.state_dict,
        },
        checkpoint_path,
    )
    run = wandb.run
    wandb.finish()
    return run


if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument("--batch-size", default=2048)
    parser.add_argument("--epochs", default=500)
    parser.add_argument("--learning-rate", default=0.001)
    args = parser.parse_args()
    train_alignment_predictor(
        paths=create_paths_from_args(args),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
    )
