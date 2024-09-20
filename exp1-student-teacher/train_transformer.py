from src.modeling import TransformerModel
from src.optional_wandb import wandb
from src.tasks.inflection import create_dataloader
from src.training import evaluate, predict, train


def train_transformer(
    train_path: str,
    eval_path: str,
    test_path: str,
    batch_size=256,
    epochs=100,
    learning_rate=0.0001,
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dropout=0.1,
    seed=0,
):
    language = train_path.split("/")[-1].split(".")[0]
    hyperparams = locals()

    # Create dataloaders
    train_dataloader, tokenizer = create_dataloader(
        data_path=train_path, batch_size=batch_size
    )
    eval_dataloader, _ = create_dataloader(
        data_path=eval_path, batch_size=batch_size, pretrained_tokenizer=tokenizer
    )
    test_dataloader, _ = create_dataloader(
        data_path=test_path, batch_size=batch_size, pretrained_tokenizer=tokenizer
    )

    model = TransformerModel(
        tokenizer=tokenizer,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
    )
    wandb.init(
        project="fst-distillation.exp1",
        config={**hyperparams},
        save_code=True,
        group=language,
    )
    wandb.watch(models=model, log_freq=1)
    train(
        project_name="fst-distillation.exp1",
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )

    # Run predictions and evaluate on test data
    pred_token_ids, label_token_ids = predict(
        model=model, dataloader=test_dataloader, tokenizer=tokenizer, max_length=64
    )
    preds = [tokenizer.decode(ids) for ids in pred_token_ids]
    labels = [tokenizer.decode(ids) for ids in label_token_ids]

    print("Predicted\tLabel")
    print("\n".join([p + "\t" + l for p, l in zip(preds, labels)]))

    preds_table = wandb.Table(
        columns=["predicted", "label", "correct"],
        data=[[p, lab, "yes" if p == lab else "no"] for p, lab in zip(preds, labels)],
    )
    wandb.log({"test_predictions": preds_table})

    metrics = {
        "accuracy": evaluate.accuracy(preds, labels),
        "levenshtein": evaluate.levenshtein(preds, labels),
    }
    print(metrics)
    wandb.log({"test": metrics})


if __name__ == "__main__":
    train_transformer(
        train_path="./task0-data/DEVELOPMENT-LANGUAGES/austronesian/hil.trn",
        eval_path="./task0-data/DEVELOPMENT-LANGUAGES/austronesian/hil.dev",
        test_path="./task0-data/GOLD-TEST/hil.tst",
        epochs=60,
    )
