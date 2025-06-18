from src.modeling import RNNModel
from src.optional_wandb import wandb
from src.tasks.inflection_classification import create_dataloader
from src.training_classifier import train


def train_rnn(
    aligned_train_path: str,
    aligned_eval_path: str,
    test_path: str,
    batch_size=256,
    epochs=100,
    learning_rate=0.0001,
    d_model=512,
    num_layers=4,
    dropout=0.1,
    seed=0,
):
    language = aligned_train_path.split("/")[-1].split(".")[0]
    hyperparams = locals()

    # Create dataloaders
    train_dataloader, tokenizer = create_dataloader(
        aligned_data_path=aligned_train_path, batch_size=batch_size
    )
    eval_dataloader, _ = create_dataloader(
        aligned_data_path=aligned_eval_path,
        batch_size=batch_size,
        pretrained_tokenizer=tokenizer,
    )

    model = RNNModel(
        tokenizer=tokenizer, d_model=d_model, num_layers=num_layers, dropout=dropout
    )
    wandb.init(
        project="fst-distillation.exp2",
        config={**hyperparams},
        save_code=True,
        group=language,
    )
    wandb.watch(models=model, log_freq=1)
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )


if __name__ == "__main__":
    train_rnn(
        aligned_train_path="exp2-clustering/aligned_data/hil.trn.aligned.jsonl",
        aligned_eval_path="./exp2-clustering/aligned_data/hil.dev.aligned.jsonl",
        test_path="./task0-data/GOLD-TEST/hil.tst",
        epochs=100,
    )
