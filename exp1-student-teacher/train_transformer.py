import wandb
from src.inflection import create_dataloader
from src.predict import predict
from src.train import train
from src.transformer import TransformerModel


def train_transformer(
    train_path: str,
    eval_path: str,
    test_path: str,
    batch_size=8,
    epochs=60,
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

    # Create dataloaderes
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
        seed=seed,
    )
    preds = predict(model=model, dataloader=test_dataloader, max_length=64)
    print(preds)


if __name__ == "__main__":
    train_transformer(
        train_path="./task0-data/DEVELOPMENT-LANGUAGES/austronesian/hil.trn",
        eval_path="./task0-data/DEVELOPMENT-LANGUAGES/austronesian/hil.dev",
        test_path="./task0-data/DEVELOPMENT-LANGUAGES/austronesian/hil.tst",
    )
