from torch import Tensor, nn

from .tokenizer import Tokenizer


class RNNModel(nn.Module):
    MODEL_TYPE = "rnn"

    def __init__(
        self,
        *args,
        tokenizer: Tokenizer | dict,
        d_model: int,
        num_layers: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, Tokenizer)
            else Tokenizer.from_state_dict(tokenizer)
        )

        vocab_size = len(self.tokenizer)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            # Do we want bidirectional?
        )
        self.out = nn.Linear(in_features=d_model, out_features=1)

    @property
    def config_dict(self):
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    @classmethod
    def load(cls, checkpoint_dict: dict, tokenizer: Tokenizer):
        assert "state_dict" in checkpoint_dict and "config_dict" in checkpoint_dict
        model = cls(**checkpoint_dict["config_dict"], tokenizer=tokenizer)
        model.load_state_dict(checkpoint_dict["state_dict"])
        return model

    def forward(
        self,
        input_ids: Tensor,
        seq_lengths: Tensor,
    ):
        """
        Arguments:
            input_ids: Tensor, shape ``[batch_size, seq_length]``
            seq_lengths: Tensor, shape ``[batch_size]``. Lengths (ignoring padding) for each sequence. Should be on cpu.
        Returns:
            torch.Tensor, shape ``[batch_size]``: Output tensor with the predicted sequence probabilities.
        """
        src_embeddings = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            src_embeddings, lengths=seq_lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden_states = self.rnn.forward(input=packed)
        return self.out(hidden_states[-1]).squeeze(-1)
