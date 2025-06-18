from torch import Tensor, nn

from .tokenizer import Tokenizer


class RNNModel(nn.Module):
    MODEL_TYPE = "rnn"

    def __init__(
        self,
        *args,
        tokenizer: Tokenizer,
        d_model: int,
        num_layers: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        vocab_size = len(tokenizer)
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
