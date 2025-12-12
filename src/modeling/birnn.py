from typing import Literal

import torch
from torch import Tensor, nn

from src.modeling.rnn import RNNModel
from src.modeling.tokenizer import Tokenizer


class BiRNN(nn.Module):
    def __init__(
        self,
        *args,
        tokenizer: Tokenizer | dict,
        d_model: int,
        num_layers: int,
        dropout: float,
        activation: Literal["relu", "tanh", "gelu"],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.tokenizer = (
            tokenizer
            if isinstance(tokenizer, Tokenizer)
            else Tokenizer.from_state_dict(tokenizer)
        )
        self.activation = activation

        vocab_size = len(self.tokenizer)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.forward_rnn = RNNModel(
            *args,
            tokenizer=self.tokenizer,
            output_head="none",
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            **kwargs,
        )
        self.backward_rnn = RNNModel(
            *args,
            tokenizer=self.tokenizer,
            output_head="none",
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            **kwargs,
        )
        self.out = nn.Linear(d_model * 3, vocab_size)

    @property
    def W_h(self):
        return self.forward_rnn.W_h + self.backward_rnn.W_h

    @property
    def config_dict(self):
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "activation": self.activation,
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
        next_input_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Computes a forward pass of our neural bimachine

        Expects a sequence of input_ids (B, S) where the first and last token are always the same, as in
        <bos> a b c <eos>

        Only the states in the middle will be trained. The next_input_ids (B, S-2) should be:
        a b c

        Will run two RNNs in either direction. Then, at each state we will train to predict the output given the input.
        E.g., we will use the state after <bos> (forward) and after 'b' (backward) to predict the output given 'a'.

        Returns:
            - out (B, S-2, d_vocab): Logits for the output vocabulary for each token
            - forward_states (B, S, d_model)
            - backward-states (B, S, d_model) - note this is in *input* order
        """
        assert next_input_ids.size(1) == input_ids.size(1) - 2
        embed = self.embedding(input_ids)  # (B, T, d_model)
        input_ids_flipped = flip_with_padding(
            input_ids, seq_lengths, self.tokenizer.pad_token_id
        )
        embed_flipped = self.embedding(input_ids_flipped)
        final_states_forward, _ = self.forward_rnn.compute_hidden_states(
            embed, seq_lengths
        )
        final_states_backward, _ = self.backward_rnn.compute_hidden_states(
            embed_flipped, seq_lengths
        )
        final_states_backward_flipped = flip_with_padding(
            final_states_backward, seq_lengths, self.tokenizer.pad_token_id
        )
        # We need to predict the next transition output in either direction
        # This means we need to cut off any states that aren't computed by both directions
        final_states = torch.concat(
            [final_states_forward[:, :-2], final_states_backward_flipped[:, 2:]], dim=2
        )  # B, T, d_model*2
        next_input_embeddings = self.embedding(next_input_ids)
        out = self.out(torch.cat([final_states, next_input_embeddings], dim=-1))
        return out, final_states_forward, final_states_backward


def flip_with_padding(x: Tensor, lengths: Tensor, pad_id: int):
    """
    T: batch_size, seq_length, ...
    lengths: batch_size
    """
    if len(x.shape) == 2:
        B, T = x.shape
        d = None
    elif len(x.shape) == 3:
        B, T, d = x.shape
    else:
        raise NotImplementedError("Requires shape (B,T) or (B,T,d)")

    idx = torch.arange(T, device=x.device).expand(B, T)
    lengths = lengths.unsqueeze(1)
    gather_idx = (lengths - 1 - idx).clamp(min=0)  # (B, T, ...)
    if d is not None:
        gather_idx = gather_idx.unsqueeze(-1).expand(B, T, d)
    reversed = torch.gather(x, 1, gather_idx)  # (B, T, ...)
    out = x.new_full(x.shape, pad_id)
    out[idx < lengths] = reversed[idx < lengths]
    return out
