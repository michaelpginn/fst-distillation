from typing import Literal

import torch
from torch import Tensor, nn

from .tokenizer import Tokenizer

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


class RNNModel(nn.Module):
    MODEL_TYPE = "rnn"

    def __init__(
        self,
        *args,
        tokenizer: Tokenizer | dict,
        output_head: Literal["classification", "lm"],
        d_model: int,
        num_layers: int,
        dropout: float,
        activation: Literal["relu", "tanh", "gelu"],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.output_head = output_head
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

        self.W_x = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.W_h = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        for i in range(num_layers):
            nn.init.xavier_uniform_(self.W_x[i].weight)
            nn.init.orthogonal_(self.W_h[i].weight)

        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(p=dropout if i < num_layers - 1 else 0)
                for i in range(num_layers)
            ]
        )
        match activation:
            case "tanh":
                self.activation_func = nn.Tanh()
            case "relu":
                self.activation_func = nn.ReLU()
            case "gelu":
                self.activation_func = nn.GELU()

        if output_head == "classification":
            self.out = nn.Linear(in_features=d_model, out_features=1)
        elif output_head == "lm":
            self.out = nn.Linear(in_features=d_model, out_features=vocab_size)

    @property
    def config_dict(self):
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_head": self.output_head,
            "activation": self.activation,
        }

    @classmethod
    def load(cls, checkpoint_dict: dict, tokenizer: Tokenizer):
        assert "state_dict" in checkpoint_dict and "config_dict" in checkpoint_dict
        model = cls(**checkpoint_dict["config_dict"], tokenizer=tokenizer)
        model.load_state_dict(checkpoint_dict["state_dict"])
        return model

    def compute_timestep(self, H_t_min1: Tensor, x_t: Tensor, mask: Tensor | None):
        """Computes a single timestep

        Args:
            H_t_min1: The prior timestep hidden states (batch_size, num_layers, d_model)
            x_t: The next input (batch_size, d_model)
            mask: Indicates whether each batch element should be masked (batch_size, 1)
        """
        H_t = []  # list of (B, d_model)
        for layer_idx in range(self.num_layers):
            if layer_idx > 0:
                x_t = H_t_min1[:, layer_idx - 1]

            # Transition using the input (previous layer hidden state) and hidden state (at previous timestep)
            H_t_layer = self.W_x[layer_idx](x_t) + self.W_h[layer_idx](
                H_t_min1[:, layer_idx]
            )
            H_t_layer = self.activation_func(H_t_layer)
            if layer_idx < self.num_layers - 1:
                H_t_layer = self.dropouts[layer_idx](H_t_layer)

            # Only update hidden state if the sequence is still going (not pad)
            if mask is not None:
                H_t_layer = H_t_layer * mask + H_t_min1[:, layer_idx] * (1 - mask)
            H_t.append(H_t_layer)
        return torch.stack(H_t, dim=1)

    def compute_hidden_states(self, embeddings: Tensor, seq_lengths: Tensor):
        """Computes the hidden states for a sequence of embeddings

        Args:
            embeddings: (batch_size, seq_length, d_model)
            seq_lengths: (batch_size)
        """
        B, T, _ = embeddings.shape

        # hidden state at current timestep (all layers), (B, L, d_model)
        H_t = torch.zeros((B, self.num_layers, self.d_model), device=embeddings.device)

        # last layer hidden state at each timestep (list of (B, d_model))
        final_hidden_states = []
        for t in range(T):
            mask = (t < seq_lengths).float().unsqueeze(1)  # B, 1
            H_t = self.compute_timestep(H_t, x_t=embeddings[:, t], mask=mask)
            final_hidden_states.append(H_t[:, -1])
        return torch.stack(final_hidden_states, dim=1), H_t

    def forward(
        self,
        input_ids: Tensor,
        seq_lengths: Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            input_ids: Tensor, shape ``[batch_size, seq_length]``
            seq_lengths: Tensor, shape ``[batch_size]``. Lengths (ignoring padding) for each sequence. Should be on cpu.
        Returns:
            torch.Tensor, shape ``[batch_size]``: Output tensor with the predicted sequence probabilities.
        """
        src_embeddings = self.embedding(input_ids)  # (B, T, d_model)
        final_hidden_states, _ = self.compute_hidden_states(src_embeddings, seq_lengths)

        if self.output_head == "classification":
            return self.out(final_hidden_states[:, -1]).squeeze(-1)
        elif self.output_head == "lm":
            return self.out(final_hidden_states)
        else:
            raise ValueError()
