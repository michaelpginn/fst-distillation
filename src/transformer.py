import math

from jaxtyping import Bool, Num
from torch import Tensor, nn

from .positional_encoding import PositionalEncoding
from .tokenizer import Tokenizer
from .utils import create_causal_mask


class TransformerModel(nn.Module):
    MODEL_TYPE = "transformer"

    def __init__(
        self,
        *args,
        tokenizer: Tokenizer,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        vocab_size = len(tokenizer)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(
        self,
        src: Num[Tensor, "batch_size src_seq_length"],
        tgt: Num[Tensor, "batch_size tgt_seq_length"],
        src_pad_mask: Bool[Tensor, "batch_size src_seq_length"] | None = None,
        tgt_pad_mask: Bool[Tensor, "batch_size tgt_seq_length"] | None = None,
    ):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, src_seq_length]``
            tgt: Tensor, shape ``[batch_size, tgt_seq_length]``
            src_pad_mask: Tensor, shape ``[batch_size, src_seq_length]``
            tgt_pad_mask: Tensor, shape ``[batch_size, tgt_seq_length]``
        Returns:
            torch.Tensor: [batch_size, tgt_seq_length, vocab_size] Output tensor with the predicted token probabilities.
        """
        src_embeddings = self.embedding(src) * math.sqrt(self.d_model)
        tgt_embeddings = self.embedding(tgt) * math.sqrt(self.d_model)
        src_embeddings = self.positional_encoding(src_embeddings)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        tgt_causal_mask = create_causal_mask(seq_length=tgt.size(1)).to(tgt.device)

        transformer_out = self.transformer.forward(
            src=src_embeddings,
            tgt=tgt_embeddings,
            tgt_mask=tgt_causal_mask,
            tgt_is_causal=True,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.out(transformer_out)
