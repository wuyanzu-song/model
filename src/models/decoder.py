import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .ffn import PositionwiseFFN
from .normalization import LayerNorm
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: Decoder input [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            self_mask: Causal mask for decoder self-attention [batch_size, tgt_seq_len, tgt_seq_len]
            cross_mask: Mask for encoder-decoder attention [batch_size, tgt_seq_len, src_seq_len]
        """
        self_attn_output = self.self_attention(x, x, x, self_mask)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)

        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=512, dropout=0.1, use_pos_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = None

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: Target sequence [batch_size, tgt_seq_len]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            self_mask: Causal mask for decoder
            cross_mask: Mask for encoder-decoder attention
        """
        x = self.token_embedding(x) * (self.d_model ** 0.5)

        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, self_mask, cross_mask)

        return x
