import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .ffn import PositionwiseFFN
from .normalization import LayerNorm
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len, seq_len]
        """
        attn_output = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
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

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
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

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tokens [batch_size, seq_len]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        """
        x = self.token_embedding(x) * (self.d_model ** 0.5)

        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x
