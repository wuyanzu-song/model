
import torch.nn as nn
from .encoder import TransformerEncoder, EncoderLayer
from .decoder import TransformerDecoder
from .positional_encoding import PositionalEncoding


class EncoderDecoderTransformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_len=512, dropout=0.1, use_pos_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers,
            d_ff, max_seq_len, dropout, use_pos_encoding
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_seq_len, dropout, use_pos_encoding
        )

        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
   
        encoder_output = self.encoder(src, src_mask)

        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

        output = self.output_layer(decoder_output)

        return output

    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        return self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

    @property
    def device(self):
        return next(self.parameters()).device


class Transformer(nn.Module):

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

        self.output_layer = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) * (self.d_model ** 0.5)

        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        logits = self.output_layer(x)

        return logits

    @property
    def device(self):
        return next(self.parameters()).device
