
import torch.nn as nn
from .encoder import TransformerEncoder, EncoderLayer
from .decoder import TransformerDecoder
from .positional_encoding import PositionalEncoding


class EncoderDecoderTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer for Sequence-to-Sequence Tasks
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_len=512, dropout=0.1, use_pos_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding

        # Encoder
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers,
            d_ff, max_seq_len, dropout, use_pos_encoding
        )

        # Decoder
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_seq_len, dropout, use_pos_encoding
        )

        # Output projection
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
            src_mask: Source mask [batch_size, src_seq_len, src_seq_len]
            tgt_mask: Target causal mask [batch_size, tgt_seq_len, tgt_seq_len]
            memory_mask: Encoder-decoder attention mask
        """
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)

        # Decode target sequence using encoder output
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

        # Project to target vocabulary
        output = self.output_layer(decoder_output)

        return output

    def encode(self, src, src_mask=None):
        """Encode source sequence (for inference)"""
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        """Decode target sequence (for inference)"""
        return self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

    @property
    def device(self):
        return next(self.parameters()).device


class Transformer(nn.Module):
    """
    Encoder-only Transformer for Language Modeling (支持消融实验)
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=512, dropout=0.1, use_pos_encoding=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (optional for ablation studies)
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = None

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Weight initialization
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

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Token embedding
        x = self.token_embedding(x) * (self.d_model ** 0.5)

        # Positional encoding (if enabled)
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Output projection
        logits = self.output_layer(x)

        return logits

    @property
    def device(self):
        return next(self.parameters()).device