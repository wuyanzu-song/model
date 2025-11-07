import torch
import torch.nn as nn
from .attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):

        batch_size, q_seq_len = Q.size(0), Q.size(1)
        k_seq_len, v_seq_len = K.size(1), V.size(1)

        Q = self.W_Q(Q).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, v_seq_len, self.num_heads, self.d_v).transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            mask = mask.expand(-1, self.num_heads, -1, -1)

        output, attn_weights = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        output = self.W_O(output)

        return output
