import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query matrix [batch_size, num_heads, seq_len, d_k]
            K: Key matrix [batch_size, num_heads, seq_len, d_k]
            V: Value matrix [batch_size, num_heads, seq_len, d_v]
            mask: Optional mask [batch_size, num_heads, seq_len, seq_len]

        Returns:
            output: [batch_size, num_heads, seq_len, d_v]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights
