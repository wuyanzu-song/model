# import torch
# import torch.nn as nn
# from .attention import ScaledDotProductAttention
#
#
# class MultiHeadAttention(nn.Module):
#     """
#     Multi-Head Attention mechanism
#     """
#
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.d_v = d_model // num_heads
#
#         # Linear projections for Q, K, V
#         self.W_Q = nn.Linear(d_model, d_model, bias=False)
#         self.W_K = nn.Linear(d_model, d_model, bias=False)
#         self.W_V = nn.Linear(d_model, d_model, bias=False)
#         self.W_O = nn.Linear(d_model, d_model, bias=False)
#
#         self.attention = ScaledDotProductAttention(dropout)
#
#     def forward(self, Q, K, V, mask=None):
#         """
#         Args:
#             Q: Query matrix [batch_size, seq_len, d_model]
#             K: Key matrix [batch_size, seq_len, d_model]
#             V: Value matrix [batch_size, seq_len, d_model]
#             mask: Optional mask [batch_size, seq_len, seq_len]
#
#         Returns:
#             output: [batch_size, seq_len, d_model]
#         """
#         batch_size, seq_len = Q.size(0), Q.size(1)
#
#         # Linear projection and reshape for multi-head
#         Q = self.W_Q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.W_K(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.W_V(V).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
#
#         # Apply mask for multi-head
#         if mask is not None:
#             mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
#
#         # Apply scaled dot-product attention
#         output, attn_weights = self.attention(Q, K, V, mask)
#
#         # Concatenate heads and put through final linear layer
#         output = output.transpose(1, 2).contiguous().view(
#             batch_size, seq_len, self.d_model
#         )
#         output = self.W_O(output)
#
#         return output

import torch
import torch.nn as nn
from .attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query matrix [batch_size, seq_len, d_model]
            K: Key matrix [batch_size, seq_len, d_model]
            V: Value matrix [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, q_seq_len = Q.size(0), Q.size(1)
        k_seq_len, v_seq_len = K.size(1), V.size(1)

        # Linear projection and reshape for multi-head
        Q = self.W_Q(Q).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, v_seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # Apply mask for multi-head
        if mask is not None:
            # 确保掩码维度正确
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            mask = mask.expand(-1, self.num_heads, -1, -1)

        # Apply scaled dot-product attention
        output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        output = self.W_O(output)

        return output