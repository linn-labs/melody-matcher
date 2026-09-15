"""Custom transformer layers with score-based attention bias.

These layers extend standard transformer architecture to incorporate
play-count scores into the attention mechanism, allowing highly-scored
songs to have more influence on the library representation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreBiasedMultiHeadAttention(nn.Module):
    """Multi-head attention with optional score-based bias.

    When scores are provided, they bias the attention logits so that
    higher-scored keys receive more attention from all queries.

    This implements: attn_logits = (Q @ K^T) / sqrt(d_k) + score_bias
    where score_bias broadcasts the score of each key across all queries.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional score bias.

        Args:
            query: (B, T_q, d_model) query tensor
            key: (B, T_k, d_model) key tensor
            value: (B, T_k, d_model) value tensor
            key_padding_mask: (B, T_k) True where keys should be masked (padding)
            scores: (B, T_k) optional scores for attention bias

        Returns:
            (B, T_q, d_model) attended output
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project to Q, K, V
        Q = self.W_q(query)  # (B, T_q, d_model)
        K = self.W_k(key)    # (B, T_k, d_model)
        V = self.W_v(value)  # (B, T_k, d_model)

        # Reshape for multi-head: (B, num_heads, T, d_k)
        Q = Q.view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention logits: (B, num_heads, T_q, T_k)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply score bias if provided
        # scores: (B, T_k) -> (B, 1, 1, T_k) to broadcast across heads and queries
        if scores is not None:
            score_bias = scores.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_k)
            attn_logits = attn_logits + score_bias

        # Apply padding mask
        # key_padding_mask: (B, T_k) True = ignore -> set to -inf
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_k)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T_q, d_k)

        # Reshape back: (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # Output projection
        return self.W_o(out)


class ScoreBiasedEncoderLayer(nn.Module):
    """Transformer encoder layer with score-biased self-attention.

    Architecture:
        x -> LayerNorm -> ScoreBiasedMHA(x, x, x) -> Dropout -> + x
        -> LayerNorm -> FFN -> Dropout -> + x
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = ScoreBiasedMultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model) input tensor
            src_key_padding_mask: (B, T) True where padded
            scores: (B, T) attention bias scores

        Returns:
            (B, T, d_model) output tensor
        """
        # Pre-norm self-attention with residual
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            scores=scores,
        )
        x = x + self.dropout1(attn_out)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        return x


class CrossAttentionDecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention to encoded library.

    Architecture:
        x -> LayerNorm -> CrossAttention(x, memory, memory) -> Dropout -> + x
        -> LayerNorm -> FFN -> Dropout -> + x

    Note: No self-attention since candidates are scored independently.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Standard multi-head attention for cross-attention (no score bias needed)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T_q, d_model) query tensor (candidates)
            memory: (B, T_k, d_model) key/value tensor (encoded library)
            memory_key_padding_mask: (B, T_k) True where memory is padded

        Returns:
            (B, T_q, d_model) output tensor
        """
        # Pre-norm cross-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.cross_attn(
            x_norm, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout1(attn_out)

        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        return x
