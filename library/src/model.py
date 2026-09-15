"""Taste-scoring neural network for music recommendations.

This model learns to predict how much a user will like a song based on
their listening history. It uses audio embeddings (MERT) and play-count
scores to learn which latent audio dimensions matter for each user's taste.

Architecture:
    1. FiLM conditioning: scores modulate embeddings per-dimension
    2. Library encoder: self-attention with score bias
    3. Candidate scorer: cross-attention to library, then regression head
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from library.src.layers import ScoreBiasedEncoderLayer, CrossAttentionDecoderLayer


class ScoreFiLM(nn.Module):
    """Score-conditioned Feature-wise Linear Modulation.

    Takes a scalar score and produces per-dimension scale (gamma) and shift (beta)
    factors for the embedding. This allows the model to learn how scores should
    modulate different embedding dimensions.

    output = gamma(score) * embedding + beta(score)
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """Initialize FiLM module.

        Args:
            embedding_dim: Dimension of input embeddings (e.g., 1024 for MERT)
            hidden_dim: Hidden dimension for scale/shift networks
        """
        super().__init__()

        # Score -> per-dimension scale factors
        self.scale_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Score -> per-dimension shift factors
        self.shift_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Initialize for stable training: gamma ≈ 1, beta ≈ 0
        # Use small weights (not zero, to allow gradient flow) with bias offset
        nn.init.normal_(self.scale_net[-1].weight, mean=0, std=0.01)
        nn.init.ones_(self.scale_net[-1].bias)
        nn.init.normal_(self.shift_net[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.shift_net[-1].bias)

    def forward(
        self,
        embedding: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Apply score-conditioned modulation.

        Args:
            embedding: (B, T, embed_dim) input embeddings
            score: (B, T, 1) scores for each embedding

        Returns:
            (B, T, embed_dim) modulated embeddings
        """
        gamma = self.scale_net(score)  # (B, T, embed_dim)
        beta = self.shift_net(score)   # (B, T, embed_dim)
        return gamma * embedding + beta


class LibraryEncoder(nn.Module):
    """Encodes a user's library into contextualized representations.

    Pipeline:
        1. FiLM: modulate embeddings by their scores
        2. Project: reduce from embed_dim to d_model
        3. Self-attention layers with score bias
        4. Final layer norm

    The score bias in self-attention allows highly-scored songs to have
    more influence on other songs' representations.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        """Initialize library encoder.

        Args:
            embed_dim: Input embedding dimension (1024 for MERT)
            d_model: Transformer hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            dropout: Dropout probability
            dim_feedforward: FFN hidden dimension
        """
        super().__init__()

        self.film = ScoreFiLM(embed_dim, hidden_dim=64)
        self.input_proj = nn.Linear(embed_dim, d_model)

        self.layers = nn.ModuleList([
            ScoreBiasedEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode library embeddings.

        Args:
            embeddings: (B, N, embed_dim) audio embeddings
            scores: (B, N, 1) normalized play-count scores
            padding_mask: (B, N) True where padded

        Returns:
            (B, N, d_model) encoded library representations
        """
        # FiLM modulation
        x = self.film(embeddings, scores)  # (B, N, embed_dim)

        # Project to transformer dimension
        x = self.input_proj(x)  # (B, N, d_model)

        # Score-biased self-attention layers
        scores_squeezed = scores.squeeze(-1)  # (B, N) for attention bias
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask, scores=scores_squeezed)

        return self.final_norm(x)


class CandidateBackbone(nn.Module):
    """Cross-attends candidates to encoded library; returns (B, M, d_model).

    The featureizer half of the original CandidateScorer — kept separate so
    the hurdle model can share it across its two heads.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()

        self.input_proj = nn.Linear(embed_dim, d_model)

        self.layers = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        library_encoded: torch.Tensor,
        library_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(candidate_embeddings)
        for layer in self.layers:
            x = layer(x, library_encoded, memory_key_padding_mask=library_padding_mask)
        return self.final_norm(x)


class CandidateScorer(nn.Module):
    """Scores candidate songs against an encoded library.

    Pipeline:
        1. CandidateBackbone -> (B, M, d_model) features
        2. Regression head -> (B, M, 1) predicted score
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        """Initialize candidate scorer.

        Args:
            embed_dim: Input embedding dimension (1024 for MERT)
            d_model: Transformer hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dropout: Dropout probability
            dim_feedforward: FFN hidden dimension
        """
        super().__init__()

        self.backbone = CandidateBackbone(
            embed_dim, d_model, num_heads, num_layers, dropout, dim_feedforward
        )

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    # Pre-refactor checkpoints stored weights under input_proj / layers /
    # final_norm directly on CandidateScorer. Map those keys onto the new
    # backbone submodule so existing best.pt files load unchanged.
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs):
        legacy_prefixes = ("input_proj.", "layers.", "final_norm.")
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix):]
            if tail.startswith(legacy_prefixes):
                new_key = prefix + "backbone." + tail
                state_dict[new_key] = state_dict.pop(key)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        library_encoded: torch.Tensor,
        library_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score candidates against encoded library.

        Args:
            candidate_embeddings: (B, M, embed_dim) candidate audio embeddings
            library_encoded: (B, N, d_model) encoded library from LibraryEncoder
            library_padding_mask: (B, N) True where library is padded

        Returns:
            (B, M, 1) predicted scores for each candidate
        """
        feats = self.backbone(candidate_embeddings, library_encoded, library_padding_mask)
        return self.score_head(feats)


class TasteScoringModel(nn.Module):
    """Complete taste-scoring model.

    Combines LibraryEncoder and CandidateScorer to predict how much a user
    will like candidate songs based on their listening history.

    Usage:
        # Training: full forward pass
        pred_scores = model(ctx_emb, ctx_scores, tgt_emb, ctx_mask)
        loss = F.mse_loss(pred_scores, tgt_scores)

        # Inference: encode library once, score many candidates
        library_enc = model.encode_library(emb, scores, mask)
        for batch in candidate_batches:
            scores = model.score_candidates(batch, library_enc, mask)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 512,
        num_heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        """Initialize taste-scoring model.

        Args:
            embed_dim: Input embedding dimension (1024 for MERT)
            d_model: Transformer hidden dimension
            num_heads: Number of attention heads
            enc_layers: Number of library encoder layers
            dec_layers: Number of candidate scorer layers
            dropout: Dropout probability
            dim_feedforward: FFN hidden dimension
        """
        super().__init__()

        self.library_encoder = LibraryEncoder(
            embed_dim, d_model, num_heads, enc_layers, dropout, dim_feedforward
        )
        self.candidate_scorer = CandidateScorer(
            embed_dim, d_model, num_heads, dec_layers, dropout, dim_feedforward
        )

        # Store config for serialization
        self.config = {
            "model_type": "scoring",
            "embed_dim": embed_dim,
            "d_model": d_model,
            "num_heads": num_heads,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "dropout": dropout,
            "dim_feedforward": dim_feedforward,
        }

    def encode_library(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode user library. Call once per user, cache result.

        Args:
            embeddings: (B, N, embed_dim) library audio embeddings
            scores: (B, N, 1) normalized play-count scores
            padding_mask: (B, N) True where padded

        Returns:
            (B, N, d_model) encoded library representations
        """
        return self.library_encoder(embeddings, scores, padding_mask)

    def score_candidates(
        self,
        candidate_embeddings: torch.Tensor,
        library_encoded: torch.Tensor,
        library_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score candidates against pre-encoded library.

        Args:
            candidate_embeddings: (B, M, embed_dim) candidate audio embeddings
            library_encoded: (B, N, d_model) from encode_library()
            library_padding_mask: (B, N) True where library is padded

        Returns:
            (B, M, 1) predicted scores
        """
        return self.candidate_scorer(
            candidate_embeddings, library_encoded, library_padding_mask
        )

    def forward(
        self,
        library_embeddings: torch.Tensor,
        library_scores: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        library_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass for training.

        Args:
            library_embeddings: (B, N, embed_dim) context library embeddings
            library_scores: (B, N, 1) context scores
            candidate_embeddings: (B, M, embed_dim) target embeddings to score
            library_padding_mask: (B, N) True where library is padded

        Returns:
            (B, M, 1) predicted scores for candidates
        """
        encoded = self.encode_library(
            library_embeddings, library_scores, library_padding_mask
        )
        return self.score_candidates(
            candidate_embeddings, encoded, library_padding_mask
        )

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience method for training with dataloader batches.

        Args:
            batch: Dict from TasteDataset containing:
                - context_embeddings: (B, N, embed_dim)
                - context_scores: (B, N, 1)
                - context_mask: (B, N)
                - target_embeddings: (B, M, embed_dim)

        Returns:
            (B, M, 1) predicted scores for targets
        """
        return self.forward(
            library_embeddings=batch["context_embeddings"],
            library_scores=batch["context_scores"],
            candidate_embeddings=batch["target_embeddings"],
            library_padding_mask=batch["context_mask"],
        )

    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters.

        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class TasteHurdleModel(nn.Module):
    """Two-head hurdle model.

    Factorizes "would the user play this?" (BCE head) from "given they would,
    how much?" (MSE head on positives only). Same backbone as TasteScoringModel,
    so the only differences live in the heads + loss.

    Heads share `CandidateBackbone` features but have separate output MLPs.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        d_model: int = 512,
        num_heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.library_encoder = LibraryEncoder(
            embed_dim, d_model, num_heads, enc_layers, dropout, dim_feedforward
        )
        self.candidate_backbone = CandidateBackbone(
            embed_dim, d_model, num_heads, dec_layers, dropout, dim_feedforward
        )
        self.play_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.config = {
            "model_type": "hurdle",
            "embed_dim": embed_dim,
            "d_model": d_model,
            "num_heads": num_heads,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "dropout": dropout,
            "dim_feedforward": dim_feedforward,
        }

    def encode_library(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.library_encoder(embeddings, scores, padding_mask)

    def score_candidates(
        self,
        candidate_embeddings: torch.Tensor,
        library_encoded: torch.Tensor,
        library_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Both heads at once. Returns a dict so callers can pick which to use.

        Returns:
            "play_logits": (B, M, 1) — logit, apply sigmoid to get P(played)
            "score_pred":  (B, M, 1) — conditional score | played
        """
        feats = self.candidate_backbone(
            candidate_embeddings, library_encoded, library_padding_mask
        )
        return {
            "play_logits": self.play_head(feats),
            "score_pred": self.score_head(feats),
        }

    def forward(
        self,
        library_embeddings: torch.Tensor,
        library_scores: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        library_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_library(
            library_embeddings, library_scores, library_padding_mask
        )
        return self.score_candidates(
            candidate_embeddings, encoded, library_padding_mask
        )

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(
            library_embeddings=batch["context_embeddings"],
            library_scores=batch["context_scores"],
            candidate_embeddings=batch["target_embeddings"],
            library_padding_mask=batch["context_mask"],
        )

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(
    model_type: str = "scoring",
    embed_dim: int = 1024,
    d_model: int = 512,
    num_heads: int = 8,
    enc_layers: int = 3,
    dec_layers: int = 2,
    dropout: float = 0.1,
    dim_feedforward: int = 2048,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Create and initialize a taste model.

    Args:
        model_type: "scoring" (single-head MSE) or "hurdle" (two-head BCE+MSE)
        embed_dim: Input embedding dimension (1024 for MERT)
        d_model: Transformer hidden dimension
        num_heads: Number of attention heads
        enc_layers: Number of library encoder layers
        dec_layers: Number of candidate scorer layers
        dropout: Dropout probability
        dim_feedforward: FFN hidden dimension
        device: Device to place model on

    Returns:
        Initialized model (TasteScoringModel or TasteHurdleModel).
    """
    arch_kwargs = dict(
        embed_dim=embed_dim,
        d_model=d_model,
        num_heads=num_heads,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
    )
    if model_type == "scoring":
        model = TasteScoringModel(**arch_kwargs)
    elif model_type == "hurdle":
        model = TasteHurdleModel(**arch_kwargs)
    else:
        raise ValueError(
            f"unknown model_type {model_type!r}; expected 'scoring' or 'hurdle'"
        )

    if device is not None:
        model = model.to(device)

    return model
