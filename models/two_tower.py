"""
Two-Tower model for candidate generation.

Architecture:
  User Tower  : user_embed(user_idx) ⊕ user_features  → MLP → L2-normalised vector
  Item Tower  : item_embed(movie_idx) ⊕ item_features → MLP → L2-normalised vector
  Score       : dot product (inner product, equivalent to cosine after normalisation)

Training objective: Bayesian Personalised Ranking (BPR) loss.
  Loss = -log σ(score(u, pos) - score(u, neg))
  With IPS weighting applied to each triplet.

After training, item embeddings are indexed in FAISS for ANN retrieval.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple feedforward block with LayerNorm and Dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    """
    Encodes a user into a fixed-dim embedding.
    Input  : (user_idx [B], user_features [B, user_feat_dim])
    Output : L2-normalised vector [B, embed_dim]
    """

    def __init__(
        self,
        num_users: int,
        user_feat_dim: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.user_embed = nn.Embedding(num_users, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.user_embed.weight)
        self.mlp = MLP(embed_dim + user_feat_dim, hidden_dims, embed_dim, dropout)

    def forward(
        self, user_idx: torch.Tensor, user_features: torch.Tensor
    ) -> torch.Tensor:
        e = self.user_embed(user_idx)  # [B, embed_dim]
        x = torch.cat([e, user_features], dim=-1)  # [B, embed_dim + feat]
        out = self.mlp(x)  # [B, embed_dim]
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes a movie into a fixed-dim embedding.
    Input  : (movie_idx [B], item_features [B, item_feat_dim])
    Output : L2-normalised vector [B, embed_dim]
    """

    def __init__(
        self,
        num_movies: int,
        item_feat_dim: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.item_embed = nn.Embedding(num_movies, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embed.weight)
        self.mlp = MLP(embed_dim + item_feat_dim, hidden_dims, embed_dim, dropout)

    def forward(
        self, movie_idx: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        e = self.item_embed(movie_idx)  # [B, embed_dim]
        x = torch.cat([e, item_features], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Combines user and item towers.

    forward() returns (user_vec, pos_item_vec, neg_item_vec) for BPR training,
    or (user_vec, item_vec) when neg_movie_idx is None (inference/scoring).
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        user_feat_dim: int,
        item_feat_dim: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_tower = UserTower(
            num_users, user_feat_dim, embed_dim, hidden_dims, dropout
        )
        self.item_tower = ItemTower(
            num_movies, item_feat_dim, embed_dim, hidden_dims, dropout
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        user_idx: torch.Tensor,
        user_features: torch.Tensor,
        pos_movie_idx: torch.Tensor,
        pos_item_features: torch.Tensor,
        neg_movie_idx: torch.Tensor | None = None,
        neg_item_features: torch.Tensor | None = None,
    ) -> tuple:
        user_vec = self.user_tower(user_idx, user_features)
        pos_item_vec = self.item_tower(pos_movie_idx, pos_item_features)

        if neg_movie_idx is not None and neg_item_features is not None:
            neg_item_vec = self.item_tower(neg_movie_idx, neg_item_features)
            return user_vec, pos_item_vec, neg_item_vec

        return user_vec, pos_item_vec

    def encode_user(
        self, user_idx: torch.Tensor, user_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode users for serving/FAISS query."""
        return self.user_tower(user_idx, user_features)

    def encode_items(
        self, movie_idx: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode items for FAISS index building."""
        return self.item_tower(movie_idx, item_features)


class BPRLoss(nn.Module):
    """
    Bayesian Personalised Ranking loss with optional IPS weighting.
    Loss = -mean( ips_weight * log σ(pos_score - neg_score) )
    """

    def forward(
        self,
        user_vec: torch.Tensor,     # [B, D]
        pos_item_vec: torch.Tensor, # [B, D]
        neg_item_vec: torch.Tensor, # [B, D]
        ips_weights: torch.Tensor | None = None,  # [B]
    ) -> torch.Tensor:
        pos_score = (user_vec * pos_item_vec).sum(dim=-1)  # [B]
        neg_score = (user_vec * neg_item_vec).sum(dim=-1)  # [B]
        loss = -F.logsigmoid(pos_score - neg_score)         # [B]

        if ips_weights is not None:
            loss = loss * ips_weights

        return loss.mean()
