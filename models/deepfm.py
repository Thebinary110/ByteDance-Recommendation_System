"""
DeepFM ranker for scoring Two-Tower candidates.

Architecture (He & Chua 2017):
  FM component  : captures 2nd-order feature interactions without feature engineering
  Deep component: MLP learns arbitrary high-order interactions from concatenated embeddings

Input fields (both sparse & dense):
  Sparse (categorical) → per-field embedding
    • user_idx   (num_users)
    • movie_idx  (num_movies)
    • year_bucket (50 buckets)
  Dense (continuous) — concatenated directly:
    • genre multi-hot (20)
    • genome PCA (32)
    • user_avg_rating, user_log_count
    • item_avg_rating, item_log_count

Output: scalar logit (sigmoid → click probability)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMLayer(nn.Module):
    """
    Factorisation Machine layer.
    Computes sum of all pairwise embedding interactions in O(kn) instead of O(n²k).
    Formula: 0.5 * ( ||Σ v_i||² - Σ||v_i||² )  summed over interaction dim.
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, num_sparse_fields, embed_k]
        sum_sq = embeddings.sum(dim=1).pow(2)   # [B, embed_k]
        sq_sum = embeddings.pow(2).sum(dim=1)    # [B, embed_k]
        interaction = 0.5 * (sum_sq - sq_sum)    # [B, embed_k]
        return interaction.sum(dim=-1, keepdim=True)  # [B, 1]


class DeepFM(nn.Module):
    """
    DeepFM: jointly trains FM and deep components, shares embedding layer.

    Parameters
    ----------
    num_users, num_movies : vocabulary sizes for sparse fields
    num_year_buckets      : number of year buckets (default 50 covers ~1920-2020)
    embed_k               : embedding dimension per sparse field (FM / deep shared)
    dense_dim             : total dimension of continuous dense features
    mlp_dims              : hidden dimensions for the deep component
    dropout               : dropout probability
    """

    NUM_GENRES = 20

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        dense_dim: int,
        num_year_buckets: int = 50,
        embed_k: int = 16,
        mlp_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        mlp_dims = mlp_dims or [400, 400, 400]
        self.embed_k = embed_k

        # Sparse field embeddings (shared between FM and Deep)
        self.user_embed = nn.Embedding(num_users + 1, embed_k, padding_idx=0)
        self.item_embed = nn.Embedding(num_movies + 1, embed_k, padding_idx=0)
        self.year_embed = nn.Embedding(num_year_buckets + 1, embed_k, padding_idx=0)
        self.num_sparse_fields = 3

        for emb in [self.user_embed, self.item_embed, self.year_embed]:
            nn.init.xavier_uniform_(emb.weight)

        # FM component — adds scalar bias per sparse field
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # FM layer
        self.fm = FMLayer()

        # Deep component
        deep_input_dim = self.num_sparse_fields * embed_k + dense_dim
        layers: list[nn.Module] = []
        in_d = deep_input_dim
        for h in mlp_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.deep = nn.Sequential(*layers)

        # First-order (linear) term weights per sparse field
        self.user_linear = nn.Embedding(num_users + 1, 1, padding_idx=0)
        self.item_linear = nn.Embedding(num_movies + 1, 1, padding_idx=0)
        self.year_linear = nn.Embedding(num_year_buckets + 1, 1, padding_idx=0)
        self.dense_linear = nn.Linear(dense_dim, 1, bias=False)

    def _get_embeddings(
        self,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor,
        year_bucket: torch.Tensor,
    ) -> torch.Tensor:
        """Returns stacked sparse embeddings [B, num_sparse, embed_k]."""
        u = self.user_embed(user_idx).unsqueeze(1)      # [B, 1, k]
        m = self.item_embed(movie_idx).unsqueeze(1)     # [B, 1, k]
        y = self.year_embed(year_bucket).unsqueeze(1)   # [B, 1, k]
        return torch.cat([u, m, y], dim=1)               # [B, 3, k]

    def forward(
        self,
        user_idx: torch.Tensor,    # [B]
        movie_idx: torch.Tensor,   # [B]
        year_bucket: torch.Tensor, # [B]
        dense_features: torch.Tensor,  # [B, dense_dim]
    ) -> torch.Tensor:
        """Returns raw logit [B] (apply sigmoid for probability)."""
        embeddings = self._get_embeddings(user_idx, movie_idx, year_bucket)

        # FM first-order terms
        linear_part = (
            self.user_linear(user_idx)
            + self.item_linear(movie_idx)
            + self.year_linear(year_bucket)
            + self.dense_linear(dense_features)
            + self.fm_bias
        )  # [B, 1]

        # FM second-order interaction term
        fm_part = self.fm(embeddings)  # [B, 1]

        # Deep component: flatten embeddings + dense
        flat_embed = embeddings.view(embeddings.size(0), -1)   # [B, 3*k]
        deep_input = torch.cat([flat_embed, dense_features], dim=-1)  # [B, 3k+dense]
        deep_part = self.deep(deep_input)  # [B, 1]

        logit = linear_part + fm_part + deep_part  # [B, 1]
        return logit.squeeze(-1)  # [B]

    def predict_proba(
        self,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor,
        year_bucket: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns click probability in [0, 1]."""
        return torch.sigmoid(self.forward(user_idx, movie_idx, year_bucket, dense_features))


def build_dense_features(
    user_features: torch.Tensor,   # [B, user_feat_dim]
    item_features: torch.Tensor,   # [B, item_feat_dim]
) -> torch.Tensor:
    """Concatenate user and item continuous features for DeepFM dense input."""
    return torch.cat([user_features, item_features], dim=-1)
