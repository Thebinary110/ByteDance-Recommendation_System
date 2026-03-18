"""
DeepFM model — wide (FM) + deep (MLP) architecture for ranking.

Input features per (user, item) pair:
  - user_id_int  : int  → nn.Embedding(NUM_USERS, EMBED_DIM)
  - item_id_int  : int  → nn.Embedding(NUM_ITEMS, EMBED_DIM)
  - user_emb     : float32[EMB_DIM]   (from two-tower, already normalised)
  - item_emb     : float32[EMB_DIM]   (from two-tower, already normalised)
  - genre_vec    : float32[GENRE_DIM] (multi-hot)

Wide part  (FM linear): learns first-order feature weights.
Deep part  (MLP):       3-layer with ReLU + Dropout(0.1).
Output     : scalar logit (BCEWithLogitsLoss during training).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEEP_DIR = os.path.join(_ROOT, "deepfm")
if _DEEP_DIR not in sys.path:
    sys.path.insert(0, _DEEP_DIR)

from feature_builder import GENRE_DIM   # 19

# ---------------------------------------------------------------------------
# Hyper-parameters — referenced by feature_builder and trainer
# ---------------------------------------------------------------------------
NUM_USERS: int = 150_000
NUM_ITEMS: int = 200_000
EMBED_DIM: int = 32          # ID-embedding dimension
EMB_DIM:   int = 32          # two-tower embedding dimension (must match embedding_model)


class DeepFM(nn.Module):
    """
    DeepFM ranker.

    Fused input dim  = EMBED_DIM + EMBED_DIM + EMB_DIM + EMB_DIM + GENRE_DIM
                     = 32 + 32 + 32 + 32 + 19 = 147

    Wide (FM linear): linear projection of fused input → 1 scalar.
    Deep (MLP):       147 → 128 → 64 → 1 with ReLU + Dropout(0.1).
    Final logit:      wide_out + deep_out  (no sigmoid — BCEWithLogitsLoss).
    """

    def __init__(self) -> None:
        super().__init__()

        self.user_embed = nn.Embedding(NUM_USERS, EMBED_DIM)
        self.item_embed = nn.Embedding(NUM_ITEMS, EMBED_DIM)

        fused_dim = EMBED_DIM + EMBED_DIM + EMB_DIM + EMB_DIM + GENRE_DIM  # 147

        # Wide part — FM linear
        self.wide = nn.Linear(fused_dim, 1, bias=True)

        # Deep part — MLP with dropout
        self.deep = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        for m in [self.wide] + list(self.deep.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_id:   torch.Tensor,   # (B,) int64
        item_id:   torch.Tensor,   # (B,) int64
        user_vec:  torch.Tensor,   # (B, EMB_DIM) float32
        item_vec:  torch.Tensor,   # (B, EMB_DIM) float32
        genre_vec: torch.Tensor,   # (B, GENRE_DIM) float32
    ) -> torch.Tensor:             # (B,) float32 logits
        u = self.user_embed(user_id)   # (B, EMBED_DIM)
        i = self.item_embed(item_id)   # (B, EMBED_DIM)

        # Fuse learned ID embeddings with two-tower embeddings (L2-normalised)
        u_fused = F.normalize(u + user_vec, dim=-1)
        i_fused = F.normalize(i + item_vec, dim=-1)

        x = torch.cat([u_fused, i_fused, user_vec, item_vec, genre_vec], dim=-1)

        logit = self.wide(x).squeeze(-1) + self.deep(x).squeeze(-1)
        return logit


# ---------------------------------------------------------------------------
# Module-level singleton — shared by trainer and inference
# ---------------------------------------------------------------------------
_model: DeepFM = DeepFM()
