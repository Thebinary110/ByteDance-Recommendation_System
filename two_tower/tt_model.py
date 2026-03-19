"""
Two-Tower model — UserEncoder + ItemEncoder in a shared 64-dim embedding space.

Architecture:
  UserEncoder: nn.Embedding(NUM_USERS, EMBED_DIM)  → L2-normalised
  ItemEncoder: nn.Embedding(NUM_ITEMS, EMBED_DIM)  → L2-normalised
  TwoTower.forward returns element-wise dot product (= cosine sim after L2-norm).

L2 normalisation pushes all vectors onto the unit hypersphere, so the dot
product equals cosine similarity — stable for BCEWithLogitsLoss training.

Module-level singleton `_tt_model` is imported by both tt_trainer and tt_inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tt_config import EMBED_DIM, NUM_ITEMS, NUM_USERS


class UserEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(NUM_USERS, EMBED_DIM)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """(B,) int64 → (B, EMBED_DIM) float32, L2-normalised."""
        return F.normalize(self.embed(user_ids), dim=-1)


class ItemEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(NUM_ITEMS, EMBED_DIM)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """(B,) int64 → (B, EMBED_DIM) float32, L2-normalised."""
        return F.normalize(self.embed(item_ids), dim=-1)


class TwoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.user_encoder = UserEncoder()
        self.item_encoder = ItemEncoder()

    def forward(
        self,
        user_ids: torch.Tensor,   # (B,) int64
        item_ids: torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:            # (B,)  float32 logits (dot product)
        u = self.user_encoder(user_ids)
        i = self.item_encoder(item_ids)
        return (u * i).sum(dim=-1)   # batched dot product

    def get_user_vec(self, user_id_tensor: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised user vector — used by trainer for caching."""
        return self.user_encoder(user_id_tensor)

    def get_item_vec(self, item_id_tensor: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised item vector — used by trainer for caching."""
        return self.item_encoder(item_id_tensor)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_tt_model: TwoTower = TwoTower()
