"""
In-memory embedding store.

Holds one torch.Tensor per user and per item.  Tensors are created on first
access (lazy initialisation) with random values and requires_grad=True so the
trainer can compute gradients against them.

All embeddings are leaf tensors — they are the model's parameters.

Note: no locking is needed here because the asyncio event loop runs on a
single thread.  If moved to multi-threaded serving, add threading.RLock().
"""

import torch
from typing import Dict, Tuple

from emb_config import EMBEDDING_DIM

user_embeddings: Dict[str, torch.Tensor] = {}
item_embeddings: Dict[str, torch.Tensor] = {}


def get_user_embedding(user_id: str) -> torch.Tensor:
    """Return (or lazily create) the embedding tensor for user_id."""
    if user_id not in user_embeddings:
        user_embeddings[user_id] = torch.randn(EMBEDDING_DIM, requires_grad=True)
    return user_embeddings[user_id]


def get_item_embedding(item_id: str) -> torch.Tensor:
    """Return (or lazily create) the embedding tensor for item_id."""
    if item_id not in item_embeddings:
        item_embeddings[item_id] = torch.randn(EMBEDDING_DIM, requires_grad=True)
    return item_embeddings[item_id]


def embedding_counts() -> Tuple[int, int]:
    """Return (n_users, n_items) currently in the store."""
    return len(user_embeddings), len(item_embeddings)
