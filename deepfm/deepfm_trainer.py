"""
DeepFM trainer — online training step called once per event.

One positive sample (user liked item) + one hard negative (random unseen item)
per event.  BCEWithLogitsLoss, Adam optimiser, lr=1e-3.

The model and optimiser are module-level singletons shared with deepfm_inference.
"""

import logging
import random

import torch
import torch.nn.functional as F
from torch import nn

from deepfm_model import _model, NUM_ITEMS, EMBED_DIM
from feature_builder import build_features

log = logging.getLogger("deepfm.trainer")

# ---------------------------------------------------------------------------
# Optimiser — shares the same _model singleton
# ---------------------------------------------------------------------------
_optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)
_criterion = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensors(feat: dict) -> tuple:
    """Convert a build_features() dict to GPU-ready tensors (all on CPU here)."""
    return (
        torch.tensor([feat["user_id_int"]], dtype=torch.long),
        torch.tensor([feat["item_id_int"]], dtype=torch.long),
        torch.tensor(feat["user_emb"],  dtype=torch.float32).unsqueeze(0),
        torch.tensor(feat["item_emb"],  dtype=torch.float32).unsqueeze(0),
        torch.tensor(feat["genre_vec"], dtype=torch.float32).unsqueeze(0),
    )


def _random_neg_item(exclude_id: str) -> str:
    """Sample a random item_id (str) that is not the positive item."""
    while True:
        neg = str(random.randint(1, NUM_ITEMS - 1))
        if neg != exclude_id:
            return neg


# ---------------------------------------------------------------------------
# Public train step
# ---------------------------------------------------------------------------

def train_step(event: dict) -> float:
    """
    Process one streaming event.

    Parameters
    ----------
    event : dict
        Must contain: user_id, item_id, event_type,
                      user_emb (detached tensor), item_emb (detached tensor).
        Only 'like' events contribute a positive training signal.

    Returns
    -------
    float
        Loss value (0.0 for non-like events).
    """
    if event.get("event_type") != "like":
        return 0.0

    user_id  = str(event["user_id"])
    item_id  = str(event["item_id"])
    user_emb = event["user_emb"]   # already detached tensor
    item_emb = event["item_emb"]   # already detached tensor

    # --- positive sample ---
    try:
        pos_feat = build_features(user_id, item_id, user_emb, item_emb)
    except Exception as exc:
        log.warning("build_features failed for pos (%s, %s): %s", user_id, item_id, exc)
        return 0.0

    # --- negative sample (random unseen item, reuse user_emb) ---
    neg_item_id = _random_neg_item(item_id)
    try:
        neg_feat = build_features(user_id, neg_item_id, user_emb, item_emb)
    except Exception as exc:
        log.warning("build_features failed for neg (%s, %s): %s", user_id, neg_item_id, exc)
        return 0.0

    # --- batch: [positive, negative] ---
    def _cat(key):
        return torch.cat([
            torch.tensor(pos_feat[key] if not isinstance(pos_feat[key], int) else [pos_feat[key]]),
            torch.tensor(neg_feat[key] if not isinstance(neg_feat[key], int) else [neg_feat[key]]),
        ], dim=0)

    user_ids   = torch.tensor([pos_feat["user_id_int"], neg_feat["user_id_int"]], dtype=torch.long)
    item_ids   = torch.tensor([pos_feat["item_id_int"], neg_feat["item_id_int"]], dtype=torch.long)
    user_vecs  = torch.tensor(
        [pos_feat["user_emb"], neg_feat["user_emb"]], dtype=torch.float32
    )
    item_vecs  = torch.tensor(
        [pos_feat["item_emb"], neg_feat["item_emb"]], dtype=torch.float32
    )
    genre_vecs = torch.tensor(
        [pos_feat["genre_vec"], neg_feat["genre_vec"]], dtype=torch.float32
    )
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)

    _model.train()
    _optimizer.zero_grad()
    logits = _model(user_ids, item_ids, user_vecs, item_vecs, genre_vecs)
    loss   = _criterion(logits, labels)
    loss.backward()
    _optimizer.step()

    return loss.item()
