"""
DeepFM trainer — mini-batch training for high-throughput pipelines.

Stage 15.6 change: per-event → mini-batch.

Events are buffered until BATCH_SIZE "like" events accumulate, then a single
forward + backward + optimizer.step() is executed over all positives and their
random negatives in one vectorised pass.  This amortises Python/PyTorch overhead
across the whole batch instead of paying it per event.

Batch layout (BATCH_SIZE=32):
  rows 0, 2, 4, ...  — positives  (label = 1)
  rows 1, 3, 5, ...  — negatives  (label = 0)
  total rows = BATCH_SIZE * 2 = 64

BCEWithLogitsLoss, Adam lr=1e-3.
The model / optimiser are module-level singletons shared with deepfm_inference.
"""

import logging
import random
from typing import List

import torch
from torch import nn

from deepfm_model import _model, NUM_ITEMS
from feature_builder import build_features

log = logging.getLogger("deepfm.trainer")

BATCH_SIZE = 32   # like-events to accumulate before one training step

_optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)
_criterion = nn.BCEWithLogitsLoss()

# Internal buffer — accumulates enriched like-events
_event_buffer: List[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_neg_item(exclude_id: str) -> str:
    while True:
        neg = str(random.randint(1, NUM_ITEMS - 1))
        if neg != exclude_id:
            return neg


def _train_batch(batch: List[dict]) -> float:
    """
    Build a tensor batch from buffered events and execute one training step.
    Each event contributes one positive row and one negative row.
    """
    user_id_ints, item_id_ints = [], []
    user_vecs, item_vecs, genre_vecs = [], [], []
    labels = []

    for event in batch:
        user_id  = str(event["user_id"])
        item_id  = str(event["item_id"])
        user_emb = event["user_emb"]
        item_emb = event["item_emb"]

        try:
            pos_feat = build_features(user_id, item_id, user_emb, item_emb)
        except Exception as exc:
            log.debug("build_features pos failed (%s, %s): %s", user_id, item_id, exc)
            continue

        neg_item_id = _random_neg_item(item_id)
        try:
            neg_feat = build_features(user_id, neg_item_id, user_emb, item_emb)
        except Exception as exc:
            log.debug("build_features neg failed: %s", exc)
            continue

        # Positive row
        user_id_ints.append(pos_feat["user_id_int"])
        item_id_ints.append(pos_feat["item_id_int"])
        user_vecs.append(pos_feat["user_emb"])
        item_vecs.append(pos_feat["item_emb"])
        genre_vecs.append(pos_feat["genre_vec"])
        labels.append(1.0)

        # Negative row
        user_id_ints.append(neg_feat["user_id_int"])
        item_id_ints.append(neg_feat["item_id_int"])
        user_vecs.append(neg_feat["user_emb"])
        item_vecs.append(neg_feat["item_emb"])
        genre_vecs.append(neg_feat["genre_vec"])
        labels.append(0.0)

    if not labels:
        return 0.0

    uid_t  = torch.tensor(user_id_ints, dtype=torch.long)
    iid_t  = torch.tensor(item_id_ints, dtype=torch.long)
    uvec_t = torch.tensor(user_vecs,  dtype=torch.float32)
    ivec_t = torch.tensor(item_vecs,  dtype=torch.float32)
    gvec_t = torch.tensor(genre_vecs, dtype=torch.float32)
    lbl_t  = torch.tensor(labels,     dtype=torch.float32)

    _model.train()
    _optimizer.zero_grad()
    logits = _model(uid_t, iid_t, uvec_t, ivec_t, gvec_t)
    loss   = _criterion(logits, lbl_t)
    loss.backward()
    _optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_step(event: dict) -> float:
    """
    Buffer one event.  Executes a training step only when BATCH_SIZE like-events
    have accumulated.  Returns the batch loss at that point, or 0.0 otherwise.
    """
    if event.get("event_type") != "like":
        return 0.0

    _event_buffer.append(event)

    if len(_event_buffer) < BATCH_SIZE:
        return 0.0

    batch = _event_buffer.copy()
    _event_buffer.clear()
    return _train_batch(batch)
