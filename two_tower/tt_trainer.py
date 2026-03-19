"""
Two-Tower trainer — mini-batch training for high-throughput pipelines.

Stage 15.6 change: per-event -> mini-batch.

Like-events are buffered until BATCH_SIZE events accumulate, then a single
forward + backward + optimizer.step() is executed over all positives and their
negatives in one vectorised pass.

Batch layout (BATCH_SIZE=32, NEGATIVE_SAMPLES=2):
  row 0           — positive for event 0  (label = 1)
  rows 1, 2       — negatives for event 0 (label = 0)
  row 3           — positive for event 1  (label = 1)
  rows 4, 5       — negatives for event 1 (label = 0)
  ...
  total rows = BATCH_SIZE * (1 + NEGATIVE_SAMPLES) = 32 * 3 = 96

After the training step, user/item embeddings for ALL unique IDs that
appeared in the batch are cached in tt_store in one inference pass.
Hard-negative cache is invalidated for each positive item in the batch.

BCEWithLogitsLoss, Adam lr = LEARNING_RATE.
"""

import logging
import os
import sys
import random
from typing import List

import torch
import torch.nn as nn

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FS_DIR = os.path.join(_ROOT, "feature_store")
if _FS_DIR not in sys.path:
    sys.path.insert(0, _FS_DIR)

from tt_config import BATCH_SIZE, LEARNING_RATE, NEGATIVE_SAMPLES, NUM_ITEMS, NUM_USERS
from tt_model import _tt_model
from tt_store import tt_item_embeddings, tt_user_embeddings
from hard_negative_sampler import invalidate_cache, sample_hard_negatives

from user_store import get_user   # feature_store

log = logging.getLogger("two_tower.trainer")

_optimizer = torch.optim.Adam(_tt_model.parameters(), lr=LEARNING_RATE)
_criterion = nn.BCEWithLogitsLoss()

# Internal buffer — accumulates like-events until BATCH_SIZE is reached
_event_buffer: List[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_int(raw_id: str, modulus: int) -> int:
    try:
        return int(raw_id) % modulus
    except (ValueError, TypeError):
        return abs(hash(raw_id)) % modulus


def _build_neg_iids(user_id: str, item_id: str, iid: int) -> list:
    """
    Build a list of NEGATIVE_SAMPLES int item IDs using hard negatives with
    random fallback.
    """
    user      = get_user(user_id)
    user_seen = (user["likes"] | user["dislikes"]) if user else set()

    hard      = sample_hard_negatives(user_id, item_id, k=NEGATIVE_SAMPLES, user_seen=user_seen)
    hard_iids = [_to_int(h, NUM_ITEMS) for h in hard]

    needed = NEGATIVE_SAMPLES - len(hard_iids)
    used   = {iid} | set(hard_iids)
    while needed > 0:
        rand = random.randint(0, NUM_ITEMS - 1)
        if rand not in used:
            hard_iids.append(rand)
            used.add(rand)
            needed -= 1

    return hard_iids


def _train_batch(batch: List[dict]) -> float:
    """
    Build a tensor batch from buffered events and execute one training step.
    Each event contributes 1 positive row + NEGATIVE_SAMPLES negative rows.
    """
    all_user_ids: List[int] = []
    all_item_ids: List[int] = []
    labels:       List[float] = []

    # Track which (user_id str, item_id str, uid int, iid int) we processed
    processed = []

    for event in batch:
        user_id = str(event["user_id"])
        item_id = str(event["item_id"])

        uid = _to_int(user_id, NUM_USERS)
        iid = _to_int(item_id, NUM_ITEMS)
        neg_iids = _build_neg_iids(user_id, item_id, iid)

        # Positive row
        all_user_ids.append(uid)
        all_item_ids.append(iid)
        labels.append(1.0)

        # Negative rows
        for neg_iid in neg_iids:
            all_user_ids.append(uid)
            all_item_ids.append(neg_iid)
            labels.append(0.0)

        processed.append((user_id, item_id, uid, iid))

    if not labels:
        return 0.0

    uid_t = torch.tensor(all_user_ids, dtype=torch.long)
    iid_t = torch.tensor(all_item_ids, dtype=torch.long)
    lbl_t = torch.tensor(labels,       dtype=torch.float32)

    _tt_model.train()
    _optimizer.zero_grad()
    logits = _tt_model(uid_t, iid_t)
    loss   = _criterion(logits, lbl_t)
    loss.backward()
    _optimizer.step()

    # Cache updated vectors for all unique user/item IDs in the batch
    unique_uids = list({uid for _, _, uid, _ in processed})
    unique_iids = list({iid for _, _, _, iid in processed})

    _tt_model.eval()
    with torch.no_grad():
        u_vecs = _tt_model.get_user_vec(torch.tensor(unique_uids, dtype=torch.long))
        i_vecs = _tt_model.get_item_vec(torch.tensor(unique_iids, dtype=torch.long))

    # Map int IDs back to string IDs for tt_store
    uid_to_str: dict = {}
    iid_to_str: dict = {}
    for user_id, item_id, uid, iid in processed:
        uid_to_str[uid] = user_id
        iid_to_str[iid] = item_id

    for idx, uid in enumerate(unique_uids):
        if uid in uid_to_str:
            tt_user_embeddings[uid_to_str[uid]] = u_vecs[idx].detach()

    for idx, iid in enumerate(unique_iids):
        if iid in iid_to_str:
            tt_item_embeddings[iid_to_str[iid]] = i_vecs[idx].detach()

    # Invalidate hard-neg cache for each positive item
    for _, item_id, _, _ in processed:
        invalidate_cache(item_id)

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
