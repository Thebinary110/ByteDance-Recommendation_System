"""
Evaluation dataset builder — temporal train/test split from live user_store.

For each user with >= min_interactions total events:
  - sort history by timestamp (already ordered in the deque)
  - first (1 - test_ratio) fraction  → train_items  (all event types)
  - last  test_ratio fraction        → test_likes   (only "like" events)

Users with no likes in their test slice are skipped.

Returns:
    {user_id: {"train": [item_id, ...], "test": [item_id, ...]}}

No data is mutated. The returned lists are independent copies.
"""

import os
import sys

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FS_DIR = os.path.join(_ROOT, "feature_store")
if _FS_DIR not in sys.path:
    sys.path.insert(0, _FS_DIR)

from user_store import _lock, _store   # feature_store

from eval_config import MIN_INTERACTIONS as _DEFAULT_MIN, TEST_RATIO as _DEFAULT_TEST


def build_eval_dataset(
    min_interactions: int   = None,
    test_ratio:       float = None,
) -> dict:
    """
    Build and return the evaluation dataset from the current user_store snapshot.

    Parameters
    ----------
    min_interactions : override eval_config.MIN_INTERACTIONS for this call
    test_ratio       : override eval_config.TEST_RATIO for this call

    Thread-safe: acquires _lock during snapshot, then releases before processing.
    """
    _min_inter = min_interactions if min_interactions is not None else _DEFAULT_MIN
    _test_rat  = test_ratio       if test_ratio       is not None else _DEFAULT_TEST

    with _lock:
        snapshot = [(uid, list(u["history"])) for uid, u in _store.items()]

    dataset = {}
    for user_id, history in snapshot:
        if len(history) < _min_inter:
            continue

        # History is a deque of (item_id, event_type, timestamp) — already time-ordered
        # but sort defensively in case of any out-of-order ingestion
        history.sort(key=lambda x: x[2])

        split_idx  = max(1, int(len(history) * (1 - _test_rat)))
        train_hist = history[:split_idx]
        test_hist  = history[split_idx:]

        # All events in train (used for exclusion during retrieval)
        train_items = [item_id for item_id, _, _ in train_hist]

        # Only "like" events in test (ground-truth positives)
        test_likes  = [item_id for item_id, etype, _ in test_hist if etype == "like"]
        if not test_likes:
            continue

        dataset[user_id] = {"train": train_items, "test": test_likes}

    return dataset
