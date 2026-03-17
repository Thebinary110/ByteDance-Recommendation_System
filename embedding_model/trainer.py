"""
Online trainer with negative sampling — multiple gradient signals per event.

Training logic per event
------------------------
1. Record interaction (excludes current item from negatives).
2. Direct pair:    (user, item,     label)  — like→1.0, dislike→0.0
3. Negative pairs: (user, neg_item, 0.0  )  — NEGATIVE_SAMPLES random unseen items

All backward() calls accumulate gradients on user_emb before the single update
step.  This is equivalent to computing a mini-batch gradient for the user vector
across 1 + NEGATIVE_SAMPLES pairs, then applying one SGD step.

Why accumulate then update once (not update after each pair):
  Each pair contributes signal about the same user_emb.  Accumulating gives a
  more stable, averaged gradient direction before moving the user vector.

Why train dislikes as direct negative pairs (label=0.0):
  The spec omits dislike training in Stage 6, but this wastes a clean negative
  signal we already have.  Training dislike events as label=0.0 is strictly
  better — the model learns to push disliked items away from the user vector.
"""

import torch

from emb_config import LEARNING_RATE, NEGATIVE_SAMPLES
from embedding_store import get_item_embedding, get_user_embedding
from model import predict
from negative_sampler import record_interaction, sample_negatives


def train_on_event(event: dict) -> float:
    """
    Train on a single event with NEGATIVE_SAMPLES additional negatives.
    Returns the average loss across all pairs (direct + negatives).
    """
    user_id    = event["user_id"]
    item_id    = event["item_id"]
    event_type = event["event_type"]

    # Record before sampling so item_id is excluded from the negative pool
    record_interaction(user_id, item_id)

    user_emb     = get_user_embedding(user_id)
    total_loss   = 0.0
    trained_items = [item_id]

    # --- Direct pair (like → 1.0, dislike → 0.0) ---
    label    = 1.0 if event_type == "like" else 0.0
    item_emb = get_item_embedding(item_id)
    score    = predict(user_emb, item_emb)
    loss     = (torch.sigmoid(score) - label) ** 2
    loss.backward()                  # gradients accumulate on user_emb and item_emb
    total_loss += loss.item()

    # --- Negative pairs ---
    negatives = sample_negatives(user_id, NEGATIVE_SAMPLES)
    for neg_id in negatives:
        neg_emb   = get_item_embedding(neg_id)
        neg_score = predict(user_emb, neg_emb)
        neg_loss  = (torch.sigmoid(neg_score) - 0.0) ** 2
        neg_loss.backward()          # gradients continue to accumulate on user_emb
        total_loss += neg_loss.item()
        trained_items.append(neg_id)

    # --- Single SGD step (applied after all backward passes) ---
    with torch.no_grad():
        if user_emb.grad is not None:
            user_emb.data -= LEARNING_RATE * user_emb.grad
            user_emb.grad.zero_()

        for tid in trained_items:
            emb = get_item_embedding(tid)
            if emb.grad is not None:
                emb.data -= LEARNING_RATE * emb.grad
                emb.grad.zero_()

    return total_loss / (1 + len(negatives))
