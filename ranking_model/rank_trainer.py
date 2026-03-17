"""
Ranking model trainer — one Adam step per streaming event.

Training flow:
  1. Fetch user & item embeddings from the embedding store.
  2. Detach them — ranker gradient must not flow back into the embedding model.
  3. Concatenate → input vector of dim 64.
  4. Forward through MLP → raw logit.
  5. BCEWithLogitsLoss vs binary label (like=1.0, dislike=0.0).
  6. Adam step.

Why detach:
  The two models (embedding + ranking) are trained independently.
  Flowing ranker gradients into the embeddings would corrupt the
  embedding model's carefully managed manual SGD updates.

Why BCEWithLogitsLoss (not BCELoss + sigmoid):
  Numerically more stable — fuses sigmoid into the loss computation,
  avoiding exp overflow for very large or small logit values.

The optimizer and loss_fn are module-level so they persist Adam moments
across all training steps (required for correct momentum/velocity tracking).
"""

import torch
import torch.nn as nn

from embedding_store import get_item_embedding, get_user_embedding
from rank_config import LEARNING_RATE
from rank_model import model

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.BCEWithLogitsLoss()


def train_ranker(event: dict) -> float:
    """Train on a single event. Returns scalar loss for monitoring."""
    user_id = event["user_id"]
    item_id = event["item_id"]
    label   = 1.0 if event["event_type"] == "like" else 0.0

    user_emb = get_user_embedding(user_id).detach()   # no grad into embedding model
    item_emb = get_item_embedding(item_id).detach()

    x      = torch.cat([user_emb, item_emb])          # (64,)
    pred   = model(x)                                 # (1,) raw logit
    target = torch.tensor([label])

    loss = loss_fn(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
