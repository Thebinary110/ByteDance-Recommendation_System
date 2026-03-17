"""
Ranking inference — scores a candidate list and returns the top-K.

Inputs:
  user_id    — used to look up the user embedding
  candidates — item_ids from FAISS retrieval (the pre-ranking pool)
  top_k      — final list size

The MLP produces a raw logit for each (user, item) pair.  Sigmoid is
monotone so we rank by logit directly — no need to apply sigmoid for ordering.

All forward passes run inside torch.no_grad() to skip graph construction,
and the model is set to eval mode (disables dropout / batchnorm if added later).
"""

import torch
from typing import List

from embedding_store import item_embeddings, user_embeddings
from rank_model import model


def rank_items(user_id: str, candidates: List[str], top_k: int = 10) -> List[str]:
    """
    Rerank candidate item_ids by MLP score and return the top_k.
    Items missing from the embedding store are silently skipped.
    """
    if user_id not in user_embeddings or not candidates:
        return []

    user_emb = user_embeddings[user_id].detach()
    scores: List[tuple] = []

    model.eval()
    with torch.no_grad():
        for item_id in candidates:
            if item_id not in item_embeddings:
                continue
            item_emb = item_embeddings[item_id].detach()
            x        = torch.cat([user_emb, item_emb])   # (64,)
            score    = model(x).item()                    # raw logit
            scores.append((item_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in scores[:top_k]]
