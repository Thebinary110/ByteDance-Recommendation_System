"""
DeepFM inference — rank a list of candidate item_ids for a given user.

Uses the shared _model singleton from deepfm_model.py.
Falls back gracefully if embeddings are missing.
"""

import logging
from typing import List

import torch

from deepfm_model import _model
from feature_builder import build_features
from embedding_store import user_embeddings, item_embeddings   # embedding_model

log = logging.getLogger("deepfm.inference")


def rank_items(user_id: str, candidates: List[str], top_k: int = 10) -> List[str]:
    """
    Score each candidate with DeepFM and return the top_k item_ids in
    descending score order.

    Parameters
    ----------
    user_id    : str
    candidates : list of item_id strings (already filtered by FAISS retrieval)
    top_k      : number of items to return

    Returns
    -------
    list of item_id strings, length <= top_k
    """
    if not candidates:
        return []

    user_tensor = user_embeddings.get(user_id)
    if user_tensor is None:
        log.debug("No user embedding for %s — returning candidates as-is", user_id)
        return candidates[:top_k]

    user_emb = user_tensor.detach()

    scored: List[tuple] = []   # (score, item_id)

    _model.eval()
    with torch.no_grad():
        for item_id in candidates:
            item_tensor = item_embeddings.get(item_id)
            if item_tensor is None:
                continue
            item_emb = item_tensor.detach()

            try:
                feat = build_features(user_id, item_id, user_emb, item_emb)
            except Exception as exc:
                log.debug("build_features failed (%s, %s): %s", user_id, item_id, exc)
                continue

            uid  = torch.tensor([feat["user_id_int"]], dtype=torch.long)
            iid  = torch.tensor([feat["item_id_int"]], dtype=torch.long)
            uvec = torch.tensor(feat["user_emb"],  dtype=torch.float32).unsqueeze(0)
            ivec = torch.tensor(feat["item_emb"],  dtype=torch.float32).unsqueeze(0)
            gvec = torch.tensor(feat["genre_vec"], dtype=torch.float32).unsqueeze(0)

            logit = _model(uid, iid, uvec, ivec, gvec)
            scored.append((logit.item(), item_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item_id for _, item_id in scored[:top_k]]
