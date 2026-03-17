"""
Recommendation service — two-stage pipeline: FAISS retrieval → MLP reranking.

Keeps all ML logic out of routes.
"""

from typing import List

from user_store import get_user              # feature_store
from embedding_store import user_embeddings  # embedding_model
from inference import recommend as retrieve  # embedding_model  (FAISS / linear)
from rank_inference import rank_items        # ranking_model    (MLP reranker)


def get_recommendations(user_id: str, top_k: int = 10) -> List[str]:
    """
    Return up to top_k reranked item IDs for user_id.
    Returns [] if the user has no embedding yet.

    Pipeline:
      1. Exclude already-interacted items (likes ∪ dislikes) from retrieval.
      2. FAISS top-50 candidates (embedding similarity).
      3. MLP reranking → final top_k.
    """
    if user_id not in user_embeddings:
        return []

    user    = get_user(user_id)
    exclude = (user["likes"] | user["dislikes"]) if user else set()

    candidates = retrieve(user_id, top_k=50, exclude=exclude)
    return rank_items(user_id, candidates, top_k=top_k)
