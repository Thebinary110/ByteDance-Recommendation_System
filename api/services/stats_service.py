"""
Stats service — system-level metrics from embedding store and FAISS index.
"""

from typing import Dict

from embedding_store import embedding_counts  # embedding_model
from faiss_index import faiss_index           # embedding_model


def get_stats() -> Dict:
    """Return aggregate system stats."""
    n_users, n_items = embedding_counts()
    return {
        "total_users":     n_users,
        "total_items":     n_items,
        "total_emb_users": n_users,
        "faiss_built":     faiss_index.built,
    }
