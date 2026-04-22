"""
MMR (Maximal Marginal Relevance) re-ranker for diversity.

MMR balances relevance and novelty:
  MMR(i) = λ * score(i)  −  (1−λ) * max_{j ∈ S} sim(i, j)

where S is the set already selected.

λ=1.0 → pure relevance (no diversity)
λ=0.0 → pure diversity (no relevance)
λ=0.7 → recommended default for production

Reference: Carbonell & Goldstein (1998) "The use of MMR, diversity-based
reranking for reordering documents and producing summaries."
"""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(
    embeddings: np.ndarray,  # [N, D]
) -> np.ndarray:
    """Pairwise cosine similarity matrix [N, N]."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms
    return normed @ normed.T  # [N, N]


def mmr_rerank(
    candidate_ids: list[int],
    relevance_scores: np.ndarray,   # [N] — higher is better
    item_embeddings: np.ndarray,    # [N, D] — embeddings for candidates only
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> list[int]:
    """
    Select top_k items from candidates using MMR.

    Parameters
    ----------
    candidate_ids     : original item IDs (e.g. movie_idx values) in same order as scores
    relevance_scores  : relevance score per candidate (DeepFM probabilities)
    item_embeddings   : embedding vectors for the candidates (NOT the full catalog)
    top_k             : number of items to return
    lambda_param      : trade-off between relevance and diversity

    Returns
    -------
    List of selected candidate IDs in re-ranked order.
    """
    n = len(candidate_ids)
    if n == 0:
        return []
    top_k = min(top_k, n)

    # Normalise relevance to [0, 1] for stable trade-off with similarity
    rel = np.asarray(relevance_scores, dtype=np.float64)
    rel_min, rel_max = rel.min(), rel.max()
    if rel_max > rel_min:
        rel = (rel - rel_min) / (rel_max - rel_min)

    # Pairwise similarity matrix
    sim = cosine_similarity_matrix(item_embeddings.astype(np.float64))  # [N, N]

    selected_indices: list[int] = []
    remaining: set[int] = set(range(n))

    for _ in range(top_k):
        if not remaining:
            break

        if not selected_indices:
            # First item: pick highest relevance
            best = max(remaining, key=lambda i: rel[i])
        else:
            # MMR score for each remaining candidate
            best = max(
                remaining,
                key=lambda i: (
                    lambda_param * rel[i]
                    - (1 - lambda_param) * max(sim[i, j] for j in selected_indices)
                ),
            )

        selected_indices.append(best)
        remaining.discard(best)

    return [candidate_ids[i] for i in selected_indices]


def diversify(
    candidate_ids: list[int],
    relevance_scores: np.ndarray,
    all_item_embeddings: np.ndarray,   # [num_movies, D] — full item embedding matrix
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> list[int]:
    """
    Convenience wrapper that slices embeddings for candidates from the full matrix.

    Parameters
    ----------
    all_item_embeddings : full catalog embedding matrix indexed by movie_idx
    """
    candidate_embeddings = all_item_embeddings[candidate_ids]  # [N, D]
    return mmr_rerank(
        candidate_ids,
        relevance_scores,
        candidate_embeddings,
        top_k=top_k,
        lambda_param=lambda_param,
    )


def intra_list_diversity(
    selected_ids: list[int],
    all_item_embeddings: np.ndarray,
) -> float:
    """
    Computes average pairwise dissimilarity of the recommended list.
    Higher = more diverse. Used as an offline metric.
    ILD = 1 - average_pairwise_cosine_similarity
    """
    if len(selected_ids) < 2:
        return 0.0
    embeds = all_item_embeddings[selected_ids]
    sim = cosine_similarity_matrix(embeds)
    n = len(selected_ids)
    # Average off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    avg_sim = sim[mask].mean()
    return float(1.0 - avg_sim)
