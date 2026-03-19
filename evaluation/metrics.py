"""
Standard ranking metrics for implicit-feedback recommendation.

All functions accept:
    predicted : list of item_id strings in ranked order
    relevant  : set  of item_id strings that are ground-truth positives
    k         : cutoff position

Definitions (implicit feedback):
    precision@k = |relevant in predicted[:k]| / k
    recall@k    = |relevant in predicted[:k]| / |relevant|
    hit_rate@k  = 1.0 if any of predicted[:k] is relevant else 0.0
"""

from typing import List, Set


def precision_at_k(predicted: List[str], relevant: Set[str], k: int) -> float:
    if k == 0 or not predicted:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in relevant)
    return hits / k


def recall_at_k(predicted: List[str], relevant: Set[str], k: int) -> float:
    if not relevant or not predicted:
        return 0.0
    hits = sum(1 for item in predicted[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(predicted: List[str], relevant: Set[str], k: int) -> float:
    if not relevant or not predicted:
        return 0.0
    return 1.0 if any(item in relevant for item in predicted[:k]) else 0.0
