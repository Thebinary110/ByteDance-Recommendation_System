"""
Two-tower model — dot product similarity.

The "two towers" are the user embedding and the item embedding.
Their dot product is the raw relevance score; passing it through sigmoid
converts it to a probability in [0, 1].

predict() returns the raw score (before sigmoid) so the caller controls
whether to apply sigmoid (training) or use the raw score (ranking).
"""

import torch


def predict(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
    """Raw dot-product score between a user and an item embedding."""
    return torch.dot(user_emb, item_emb)
