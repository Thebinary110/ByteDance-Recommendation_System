"""
MLP ranking model — scores (user, item) pairs for final reranking.

Architecture:
  input  : [user_emb || item_emb]  — dim 64 (32 + 32 concatenated)
  layer 1: Linear(64 → 64) + ReLU
  layer 2: Linear(64 → 32) + ReLU
  layer 3: Linear(32 →  1)          — raw logit (no sigmoid here)

The output is a raw logit.  The trainer applies BCEWithLogitsLoss (sigmoid
is fused into the loss for numerical stability).  The inference function
uses the raw score for ranking (monotone, so sigmoid is not needed).

The global `model` instance is shared between rank_trainer.py and
rank_inference.py via module caching — both import the same object.
"""

import torch
import torch.nn as nn

from rank_config import HIDDEN_DIM, INPUT_DIM


class RankingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Shared global — imported by rank_trainer and rank_inference
model = RankingModel()
