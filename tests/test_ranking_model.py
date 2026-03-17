"""
Unit tests for the ranking_model package.

Tests cover:
  - RankingModel architecture and forward pass
  - train_ranker: returns float loss, responds correctly to like/dislike labels
  - rank_items: top-k respected, unknown user / missing items handled, order is correct

sys.path is set up at import time so pytest can be run from the project root:
    pytest tests/test_ranking_model.py -v
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup — must happen before any project imports
# ---------------------------------------------------------------------------
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RANKING_DIR = os.path.join(_ROOT, "ranking_model")
_EMB_DIR     = os.path.join(_ROOT, "embedding_model")

for _p in (_EMB_DIR, _RANKING_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project imports (after path setup)
# ---------------------------------------------------------------------------
import torch
import pytest

from rank_config import HIDDEN_DIM, INPUT_DIM, TOP_K_FINAL
from rank_model import RankingModel
from embedding_store import user_embeddings, item_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(user_id="u_test", item_id="i_test", event_type="like"):
    return {"user_id": user_id, "item_id": item_id, "event_type": event_type}


def _seed_embeddings(user_id: str, item_ids: list):
    """Inject deterministic fixed embeddings directly into the global store."""
    from emb_config import EMBEDDING_DIM
    torch.manual_seed(42)
    user_embeddings[user_id] = torch.randn(EMBEDDING_DIM, requires_grad=True)
    for iid in item_ids:
        item_embeddings[iid] = torch.randn(EMBEDDING_DIM, requires_grad=True)


# ---------------------------------------------------------------------------
# RankingModel architecture tests
# ---------------------------------------------------------------------------

class TestRankingModelArchitecture:

    def test_output_shape_batched(self):
        """Forward pass on a batch: (B, 64) → (B, 1)."""
        m = RankingModel()
        x = torch.randn(8, INPUT_DIM)
        out = m(x)
        assert out.shape == (8, 1)

    def test_output_shape_single(self):
        """Forward pass on a single vector: (64,) → (1,)."""
        m = RankingModel()
        x = torch.randn(INPUT_DIM)
        out = m(x)
        assert out.shape == (1,)

    def test_output_is_raw_logit(self):
        """Output should be an unbounded logit — not squashed to [0, 1]."""
        m = RankingModel()
        torch.manual_seed(0)
        x = torch.randn(1000, INPUT_DIM)
        out = m(x).detach()
        # A sigmoid-clamped output would always be in [0,1];
        # raw logits from a random net span a much wider range.
        assert out.min().item() < 0.0 or out.max().item() > 1.0

    def test_input_dim_matches_config(self):
        """First linear layer must accept INPUT_DIM (= 64) features."""
        m = RankingModel()
        first_linear = m.net[0]
        assert first_linear.in_features == INPUT_DIM

    def test_hidden_dim_matches_config(self):
        """First linear layer output width must match HIDDEN_DIM."""
        m = RankingModel()
        first_linear = m.net[0]
        assert first_linear.out_features == HIDDEN_DIM

    def test_final_layer_outputs_one(self):
        """Last linear layer must produce a scalar (out_features=1)."""
        m = RankingModel()
        last_linear = m.net[-1]
        assert last_linear.out_features == 1

    def test_parameters_are_trainable(self):
        """All parameters must require gradients by default."""
        m = RankingModel()
        for name, param in m.named_parameters():
            assert param.requires_grad, f"{name} has requires_grad=False"

    def test_forward_is_deterministic_in_eval(self):
        """eval() mode must give identical outputs for the same input."""
        m = RankingModel()
        m.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            out1 = m(x)
            out2 = m(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# train_ranker tests
# ---------------------------------------------------------------------------

class TestTrainRanker:

    def setup_method(self):
        """Each test gets fresh embeddings and a fresh ranker model + optimizer."""
        # Import fresh module-level singletons
        import rank_model as _rm
        import rank_trainer as _rt
        # Reset model weights so tests are independent
        torch.manual_seed(0)
        for layer in _rm.model.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        # Reinitialise optimizer state
        _rt.optimizer = torch.optim.Adam(_rm.model.parameters(), lr=0.001)

    def test_returns_float(self):
        """train_ranker must return a Python float."""
        from rank_trainer import train_ranker
        _seed_embeddings("u1", ["i1"])
        loss = train_ranker(_make_event("u1", "i1", "like"))
        assert isinstance(loss, float)

    def test_loss_is_non_negative(self):
        """BCEWithLogitsLoss is always >= 0."""
        from rank_trainer import train_ranker
        _seed_embeddings("u2", ["i2"])
        loss = train_ranker(_make_event("u2", "i2", "like"))
        assert loss >= 0.0

    def test_like_label_target_is_one(self):
        """
        After many steps on a like event the loss should decrease toward 0,
        because the MLP is being trained to predict 1 for this (user, item) pair.
        """
        from rank_trainer import train_ranker
        _seed_embeddings("u_like", ["i_like"])
        first = train_ranker(_make_event("u_like", "i_like", "like"))
        for _ in range(50):
            train_ranker(_make_event("u_like", "i_like", "like"))
        last = train_ranker(_make_event("u_like", "i_like", "like"))
        assert last < first, "Loss should decrease when repeatedly training a like event"

    def test_dislike_label_target_is_zero(self):
        """
        After many steps on a dislike event the loss should decrease toward 0,
        because the MLP is being trained to predict 0 for this (user, item) pair.
        """
        from rank_trainer import train_ranker
        _seed_embeddings("u_dis", ["i_dis"])
        first = train_ranker(_make_event("u_dis", "i_dis", "dislike"))
        for _ in range(50):
            train_ranker(_make_event("u_dis", "i_dis", "dislike"))
        last = train_ranker(_make_event("u_dis", "i_dis", "dislike"))
        assert last < first, "Loss should decrease when repeatedly training a dislike event"

    def test_model_weights_change_after_step(self):
        """A single train step must update model weights (gradient != 0)."""
        import rank_model as _rm
        from rank_trainer import train_ranker
        _seed_embeddings("u_w", ["i_w"])
        before = [p.clone() for p in _rm.model.parameters()]
        train_ranker(_make_event("u_w", "i_w", "like"))
        after  = [p.clone() for p in _rm.model.parameters()]
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "Model parameters must change after a training step"

    def test_embeddings_not_modified_by_ranker(self):
        """
        Detach ensures ranker backward does not alter embedding store tensors.
        """
        from rank_trainer import train_ranker
        _seed_embeddings("u_det", ["i_det"])
        u_before = user_embeddings["u_det"].clone().detach()
        i_before = item_embeddings["i_det"].clone().detach()
        train_ranker(_make_event("u_det", "i_det", "like"))
        assert torch.equal(user_embeddings["u_det"].detach(), u_before), \
            "User embedding must not be changed by rank_trainer"
        assert torch.equal(item_embeddings["i_det"].detach(), i_before), \
            "Item embedding must not be changed by rank_trainer"


# ---------------------------------------------------------------------------
# rank_items (inference) tests
# ---------------------------------------------------------------------------

class TestRankItems:

    def test_returns_list(self):
        """rank_items must return a list."""
        from rank_inference import rank_items
        _seed_embeddings("u_ri", ["i_a", "i_b", "i_c"])
        result = rank_items("u_ri", ["i_a", "i_b", "i_c"])
        assert isinstance(result, list)

    def test_top_k_respected(self):
        """Result length must not exceed top_k."""
        from rank_inference import rank_items
        items = [f"item_{n}" for n in range(20)]
        _seed_embeddings("u_topk", items)
        result = rank_items("u_topk", items, top_k=5)
        assert len(result) <= 5

    def test_returns_all_when_fewer_than_k(self):
        """When candidates < top_k, return all scored candidates."""
        from rank_inference import rank_items
        _seed_embeddings("u_few", ["x1", "x2"])
        result = rank_items("u_few", ["x1", "x2"], top_k=10)
        assert len(result) == 2

    def test_unknown_user_returns_empty(self):
        """If the user has no embedding, return an empty list."""
        from rank_inference import rank_items
        result = rank_items("nonexistent_user_xyz", ["i_a", "i_b"])
        assert result == []

    def test_empty_candidates_returns_empty(self):
        """If the candidate list is empty, return an empty list."""
        from rank_inference import rank_items
        _seed_embeddings("u_empty_cands", [])
        result = rank_items("u_empty_cands", [], top_k=10)
        assert result == []

    def test_missing_item_embeddings_skipped(self):
        """Items not in item_embeddings are silently dropped."""
        from rank_inference import rank_items
        _seed_embeddings("u_skip", ["real_item"])
        result = rank_items("u_skip", ["real_item", "ghost_item_xyz"], top_k=10)
        assert "ghost_item_xyz" not in result
        assert "real_item" in result

    def test_result_contains_only_input_candidates(self):
        """Output items must be a subset of the input candidate list."""
        from rank_inference import rank_items
        items = ["c1", "c2", "c3", "c4", "c5"]
        _seed_embeddings("u_subset", items)
        result = rank_items("u_subset", items, top_k=3)
        assert all(r in items for r in result)

    def test_ordering_by_score(self):
        """
        We can verify ordering indirectly: a model trained hard on one item
        should rank that item higher than an untrained item.
        """
        import rank_model as _rm
        import rank_trainer as _rt
        import torch.nn as nn

        # Fresh model + optimizer
        torch.manual_seed(7)
        for layer in _rm.model.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        _rt.optimizer = torch.optim.Adam(_rm.model.parameters(), lr=0.01)

        from rank_trainer import train_ranker
        from rank_inference import rank_items

        _seed_embeddings("u_order", ["hot_item", "cold_item"])

        # Train heavily on hot_item as a like
        for _ in range(200):
            train_ranker(_make_event("u_order", "hot_item", "like"))

        result = rank_items("u_order", ["hot_item", "cold_item"], top_k=2)
        assert result[0] == "hot_item", \
            f"hot_item should rank #1 after 200 like steps, got: {result}"

    def test_no_duplicates_in_result(self):
        """Returned list must have no duplicate item_ids."""
        from rank_inference import rank_items
        items = [f"d{n}" for n in range(10)]
        _seed_embeddings("u_nodup", items)
        result = rank_items("u_nodup", items, top_k=10)
        assert len(result) == len(set(result))
