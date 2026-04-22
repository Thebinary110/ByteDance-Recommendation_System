"""
RecommendationEngine — full inference pipeline.

Request flow:
  user_idx
    → FeatureStore.get_user_embedding()        (or compute on-the-fly)
    → FAISSIndex.search(k=200)                 (candidate generation)
    → filter seen items
    → DeepFM.predict_proba() for all candidates (ranking)
    → sort by DeepFM score, take top-50
    → MMR re-rank to top-k                     (diversity)
    → generate explanation strings
    → return recommendations

Also exposes:
  • search(query) — text search over movie titles/genres
  • get_similar(movie_idx) — similar movies via FAISS
  • get_explanation(user_idx, movie_idx) — human-readable reason
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from data.preprocessor import Preprocessor
from models.deepfm import DeepFM, build_dense_features
from models.reranker import diversify
from models.two_tower import TwoTowerModel
from serving.faiss_index import FAISSIndex
from serving.feature_store import FeatureStore

logger = logging.getLogger(__name__)


def _year_to_bucket(year: int, n_buckets: int = 50) -> int:
    return max(0, min(n_buckets - 1, (year - 1920) // 2))


class RecommendationEngine:
    """
    Single object owning all models and indexes needed for serving.
    Loaded once at API startup; methods are called per-request.
    """

    def __init__(
        self,
        prep: Preprocessor,
        two_tower: TwoTowerModel,
        deepfm: DeepFM,
        faiss_index: FAISSIndex,
        item_embeddings: np.ndarray,  # (num_movies, embed_dim) — for MMR
        feature_store: FeatureStore,
        device: torch.device,
    ):
        self.prep = prep
        self.two_tower = two_tower.to(device).eval()
        self.deepfm = deepfm.to(device).eval()
        self.faiss_index = faiss_index
        self.item_embeddings = item_embeddings
        self.feature_store = feature_store
        self.device = device

        # Precompute movie_idx → metadata map for fast lookups
        self._build_movie_meta()

    def _build_movie_meta(self) -> None:
        """Build a dict movie_idx → {title, year, genres, movie_id}."""
        self.movie_meta: dict[int, dict] = {}
        if self.prep.movie_df is None:
            return
        for _, row in self.prep.movie_df.iterrows():
            idx = self.prep.movie_id_map.get(row["movieId"])
            if idx is None:
                continue
            self.movie_meta[idx] = {
                "movie_id": int(row["movieId"]),
                "title": str(row["title"]),
                "year": int(row.get("year", 0)),
                "genres": row.get("genre_list", []),
            }

    # ------------------------------------------------------------------
    # User embedding retrieval / computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get or compute user embedding, caching in feature store."""
        cached = self.feature_store.get_user_embedding(user_idx)
        if cached is not None:
            return cached

        # Compute from scratch
        user_feats = self.feature_store.get_user_features(user_idx)
        if user_feats is None:
            if user_idx < len(self.prep.user_features):
                user_feats = self.prep.user_features[user_idx]
            else:
                user_feats = np.zeros(self.prep.user_features.shape[1], dtype=np.float32)

        uid_t = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        uf_t = torch.tensor(user_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        emb = self.two_tower.encode_user(uid_t, uf_t).cpu().numpy()[0]  # (D,)

        self.feature_store.set_user_embedding(user_idx, emb)
        return emb

    # ------------------------------------------------------------------
    # Core recommendation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend_by_idx(
        self,
        user_idx: int,
        limit: int = 20,
        seen_movie_idxs: Optional[set[int]] = None,
        faiss_k: int = 200,
        mmr_pool: int = 50,
        lambda_mmr: float = 0.7,
    ) -> list[int]:
        """Return a list of recommended movie_idx values."""
        if seen_movie_idxs is None:
            seen_movie_idxs = set()

        # Step 1: candidate retrieval via FAISS
        user_emb = self._get_user_embedding(user_idx)
        _, candidate_idxs = self.faiss_index.search(user_emb, k=faiss_k)
        candidate_idxs = candidate_idxs[0].tolist()

        # Step 2: filter seen
        candidate_idxs = [c for c in candidate_idxs if c not in seen_movie_idxs]
        if not candidate_idxs:
            return list(range(min(limit, self.prep.num_movies)))

        # Step 3: DeepFM scoring
        scores = self._score_candidates(user_idx, candidate_idxs)

        # Step 4: sort by DeepFM, take top mmr_pool
        pool_size = min(mmr_pool, len(candidate_idxs))
        top_idxs = np.argsort(-scores)[:pool_size]
        pool_candidates = [candidate_idxs[i] for i in top_idxs]
        pool_scores = scores[top_idxs]

        # Step 5: MMR re-rank
        final_idxs = diversify(
            pool_candidates,
            pool_scores,
            self.item_embeddings,
            top_k=limit,
            lambda_param=lambda_mmr,
        )
        return final_idxs

    @torch.no_grad()
    def _score_candidates(
        self, user_idx: int, candidate_idxs: list[int]
    ) -> np.ndarray:
        """Score candidates with DeepFM. Returns (N,) float32 array."""
        user_feats = self.prep.user_features[min(user_idx, len(self.prep.user_features) - 1)]
        item_feats = self.prep.item_features

        uid_t = torch.tensor([user_idx] * len(candidate_idxs), dtype=torch.long, device=self.device)
        mid_t = torch.tensor(candidate_idxs, dtype=torch.long, device=self.device)

        # Year buckets
        year_buckets = []
        for c in candidate_idxs:
            meta = self.movie_meta.get(c, {})
            year_buckets.append(_year_to_bucket(meta.get("year", 2000)))
        yb_t = torch.tensor(year_buckets, dtype=torch.long, device=self.device)

        uf_t = torch.tensor(
            np.tile(user_feats, (len(candidate_idxs), 1)), dtype=torch.float32, device=self.device
        )
        if_t = torch.tensor(
            item_feats[candidate_idxs], dtype=torch.float32, device=self.device
        )
        dense = build_dense_features(uf_t, if_t)

        probs = self.deepfm.predict_proba(uid_t, mid_t, yb_t, dense).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        limit: int = 20,
        seen_movie_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Main recommendation endpoint.
        Accepts and returns original MovieLens IDs (not internal indices).
        """
        user_idx = self.prep.user_id_map.get(user_id)
        if user_idx is None:
            # Cold-start: return popular items by falling back to sorted catalog
            return self._popular_fallback(limit)

        seen_idxs = set()
        if seen_movie_ids:
            seen_idxs = {
                self.prep.movie_id_map[m]
                for m in seen_movie_ids
                if m in self.prep.movie_id_map
            }

        rec_idxs = self.recommend_by_idx(user_idx, limit=limit, seen_movie_idxs=seen_idxs)

        return [
            {
                **self._idx_to_movie(idx),
                "score": float(i + 1),  # rank order as score
                "explanation": self.get_explanation(user_idx, idx),
            }
            for i, idx in enumerate(rec_idxs)
        ]

    def get_similar(self, movie_id: int, limit: int = 10) -> list[dict]:
        """Find similar movies using FAISS item–item nearest neighbours."""
        movie_idx = self.prep.movie_id_map.get(movie_id)
        if movie_idx is None or movie_idx >= len(self.item_embeddings):
            return []

        query = self.item_embeddings[movie_idx]
        _, indices = self.faiss_index.search(query, k=limit + 1)
        similar = [
            self._idx_to_movie(idx)
            for idx in indices[0].tolist()
            if idx != movie_idx
        ]
        return similar[:limit]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over movie titles and genres (in-memory trie-free scan)."""
        q = query.lower().strip()
        results = []
        for idx, meta in self.movie_meta.items():
            title_lower = meta["title"].lower()
            genres_lower = " ".join(meta.get("genres", [])).lower()
            if q in title_lower or q in genres_lower:
                results.append((idx, meta))

        # Rank: exact title start > title contains > genre contains
        def rank_key(item):
            idx, meta = item
            t = meta["title"].lower()
            if t.startswith(q):
                return 0
            if q in t:
                return 1
            return 2

        results.sort(key=rank_key)
        return [self._idx_to_movie(idx) for idx, _ in results[:limit]]

    def get_movie(self, movie_id: int) -> Optional[dict]:
        """Get full movie metadata by MovieLens ID."""
        movie_idx = self.prep.movie_id_map.get(movie_id)
        if movie_idx is None:
            return None
        return self._idx_to_movie(movie_idx)

    def get_taste_profile(self, user_id: int) -> dict:
        """Return a user's genre preferences and activity statistics."""
        user_idx = self.prep.user_id_map.get(user_id)
        if user_idx is None:
            return {"error": "User not found"}

        if user_idx < len(self.prep.user_features):
            feats = self.prep.user_features[user_idx]
        else:
            feats = np.zeros(self.prep.user_features.shape[1], dtype=np.float32)

        from data.loader import ALL_GENRES
        genre_prefs = {
            genre: float(feats[i])
            for i, genre in enumerate(ALL_GENRES)
            if i < len(feats)
        }
        avg_rating = float(feats[len(ALL_GENRES)]) * 5.0 if len(feats) > len(ALL_GENRES) else 3.5
        import math
        log_count = float(feats[len(ALL_GENRES) + 1]) if len(feats) > len(ALL_GENRES) + 1 else 0.0
        rating_count = int(math.expm1(log_count * 10))

        return {
            "user_id": user_id,
            "genre_preferences": genre_prefs,
            "avg_rating": round(avg_rating, 2),
            "rating_count": rating_count,
            "top_genres": sorted(genre_prefs, key=genre_prefs.get, reverse=True)[:5],
        }

    def get_explanation(self, user_idx: int, movie_idx: int) -> str:
        """Generate a human-readable explanation for a recommendation."""
        meta = self.movie_meta.get(movie_idx, {})
        genres = meta.get("genres", [])

        if user_idx < len(self.prep.user_features):
            user_feats = self.prep.user_features[user_idx]
            from data.loader import ALL_GENRES
            genre_scores = {
                g: float(user_feats[i])
                for i, g in enumerate(ALL_GENRES)
                if i < len(user_feats)
            }
            matching = [g for g in genres if genre_scores.get(g, 0) > 0.3]
        else:
            matching = genres[:2]

        if matching:
            return f"Because you enjoy {', '.join(matching[:2])} films"
        elif genres:
            return f"A top-rated {genres[0]} film you haven't seen"
        return "Highly rated and popular with viewers like you"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _idx_to_movie(self, movie_idx: int) -> dict:
        meta = self.movie_meta.get(movie_idx, {})
        return {
            "movie_id": meta.get("movie_id", movie_idx),
            "title": meta.get("title", f"Movie {movie_idx}"),
            "year": meta.get("year", 0),
            "genres": meta.get("genres", []),
        }

    def _popular_fallback(self, limit: int) -> list[dict]:
        """Return a fixed popular-movie list for cold-start users."""
        popular_idxs = list(range(min(limit, self.prep.num_movies)))
        return [self._idx_to_movie(i) for i in popular_idxs]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        device_str: str = "cpu",
        feature_store: Optional[FeatureStore] = None,
    ) -> "RecommendationEngine":
        artifact_dir = Path(artifact_dir).resolve()  # always absolute
        device = torch.device(device_str)

        prep = Preprocessor.load(artifact_dir)

        # Two-Tower
        tt_ckpt = torch.load(artifact_dir / "two_tower.pt", map_location=device)
        tt_cfg = tt_ckpt["config"]
        two_tower = TwoTowerModel(**tt_cfg)
        two_tower.load_state_dict(tt_ckpt["state_dict"])

        # DeepFM
        dfm_ckpt = torch.load(artifact_dir / "deepfm_best.pt", map_location=device)
        dfm_cfg = dfm_ckpt["config"]
        deepfm = DeepFM(**dfm_cfg)
        deepfm.load_state_dict(dfm_ckpt["state_dict"])

        # FAISS index
        faiss_index = FAISSIndex().load(artifact_dir / "faiss.index")

        # Item embeddings (for MMR)
        item_embeddings = np.load(artifact_dir / "item_embeddings.npy")

        # Feature store
        if feature_store is None:
            feature_store = FeatureStore(
                sqlite_path=artifact_dir / "feature_store.db"
            )

        engine = cls(
            prep=prep,
            two_tower=two_tower,
            deepfm=deepfm,
            faiss_index=faiss_index,
            item_embeddings=item_embeddings,
            feature_store=feature_store,
            device=device,
        )
        logger.info("RecommendationEngine loaded successfully.")
        return engine
