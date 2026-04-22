"""
Feature Store — abstracts real-time feature retrieval.

Primary backend : Redis (in-memory, sub-millisecond reads)
Fallback backend: SQLite (file-based, always available locally)

The interface is identical regardless of backend, so the rest of the
system doesn't need to know which is running.

Keys:
  user:{user_idx}:features   → JSON-encoded float list
  user:{user_idx}:embedding  → JSON-encoded float list
  item:{movie_idx}:features  → JSON-encoded float list
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

_REDIS_AVAILABLE = False
try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    pass


# ------------------------------------------------------------------
# SQLite backend
# ------------------------------------------------------------------

class _SQLiteBackend:
    """Lightweight KV store backed by SQLite — mirrors Redis get/set/delete."""

    def __init__(self, db_path: str | Path = "feature_store.db"):
        self.db_path = str(db_path)
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._con.commit()

    def get(self, key: str) -> Optional[str]:
        cur = self._con.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        # ex (TTL) is accepted but ignored — SQLite has no native TTL
        self._con.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value)
        )
        self._con.commit()

    def delete(self, key: str) -> None:
        self._con.execute("DELETE FROM kv WHERE key = ?", (key,))
        self._con.commit()

    def ping(self) -> bool:
        return True


# ------------------------------------------------------------------
# Feature Store
# ------------------------------------------------------------------

class FeatureStore:
    """
    Unified interface for reading and writing features.
    Tries Redis first; automatically falls back to SQLite.
    """

    TTL = 3600 * 24  # 24-hour cache for user features

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        sqlite_path: str | Path = "feature_store.db",
    ):
        self._backend = self._connect(redis_host, redis_port, redis_db, sqlite_path)

    @staticmethod
    def _build_upstash_url() -> Optional[str]:
        """
        Resolution order for the Redis connection URL:

        1. REDIS_URL        — full rediss:// TCP URL (preferred, works directly with redis-py)
                              e.g. rediss://default:TOKEN@host:6379
        2. UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN
                            — Upstash REST URL + separate token; we convert to rediss://
                              e.g. https://host  →  rediss://default:TOKEN@host:6379
        3. Neither set      — fall through to local Redis host:port or SQLite
        """
        # Priority 1: direct TCP URL already in redis-py format
        direct = os.getenv("REDIS_URL", "").strip()
        if direct.startswith("rediss://") or direct.startswith("redis://"):
            return direct

        # Priority 2: Upstash REST URL + token → construct TCP URL
        rest_url = os.getenv("UPSTASH_REDIS_URL", "").strip()
        token = os.getenv("UPSTASH_REDIS_TOKEN", "").strip()
        if rest_url and token:
            host = rest_url.replace("https://", "").replace("http://", "").rstrip("/")
            return f"rediss://default:{token}@{host}:6379"

        return None

    @staticmethod
    def _connect(host, port, db, sqlite_path):
        if _REDIS_AVAILABLE:
            try:
                upstash_url = FeatureStore._build_upstash_url()
                if upstash_url:
                    # Upstash via TLS Redis protocol — token embedded in URL
                    client = _redis_lib.from_url(
                        upstash_url, decode_responses=True, socket_connect_timeout=5
                    )
                    client.ping()
                    logger.info("Feature Store connected to Upstash Redis ✓")
                else:
                    client = _redis_lib.Redis(
                        host=host, port=port, db=db,
                        socket_connect_timeout=1, decode_responses=True
                    )
                    client.ping()
                    logger.info(f"Feature Store connected to Redis at {host}:{port}")
                return client
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), falling back to SQLite.")
        backend = _SQLiteBackend(sqlite_path)
        logger.info(f"Feature Store using SQLite at {sqlite_path}")
        return backend

    # ------ Internal serialisation ------

    @staticmethod
    def _encode(arr: np.ndarray) -> str:
        return json.dumps(arr.tolist())

    @staticmethod
    def _decode(raw: str | None) -> Optional[np.ndarray]:
        if raw is None:
            return None
        return np.array(json.loads(raw), dtype=np.float32)

    # ------ User features ------

    def get_user_features(self, user_idx: int) -> Optional[np.ndarray]:
        return self._decode(self._backend.get(f"user:{user_idx}:features"))

    def set_user_features(self, user_idx: int, features: np.ndarray) -> None:
        self._backend.set(
            f"user:{user_idx}:features", self._encode(features), ex=self.TTL
        )

    # ------ User embedding ------

    def get_user_embedding(self, user_idx: int) -> Optional[np.ndarray]:
        return self._decode(self._backend.get(f"user:{user_idx}:embedding"))

    def set_user_embedding(self, user_idx: int, embedding: np.ndarray) -> None:
        self._backend.set(
            f"user:{user_idx}:embedding", self._encode(embedding), ex=self.TTL
        )

    # ------ Item features ------

    def get_item_features(self, movie_idx: int) -> Optional[np.ndarray]:
        return self._decode(self._backend.get(f"item:{movie_idx}:features"))

    def set_item_features(self, movie_idx: int, features: np.ndarray) -> None:
        self._backend.set(
            f"item:{movie_idx}:features", self._encode(features), ex=self.TTL
        )

    # ------ Bulk population ------

    def populate_item_features(self, item_features: np.ndarray) -> None:
        """Bulk-load all item features into the store at startup."""
        logger.info(f"Populating item features for {len(item_features)} items …")
        for idx, feats in enumerate(item_features):
            self.set_item_features(idx, feats)
        logger.info("Item features populated.")

    def populate_user_features(self, user_features: np.ndarray) -> None:
        """Bulk-load all user features into the store at startup."""
        logger.info(f"Populating user features for {len(user_features)} users …")
        for idx, feats in enumerate(user_features):
            self.set_user_features(idx, feats)
        logger.info("User features populated.")
