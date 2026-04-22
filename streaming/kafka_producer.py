"""
Kafka producer with SQLite fallback.

Used by the event logger and any component that needs to publish events.
The producer is thread-safe and can be shared across FastAPI request handlers.

Topics:
  rec_events   — all user interaction events
  model_updates — signals to trigger model retraining
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_KAFKA_AVAILABLE = False
try:
    from kafka import KafkaProducer as _KP
    from kafka.errors import KafkaError
    _KAFKA_AVAILABLE = True
except ImportError:
    pass


class KafkaProducerWrapper:
    """
    Thin wrapper around kafka-python KafkaProducer.
    Automatically falls back to SQLite if Kafka is unavailable.

    Interface mirrors a subset of kafka-python's KafkaProducer API
    so callers don't need to know which backend is active.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        sqlite_path: str | Path = "kafka_fallback.db",
        connect_timeout_ms: int = 2000,
    ):
        self._lock = threading.Lock()
        self._producer = None
        self._sqlite_path = str(sqlite_path)
        self._backend = "sqlite"

        if _KAFKA_AVAILABLE:
            try:
                self._producer = _KP(
                    bootstrap_servers=[bootstrap_servers],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    retries=3,
                    request_timeout_ms=connect_timeout_ms,
                )
                self._backend = "kafka"
                logger.info(f"KafkaProducer connected to {bootstrap_servers}")
            except Exception as e:
                logger.warning(f"Cannot connect to Kafka ({e}). Using SQLite fallback.")

        if self._backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        con = sqlite3.connect(self._sqlite_path)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                topic     TEXT NOT NULL,
                key       TEXT,
                value     TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                consumed  INTEGER DEFAULT 0
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_topic_consumed ON messages(topic, consumed)")
        con.commit()
        con.close()

    def produce(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        """Publish a message to a topic."""
        with self._lock:
            if self._backend == "kafka" and self._producer:
                future = self._producer.send(topic, value=value, key=key)
                try:
                    future.get(timeout=5)
                except Exception as e:
                    logger.error(f"Kafka send failed: {e}, writing to SQLite")
                    self._write_sqlite(topic, key, value)
            else:
                self._write_sqlite(topic, key, value)

    def _write_sqlite(self, topic: str, key: Optional[str], value: dict) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        con = sqlite3.connect(self._sqlite_path)
        con.execute(
            "INSERT INTO messages (topic, key, value, timestamp) VALUES (?, ?, ?, ?)",
            (topic, key, json.dumps(value), ts),
        )
        con.commit()
        con.close()

    def flush(self) -> None:
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        if self._producer:
            self._producer.flush()
            self._producer.close()

    # ------ Consumer helper (SQLite backend) ------

    def poll_sqlite(self, topic: str, batch_size: int = 100) -> list[dict]:
        """Fetch unconsumed messages from the SQLite fallback store."""
        if self._backend != "sqlite":
            return []
        con = sqlite3.connect(self._sqlite_path)
        rows = con.execute(
            "SELECT id, value FROM messages WHERE topic = ? AND consumed = 0 LIMIT ?",
            (topic, batch_size),
        ).fetchall()

        ids = [r[0] for r in rows]
        if ids:
            con.execute(
                f"UPDATE messages SET consumed = 1 WHERE id IN ({','.join('?'*len(ids))})",
                ids,
            )
            con.commit()
        con.close()

        return [json.loads(r[1]) for r in rows]


# Module-level singleton — import and use directly
_default_producer: Optional[KafkaProducerWrapper] = None


def get_producer(
    bootstrap_servers: str = "localhost:9092",
    sqlite_path: str | Path = "kafka_fallback.db",
) -> KafkaProducerWrapper:
    global _default_producer
    if _default_producer is None:
        _default_producer = KafkaProducerWrapper(bootstrap_servers, sqlite_path)
    return _default_producer
