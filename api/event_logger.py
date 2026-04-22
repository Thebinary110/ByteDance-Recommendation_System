"""
Event Logger — records user interactions for offline analysis and model updates.

Events are published to Kafka when available, otherwise persisted in SQLite.
The streaming layer (flink_consumer.py) picks them up and updates the feature store.

Event schema:
  {
    "event_id": str (uuid4),
    "user_id": int,
    "event_type": "view" | "click" | "rating" | "search" | "feedback",
    "movie_id": int | null,
    "timestamp": ISO-8601 string,
    "metadata": {}
  }
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_KAFKA_AVAILABLE = False
try:
    from kafka import KafkaProducer as _KafkaProducer
    _KAFKA_AVAILABLE = True
except ImportError:
    pass


EVENTS_TOPIC = "rec_events"


class EventLogger:
    """
    Publishes events to Kafka (preferred) or SQLite (fallback).
    Constructed once at API startup and injected via FastAPI dependency.
    """

    def __init__(
        self,
        kafka_bootstrap: str = "localhost:9092",
        sqlite_path: str | Path = "events.db",
    ):
        self._producer = None
        self._sqlite_path = str(sqlite_path)
        self._backend: str

        if _KAFKA_AVAILABLE:
            try:
                self._producer = _KafkaProducer(
                    bootstrap_servers=[kafka_bootstrap],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    request_timeout_ms=2000,
                )
                # Verify connectivity
                self._producer.bootstrap_connected()
                self._backend = "kafka"
                logger.info(f"EventLogger connected to Kafka at {kafka_bootstrap}")
            except Exception as e:
                logger.warning(f"Kafka unavailable ({e}), falling back to SQLite.")
                self._producer = None

        if self._producer is None:
            self._backend = "sqlite"
            self._init_sqlite()
            logger.info(f"EventLogger using SQLite at {sqlite_path}")

    def _init_sqlite(self) -> None:
        con = sqlite3.connect(self._sqlite_path)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id   TEXT PRIMARY KEY,
                user_id    INTEGER,
                event_type TEXT,
                movie_id   INTEGER,
                timestamp  TEXT,
                metadata   TEXT
            )
            """
        )
        con.commit()
        con.close()

    def log(
        self,
        user_id: int,
        event_type: str,
        movie_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        event = {
            "event_id": event_id,
            "user_id": user_id,
            "event_type": event_type,
            "movie_id": movie_id,
            "timestamp": ts,
            "metadata": metadata or {},
        }

        if self._backend == "kafka" and self._producer:
            self._producer.send(EVENTS_TOPIC, value=event)
        else:
            self._write_sqlite(event)

        return event_id

    def _write_sqlite(self, event: dict) -> None:
        con = sqlite3.connect(self._sqlite_path)
        con.execute(
            """
            INSERT OR IGNORE INTO events
              (event_id, user_id, event_type, movie_id, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event["event_id"],
                event["user_id"],
                event["event_type"],
                event.get("movie_id"),
                event["timestamp"],
                json.dumps(event.get("metadata", {})),
            ),
        )
        con.commit()
        con.close()

    def recent_events(self, limit: int = 100) -> list[dict]:
        """Return the most recent events (SQLite backend only)."""
        if self._backend != "sqlite":
            return []
        con = sqlite3.connect(self._sqlite_path)
        rows = con.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        return [
            {
                "event_id": r[0], "user_id": r[1], "event_type": r[2],
                "movie_id": r[3], "timestamp": r[4], "metadata": json.loads(r[5]),
            }
            for r in rows
        ]

    def close(self) -> None:
        if self._producer:
            self._producer.flush()
            self._producer.close()
