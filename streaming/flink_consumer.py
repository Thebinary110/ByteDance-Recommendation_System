"""
Flink-style streaming consumer for real-time feature updates.

Because Apache Flink requires a JVM cluster, this module implements
the same processing logic using Python threads — making it runnable
locally without any cluster infrastructure while preserving the
exact same feature-update semantics.

To replace with real Flink: implement the same process_event() logic
inside a Flink PyFlink DataStream map/process function.

What this consumer does:
  1. Poll events from Kafka (or SQLite fallback)
  2. For each event, update the user's genre preference vector
     in the feature store (incremental exponential moving average)
  3. Invalidate the user's cached embedding so it's recomputed fresh
  4. Accumulate model-update signals — if a user has N new interactions,
     emit a retraining trigger to the model_updates topic

Usage:
  python -m streaming.flink_consumer --artifact_dir artifacts/ --interval 10
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StreamingFeatureUpdater:
    """
    Consumes events and applies incremental updates to the feature store.

    Updates are applied using an Exponential Moving Average (EMA) so recent
    interactions have higher weight than old ones without re-scanning history.
    """

    EMA_ALPHA = 0.1  # weight given to the new observation

    def __init__(
        self,
        feature_store,          # FeatureStore instance
        producer,               # KafkaProducerWrapper instance
        genre_list: list[str],  # ordered list of genre names
        retrain_threshold: int = 500,
    ):
        self.feature_store = feature_store
        self.producer = producer
        self.genre_list = genre_list
        self.genre_index = {g: i for i, g in enumerate(genre_list)}
        self.retrain_threshold = retrain_threshold
        self._interaction_counter: dict[int, int] = {}
        self._lock = threading.Lock()

    def process_event(self, event: dict[str, Any]) -> None:
        """Process a single event from the stream."""
        event_type = event.get("event_type")
        user_id = event.get("user_id")
        movie_id = event.get("movie_id")
        metadata = event.get("metadata", {})

        if user_id is None:
            return

        if event_type in ("click", "rating", "feedback", "watch"):
            self._update_user_features(user_id, movie_id, metadata)
            self._maybe_trigger_retrain(user_id)

    def _update_user_features(
        self,
        user_id: int,
        movie_id: int | None,
        metadata: dict,
    ) -> None:
        """
        Update user genre preference vector using EMA.
        Fetches current features, blends in new signal, writes back.
        """
        if movie_id is None:
            return

        # Get current user feature vector
        current = self.feature_store.get_user_features(user_id)
        if current is None:
            current = np.zeros(len(self.genre_list) + 2, dtype=np.float32)

        # Determine genre signal for this movie
        # (in production, fetch from item feature store)
        movie_genres = metadata.get("genres", [])
        genre_signal = np.zeros(len(self.genre_list), dtype=np.float32)
        for genre in movie_genres:
            idx = self.genre_index.get(genre)
            if idx is not None:
                genre_signal[idx] = 1.0

        # Rating signal for preference strength
        rating = float(metadata.get("rating", 3.5))
        strength = (rating - 1.0) / 4.0  # normalise to [0, 1]

        # EMA update on the genre portion
        n_genres = len(self.genre_list)
        current[:n_genres] = (
            (1 - self.EMA_ALPHA) * current[:n_genres]
            + self.EMA_ALPHA * genre_signal * strength
        )

        self.feature_store.set_user_features(user_id, current)

        # Invalidate cached embedding so it's recomputed with new features
        self.feature_store._backend.delete(f"user:{user_id}:embedding")

    def _maybe_trigger_retrain(self, user_id: int) -> None:
        with self._lock:
            self._interaction_counter[user_id] = (
                self._interaction_counter.get(user_id, 0) + 1
            )
            total = sum(self._interaction_counter.values())
            if total >= self.retrain_threshold:
                self.producer.produce(
                    "model_updates",
                    {
                        "trigger": "interaction_threshold",
                        "total_new_interactions": total,
                        "affected_users": len(self._interaction_counter),
                    },
                )
                self._interaction_counter.clear()
                logger.info(
                    f"Retrain trigger emitted — {total} new interactions accumulated."
                )


class FlinkConsumerThread:
    """
    Runs StreamingFeatureUpdater in a background daemon thread,
    polling for new events every `interval` seconds.
    """

    def __init__(
        self,
        updater: StreamingFeatureUpdater,
        topic: str = "rec_events",
        interval: float = 5.0,
        batch_size: int = 200,
    ):
        self.updater = updater
        self.topic = topic
        self.interval = interval
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FlinkConsumer")

    def start(self) -> None:
        self._thread.start()
        logger.info(
            f"FlinkConsumerThread started — polling '{self.topic}' every {self.interval}s"
        )

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                events = self.updater.producer.poll_sqlite(
                    self.topic, batch_size=self.batch_size
                )
                for event in events:
                    self.updater.process_event(event)
                if events:
                    logger.debug(f"Processed {len(events)} events from '{self.topic}'")
            except Exception as e:
                logger.error(f"FlinkConsumerThread error: {e}")
            self._stop.wait(self.interval)


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    from data.loader import ALL_GENRES
    from serving.feature_store import FeatureStore
    from streaming.kafka_producer import KafkaProducerWrapper

    feature_store = FeatureStore(
        sqlite_path=Path(args.artifact_dir) / "feature_store.db"
    )
    producer = KafkaProducerWrapper(
        sqlite_path=Path(args.artifact_dir) / "kafka_fallback.db"
    )

    updater = StreamingFeatureUpdater(feature_store, producer, ALL_GENRES)
    consumer = FlinkConsumerThread(
        updater, interval=args.interval, batch_size=args.batch_size
    )
    consumer.start()

    logger.info("Consumer running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        consumer.stop()
        logger.info("Consumer stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default="artifacts")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=200)
    main(parser.parse_args())
