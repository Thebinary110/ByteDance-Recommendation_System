"""
Event generator — lazy iterator over clean interaction events.

Pulls one chunk at a time and applies rating filters using vectorised
pandas operations (no Python-level row loop). The outer interface is
still a generator so the pipeline stays memory-bounded and streaming-ready.

Row-level transform_row is preserved and used as a fallback if needed
for custom per-row logic in future stages.
"""

from typing import Iterator

import pandas as pd

from transformer import Event, _parse_timestamp


def generate_events(chunks: Iterator[pd.DataFrame]) -> Iterator[Event]:
    """
    Yield one Event per valid rating row across all chunks.
    Neutral ratings (== 3) are filtered out via vectorised masking.
    Timestamps are converted to Unix seconds in one vectorised pass.
    """
    for chunk in chunks:
        # ── filter: keep only likes (>=4) and dislikes (<=2) ────────────────
        mask = chunk["rating"] != 3.0
        filtered = chunk[mask].copy()

        if filtered.empty:
            continue

        # ── label event types ────────────────────────────────────────────────
        filtered["event_type"] = "dislike"
        filtered.loc[filtered["rating"] >= 4.0, "event_type"] = "like"

        # ── normalise timestamps to Unix int ─────────────────────────────────
        sample = filtered["timestamp"].iloc[0]
        if isinstance(sample, str):
            filtered["timestamp"] = (
                pd.to_datetime(filtered["timestamp"], utc=True)
                .astype("int64") // 10 ** 9
            )
        else:
            filtered["timestamp"] = filtered["timestamp"].astype(int)

        # ── yield as Event dicts ─────────────────────────────────────────────
        for row in filtered[["userId", "movieId", "event_type", "timestamp"]].itertuples(index=False):
            yield Event(
                user_id=str(row.userId),
                item_id=str(row.movieId),
                event_type=row.event_type,
                timestamp=int(row.timestamp),
            )
