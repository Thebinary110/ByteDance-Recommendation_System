"""
Transformer — converts a single ratings row into a typed interaction event.

Rating rules:
  >= 4   →  like
  <= 2   →  dislike
  == 3   →  ignored (neutral, not useful signal)

Timestamp note: the dataset stores timestamps as datetime strings
(e.g. "2005-04-02 23:53:47"). We normalise them to Unix seconds (int)
so the output schema is consistent and streaming-ready.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict


class Event(TypedDict):
    user_id: str
    item_id: str
    event_type: str
    timestamp: int


def _parse_timestamp(raw: str | int | float) -> int:
    """
    Accept either a Unix numeric timestamp or an ISO-style datetime string
    and always return an integer Unix timestamp (UTC seconds).
    """
    if isinstance(raw, (int, float)):
        return int(raw)
    # datetime string  →  UTC epoch seconds
    dt = datetime.strptime(str(raw).strip(), "%Y-%m-%d %H:%M:%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def transform_row(row: "pd.Series") -> Event | None:  # type: ignore[name-defined]
    """
    Map one rating row to an Event dict, or return None to skip it.
    """
    rating: float = float(row["rating"])

    if rating >= 4.0:
        event_type = "like"
    elif rating <= 2.0:
        event_type = "dislike"
    else:
        return None  # neutral — discard

    return Event(
        user_id=str(row["userId"]),
        item_id=str(row["movieId"]),
        event_type=event_type,
        timestamp=_parse_timestamp(row["timestamp"]),
    )
