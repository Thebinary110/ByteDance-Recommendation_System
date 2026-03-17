"""
Feature updater — applies business logic on each incoming event.

Thin pass-through for now: routes the event directly to user_store.update_user.
Will grow when feature engineering logic is added (e.g. session detection,
recency weighting, content-based signals).
"""

from user_store import update_user


def process_event(event: dict) -> None:
    """Extract fields from a raw event dict and update the user store."""
    update_user(
        user_id=event["user_id"],
        item_id=event["item_id"],
        event_type=event["event_type"],
        timestamp=event["timestamp"],
    )
