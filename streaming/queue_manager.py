"""
Queue registry — one dedicated async queue per consumer (fan-out pattern).

Usage:
    q = register_consumer("model")   # returns a new Queue for that consumer
    queues = get_all_queues()        # producer calls this to broadcast to all
"""

from asyncio import Queue
from typing import Dict

from config import QUEUE_MAXSIZE

_consumer_queues: Dict[str, Queue] = {}


def register_consumer(name: str) -> Queue:
    """Register a named consumer and return its dedicated queue."""
    q: Queue = Queue(maxsize=QUEUE_MAXSIZE)
    _consumer_queues[name] = q
    return q


def get_all_queues() -> list:
    """Return all registered consumer queues for fan-out broadcasting."""
    return list(_consumer_queues.values())
