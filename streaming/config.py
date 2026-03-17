"""
Streaming configuration — all tunables in one place.
"""

EVENTS_PATH: str = "data/processed/events.csv"

# --- Timing ---
# REPLAY_MODE = True  → sleep proportional to timestamp gaps (time-aware)
# REPLAY_MODE = False → fixed STREAM_DELAY between every event
REPLAY_MODE: bool = True
SCALE_FACTOR: float = 1_000_000   # compress 1M real seconds into 1 simulated second
MAX_REPLAY_DELAY: float = 0.005    # cap any single gap at 5ms so large jumps don't hang
STREAM_DELAY: float = 0.001        # fallback fixed delay (used when REPLAY_MODE = False)

# --- Queue ---
QUEUE_MAXSIZE: int = 10_000        # per-consumer queue depth; producer blocks when full

# --- Logging ---
LOG_INTERVAL: int = 1_000          # consumers log a status line every N events
