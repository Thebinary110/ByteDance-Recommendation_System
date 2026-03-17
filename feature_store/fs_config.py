"""
Feature store configuration.

Named fs_config (not config) to avoid shadowing streaming/config.py
when both directories share sys.path in run_feature_store.py.
"""

MAX_HISTORY: int = 50        # max interaction records kept per user (deque maxlen)
LOG_INTERVAL: int = 10_000   # log a stats line every N processed events

