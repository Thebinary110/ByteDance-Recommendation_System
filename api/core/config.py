"""
API configuration and sys.path bootstrapping.

Imported first by main.py so all backend packages are resolvable
before any service module is imported.
"""

import os
import sys

API_TITLE   = "Real-Time Recommendation API"
API_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# sys.path — same strategy used across all run_xxx.py entry points
# ---------------------------------------------------------------------------
_ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FS_DIR      = os.path.join(_ROOT, "feature_store")
_EMB_DIR     = os.path.join(_ROOT, "embedding_model")
_RANKING_DIR = os.path.join(_ROOT, "ranking_model")
_STREAM_DIR  = os.path.join(_ROOT, "streaming")

for _p in (_STREAM_DIR, _RANKING_DIR, _EMB_DIR, _FS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
