"""
Embedding model configuration.

Named emb_config (not config) — streaming/config.py already occupies the
"config" module slot when both directories share sys.path.
"""

EMBEDDING_DIM: int = 32       # vector size for user and item embeddings
LEARNING_RATE: float = 0.01   # online SGD step size
NEGATIVE_SAMPLES: int = 5           # random negatives sampled per event
FAISS_REBUILD_INTERVAL: int = 20_000 # rebuild FAISS index every N events
LOG_INTERVAL: int = 10_000           # log stats + sample recommendation every N events
