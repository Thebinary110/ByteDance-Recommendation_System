"""
Ranking model configuration.

Named rank_config — consistent with emb_config / fs_config / rec_config
to avoid occupying the "config" module slot already owned by streaming/.
"""

INPUT_DIM: int   = 64      # user_emb (32) + item_emb (32) concatenated
HIDDEN_DIM: int  = 64      # first hidden layer width
LEARNING_RATE: float = 0.001   # Adam learning rate

TOP_K_CANDIDATES: int = 50   # items retrieved from FAISS before reranking
TOP_K_FINAL: int      = 10   # items returned after MLP reranking

FAISS_REBUILD_INTERVAL: int = 50_000  # rebuild FAISS index every N events
LOG_INTERVAL: int           = 20_000  # log + sample recommendations every N events
