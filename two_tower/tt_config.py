# Two-Tower retrieval hyper-parameters
#
# These are separate from emb_config (Stage 5) which drives the 32-dim two-tower
# that feeds DeepFM. This model is the 64-dim retrieval tower.

EMBED_DIM        = 64
LEARNING_RATE    = 1e-3
NEGATIVE_SAMPLES = 2     # reduced from 5 (Stage 15.6 throughput optimisation)
TOP_K            = 50
BATCH_SIZE       = 32    # accumulate N like-events before one forward+backward pass

NUM_USERS        = 150_000
NUM_ITEMS        = 200_000
