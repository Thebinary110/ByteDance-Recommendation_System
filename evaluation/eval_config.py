# Evaluation hyper-parameters

TOP_K             = 10      # recommendation list length to evaluate
MIN_INTERACTIONS  = 10      # skip users with fewer total events
TEST_RATIO        = 0.2     # fraction of history held out as test
TOP_K_CANDIDATES  = 100     # FAISS retrieval pool for evaluation
EVAL_INTERVAL     = 50_000  # run evaluation every N streaming events
