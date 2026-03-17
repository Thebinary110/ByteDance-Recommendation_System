"""
Recommender configuration.

Named rec_config (not config) — streaming/config.py already occupies the
"config" module slot when all three directories share sys.path.
"""

TOP_K: int = 10               # recommendations returned per user
MIN_HISTORY: int = 3          # minimum interactions before recommendations are generated
SIMILARITY_TOP_N: int = 20    # similar items retrieved per liked item during candidate generation
