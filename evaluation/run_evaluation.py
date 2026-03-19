"""
Standalone offline evaluation — loads events from CSV, trains the embedding
model, builds FAISS, then measures ranking quality.

Why this is needed
------------------
All model state (user_store, embeddings, FAISS) lives in-memory only.
Running this script in a fresh process starts with an empty state.
The script replays events.csv to rebuild that state before evaluating.

Processing stages
-----------------
1. Load SAMPLE_SIZE events from events.csv (sorted by user_id in the CSV,
   so the first N rows are the first ~N/114 users with complete histories).
2. For each event: update_user() + train_on_event() to populate both the
   feature store and embedding model simultaneously.
3. Rebuild FAISS index from trained item embeddings.
4. Call evaluate() with reduced thresholds (MIN_INTERACTIONS=5, TEST_RATIO=0.2)
   to handle the limited number of events per user in a sampled dataset.

Speed
-----
~1400 events/s for train_on_event (6 backward passes each).
SAMPLE_SIZE = 50_000 → ~35 seconds of training.

Run from the project root:
    python evaluation/run_evaluation.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # suppress OpenMP multi-runtime conflict

import logging
import sys
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("feature_store", "embedding_model", "ranking_model", "deepfm",
           "evaluation", "utils", "two_tower"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_evaluation")

# ── imports (after sys.path is set up) ────────────────────────────────────────
from user_store import update_user, _store                   # feature_store
from trainer import train_on_event                           # embedding_model
from faiss_index import faiss_index                          # embedding_model
from embedding_store import user_embeddings, item_embeddings # embedding_model
from dataset_builder import build_eval_dataset
from evaluator import evaluate
from eval_config import TOP_K

# ── configuration ─────────────────────────────────────────────────────────────
EVENTS_PATH  = os.path.join(_ROOT, "data", "processed", "events.csv")
SAMPLE_SIZE  = 50_000    # events to load+train (≈440 users with full histories)
EVAL_MIN_INT = 5         # reduced from default 10 for offline/sampled evaluation
EVAL_TEST_R  = 0.2       # test fraction


def load_and_train(path: str, n: int) -> int:
    """
    Read n events from path, call update_user() and train_on_event() for each.
    Returns the number of events processed.
    """
    log.info("Loading %d events from %s ...", n, path)
    df = pd.read_csv(path, nrows=n)
    total = len(df)
    log.info("  Read %d rows", total)

    t0 = time.perf_counter()
    for i, row in enumerate(df.itertuples(index=False), start=1):
        uid   = str(row.user_id)
        iid   = str(row.item_id)
        etype = str(row.event_type)
        ts    = str(row.timestamp)

        update_user(uid, iid, etype, ts)
        train_on_event({"user_id": uid, "item_id": iid,
                        "event_type": etype, "timestamp": ts})

        if i % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            rate    = i / elapsed
            log.info("  %d / %d events  (%.0f events/s)", i, total, rate)

    elapsed = time.perf_counter() - t0
    log.info("  Done — %.1fs  (%.0f events/s avg)", elapsed, total / elapsed)
    return total


def main() -> None:
    # ── Step 1: populate user_store + train embeddings ────────────────────────
    load_and_train(EVENTS_PATH, SAMPLE_SIZE)

    log.info("user_store  : %d users",  len(_store))
    log.info("user_embeds : %d users",  len(user_embeddings))
    log.info("item_embeds : %d items",  len(item_embeddings))

    # ── Step 2: rebuild FAISS from trained item embeddings ────────────────────
    log.info("Building FAISS index ...")
    n_indexed = faiss_index.build()
    log.info("  FAISS indexed %d items", n_indexed)

    # ── Step 3: build eval dataset with potentially reduced thresholds ─────────
    dataset = build_eval_dataset(
        min_interactions=EVAL_MIN_INT,
        test_ratio=EVAL_TEST_R,
    )
    log.info(
        "Eval dataset: %d eligible users "
        "(MIN_INTERACTIONS=%d, TEST_RATIO=%.2f)",
        len(dataset), EVAL_MIN_INT, EVAL_TEST_R,
    )

    if not dataset:
        # Last resort: drop to minimal thresholds
        log.warning("Dataset still empty — dropping to MIN_INTERACTIONS=3, TEST_RATIO=0.1")
        dataset = build_eval_dataset(min_interactions=3, test_ratio=0.1)
        log.info("After emergency reduction: %d eligible users", len(dataset))

    # ── Step 4: evaluate ──────────────────────────────────────────────────────
    log.info("Running evaluation (TOP_K=%d) ...", TOP_K)
    metrics = evaluate(
        min_interactions=EVAL_MIN_INT,
        test_ratio=EVAL_TEST_R,
    )

    # ── Step 5: print results ─────────────────────────────────────────────────
    print()
    print("--- EVALUATION RESULTS ---")
    print(f"  precision@{TOP_K}  : {metrics[f'precision@{TOP_K}']:.4f}")
    print(f"  recall@{TOP_K}     : {metrics[f'recall@{TOP_K}']:.4f}")
    print(f"  hit_rate@{TOP_K}   : {metrics[f'hit_rate@{TOP_K}']:.4f}")
    print(f"  users evaluated  : {metrics['users_evaluated']:,}")
    print("--------------------------")
    print()


if __name__ == "__main__":
    main()
