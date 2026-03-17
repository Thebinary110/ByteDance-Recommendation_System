"""
Pipeline orchestrator — wires parser → event_generator → CSV writer.

Run from the project root:
    python data_pipeline/run_pipeline.py
"""

import csv
import os
import sys
import time

# Ensure sibling modules resolve when invoked from the project root.
sys.path.insert(0, os.path.dirname(__file__))

from config import CHUNK_SIZE, INPUT_PATH, OUTPUT_PATH
from event_generator import generate_events
from parser import read_data_in_chunks

FIELDNAMES = ["user_id", "item_id", "event_type", "timestamp"]
LOG_INTERVAL = 100_000


def run_pipeline() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"[START] Reading  : {INPUT_PATH}")
    print(f"        Writing  : {OUTPUT_PATH}")
    print(f"        Chunk    : {CHUNK_SIZE:,} rows")
    print()

    chunks = read_data_in_chunks(INPUT_PATH, CHUNK_SIZE)
    events = generate_events(chunks)

    t0 = time.perf_counter()
    count = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for event in events:
            writer.writerow(event)
            count += 1

            if count % LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - t0
                rate = count / elapsed
                print(f"  [+] {count:>10,} events  |  {rate:,.0f} rows/s")

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

    print()
    print(f"[DONE] Total events written : {count:,}")
    print(f"       Output size          : {size_mb:.1f} MB")
    print(f"       Time elapsed         : {elapsed:.1f}s")
    print(f"       Avg throughput       : {count / elapsed:,.0f} rows/s")


if __name__ == "__main__":
    run_pipeline()
