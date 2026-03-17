"""
Pipeline configuration — all paths and tunables in one place.
"""

INPUT_PATH: str = "data/raw/rating.csv"
OUTPUT_PATH: str = "data/processed/events.csv"
CHUNK_SIZE: int = 50_000
