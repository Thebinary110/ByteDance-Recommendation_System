"""
Parser — reads the raw ratings CSV in fixed-size chunks.
Memory footprint is bounded by CHUNK_SIZE, never the full file.
"""

from typing import Iterator

import pandas as pd


def read_data_in_chunks(file_path: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    """
    Yield successive DataFrame chunks from a CSV file.
    No data is held in memory beyond one chunk at a time.
    """
    return pd.read_csv(
        file_path,
        chunksize=chunk_size,
        dtype={
            "userId": str,
            "movieId": str,
            "rating": float,
        },
    )
