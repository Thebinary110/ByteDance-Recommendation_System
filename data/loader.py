"""
Data loading utilities for MovieLens 20M dataset.
Handles chunked loading for the large ratings file and provides a unified interface.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# All 20 MovieLens genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)",
]


def load_ratings(
    path: Path,
    sample_frac: Optional[float] = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Load ratings.csv with optional sampling for development.
    Uses chunked reading to handle the 20M row file efficiently.
    """
    path = Path(path)
    logger.info(f"Loading ratings from {path} …")

    if sample_frac and sample_frac < 1.0:
        # Fast path: estimate rows, then sample
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(chunk.sample(frac=sample_frac, random_state=42))
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Sampled {len(df):,} ratings (frac={sample_frac})")
    else:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df):,} ratings")

    # Normalise column types
    df["userId"] = df["userId"].astype(np.int32)
    df["movieId"] = df["movieId"].astype(np.int32)
    df["rating"] = df["rating"].astype(np.float32)

    # Parse timestamp — MovieLens 20M already has readable timestamps
    if df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df.sort_values("timestamp").reset_index(drop=True)


def load_movies(path: Path) -> pd.DataFrame:
    """
    Load movies.csv and expand genres into a list column.
    Extracts release year from the title string.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df["movieId"] = df["movieId"].astype(np.int32)

    # Extract year from title "(YYYY)"
    df["year"] = (
        df["title"]
        .str.extract(r"\((\d{4})\)\s*$", expand=False)
        .fillna("0")
        .astype(np.int16)
    )

    # Split pipe-delimited genres into list
    df["genre_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    # Multi-hot genre encoding (one column per genre)
    for genre in ALL_GENRES:
        safe = genre.replace("-", "_").replace("(", "").replace(")", "").replace(" ", "_")
        df[f"g_{safe}"] = df["genre_list"].apply(lambda lst: int(genre in lst)).astype(np.uint8)

    logger.info(f"Loaded {len(df):,} movies")
    return df


def load_genome(scores_path: Path, tags_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load genome tag scores and tag names.
    Returns (scores_df, tags_df).
    scores_df has columns: movieId, tagId, relevance
    """
    tags = pd.read_csv(Path(tags_path))
    scores = pd.read_csv(Path(scores_path))
    scores["movieId"] = scores["movieId"].astype(np.int32)
    scores["tagId"] = scores["tagId"].astype(np.int32)
    scores["relevance"] = scores["relevance"].astype(np.float32)
    logger.info(
        f"Loaded genome: {len(tags):,} tags, {len(scores):,} tag-movie scores"
    )
    return scores, tags


def load_links(path: Path) -> pd.DataFrame:
    """Load link.csv mapping movieId → imdbId / tmdbId."""
    df = pd.read_csv(Path(path))
    df["movieId"] = df["movieId"].astype(np.int32)
    return df


def load_all(
    data_dir: str | Path,
    sample_frac: Optional[float] = None,
) -> dict:
    """
    Load the full MovieLens 20M dataset from data_dir.
    Returns a dict with keys: ratings, movies, genome_scores, genome_tags, links.
    """
    data_dir = Path(data_dir)
    return {
        "ratings": load_ratings(data_dir / "rating.csv", sample_frac=sample_frac),
        "movies": load_movies(data_dir / "movie.csv"),
        "genome_scores": load_genome(
            data_dir / "genome_scores.csv",
            data_dir / "genome_tags.csv",
        )[0],
        "genome_tags": load_genome(
            data_dir / "genome_scores.csv",
            data_dir / "genome_tags.csv",
        )[1],
        "links": load_links(data_dir / "link.csv"),
    }
