"""
Batch processing with PySpark (or pandas fallback).

Runs periodically (e.g., nightly) to:
  1. Recompute user feature vectors from the full interaction history
  2. Detect drift — compare new feature distributions with stored ones
  3. Emit a retraining signal if drift exceeds threshold
  4. Optionally trigger full model retraining

PySpark is optional — if not available, the same logic runs with pandas.
This keeps the system runnable on a laptop while staying production-ready.

Usage:
  python -m streaming.batch_spark \
      --data_dir ../ \
      --artifact_dir artifacts/ \
      --output_dir artifacts/

Or scheduled via cron / Windows Task Scheduler for nightly runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    _SPARK_AVAILABLE = True
except ImportError:
    pass


# ------------------------------------------------------------------
# Pandas implementation (always available)
# ------------------------------------------------------------------

def compute_user_stats_pandas(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    genre_cols: list[str],
) -> pd.DataFrame:
    """
    Computes per-user genre preferences and rating stats using pandas.
    Returns DataFrame indexed by user_idx with feature columns.
    """
    # Merge ratings with movie genre flags
    merged = ratings_df.merge(
        movies_df[["movieId"] + genre_cols],
        left_on="movieId",
        right_on="movieId",
        how="left",
    )

    # Weighted genre preference (weight = rating / 5.0)
    for col in genre_cols:
        merged[f"w_{col}"] = merged[col] * merged["rating"] / 5.0

    weighted_cols = [f"w_{c}" for c in genre_cols]
    user_genre = merged.groupby("userId")[weighted_cols].mean()
    user_genre.columns = genre_cols  # rename back

    user_stats = merged.groupby("userId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )

    result = user_genre.join(user_stats, how="left")
    result["avg_rating"] = result["avg_rating"] / 5.0
    result["log_count"] = np.log1p(result["rating_count"]) / 10.0
    return result.fillna(0.0)


def compute_user_stats_spark(
    spark: "SparkSession",
    ratings_path: str,
    movies_path: str,
    genre_cols: list[str],
):
    """Spark implementation for large-scale recomputation."""
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)

    joined = ratings.join(movies.select(["movieId"] + genre_cols), on="movieId", how="left")

    # Weighted genre aggregation
    agg_exprs = [
        F.mean(F.col(c) * F.col("rating") / 5.0).alias(c)
        for c in genre_cols
    ]
    agg_exprs += [
        F.mean("rating").alias("avg_rating"),
        F.count("rating").alias("rating_count"),
    ]

    user_features = joined.groupBy("userId").agg(*agg_exprs)
    return user_features


# ------------------------------------------------------------------
# Drift detection
# ------------------------------------------------------------------

def detect_feature_drift(
    new_features: np.ndarray,
    old_features: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """
    Computes per-dimension mean absolute change.
    Returns drift report and whether retraining is recommended.
    """
    if new_features.shape != old_features.shape:
        return {"drift_detected": True, "reason": "shape_mismatch"}

    mean_change = np.abs(new_features - old_features).mean(axis=0)
    max_drift = float(mean_change.max())
    avg_drift = float(mean_change.mean())

    return {
        "max_feature_drift": max_drift,
        "avg_feature_drift": avg_drift,
        "drift_detected": max_drift > threshold,
        "n_users": len(new_features),
    }


# ------------------------------------------------------------------
# Main batch job
# ------------------------------------------------------------------

def run_batch_job(
    data_dir: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
    use_spark: bool = False,
) -> dict:
    """
    Full batch pipeline:
      1. Load full interaction history
      2. Recompute user features
      3. Detect drift vs stored features
      4. Save updated features
      5. Return drift report
    """
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from data.loader import ALL_GENRES, load_movies, load_ratings
    from data.preprocessor import Preprocessor

    logger.info("Loading preprocessor …")
    prep = Preprocessor.load(artifact_dir)

    logger.info("Loading full ratings dataset …")
    ratings = load_ratings(data_dir / "rating.csv")
    movies = load_movies(data_dir / "movie.csv")

    from data.loader import GENRE_COLS
    genre_cols = [
        f"g_{g.replace('-','_').replace('(','').replace(')','').replace(' ','_')}"
        for g in ALL_GENRES
    ]

    if use_spark and _SPARK_AVAILABLE:
        logger.info("Running Spark batch job …")
        spark = (
            SparkSession.builder
            .appName("RecommenderBatchJob")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )
        user_stats_spark = compute_user_stats_spark(
            spark,
            str(data_dir / "rating.csv"),
            str(data_dir / "movie.csv"),
            genre_cols,
        )
        user_stats = user_stats_spark.toPandas()
    else:
        logger.info("Running pandas batch job …")
        user_stats = compute_user_stats_pandas(ratings, movies, genre_cols)

    # Re-encode user IDs and build new feature matrix
    new_user_features = np.zeros(
        (prep.num_users, len(ALL_GENRES) + 2), dtype=np.float32
    )
    for orig_id, row in user_stats.iterrows():
        idx = prep.user_id_map.get(orig_id)
        if idx is None:
            continue
        genre_vals = [float(row.get(c, 0.0)) for c in genre_cols]
        new_user_features[idx, : len(ALL_GENRES)] = genre_vals
        new_user_features[idx, len(ALL_GENRES)] = float(row.get("avg_rating", 0.7))
        new_user_features[idx, len(ALL_GENRES) + 1] = float(row.get("log_count", 0.0))

    # Load old features for drift comparison
    old_path = artifact_dir / "user_features.npy"
    drift_report = {}
    if old_path.exists():
        old_features = np.load(str(old_path))
        drift_report = detect_feature_drift(new_user_features, old_features)
        logger.info(f"Drift report: {drift_report}")
    else:
        drift_report = {"drift_detected": False, "reason": "first_run"}

    # Save updated features
    np.save(str(output_dir / "user_features.npy"), new_user_features)
    logger.info(f"Updated user features saved to {output_dir / 'user_features.npy'}")

    if drift_report.get("drift_detected"):
        logger.warning(
            f"Significant feature drift detected (max={drift_report.get('max_feature_drift', 0):.4f}). "
            "Consider retraining the models."
        )

    return drift_report


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Batch feature recomputation")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--artifact_dir", default="artifacts")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--spark", action="store_true")
    args = parser.parse_args()

    report = run_batch_job(args.data_dir, args.artifact_dir, args.output_dir, args.spark)
    print(f"\nBatch job complete. Drift report:\n{report}")
