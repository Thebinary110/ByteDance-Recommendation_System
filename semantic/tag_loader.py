"""
Build movie_tags.pkl from MovieLens genome data.

Genome scores map (movieId, tagId) → relevance in [0, 1].
We pivot this into a per-movie dict: movieId (str) → {tag (str): relevance (float)}.

Run once:
    python semantic/tag_loader.py

Input files:
    data/raw/genome_tags.csv   — tagId, tag
    data/raw/genome_scores.csv — movieId, tagId, relevance

Output:
    data/processed/movie_tags.pkl
"""

import os
import pickle
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_tag_data(
    genome_tags_csv: str,
    genome_scores_csv: str,
    output_path: str,
) -> None:
    print("Loading genome tags...")
    tags_df = pd.read_csv(genome_tags_csv)
    tag_map = dict(zip(tags_df["tagId"].astype(int), tags_df["tag"]))
    print(f"  {len(tag_map)} unique tags")

    print("Loading genome scores (vectorized)...")
    scores_df = pd.read_csv(genome_scores_csv)
    scores_df["tagId"]   = scores_df["tagId"].astype(int)
    scores_df["movieId"] = scores_df["movieId"].astype(int).astype(str)
    scores_df["tag"]     = scores_df["tagId"].map(tag_map)
    scores_df.dropna(subset=["tag"], inplace=True)

    print("Building per-movie dicts...")
    movie_tags: dict = {}
    for movie_id, grp in scores_df.groupby("movieId"):
        movie_tags[movie_id] = dict(zip(grp["tag"], grp["relevance"].astype(float)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(movie_tags, f)

    print(f"Saved semantic tag data: {len(movie_tags)} movies -> {output_path}")


if __name__ == "__main__":
    build_tag_data(
        genome_tags_csv   = os.path.join(_ROOT, "data", "raw",       "genome_tags.csv"),
        genome_scores_csv = os.path.join(_ROOT, "data", "raw",       "genome_scores.csv"),
        output_path       = os.path.join(_ROOT, "data", "processed", "movie_tags.pkl"),
    )
