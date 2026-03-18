"""
Build link_map.pkl from MovieLens link.csv.

Maps movieId (str) -> {imdb: str (7-digit zero-padded), tmdb: str or None}

Run once:
    python utils/link_loader.py

Input:  data/raw/link.csv
Output: data/processed/link_map.pkl
"""

import os
import pickle

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_link_map(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    link_map = {}
    for _, row in df.iterrows():
        movie_id = str(int(row["movieId"]))

        # IMDb IDs need 7-digit zero-padding (tt0114709 format)
        imdb = str(int(row["imdbId"])).zfill(7)

        # tmdbId can be NaN (252 movies have no TMDB entry)
        tmdb = str(int(row["tmdbId"])) if pd.notna(row["tmdbId"]) else None

        link_map[movie_id] = {"imdb": imdb, "tmdb": tmdb}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(link_map, f)

    print(f"Saved link map: {len(link_map)} movies -> {output_path}")


if __name__ == "__main__":
    build_link_map(
        input_path  = os.path.join(_ROOT, "data", "raw",       "link.csv"),
        output_path = os.path.join(_ROOT, "data", "processed", "link_map.pkl"),
    )
