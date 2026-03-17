"""
One-time script — builds and saves the movie metadata lookup map.

Input:  data/raw/movie.csv      (movieId, title, genres)
Output: data/processed/movie_map.pkl

Run from project root:
    python utils/movie_loader.py
"""

import os
import pickle

import pandas as pd

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(_ROOT, "data", "raw", "movie.csv")
OUTPUT_PATH = os.path.join(_ROOT, "data", "processed", "movie_map.pkl")


def build_movie_map(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> dict:
    df = pd.read_csv(input_path)

    movie_map = {}
    for _, row in df.iterrows():
        movie_id = str(row["movieId"])
        genres   = row["genres"]
        movie_map[movie_id] = {
            "title":  row["title"],
            "genres": genres.split("|") if genres and genres != "(no genres listed)" else [],
        }

    with open(output_path, "wb") as f:
        pickle.dump(movie_map, f)

    print(f"Saved movie map: {len(movie_map):,} movies -> {output_path}")
    return movie_map


if __name__ == "__main__":
    build_movie_map()
