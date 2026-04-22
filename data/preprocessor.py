"""
Feature engineering and dataset preparation for the recommendation pipeline.

Steps:
1. Encode user/movie IDs to contiguous 0-based integers
2. Compute user features (genre preferences, activity stats)
3. Compute item features (genre encoding, genome PCA embeddings, year)
4. Temporal train/val/test split
5. Generate negative samples for training
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .loader import ALL_GENRES, load_all

logger = logging.getLogger(__name__)

GENRE_COLS = [
    f"g_{g.replace('-','_').replace('(','').replace(')','').replace(' ','_')}"
    for g in ALL_GENRES
]


class Preprocessor:
    """
    Encapsulates all feature-engineering logic.
    Call fit() on training data, then transform() on any split.
    Artifacts are saved to output_dir for serving-time use.
    """

    def __init__(self, genome_n_components: int = 32):
        self.genome_n_components = genome_n_components
        self.user_id_map: dict[int, int] = {}
        self.movie_id_map: dict[int, int] = {}
        self.user_id_unmap: dict[int, int] = {}
        self.movie_id_unmap: dict[int, int] = {}
        self.num_users: int = 0
        self.num_movies: int = 0
        self.genre_scaler = StandardScaler()
        self.genome_svd: Optional[TruncatedSVD] = None
        self.item_features: Optional[np.ndarray] = None  # (num_movies, feature_dim)
        self.user_features: Optional[np.ndarray] = None  # (num_users, feature_dim)
        self.movie_df: Optional[pd.DataFrame] = None
        self.fitted = False

    # ------------------------------------------------------------------
    # ID encoding
    # ------------------------------------------------------------------

    def _build_id_maps(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        unique_users = sorted(ratings["userId"].unique())
        unique_movies = sorted(movies["movieId"].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.user_id_unmap = {v: k for k, v in self.user_id_map.items()}
        self.movie_id_unmap = {v: k for k, v in self.movie_id_map.items()}
        self.num_users = len(unique_users)
        self.num_movies = len(unique_movies)
        logger.info(f"ID maps: {self.num_users:,} users, {self.num_movies:,} movies")

    def encode_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        df = ratings.copy()
        df["user_idx"] = df["userId"].map(self.user_id_map)
        df["movie_idx"] = df["movieId"].map(self.movie_id_map)
        # Drop rows where IDs aren't in training vocabulary
        df = df.dropna(subset=["user_idx", "movie_idx"])
        df["user_idx"] = df["user_idx"].astype(np.int32)
        df["movie_idx"] = df["movie_idx"].astype(np.int32)
        return df

    # ------------------------------------------------------------------
    # Item feature matrix
    # ------------------------------------------------------------------

    def _build_item_features(
        self, movies: pd.DataFrame, genome_scores: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Returns (num_movies, feature_dim) float32 array."""
        # Genre multi-hot — shape (num_movies, 20)
        genre_matrix = np.zeros((self.num_movies, len(GENRE_COLS)), dtype=np.float32)
        for _, row in movies.iterrows():
            idx = self.movie_id_map.get(row["movieId"])
            if idx is None:
                continue
            for i, col in enumerate(GENRE_COLS):
                genre_matrix[idx, i] = row.get(col, 0)

        # Normalised release year — shape (num_movies, 1)
        year_vec = np.zeros((self.num_movies, 1), dtype=np.float32)
        for _, row in movies.iterrows():
            idx = self.movie_id_map.get(row["movieId"])
            if idx is None:
                continue
            year_vec[idx, 0] = max(0, row.get("year", 2000) - 1900) / 100.0

        features = [genre_matrix, year_vec]

        # Genome tag embeddings via SVD — shape (num_movies, genome_n_components)
        if genome_scores is not None:
            genome_matrix = self._build_genome_matrix(movies, genome_scores)
            features.append(genome_matrix)

        item_feats = np.concatenate(features, axis=1)
        logger.info(f"Item feature dim: {item_feats.shape[1]}")
        return item_feats

    def _build_genome_matrix(
        self, movies: pd.DataFrame, genome_scores: pd.DataFrame
    ) -> np.ndarray:
        """Pivot genome scores to (num_movies, num_tags), then reduce with SVD."""
        pivot = genome_scores.pivot_table(
            index="movieId", columns="tagId", values="relevance", fill_value=0.0
        )
        # Only keep movies in our vocabulary
        valid_ids = [
            mid for mid in pivot.index if mid in self.movie_id_map
        ]
        pivot = pivot.loc[valid_ids]

        if self.genome_svd is None:
            n = min(self.genome_n_components, pivot.shape[1] - 1, pivot.shape[0] - 1)
            self.genome_svd = TruncatedSVD(n_components=n, random_state=42)
            reduced = self.genome_svd.fit_transform(pivot.values)
        else:
            reduced = self.genome_svd.transform(pivot.values)

        genome_matrix = np.zeros(
            (self.num_movies, self.genome_svd.n_components), dtype=np.float32
        )
        for orig_id, row in zip(valid_ids, reduced):
            idx = self.movie_id_map[orig_id]
            genome_matrix[idx] = row.astype(np.float32)

        logger.info(f"Genome SVD shape: {genome_matrix.shape}")
        return genome_matrix

    # ------------------------------------------------------------------
    # User feature matrix
    # ------------------------------------------------------------------

    def _build_user_features(
        self, ratings: pd.DataFrame, encoded: pd.DataFrame
    ) -> np.ndarray:
        """
        Returns (num_users, feature_dim) float32 array.
        Features: genre preferences (20), avg_rating (1), log_count (1) → 22-dim.

        Fully vectorised: uses np.bincount (compiled C) per genre column instead
        of a Python groupby loop, reducing runtime from minutes to seconds on 2M rows.
        """
        n_genres = len(GENRE_COLS)
        genre_pref = np.zeros((self.num_users, n_genres), dtype=np.float64)
        avg_rating = np.zeros((self.num_users, 1), dtype=np.float32)
        log_count = np.zeros((self.num_users, 1), dtype=np.float32)

        if self.item_features is not None:
            user_idxs = encoded["user_idx"].values.astype(np.int64)
            movie_idxs = encoded["movie_idx"].values.astype(np.int64)
            rating_vals = encoded["rating"].values.astype(np.float64)

            item_genre = self.item_features[:, :n_genres].astype(np.float64)

            # Weighted genre signal per interaction: [N, n_genres]
            genre_weighted = item_genre[movie_idxs] * rating_vals[:, None]

            # Accumulate per user with np.bincount (fast C loop, one call per genre)
            for g in range(n_genres):
                genre_pref[:, g] = np.bincount(
                    user_idxs, weights=genre_weighted[:, g], minlength=self.num_users
                )

            # Per-user interaction counts and rating sums
            counts = np.bincount(user_idxs, minlength=self.num_users).astype(np.float64)
            rating_sum = np.bincount(user_idxs, weights=rating_vals, minlength=self.num_users)

            mask = counts > 0
            genre_pref[mask] /= counts[mask, None]
            avg_rating[mask, 0] = (rating_sum[mask] / counts[mask] / 5.0).astype(np.float32)
            log_count[mask, 0] = (np.log1p(counts[mask]) / 10.0).astype(np.float32)

        user_feats = np.concatenate([genre_pref.astype(np.float32), avg_rating, log_count], axis=1)
        logger.info(f"User feature dim: {user_feats.shape[1]}")
        return user_feats

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------

    def temporal_split(
        self,
        encoded: pd.DataFrame,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by timestamp: oldest interactions → train, newest → test.
        This avoids data leakage from the future.
        """
        n = len(encoded)
        test_start = int(n * (1 - test_frac))
        val_start = int(n * (1 - val_frac - test_frac))

        train = encoded.iloc[:val_start].copy()
        val = encoded.iloc[val_start:test_start].copy()
        test = encoded.iloc[test_start:].copy()

        logger.info(
            f"Split — train: {len(train):,}, val: {len(val):,}, test: {len(test):,}"
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def generate_negatives(
        self,
        encoded: pd.DataFrame,
        n_neg: int = 4,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        For each positive (user, movie) pair, sample n_neg random negatives.
        Returns DataFrame with columns: user_idx, pos_movie_idx, neg_movie_idx.

        Fully vectorised: samples negatives in one numpy call, then fixes the rare
        case where a sampled negative equals the positive (collision rate ≈ 0.5% on
        MovieLens given ~145 ratings per user vs 27K movies).  Full seen-set filtering
        is skipped intentionally — the sparsity of the dataset means almost all random
        items are true negatives, which is standard practice for large-scale BPR.
        """
        rng = np.random.default_rng(seed)

        user_idxs = encoded["user_idx"].values.astype(np.int32)
        pos_idxs = encoded["movie_idx"].values.astype(np.int32)
        N = len(user_idxs)

        # Sample n_neg negatives per interaction in one vectorised call
        neg_samples = rng.integers(0, self.num_movies, size=(N, n_neg), dtype=np.int32)

        # Fix the trivial collision: neg == pos (rare but easy to handle vectorially)
        for k in range(n_neg):
            collision = neg_samples[:, k] == pos_idxs
            neg_samples[collision, k] = (pos_idxs[collision] + 1) % self.num_movies

        # Flatten: each positive gets n_neg rows
        uid_rep = np.repeat(user_idxs, n_neg)
        pos_rep = np.repeat(pos_idxs, n_neg)
        neg_flat = neg_samples.ravel()

        df = pd.DataFrame({
            "user_idx": uid_rep,
            "pos_movie_idx": pos_rep,
            "neg_movie_idx": neg_flat,
        })
        logger.info(f"Generated {len(df):,} training triplets")
        return df

    # ------------------------------------------------------------------
    # Fit / transform / save
    # ------------------------------------------------------------------

    def fit(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        genome_scores: Optional[pd.DataFrame] = None,
    ) -> "Preprocessor":
        self.movie_df = movies
        self._build_id_maps(ratings, movies)
        encoded = self.encode_ratings(ratings)

        self.item_features = self._build_item_features(movies, genome_scores)
        self.user_features = self._build_user_features(ratings, encoded)
        self.fitted = True
        return self

    def fit_transform(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        genome_scores: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        self.fit(ratings, movies, genome_scores)
        return self.encode_ratings(ratings)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily strip movie_df before pickling.
        # Pandas DataFrames with list/object columns trigger a NDArrayBacked
        # __setstate__ incompatibility across pandas 2.x minor versions.
        # We save movie metadata as a plain CSV instead — always portable.
        movie_df_backup = self.movie_df
        self.movie_df = None

        with open(output_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(self, f)

        self.movie_df = movie_df_backup  # restore in-memory object

        # Save movie metadata as version-safe CSV
        if movie_df_backup is not None:
            meta_cols = [c for c in ["movieId", "title", "genres", "year"] if c in movie_df_backup.columns]
            movie_df_backup[meta_cols].to_csv(output_dir / "movie_meta.csv", index=False)

        np.save(output_dir / "item_features.npy", self.item_features)
        np.save(output_dir / "user_features.npy", self.user_features)
        logger.info(f"Preprocessor saved to {output_dir}")

    @classmethod
    def load(cls, output_dir: str | Path) -> "Preprocessor":
        output_dir = Path(output_dir).resolve()  # always absolute — safe regardless of CWD
        with open(output_dir / "preprocessor.pkl", "rb") as f:
            obj = pickle.load(f)

        # Reload movie_df from CSV (bypasses pandas pickle version issues)
        meta_path = output_dir / "movie_meta.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            df["genre_list"] = df["genres"].apply(
                lambda g: [] if str(g) == "(no genres listed)" else str(g).split("|")
            )
            obj.movie_df = df
        else:
            obj.movie_df = None

        # Also reload the numpy feature matrices in case they were updated
        item_path = output_dir / "item_features.npy"
        user_path = output_dir / "user_features.npy"
        if item_path.exists():
            obj.item_features = np.load(str(item_path))
        if user_path.exists():
            obj.user_features = np.load(str(user_path))

        return obj


# ------------------------------------------------------------------
# Convenience entry-point
# ------------------------------------------------------------------

def preprocess_all(
    data_dir: str | Path,
    output_dir: str | Path,
    sample_frac: Optional[float] = None,
    genome_n_components: int = 32,
) -> tuple["Preprocessor", dict]:
    """
    Full preprocessing pipeline: load → encode → features → split → save.
    Returns (preprocessor, splits) where splits = {train, val, test, negatives}.
    """
    data = load_all(data_dir, sample_frac=sample_frac)

    prep = Preprocessor(genome_n_components=genome_n_components)
    encoded = prep.fit_transform(
        data["ratings"], data["movies"], data["genome_scores"]
    )
    train, val, test = prep.temporal_split(encoded)
    negatives = prep.generate_negatives(train, n_neg=4)

    prep.save(output_dir)

    return prep, {
        "train": train,
        "val": val,
        "test": test,
        "negatives": negatives,
        "movies": data["movies"],
    }
