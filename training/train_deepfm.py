"""
DeepFM ranker training pipeline.

Treats recommendation as a binary classification task:
  label = 1 if rating >= 3.5 (positive engagement)
  label = 0 otherwise

Loss = IPS-weighted Binary Cross-Entropy

Usage:
  python -m training.train_deepfm --data_dir ../  --output_dir artifacts/
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from data.preprocessor import Preprocessor
from models.deepfm import DeepFM, build_dense_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

def _year_to_bucket(year: int, n_buckets: int = 50) -> int:
    """Map release year to a bucket index (1920–2020 → 0–49)."""
    return max(0, min(n_buckets - 1, (year - 1920) // 2))


class DeepFMDataset(Dataset):
    def __init__(
        self,
        ratings,          # encoded DataFrame: user_idx, movie_idx, rating
        user_features: np.ndarray,
        item_features: np.ndarray,
        movie_years: np.ndarray,   # (num_movies,) int16
        ips_weights: np.ndarray | None = None,
    ):
        self.user_idx = torch.from_numpy(ratings["user_idx"].values.astype(np.int64))
        self.movie_idx = torch.from_numpy(ratings["movie_idx"].values.astype(np.int64))
        # Binary label: engaged = rating >= 3.5
        labels = (ratings["rating"].values >= 3.5).astype(np.float32)
        self.labels = torch.from_numpy(labels)

        year_arr = np.array(
            [_year_to_bucket(int(movie_years[m])) for m in ratings["movie_idx"].values],
            dtype=np.int64,
        )
        self.year_bucket = torch.from_numpy(year_arr)

        self.user_features = torch.from_numpy(user_features).float()
        self.item_features = torch.from_numpy(item_features).float()

        if ips_weights is not None:
            self.ips = torch.from_numpy(ips_weights).float()
        else:
            self.ips = None

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int):
        uid = self.user_idx[idx]
        mid = self.movie_idx[idx]
        item = {
            "user_idx": uid,
            "movie_idx": mid,
            "year_bucket": self.year_bucket[idx],
            "user_features": self.user_features[uid],
            "item_features": self.item_features[mid],
            "label": self.labels[idx],
        }
        if self.ips is not None:
            item["ips_weight"] = self.ips[idx]
        return item


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class DeepFMTrainer:
    def __init__(self, model: DeepFM, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        clip_grad: float = 1.0,
    ) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()

            dense = build_dense_features(batch["user_features"], batch["item_features"])
            logits = self.model(
                batch["user_idx"],
                batch["movie_idx"],
                batch["year_bucket"],
                dense,
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["label"], reduction="none"
            )
            if "ips_weight" in batch:
                loss = loss * batch["ips_weight"]
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item() * len(logits)
            n += len(logits)

        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_probs, all_labels = [], []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            dense = build_dense_features(batch["user_features"], batch["item_features"])
            probs = self.model.predict_proba(
                batch["user_idx"], batch["movie_idx"], batch["year_bucket"], dense
            )
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Binary cross-entropy
        eps = 1e-7
        bce = -np.mean(
            all_labels * np.log(all_probs + eps)
            + (1 - all_labels) * np.log(1 - all_probs + eps)
        )
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

        return {"bce": float(bce), "auc": float(auc)}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def train(config: dict) -> DeepFM:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    prep = Preprocessor.load(output_dir)

    with open(output_dir / "splits.pkl", "rb") as f:
        splits = pickle.load(f)
    train_df = splits["train"]
    val_df = splits["val"]

    # IPS weights for training
    ips_arr = None
    if config.get("use_ips", True):
        from data.ips_weights import compute_ips_weights
        ips_arr = compute_ips_weights(
            train_df, num_movies=prep.num_movies, alpha=0.5, cap=10.0
        )

    # Extract year per movie
    movie_years = np.zeros(prep.num_movies, dtype=np.int16)
    if prep.movie_df is not None:
        for _, row in prep.movie_df.iterrows():
            idx = prep.movie_id_map.get(row["movieId"])
            if idx is not None:
                movie_years[idx] = int(row.get("year", 2000))

    train_dataset = DeepFMDataset(
        train_df, prep.user_features, prep.item_features, movie_years, ips_arr
    )
    val_dataset = DeepFMDataset(
        val_df, prep.user_features, prep.item_features, movie_years
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4096),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4096),
        shuffle=False,
        num_workers=0,
    )

    dense_dim = prep.user_features.shape[1] + prep.item_features.shape[1]

    model = DeepFM(
        num_users=prep.num_users,
        num_movies=prep.num_movies,
        dense_dim=dense_dim,
        embed_k=config.get("embed_k", 16),
        mlp_dims=config.get("mlp_dims", [400, 400, 400]),
        dropout=config.get("dropout", 0.2),
    )
    logger.info(f"DeepFM parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = DeepFMTrainer(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_auc = 0.0
    for epoch in range(1, config.get("epochs", 10) + 1):
        loss = trainer.train_epoch(train_loader, optimizer)
        val_metrics = trainer.validate(val_loader)
        scheduler.step(val_metrics["bce"])

        logger.info(
            f"Epoch {epoch:02d} | loss={loss:.4f} | "
            f"val_bce={val_metrics['bce']:.4f} | val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "num_users": prep.num_users,
                    "num_movies": prep.num_movies,
                    "dense_dim": dense_dim,
                    "embed_k": config.get("embed_k", 16),
                    "mlp_dims": config.get("mlp_dims", [400, 400, 400]),
                },
            }, output_dir / "deepfm_best.pt")
            logger.info(f"  ↳ New best AUC: {best_auc:.4f} — checkpoint saved")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                        help="Path to raw CSVs (not required — DeepFM loads from --output_dir artifacts)")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--embed_k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_ips", action="store_true", default=True)
    args = parser.parse_args()
    train(vars(args))
