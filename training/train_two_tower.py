"""
Two-Tower model training pipeline.

Usage:
  python -m training.train_two_tower --data_dir ../  --output_dir artifacts/
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.preprocessor import Preprocessor, preprocess_all
from data.ips_weights import attach_ips_weights
from models.two_tower import TwoTowerModel, BPRLoss
from serving.faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class TripletDataset(Dataset):
    """
    Dataset of (user, positive, negative) triplets.
    Loads user and item feature matrices from the preprocessor.
    """

    def __init__(
        self,
        triplets,          # DataFrame with user_idx, pos_movie_idx, neg_movie_idx
        user_features: np.ndarray,   # [num_users, uf]
        item_features: np.ndarray,   # [num_movies, if]
        ips_weights: np.ndarray | None = None,  # [len(triplets)]
    ):
        self.user_idx = torch.from_numpy(triplets["user_idx"].values.astype(np.int64))
        self.pos_idx = torch.from_numpy(triplets["pos_movie_idx"].values.astype(np.int64))
        self.neg_idx = torch.from_numpy(triplets["neg_movie_idx"].values.astype(np.int64))
        self.user_features = torch.from_numpy(user_features).float()
        self.item_features = torch.from_numpy(item_features).float()
        if ips_weights is not None:
            self.ips_weights = torch.from_numpy(ips_weights).float()
        else:
            self.ips_weights = None

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int):
        uid = self.user_idx[idx]
        pid = self.pos_idx[idx]
        nid = self.neg_idx[idx]
        item = {
            "user_idx": uid,
            "pos_movie_idx": pid,
            "neg_movie_idx": nid,
            "user_features": self.user_features[uid],
            "pos_item_features": self.item_features[pid],
            "neg_item_features": self.item_features[nid],
        }
        if self.ips_weights is not None:
            item["ips_weight"] = self.ips_weights[idx]
        return item


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class TwoTowerTrainer:
    def __init__(self, model: TwoTowerModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.criterion = BPRLoss()

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

            user_vec, pos_vec, neg_vec = self.model(
                batch["user_idx"],
                batch["user_features"],
                batch["pos_movie_idx"],
                batch["pos_item_features"],
                batch["neg_movie_idx"],
                batch["neg_item_features"],
            )
            ips = batch.get("ips_weight")
            loss = self.criterion(user_vec, pos_vec, neg_vec, ips_weights=ips)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item() * len(batch["user_idx"])
            n += len(batch["user_idx"])

        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
        item_features: torch.Tensor,
        prep: Preprocessor,
        k: int = 10,
    ) -> dict[str, float]:
        """Quick NDCG@k proxy on the validation set."""
        self.model.eval()
        all_movie_idxs = torch.arange(prep.num_movies, device=self.device)
        all_item_feats = item_features.to(self.device)

        # Encode all items once
        item_vecs = self.model.encode_items(all_movie_idxs, all_item_feats)  # [M, D]

        hits, ndcg_vals = [], []
        seen_users: set = set()
        for batch in loader:
            batch = {kk: v.to(self.device) for kk, v in batch.items()}
            uids = batch["user_idx"]
            pos_ids = batch["pos_movie_idx"]

            user_vecs = self.model.encode_user(uids, batch["user_features"])  # [B, D]
            scores = user_vecs @ item_vecs.T  # [B, M]

            for i, (uid, pos) in enumerate(zip(uids.tolist(), pos_ids.tolist())):
                if uid in seen_users:
                    continue
                seen_users.add(uid)
                top_k = scores[i].topk(k).indices.tolist()
                hit = int(pos in top_k)
                hits.append(hit)
                if hit:
                    rank = top_k.index(pos) + 1
                    ndcg_vals.append(1.0 / np.log2(rank + 1))
                else:
                    ndcg_vals.append(0.0)

        return {
            "hit_rate": float(np.mean(hits)) if hits else 0.0,
            "ndcg": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        }


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(config: dict) -> TwoTowerModel:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load or build preprocessed data
    prep_path = output_dir / "preprocessor.pkl"
    splits_path = output_dir / "splits.pkl"

    if prep_path.exists() and splits_path.exists() and not config.get("force_reprocess"):
        logger.info("Loading cached preprocessor and splits …")
        prep = Preprocessor.load(output_dir)
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
    else:
        logger.info("Preprocessing data …")
        prep, splits = preprocess_all(
            config["data_dir"],
            output_dir,
            sample_frac=config.get("sample_frac"),
        )
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)

    train_df = splits["train"]
    val_df = splits["val"]
    negatives = splits["negatives"]

    # IPS weights for training triplets
    ips_arr = None
    if config.get("use_ips", True):
        from data.ips_weights import compute_ips_weights
        # Align IPS weights to the training set (pos_movie_idx)
        pos_movie_arr = negatives["pos_movie_idx"].values
        pop_counts = train_df.groupby("movie_idx")["rating"].count().to_dict()
        num_movies = prep.num_movies
        counts = np.array(
            [pop_counts.get(m, 1) for m in range(num_movies)], dtype=np.float32
        )
        prop = counts ** 0.5
        prop /= prop.max()
        w = 1.0 / prop[pos_movie_arr]
        w = np.clip(w, 1.0, 10.0)
        ips_arr = (w / w.mean()).astype(np.float32)

    user_features = torch.from_numpy(prep.user_features).float()
    item_features = torch.from_numpy(prep.item_features).float()

    train_dataset = TripletDataset(
        negatives, prep.user_features, prep.item_features, ips_weights=ips_arr
    )
    val_dataset = TripletDataset(val_df.rename(columns={
        "movie_idx": "pos_movie_idx"
    }).assign(neg_movie_idx=0),
        prep.user_features, prep.item_features
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 2048),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 2048),
        shuffle=False,
        num_workers=0,
    )

    user_feat_dim = prep.user_features.shape[1]
    item_feat_dim = prep.item_features.shape[1]

    model = TwoTowerModel(
        num_users=prep.num_users,
        num_movies=prep.num_movies,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        embed_dim=config.get("embed_dim", 64),
        hidden_dims=config.get("hidden_dims", [256, 128]),
        dropout=config.get("dropout", 0.2),
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = TwoTowerTrainer(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 10)
    )

    best_ndcg = 0.0
    for epoch in range(1, config.get("epochs", 10) + 1):
        train_loss = trainer.train_epoch(train_loader, optimizer)
        scheduler.step()

        if epoch % config.get("eval_every", 2) == 0:
            val_metrics = trainer.validate(val_loader, item_features, prep, k=10)
            logger.info(
                f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
                f"hit@10={val_metrics['hit_rate']:.4f} | ndcg@10={val_metrics['ndcg']:.4f}"
            )
            if val_metrics["ndcg"] > best_ndcg:
                best_ndcg = val_metrics["ndcg"]
                torch.save(model.state_dict(), output_dir / "two_tower_best.pt")
                logger.info(f"  ↳ New best NDCG@10: {best_ndcg:.4f} — checkpoint saved")
        else:
            logger.info(f"Epoch {epoch:02d} | loss={train_loss:.4f}")

    # Load best checkpoint and build FAISS index
    model.load_state_dict(torch.load(output_dir / "two_tower_best.pt", map_location=device))
    model.eval()

    logger.info("Building FAISS index from item embeddings …")
    all_item_idxs = torch.arange(prep.num_movies, device=device)
    all_item_feats = item_features.to(device)
    with torch.no_grad():
        item_embeddings = model.encode_items(all_item_idxs, all_item_feats).cpu().numpy()

    np.save(output_dir / "item_embeddings.npy", item_embeddings)

    faiss_idx = FAISSIndex()
    faiss_idx.build(item_embeddings)
    faiss_idx.save(output_dir / "faiss.index")
    logger.info("FAISS index saved.")

    # Save model config for loading at serving time
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "num_users": prep.num_users,
            "num_movies": prep.num_movies,
            "user_feat_dim": user_feat_dim,
            "item_feat_dim": item_feat_dim,
            "embed_dim": config.get("embed_dim", 64),
            "hidden_dims": config.get("hidden_dims", [256, 128]),
        }
    }, output_dir / "two_tower.pt")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_ips", action="store_true", default=True)
    args = parser.parse_args()

    train(vars(args))
