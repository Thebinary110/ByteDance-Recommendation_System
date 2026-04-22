#!/usr/bin/env bash
# Full training pipeline: preprocess → Two-Tower → DeepFM → evaluate
# Run from inside the recommender/ directory with the venv activated.
#
# Usage:
#   ./run_training.sh                    # 10% sample, fast dev run
#   SAMPLE=1.0 ./run_training.sh         # full 20M dataset
#   EPOCHS=20 ./run_training.sh          # more epochs

set -euo pipefail

DATA_DIR="${DATA_DIR:-../}"          # directory containing the CSV files
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"
SAMPLE="${SAMPLE:-0.1}"              # fraction of ratings to use (0.1 = 10% ≈ 2M rows)
EPOCHS="${EPOCHS:-10}"
EMBED_DIM="${EMBED_DIM:-64}"

echo "================================================"
echo "  CineMatch Training Pipeline"
echo "================================================"
echo "Data    : $DATA_DIR"
echo "Artifacts: $ARTIFACT_DIR"
echo "Sample  : $SAMPLE"
echo "Epochs  : $EPOCHS"
echo ""

# --- Step 1: Two-Tower ---
echo "[1/3] Training Two-Tower model …"
python -m training.train_two_tower \
  --data_dir "$DATA_DIR" \
  --output_dir "$ARTIFACT_DIR" \
  --sample_frac "$SAMPLE" \
  --embed_dim "$EMBED_DIM" \
  --epochs "$EPOCHS" \
  --batch_size 2048 \
  --lr 0.001 \
  --use_ips

echo ""
echo "[2/3] Training DeepFM ranker …"
python -m training.train_deepfm \
  --data_dir "$DATA_DIR" \
  --output_dir "$ARTIFACT_DIR" \
  --embed_k 16 \
  --epochs "$EPOCHS" \
  --batch_size 4096 \
  --lr 0.001 \
  --use_ips

echo ""
echo "[3/3] Running offline evaluation …"
python -m training.evaluate \
  --output_dir "$ARTIFACT_DIR" \
  --top_k 10 \
  --n_users 500

echo ""
echo "================================================"
echo "  Training complete! Artifacts saved to $ARTIFACT_DIR/"
echo "  Start the API with:  uvicorn api.main:app --reload --port 8000"
echo "================================================"
