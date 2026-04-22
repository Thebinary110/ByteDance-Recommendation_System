#!/usr/bin/env bash
# Start all services for local development.
# Run from inside the recommender/ directory.
#
# Services started:
#   Redis (optional — SQLite fallback used if unavailable)
#   Kafka (optional — SQLite fallback used if unavailable)
#   FastAPI backend on :8000
#   Flink consumer (background thread)
#   React frontend dev server on :5173

set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"

echo "================================================"
echo "  CineMatch — Starting services"
echo "================================================"

# --- Redis (optional) ---
if command -v redis-server &>/dev/null; then
  echo "[Redis] Starting …"
  redis-server --daemonize yes --port 6379 --loglevel warning
  echo "[Redis] Running on :6379"
else
  echo "[Redis] Not found — feature store will use SQLite fallback."
fi

# --- Kafka (optional) ---
if command -v kafka-server-start.sh &>/dev/null; then
  echo "[Kafka] Start manually if needed:"
  echo "  kafka-server-start.sh config/server.properties"
else
  echo "[Kafka] Not found — event logger will use SQLite fallback."
fi

echo ""

# --- API check ---
if [ ! -f "$ARTIFACT_DIR/two_tower.pt" ]; then
  echo "[WARN] No trained model found at $ARTIFACT_DIR/two_tower.pt"
  echo "       Run ./run_training.sh first."
  echo ""
fi

# --- Start API ---
echo "[API] Starting FastAPI on http://localhost:8000 …"
echo "[API] Docs available at http://localhost:8000/docs"
echo ""

# Start streaming consumer in background
python -m streaming.flink_consumer \
  --artifact_dir "$ARTIFACT_DIR" \
  --interval 5 &
CONSUMER_PID=$!
echo "[Streaming] Flink consumer started (PID $CONSUMER_PID)"

# Start API (foreground)
ARTIFACT_DIR="$ARTIFACT_DIR" uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info &
API_PID=$!
echo "[API] FastAPI started (PID $API_PID)"

# Start frontend
echo "[Frontend] Starting React dev server on http://localhost:5173 …"
cd frontend && npm run dev &
FRONTEND_PID=$!
echo "[Frontend] Started (PID $FRONTEND_PID)"

echo ""
echo "================================================"
echo "  All services running."
echo "  API    → http://localhost:8000/docs"
echo "  App    → http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "================================================"

# Clean up on exit
cleanup() {
  echo ""
  echo "Stopping services …"
  kill $API_PID $CONSUMER_PID $FRONTEND_PID 2>/dev/null || true
  if command -v redis-cli &>/dev/null; then redis-cli shutdown 2>/dev/null || true; fi
}
trap cleanup INT TERM

wait
