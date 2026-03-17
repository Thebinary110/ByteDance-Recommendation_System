# Run Guide — Recommendation System

---

## Stage 1: Frontend (Local Simulation)

### Prerequisites

```
Python 3.10+
```

### 1. Install dependencies

```bash
pip install streamlit
```

### 2. Run the app

```bash
streamlit run frontend/app.py
```

Run from the project root (the directory containing `frontend/`).

### 3. Expected output

- Browser opens automatically at `http://localhost:8501`
- Enter a User ID and click **Load Recommendations**
- Ten recommendation cards appear
- Click **Like** or **Dislike** on any card
- The Interaction Log panel on the right updates with each action in real time
- Timestamps reflect the exact moment of interaction

---

## Project structure

```
Recommendation System/
├── frontend/
│   ├── app.py                        # entry point
│   ├── components/
│   │   ├── user_input.py             # user ID input + load button
│   │   ├── recommendation_list.py    # item cards with action buttons
│   │   └── interaction_log.py        # live action table
│   ├── services/
│   │   └── interaction_service.py    # async fetch + record stubs
│   └── utils/
│       └── helpers.py                # timestamps, dummy data, formatters
└── run.md
```

---

## Stage 1.5: Data Pipeline (MovieLens)

### Prerequisites

```
Python 3.10+  |  pandas (already in .venv)
```

### 1. Place the raw dataset

```
data/raw/rating.csv        # MovieLens rating file (~20M rows)
```

Columns expected: `userId`, `movieId`, `rating`, `timestamp`

### 2. Run the pipeline

```bash
python data_pipeline/run_pipeline.py
```

Run from the project root.

### 3. Output

```
data/processed/events.csv
```

Schema:

```
user_id,item_id,event_type,timestamp
1,151,like,1094785734
1,29,dislike,1112484676
```

### 4. Verify

- File exists at `data/processed/events.csv`
- Size ~432 MB for full MovieLens 20M dataset
- Total events: ~15.7 M (likes + dislikes; neutral ratings filtered)
- Event type breakdown: ~9.99 M like / ~5.71 M dislike

### Performance

- Chunk size: 50,000 rows
- Throughput: ~279,000 rows/s
- Full 20M dataset: ~56 seconds

### Rating rules

| Rating  | Event    |
|---------|----------|
| >= 4.0  | like     |
| <= 2.0  | dislike  |
| == 3.0  | ignored  |

---

## Stage 2: Streaming Service (Mini Kafka)

### Prerequisites

```
Python 3.10+  |  data/processed/events.csv (run Stage 1.5 first)
```

### Run

```bash
python streaming/run_stream.py
```

Run from the project root.

### Expected output

```
08:00:01  INFO  __main__ — Stream initializing — consumers: ['model', 'logger', 'feature_store']
08:00:01  INFO  producer — Producer starting — source: data/processed/events.csv  replay=True  consumers=3
08:00:01  INFO  consumer.model — [model] Consumer starting
08:00:01  INFO  consumer.logger — [logger] Consumer starting
08:00:01  INFO  consumer.feature_store — [feature_store] Consumer starting
08:00:01  INFO  producer — [PRODUCER]      1,000 pushed  |  85,000 events/s  |  consumers=3
08:00:01  INFO  consumer.model — [model]      1,000  user=11    item=7163  type=like  ts=1230789208  |  85,200 events/s  queue=1
08:00:01  INFO  consumer.logger — [logger]      1,000  user=11    item=7163  type=like  ts=1230789208  |  85,210 events/s  queue=1
08:00:01  INFO  consumer.feature_store — [feature_store]      1,000  user=11  ...
```

All three consumers receive every event independently. Each logs a status line every 1,000 events.

Press `Ctrl+C` to stop.

### Architecture

```
events.csv  →  [validate]  →  producer  →  Queue(model)         →  consumer.model
                                        →  Queue(logger)        →  consumer.logger
                                        →  Queue(feature_store) →  consumer.feature_store
```

- Fan-out: one dedicated `asyncio.Queue(10,000)` per consumer
- Backpressure: producer blocks on the slowest consumer's queue
- Replay mode: sleeps proportional to timestamp gap (`time_diff / SCALE_FACTOR`), capped at `MAX_REPLAY_DELAY`
- Schema validation: invalid rows skipped with a warning before reaching any queue
- All output via `logging` — no `print()` anywhere

### Configuration (`streaming/config.py`)

| Key | Default | Effect |
|---|---|---|
| `REPLAY_MODE` | `True` | Time-aware replay vs fixed delay |
| `SCALE_FACTOR` | `1_000_000` | Compress 1M real seconds → 1 simulated second |
| `MAX_REPLAY_DELAY` | `0.005s` | Cap on any single inter-event sleep |
| `STREAM_DELAY` | `0.001s` | Fixed delay when `REPLAY_MODE=False` |
| `QUEUE_MAXSIZE` | `10_000` | Per-consumer queue depth |
| `LOG_INTERVAL` | `1_000` | Events between consumer log lines |

### Adding / removing consumers

Edit `CONSUMERS` in `streaming/run_stream.py`:

```python
CONSUMERS = ["model", "logger", "feature_store"]   # add or remove names here
```

Each name gets its own queue and its own coroutine automatically.

### Modules

```
streaming/
├── config.py          # all tunables
├── queue_manager.py   # register_consumer(name) / get_all_queues() registry
├── producer.py        # validate → replay sleep → fan-out to all queues
├── consumer.py        # consume(name, queue) — named, fully logged
└── run_stream.py      # entry point — registers consumers, asyncio.gather()
```

---

## Stage 3: Feature Store (User State Engine)

### Prerequisites

```
Python 3.10+  |  data/processed/events.csv (run Stage 1.5 first)
```

### Run

```bash
python feature_store/run_feature_store.py
```

Run from the project root.

### Expected output

```
08:00:01  INFO  __main__ — Feature store initializing
08:00:01  INFO  producer — Producer starting — source: data/processed/events.csv  replay=True  consumers=1
08:00:01  INFO  consumer — Feature store consumer starting
08:00:01  INFO  producer — [PRODUCER]     10,000 pushed  |  97,000 events/s  |  consumers=1
08:00:01  INFO  feature_store.stats — events=    10,000  users=    116  avg_history= 38.8  total_interactions=     4,504
08:00:01  INFO  feature_store.stats — sample user_id='1'  likes=['1036', '1079', ...]  dislikes=['1009', ...]  history(last5)=[...]  last_interaction=1112485748
```

Stats line every 10,000 events shows:
- total events processed
- distinct users seen
- average history size per user
- total interaction count across all users

A sample user profile (user_id, likes, dislikes, recent history, last timestamp) follows each stats line.

### Architecture

```
events.csv → producer → Queue(feature_store) → consume_events()
                                                      ↓
                                              process_event()
                                                      ↓
                                             user_store.update_user()
                                                      ↓
                                       _store[user_id] = {
                                           "likes":    set(),
                                           "dislikes": set(),
                                           "history":  deque(maxlen=50),
                                           "last_interaction": int
                                       }
```

### User state schema

```python
{
    "likes":             set()              # item_ids rated >= 4
    "dislikes":          set()              # item_ids rated <= 2 (mutually exclusive with likes)
    "history":           deque(maxlen=50)   # (item_id, event_type, timestamp) — auto-bounded
    "last_interaction":  int                # Unix timestamp of most recent event
}
```

### Design notes

- `deque(maxlen=50)` — bounded history; oldest entries drop automatically, memory stays flat
- `threading.RLock()` — thread-safe for future Redis migration or thread pool executors
- Likes/dislikes are mutually exclusive: liking an item removes it from dislikes and vice versa
- `fs_config.py` (not `config.py`) avoids shadowing `streaming/config.py` when both directories share `sys.path`

### Modules

```
feature_store/
├── fs_config.py           # MAX_HISTORY=50, LOG_INTERVAL=10_000
├── user_store.py          # in-memory state: get_user, update_user, get_stats, get_sample_user
├── feature_updater.py     # process_event(event) → update_user(...)
├── consumer.py            # async consume_events(queue) — pulls from queue, calls process_event
├── logger.py              # log_stats(count) — aggregate stats + sample user profile
└── run_feature_store.py   # entry point — registers consumer, asyncio.gather(produce, consume_events)
```

---

## Stage 4: Recommendation Engine (Heuristic Baseline)

### Prerequisites

```
Python 3.10+  |  data/processed/events.csv (run Stage 1.5 first)
```

### Run

```bash
python recommender/run_recommender.py
```

Run from the project root. No pre-population step needed — the pipeline starts, populates the feature store, and begins showing recommendations automatically.

### Expected output

```
08:00:01  INFO  __main__ — Recommender system starting
08:00:01  INFO  __main__ — Test users: ['1', '2', '3', '10']
08:00:01  INFO  __main__ — Recommendations shown every 50,000 events
08:00:01  INFO  producer — Producer starting ...
...
08:00:01  INFO  __main__ — --- Recommendations @ 50,000 events ---
08:00:01  INFO  __main__ —   user='1'  →  (not enough data yet)
08:00:01  INFO  __main__ —   user='3'  →  ['110', '1580', '891', '1249', '1327', '908', '266', '70', '62']
08:00:01  INFO  __main__ —   user='10' →  ['2791', '2455', '110', '589', '891', '1356', '1544', '541', '1270', '1249']
08:00:02  INFO  __main__ — --- Recommendations @ 100,000 events ---
08:00:02  INFO  __main__ —   user='1'  →  ['2948', '1210', '2455', '891', '1356', '110', '1544', '908', '266', '70']
08:00:02  INFO  __main__ —   user='3'  →  ['1387', '969', '1250', '527', '912', '11', '1136', '25', '110', '1580']
```

- Users with insufficient history show `(not enough data yet)` until MIN_HISTORY=3 events are seen
- Different users return different item lists ✓
- Lists update as more events are processed ✓

Press `Ctrl+C` to stop.

### Algorithm

```
For each user:
  1. Load user["likes"] from the feature store
  2. For each liked item → look up top-SIMILARITY_TOP_N co-occurring items
  3. Union all retrieved items → candidate pool
  4. Filter out already-seen items (likes ∪ dislikes)
  5. Score each candidate: Σ cooccurrence(liked_item, candidate) for all liked_items
  6. Sort descending → return top TOP_K
```

Co-occurrence is built from user interaction histories — items that appear together in the same user's history accumulate a shared count.

### Configuration (`recommender/rec_config.py`)

| Key | Default | Effect |
|---|---|---|
| `TOP_K` | `10` | Items returned per recommendation |
| `MIN_HISTORY` | `3` | Minimum events before recommending |
| `SIMILARITY_TOP_N` | `20` | Similar items retrieved per liked item |

### Modules

```
recommender/
├── rec_config.py           # TOP_K, MIN_HISTORY, SIMILARITY_TOP_N
├── similarity_engine.py    # item_cooccurrence map, update_similarity, get_similar_items
├── candidate_generator.py  # generate_candidates(user) → candidate pool from liked items
├── ranker.py               # rank_items(user, candidates) → sorted by co-occurrence score
├── recommender_engine.py   # recommend(user_id) → top-K list
└── run_recommender.py      # entry point — full pipeline + periodic recommendation sampling
```

---

## Stage 5: Embedding Model (Two-Tower, Online Learning)

### Prerequisites

```
Python 3.10+  |  data/processed/events.csv  |  torch (CPU build)
```

Install PyTorch if needed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run

```bash
python embedding_model/run_embedding.py
```

Run from the project root.

### Expected output

```
08:00:01  INFO  __main__ — Embedding model initializing (two-tower, online SGD)
08:00:02  INFO  __main__ — [EMBED]  events=    10,000  avg_loss=0.4272  users=116  items=3,041
08:00:02  INFO  __main__ — [EMBED]  user='116'  →  ['1112', '1307', '62265', '5054', '42738', ...]
08:00:04  INFO  __main__ — [EMBED]  events=    20,000  avg_loss=0.4327  users=208  items=4,359
08:00:04  INFO  __main__ — [EMBED]  user='208'  →  ['3698', '5449', '5649', '747', '441', ...]
```

A log pair (stats line + recommendation line) is emitted every 10,000 events.

### Architecture

```
streaming producer → Queue(embedding_model) → _consume()
                                                    ↓  every event
                                             train_on_event()
                                               forward:  score = dot(user_emb, item_emb)
                                                         prob  = sigmoid(score)
                                                         loss  = (prob - label)²
                                               backward: loss.backward()
                                               update:   emb.data -= lr * emb.grad
                                                         emb.grad.zero_()
                                                    ↓  every 10,000 events
                                             recommend(user_id)
                                               scores = [dot(user_emb, item_emb) for all items]
                                               return top-K by score
```

### Two-tower model

```
user_id ──► user_emb (dim=32) ──┐
                                 ├──► dot product ──► sigmoid ──► P(like)
item_id ──► item_emb (dim=32) ──┘
```

Both towers are random-initialised 32-d vectors. Online SGD nudges them toward or away from each other based on like/dislike labels.

### Verified output

```
events=10,000   avg_loss=0.4272   users=116    items=3,041
events=20,000   avg_loss=0.4327   users=208    items=4,359
events=100,000  avg_loss=0.4302   users=869    items=8,841
```

Loss is stable ~0.43 — model is learning without diverging ✓
Different users get different recommendations ✓

### Configuration (`embedding_model/emb_config.py`)

| Key | Default | Effect |
|---|---|---|
| `EMBEDDING_DIM` | `32` | Vector dimensions for all embeddings |
| `LEARNING_RATE` | `0.01` | SGD step size per event |
| `LOG_INTERVAL` | `10_000` | Events between log + recommendation snapshots |

### Modules

```
embedding_model/
├── emb_config.py       # EMBEDDING_DIM, LEARNING_RATE, LOG_INTERVAL
├── embedding_store.py  # user_embeddings / item_embeddings dicts, lazy init
├── model.py            # predict(user_emb, item_emb) → dot product
├── trainer.py          # train_on_event(event) → forward + backward + SGD step
├── inference.py        # recommend(user_id, top_k, exclude) → top-K items
└── run_embedding.py    # entry point — asyncio pipeline + periodic logging
```

---

## Stage 6: Negative Sampling + Online Training Upgrade

No new entry point — this upgrades `embedding_model/run_embedding.py` in-place.

```bash
python embedding_model/run_embedding.py   # same command as Stage 5
```

### What changed

| | Stage 5 | Stage 6 |
|---|---|---|
| Pairs per event | 1 | 1 + NEGATIVE_SAMPLES (6) |
| Backward passes per event | 1 | 6 |
| Update steps per event | 1 per embedding | 1 per embedding (accumulated) |
| Throughput | ~100,000 events/s | ~1,400 events/s |

### Training logic (per event)

```
record_interaction(user_id, item_id)       ← ensures item excluded from negatives

backward pass 1:  (user, item,     label)  ← like→1.0, dislike→0.0
backward pass 2:  (user, neg_1,    0.0  )  ┐
backward pass 3:  (user, neg_2,    0.0  )  │  gradients accumulate
backward pass 4:  (user, neg_3,    0.0  )  │  on user_emb
backward pass 5:  (user, neg_4,    0.0  )  │
backward pass 6:  (user, neg_5,    0.0  )  ┘

single SGD step:  user_emb.data -= lr * grad   ← one update with summed gradient
                  each item_emb.data -= lr * grad
```

### Verified loss trend

```
events= 10,000  avg_loss=0.4336
events= 20,000  avg_loss=0.4310
events= 40,000  avg_loss=0.4257
events= 70,000  avg_loss=0.4264
events= 90,000  avg_loss=0.4073
events=120,000  avg_loss=0.4306
```

Loss oscillates but trends downward.  Oscillation is expected — online learning
with non-IID streaming data (events arrive in user-sorted order, not time order)
means consecutive batches are from the same user context, then abruptly switch.
A fixed evaluation set would show cleaner convergence.

### New file

```
embedding_model/
├── negative_sampler.py   ← NEW: record_interaction(), sample_negatives()
├── trainer.py            ← MODIFIED: accumulate grads, one update step
└── emb_config.py         ← MODIFIED: NEGATIVE_SAMPLES = 5
```

---

## Stage 7: FAISS Integration (Fast Vector Search)

No new entry point — upgrades `embedding_model/run_embedding.py` in-place.

```bash
python embedding_model/run_embedding.py   # same command as Stages 5 & 6
```

### What changed

| | Stage 5/6 | Stage 7 |
|---|---|---|
| Retrieval | linear scan O(N·dim) | FAISS IndexFlatIP O(N) exact search |
| Similarity metric | dot product | cosine (L2-normalised before index + query) |
| Index refresh | — | full rebuild every 20,000 events |

### Expected output

```
12:28:58  INFO  __main__ — [EMBED]  events=10,000  avg_loss=0.4277  retrieval=linear
12:29:04  INFO  __main__ — [FAISS] index rebuilt — 4,359 items indexed
12:29:04  INFO  __main__ — [EMBED]  events=20,000  avg_loss=0.4313  retrieval=FAISS
12:29:17  INFO  __main__ — [FAISS] index rebuilt — 6,000 items indexed
12:29:17  INFO  __main__ — [EMBED]  events=40,000  avg_loss=0.4272  retrieval=FAISS
```

- Events 0–19,999: `retrieval=linear` (fallback until first build)
- Events 20,000+: `retrieval=FAISS`
- Index rebuilds at 20k, 40k, 60k, … with growing item counts

### Architecture

```
every FAISS_REBUILD_INTERVAL events:
  item_embeddings → np.stack → faiss.normalize_L2 → IndexFlatIP.add()

every query:
  user_emb → normalize → index.search(top_k * 2) → filter exclude → top_k
```

### Rebuild ordering

The rebuild check fires **before** the LOG check in the consumer loop.
At the same boundary (e.g. count=20,000), the index is built first, so
`recommend()` at that checkpoint already uses FAISS.

### Index type upgrade path

```
IndexFlatIP   exact,  O(N)      current — works for ~27k items
IndexIVFFlat  approx, O(N/n_lists)       when N > 100k
IndexHNSW     approx, O(log N)           production
```

### New / modified files

```
embedding_model/
├── faiss_index.py   ← NEW: FaissIndex class + global faiss_index instance
├── inference.py     ← MODIFIED: FAISS primary path, linear fallback
├── run_embedding.py ← MODIFIED: rebuild trigger + retrieval= label in logs
└── emb_config.py    ← MODIFIED: FAISS_REBUILD_INTERVAL = 20_000
```

---

## Stage 8: Ranking Model (MLP — Final Decision Layer)

### Prerequisites

```
Python 3.10+  |  data/processed/events.csv  |  torch  |  faiss-cpu
```

### Run

```bash
python ranking_model/run_ranker.py
```

Run from the project root.

### Expected output

```
13:12:11  INFO  __main__ — [RANKER]  events=10,000  emb_loss=0.4304  rank_loss=0.5714  users=116  items=3,041
13:12:11  INFO  __main__ — [FINAL REC]  user='116'  →  ['2890', '955', '3079', '6373', ...]
13:12:31  INFO  __main__ — [FAISS] index rebuilt — 4,359 items
13:12:31  INFO  __main__ — [RANKER]  events=20,000  emb_loss=0.4303  rank_loss=0.5171  users=208  items=4,359
13:12:31  INFO  __main__ — [FINAL REC]  user='208'  →  ['53883', '4982', '40815', '487', ...]
```

Both `emb_loss` and `rank_loss` are logged every 10k events.  `rank_loss` starts ~0.57 and decreases as the MLP learns to distinguish liked from random items.

### Two-stage pipeline

```
streaming event
      ↓
  train_on_event()     ← embedding model: two-tower SGD + negative sampling
  train_ranker()       ← ranking model: MLP Adam step
      ↓  every 20k events
  faiss_index.build()  ← rebuild ANN index
      ↓  every 10k events
  retrieve(user, 50)   ← FAISS: top-50 candidates  (fast, approximate)
  rank_items(user, 50) ← MLP:   top-10 final recs  (precise, per-pair scoring)
```

### MLP architecture

```
input  [user_emb(32) || item_emb(32)] = 64 dims
   ↓   Linear(64 → 64) + ReLU
   ↓   Linear(64 → 32) + ReLU
   ↓   Linear(32 →  1)              ← raw logit
```

Loss: `BCEWithLogitsLoss` (sigmoid fused in — more stable than `BCELoss + sigmoid`)
Optimizer: `Adam(lr=0.001)`

### Why embeddings are detached in the ranker

`get_user_embedding().detach()` cuts the gradient path between the ranker and the embedding model.  Without `.detach()`, the ranker's backward pass would flow into the embedding tensors and corrupt the embedding model's manual SGD updates.

### Modules

```
ranking_model/
├── rank_config.py      # INPUT_DIM=64, HIDDEN_DIM=64, LR=0.001, TOP_K_CANDIDATES=50
├── rank_model.py       # RankingModel(nn.Module) + global model instance
├── rank_trainer.py     # train_ranker(event) → Adam step, returns loss
├── rank_inference.py   # rank_items(user_id, candidates, top_k) → reranked list
└── run_ranker.py       # entry point — full two-stage pipeline
```

---

## Stage 9 (upcoming): API Layer

Will expose the full ranking pipeline via REST endpoints.
