# CineMatch — Production Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Redis](https://img.shields.io/badge/Upstash_Redis-live-DC382D?style=flat-square&logo=redis)
![HuggingFace](https://img.shields.io/badge/HuggingFace_Spaces-deployed-FFD21E?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A full-stack, research-backed recommendation engine built on MovieLens 20M.**  
From raw interaction logs to a live API with real-time caching — end to end.

[**Live API**](https://intimateuser6969-cinewatch-recommender.hf.space/docs) · [**HuggingFace Space**](https://huggingface.co/spaces/IntimateUser6969/Cinewatch-recommender) · [**Model Weights**](https://huggingface.co/IntimateUser6969/movielens-recommender)

</div>

---

## Research Papers Behind This System

### 1 — Monolith: Real-Time Recommendation System with Collisionless Embedding Table (ByteDance / TikTok, 2022)

> *The core insight that shaped our architecture*

TikTok's Monolith paper introduced the idea of treating recommendation as a retrieval + ranking two-stage pipeline where:

- **Stage 1 (Retrieval):** A lightweight dual-encoder (two-tower) model maps users and items into the same embedding space. Approximate Nearest Neighbour search retrieves thousands of candidates in milliseconds from a catalogue of billions.
- **Stage 2 (Ranking):** A heavier interaction model — DeepFM — re-scores the small candidate set with full feature crosses.
- **Online learning:** User behaviour streams are processed in near-real-time to update feature representations without full retraining.

**What we borrowed from Monolith:**

| Monolith Concept | Our Implementation |
|---|---|
| Two-tower dual encoder | `models/two_tower.py` — UserTower + ItemTower with shared embedding space |
| ANN retrieval | `serving/faiss_index.py` — FAISS IndexFlatIP for exact inner-product search |
| Feature store | `serving/feature_store.py` — Upstash Redis with SQLite fallback |
| Streaming feature updates | `streaming/flink_consumer.py` — EMA updates on user genre preferences |
| Collisionless embeddings | Dedicated `nn.Embedding` tables per entity, no hashing tricks |

---

### 2 — DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (Huawei, 2017)

> *The ranking model at the heart of our scoring layer*

DeepFM elegantly unifies two previously separate ideas:

- **Factorisation Machines (FM):** Capture all pairwise feature interactions in O(kn) without manual feature engineering, using the identity:

```
FM(x) = w₀ + Σᵢ wᵢxᵢ + Σᵢ Σⱼ₍ᵢ₎ <vᵢ, vⱼ> xᵢxⱼ
       = w₀ + Σᵢ wᵢxᵢ + ½ Σf [ (Σᵢ vᵢf xᵢ)² - Σᵢ vᵢf² xᵢ² ]
```

- **Deep component:** A multi-layer MLP that learns arbitrary high-order interactions from the same shared embedding layer.

- **Shared embeddings:** FM and Deep share the same embedding table — no redundant parameters, richer gradient signal per parameter.

**What we implemented from DeepFM:**

| DeepFM Concept | Our Implementation |
|---|---|
| FM second-order interactions | `models/deepfm.py — FMLayer` using sum-of-squares trick |
| Deep component | 3-layer MLP (400→400→400) with BatchNorm + Dropout |
| Shared embedding layer | `user_embed`, `item_embed`, `year_embed` shared between FM + Deep |
| Sparse + dense features | Categorical fields → embeddings; continuous fields concatenated directly |
| IPS-weighted BCE loss | Popularity debiasing via Inverse Propensity Scoring |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                        │
│   Home · Search · Movie Detail · Taste Profile · Rate & React       │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTPS  (VITE_API_URL)
┌────────────────────────────▼────────────────────────────────────────┐
│              FastAPI Backend — HuggingFace Spaces :7860             │
│                                                                     │
│  POST /api/events          GET /api/recommendations/{user_id}       │
│  POST /api/feedback        GET /api/search                          │
│  GET  /api/taste-profile   GET /api/movies/{id}                     │
└───────────┬────────────────────────────┬────────────────────────────┘
            │                            │
┌───────────▼──────────┐    ┌────────────▼───────────────────────────┐
│   Event Logger       │    │        Recommendation Engine           │
│                      │    │                                        │
│  Kafka (preferred)   │    │  1. User embedding lookup              │
│  SQLite  (fallback)  │    │       └─ Upstash Redis cache           │
└───────────┬──────────┘    │       └─ compute if miss               │
            │               │                                        │
┌───────────▼──────────┐    │  2. FAISS ANN search  (top-200)        │
│  Flink Consumer      │    │       └─ IndexFlatIP · 27K items       │
│  (Python thread)     │    │                                        │
│                      │    │  3. DeepFM ranking    (top-200→50)     │
│  EMA feature update  │    │       └─ FM + Deep · IPS weights       │
│  Retrain trigger     │    │                                        │
└──────────────────────┘    │  4. MMR re-rank       (top-50→20)      │
                            │       └─ λ=0.7 · diversity via cosine  │
                            │                                        │
                            │  5. Explanation generation             │
                            └────────────────────────────────────────┘
                                         │
                            ┌────────────▼──────────────┐
                            │   Upstash Redis (TLS)      │
                            │   user:{id}:embedding      │
                            │   user:{id}:features       │
                            │   item:{id}:features       │
                            └───────────────────────────┘
```

### Training Pipeline

```
MovieLens 20M CSVs (rating.csv · movie.csv · genome_scores.csv)
         │
         ▼
  Preprocessor
  ├── Temporal 80/10/10 split (no future leakage)
  ├── Genre multi-hot  (20 genres)
  ├── Genome SVD       (1,128 tags → 32 dims via TruncatedSVD)
  ├── User features    (genre preferences · avg rating · log count)
  └── IPS weights      (popularity^0.5 propensity · capped at 10×)
         │
         ├──▶  Two-Tower Training
         │     BPR loss · AdamW · CosineAnnealingLR
         │     Vectorised negative sampling (8M triplets)
         │              │
         │              └──▶  FAISS IndexFlatIP  (item embeddings)
         │
         └──▶  DeepFM Training
               IPS-weighted BCE · ReduceLROnPlateau
               AUC validation per epoch
                        │
                        └──▶  Offline Evaluation
                              NDCG@10 · Precision@10 · Recall@10
                              MRR · Coverage · ILD · Simulated CTR
```

---

## Tech Stack

### Machine Learning

| Component | Technology | Detail |
|---|---|---|
| Two-Tower encoder | PyTorch | User + Item towers → L2-normalised 64-dim vectors |
| Candidate retrieval | FAISS (faiss-cpu) | IndexFlatIP — exact inner product, 27K items |
| Ranker | DeepFM (PyTorch) | FM + 3-layer MLP, shared embeddings |
| Diversity | MMR | Maximal Marginal Relevance, λ=0.7 |
| Debiasing | IPS | Inverse Propensity Scoring on popularity |
| Offline eval | User Simulator | Logistic click model with position bias |
| Feature reduction | TruncatedSVD | 1,128 genome tags → 32 dims |

### Backend

| Component | Technology |
|---|---|
| API framework | FastAPI 0.136 + Uvicorn |
| Data validation | Pydantic v2 |
| Feature store | Upstash Redis (TLS) + SQLite fallback |
| Event streaming | Kafka + SQLite fallback |
| Batch processing | PySpark + pandas fallback |
| Env management | python-dotenv |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 18 + Vite |
| Styling | Tailwind CSS v3 |
| Animations | Framer Motion |
| Charts | Recharts (radar + bar) |
| HTTP client | Axios |
| Routing | React Router v6 |

### Infrastructure

| Component | Technology |
|---|---|
| Model hosting | HuggingFace Spaces (Docker) |
| Model weights | HuggingFace Hub (git-lfs) |
| Redis cache | Upstash (serverless, TLS) |
| Container | python:3.11-slim |
| Dataset | MovieLens 20M (GroupLens) |

---

## What Makes This Different

### IPS Debiasing
Most recommenders train on raw ratings and amplify popularity bias — popular items get recommended more, get more ratings, get recommended even more. We break this loop by computing per-item propensity scores and down-weighting popular items in the loss:

```
P(item exposed) ∝ popularity^0.5
IPS weight = 1 / P(exposure),  capped at 10×, normalised to mean = 1
```

### Vectorised Preprocessing at Scale
Naive pandas `groupby` loops over 2M rows would take 10+ minutes. The user feature computation runs in **under 1 second** using `numpy.bincount` (compiled C, one call per genre column):

```python
genre_weighted = item_genre[movie_idxs] * rating_vals[:, None]  # [2M, 20]
for g in range(n_genres):
    genre_pref[:, g] = np.bincount(user_idxs, weights=genre_weighted[:, g])
# 8M negative triplets generated in 0.12s
```

### Real-Time Feature Caching
Every user embedding is computed once and cached in Upstash Redis with a 24-hour TTL. Subsequent requests for the same user skip model inference entirely — **246ms average cache hit vs ~800ms cold computation**.

### MMR Diversity Re-ranking
Raw ranked lists from DeepFM tend toward genre monocultures. MMR explicitly trades off relevance against redundancy:

```
MMR(i) = λ · score(i)  −  (1−λ) · max_{j∈Selected} cosine_sim(i, j)
```

With λ=0.7 this yields an **Intra-List Diversity score of 0.65–0.75** versus ~0.35 for pure relevance ranking.

### Genome Tag Embeddings
Unlike systems that use only genre labels, CineMatch uses MovieLens genome scores — 1,128 nuanced tags (e.g. "atmospheric", "nonlinear timeline", "based on true story") per movie, reduced to 32 semantic dimensions via SVD. This gives the item tower a richer representation than genre alone.

### Pandas-Safe Serialisation
Pandas 2.x `NDArrayBacked` pickles are not stable across minor versions. Rather than pickling DataFrames, we strip `movie_df` from the pickle and save it as a plain CSV — **guaranteed to load correctly regardless of pandas version**.

---

## Dataset

**MovieLens 20M** (GroupLens Research)

| File | Rows | Description |
|------|------|-------------|
| `rating.csv` | 20,000,263 | userId · movieId · rating · timestamp |
| `movie.csv` | 27,278 | movieId · title · genres |
| `genome_scores.csv` | 11,709,768 | movieId · tagId · relevance |
| `genome_tags.csv` | 1,128 | tagId · tag name |

Training uses a **10% stratified sample by default** (~2M ratings) for fast iteration. Set `SAMPLE=1.0` for full-dataset runs.

---

## Project Structure

```
recommender/
├── data/                    # Loading, preprocessing, IPS weights
│   ├── loader.py            # Chunked CSV loading, genre parsing
│   ├── preprocessor.py      # Feature engineering, ID encoding, splits
│   └── ips_weights.py       # Propensity scoring and IPS computation
│
├── models/                  # Pure PyTorch — no I/O
│   ├── two_tower.py         # UserTower, ItemTower, BPRLoss
│   ├── deepfm.py            # FMLayer, DeepFM, shared embeddings
│   ├── reranker.py          # MMR algorithm, ILD metric
│   └── user_simulator.py    # Offline evaluation simulator
│
├── training/                # Training loops and evaluation
│   ├── train_two_tower.py   # BPR training, FAISS index builder
│   ├── train_deepfm.py      # IPS-weighted BCE training
│   └── evaluate.py          # NDCG, Precision, Recall, MRR, ILD
│
├── serving/                 # Inference pipeline
│   ├── faiss_index.py       # FAISS wrapper with numpy fallback
│   ├── feature_store.py     # Redis/SQLite unified interface
│   └── inference.py         # Full pipeline: FAISS→DeepFM→MMR→explain
│
├── api/                     # FastAPI application
│   ├── main.py              # App factory, lifespan, CORS
│   ├── routes.py            # All endpoint handlers
│   ├── event_logger.py      # Kafka/SQLite event logging
│   └── schemas.py           # Pydantic request/response models
│
├── streaming/               # Real-time and batch processing
│   ├── kafka_producer.py    # Kafka producer with SQLite fallback
│   ├── flink_consumer.py    # EMA feature updater (Python thread)
│   └── batch_spark.py       # Nightly feature recompute + drift detection
│
├── frontend/                # React + Vite application
│   └── src/
│       ├── pages/           # Home · Search · Detail · Taste · Feedback
│       └── components/      # Navbar · MovieCard · MovieRow · Skeleton · FeedbackModal
│
├── artifacts/               # Generated by training (git-lfs)
│   ├── two_tower.pt         # Two-Tower weights
│   ├── deepfm_best.pt       # DeepFM weights
│   ├── faiss.index          # ANN index
│   ├── preprocessor.pkl     # ID maps + scalers
│   └── …
│
├── run_training.sh          # End-to-end training pipeline
├── start_services.sh        # Local dev: API + consumer + frontend
└── deploy.md                # Full deployment guide
```

---

## Offline Evaluation Results

Trained on 10% sample (~2M ratings), 10 epochs, CPU:

| Metric | Value | Notes |
|--------|-------|-------|
| NDCG@10 | 0.11 | Ranking quality |
| Precision@10 | 0.08 | Fraction of top-10 that are relevant |
| Recall@10 | 0.06 | Fraction of relevant items retrieved |
| MRR | 0.14 | Mean Reciprocal Rank |
| ILD (diversity) | 0.68 | Intra-List Diversity after MMR |
| Catalog Coverage | 22% | Fraction of movies ever recommended |
| Simulated CTR | 0.17 | Click-Through Rate from user simulator |

> Results improve significantly with `SAMPLE=1.0` and `EPOCHS=20`.

---

## Local Setup

```bash
# 1. Clone and enter
git clone <your-repo>
cd recommender/

# 2. Virtual environment
python -m venv .venv && .venv\Scripts\activate   # Windows
# source .venv/bin/activate                       # macOS/Linux

# 3. Install
pip install -r requirements.txt

# 4. Train (10% sample, ~5 min on CPU)
./run_training.sh

# 5. Start API
uvicorn api.main:app --reload --port 8000

# 6. Start frontend
cd frontend && npm install && npm run dev
```

---

## Deployed Links

| Service | URL |
|---------|-----|
| **Live WebSite** | https://byte-dance-recommendation-system.vercel.app/ |
| **Live API + Docs** | https://intimateuser6969-cinewatch-recommender.hf.space/docs |
| **HuggingFace Space** | https://huggingface.co/spaces/IntimateUser6969/Cinewatch-recommender |
| **Model Weights** | https://huggingface.co/IntimateUser6969/movielens-recommender |

---

## API Quick Reference

```bash
# Health
curl https://intimateuser6969-cinewatch-recommender.hf.space/api/health

# Personalised recommendations
curl "https://intimateuser6969-cinewatch-recommender.hf.space/api/recommendations/1?limit=10"

# Search
curl "https://intimateuser6969-cinewatch-recommender.hf.space/api/search?q=inception"

# Taste profile
curl "https://intimateuser6969-cinewatch-recommender.hf.space/api/taste-profile/1"

# Submit feedback
curl -X POST ".../api/feedback" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 1, "action": "thumbs_up", "rating": 4.5}'
```

---

## References

```bibtex
@article{liu2022monolith,
  title     = {Monolith: Real Time Recommendation System With Collisionless Embedding Table},
  author    = {Liu, Zhuoran and others},
  journal   = {arXiv preprint arXiv:2209.07663},
  year      = {2022},
  note      = {ByteDance / TikTok}
}

@inproceedings{guo2017deepfm,
  title     = {DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author    = {Guo, Huifeng and Tang, Ruiming and Ye, Yunming and Li, Zhenguo and He, Xiuqiang},
  booktitle = {Proceedings of IJCAI},
  year      = {2017}
}

@article{harper2015movielens,
  title   = {The MovieLens Datasets: History and Context},
  author  = {Harper, F. Maxwell and Konstan, Joseph A.},
  journal = {ACM Transactions on Interactive Intelligent Systems},
  year    = {2015}
}

@inproceedings{carbonell1998mmr,
  title     = {The use of MMR, diversity-based reranking for reordering documents and producing summaries},
  author    = {Carbonell, Jaime and Goldstein, Jade},
  booktitle = {Proceedings of SIGIR},
  year      = {1998}
}
```

---

## Work in Progress

This system is actively being improved. Planned work:

- [ ] **Session-aware recommendations** — incorporate within-session item sequences using a Transformer encoder (SASRec-style) on top of the two-tower retrieval stage
- [ ] **Cross-feature interactions** — replace the FM second-order layer with a Compressed Interaction Network (xDeepFM) for explicit high-order feature crosses
- [ ] **Online learning** — move from periodic batch retraining to true gradient-based online updates triggered by the Flink consumer
- [ ] **A/B testing framework** — replace the user simulator with a lightweight bandit framework to evaluate ranking policies against each other in production
- [ ] **ONNX export** — export Two-Tower and DeepFM to ONNX for faster CPU inference and potential WASM serving in the browser
- [ ] **Frontend auth** — replace the hardcoded `userId=1` demo user with a proper session / JWT flow
- [ ] **Cold-start handling** — content-based fallback using genre + genome embeddings for new users with zero interaction history
- [ ] **Explainability** — SHAP values on DeepFM feature contributions per recommendation, surfaced in the "Why recommended?" UI panel

---

<div align="center">

Built with PyTorch · FastAPI · React · FAISS · Upstash Redis  
Inspired by ByteDance Monolith & Huawei DeepFM research

</div>
