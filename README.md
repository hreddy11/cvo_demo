# cvo_demo: a local Customer Value Optimization project

Goal: build, serve and test one CVO model end to end on a Mac with Docker.
Model: predict each customer's **spend in the next 90 days** (CLV-style score).
Data: UCI Online Retail II (CC BY 4.0).

## Roadmap
| Step | What | Status |
|---|---|---|
| 1 | Download data, explore, define target and cutoff | **this drop** |
| 2 | Feature engineering (RFM, tenure, returns) with a time-safe cutoff -> `features.parquet` | next |
| 3 | Train baseline + gradient boosting, log metrics to `models/metrics.json` | next |
| 4 | FastAPI service: `/health`, `/predict`, `/predict/batch`, model version in response | next |
| 5 | Dockerfile + docker-compose (api + feature store volume) | next |
| 6 | pytest/requests scripts: contract, latency, bad-input tests | next |

## Run step 1 (uv)
```bash
# install uv once, if needed: brew install uv
cd cvo_demo
uv sync                                                # creates .venv, installs deps + dev group
uv run python scripts/download_data.py                 # or add --max-customers 2000 to shrink
uv run jupyter lab                                     # open notebooks/01_explore.ipynb
```
Later steps use the same tool: `uv run pytest`, `uv add <pkg>`, and `uv sync --frozen --no-dev` inside the Docker image.

The full dataset is ~1M rows (~45 MB Excel), which is comfortable on a laptop;
`--max-customers` is there if you want faster iteration.


## Run step 2
```bash
uv add pytest
uv sync
uv run pytest                                  # 4 feature tests incl. a no-leakage test
uv run python scripts/build_features.py        # -> data/processed/features.parquet, serving_features.parquet
uv run jupyter lab                             # open notebooks/02_features.ipynb
```
Snapshots: 4 train cutoffs spaced 90 days before the test cutoff (2011-09-09), so every train target window
ends at or before the test cutoff. `serving_features.parquet` is the customer feature table the API will read.

## Run step 4 (API, without Docker for now)
```bash
uv run pytest                                          # 23 tests, incl. API contract tests
uv run uvicorn cvo.service:app --port 8000             # needs models/model.joblib + serving_features.parquet
# open http://localhost:8000/docs for the interactive API docs
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"customer_id": 12347}'
```
| Endpoint | Purpose |
|---|---|
| `POST /predict` | score one customer by id (features looked up from the serving table) |
| `POST /predict/batch` | up to 1000 ids; unknown ids are returned, not fatal |
| `POST /predict/features` | caller supplies all features; strict contract (missing or unknown names -> 422) |
| `GET /health`, `/ready` | liveness vs readiness (model + features loaded) |
| `GET /model` | version, params, calibration factor, offline test metrics, feature list |
| `GET /stats` | request counts, 4xx/5xx, latency p50/p95, predictions served |

Responses include `predicted_spend_90d`, `percentile` and `value_tier` (high = top 10%, medium = next 40%, low = rest,
relative to the serving population) plus `model_version` and `features_as_of`. Every request gets an `X-Request-ID`
and one JSON log line. Config via env: `MODEL_PATH`, `FEATURES_PATH`, `META_PATH`.

## Run step 5 (Docker)
Prereqs: Docker Desktop for Mac running, and steps 2-3 done (`models/model.joblib` and the two files in `data/processed/` exist).
```bash
make up            # docker compose up -d --build, then shows status (healthy after ~10-20 s)
make status        # GET /model: which model version is live
make logs          # follow the JSON request logs
make down
```
**Roll out / roll back a model without rebuilding the image:** `train_model.py` writes `models/model.joblib` (latest) plus an
immutable copy in `models/registry/<version>/`. Compose reads `MODEL_FILE`:
```bash
uv run python scripts/train_model.py                                   # new version -> models/model.joblib
docker compose restart cvo-api                                         # roll out the latest
MODEL_FILE=registry/gbm-YYYYMMDD-HHMMSS/model.joblib docker compose up -d   # pin / roll back to a specific version
curl -s localhost:8000/model | python3 -m json.tool | grep model_version
```
Image design: multi-stage build (uv resolves with `--frozen --no-dev`, package installed non-editable), slim runtime,
non-root user, read-only root filesystem, healthcheck on `/ready`. The model and feature table are mounted read-only, not baked in.
If a file is missing the app exits at startup instead of serving errors (Compose will keep restarting it; see `make logs`).

## Design choices 
- **Time-safe features:** computed only from data before the cutoff (no leakage).
- **Feature store, poor-man's edition:** a precomputed customer feature table the API reads at request time, so callers send a `customer_id`, not 15 raw features.
- **Model versioning + metrics:** saved with the artifact and returned by the API.
- **Observability:** request logging, latency, prediction distribution.