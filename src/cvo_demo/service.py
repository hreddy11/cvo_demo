"""CVO scoring service (FastAPI).

Two ways to score a customer:
  POST /predict            {"customer_id": 12347}      -> features looked up from the serving table
                                                           (our stand-in for an online feature store)
  POST /predict/features   {"features": {...}}         -> caller supplies all features
  POST /predict/batch      {"customer_ids": [...]}     -> many customers, one model call

Operational endpoints: /health (liveness), /ready (model + features loaded), /model (metadata),
/stats (request counts, latency percentiles). Every response carries X-Request-ID and
X-Process-Time-ms headers and every request is logged as one JSON line.

Config (env): MODEL_PATH, FEATURES_PATH, META_PATH.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cvo_demo.modeling import predict_spend

log = logging.getLogger("cvo.api")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

MAX_BATCH = 1000


# ---------- schemas ----------
class PredictRequest(BaseModel):
    customer_id: int = Field(..., examples=[12347])


class BatchRequest(BaseModel):
    customer_ids: list[int] = Field(..., min_length=1, max_length=MAX_BATCH)


class FeaturesRequest(BaseModel):
    features: dict[str, float | None] = Field(
        ..., description="All model features by name. null is allowed (e.g. avg_gap_days for one-time buyers).")


class Prediction(BaseModel):
    customer_id: int | None = None
    predicted_spend_90d: float
    percentile: float = Field(..., description="Share of the serving population with a score <= this one (0-100)")
    value_tier: str = Field(..., description="high (top 10%), medium (next 40%), low (bottom 50%)")


class PredictResponse(Prediction):
    model_version: str
    horizon_days: int
    features_as_of: str | None = None


class BatchResponse(BaseModel):
    model_version: str
    horizon_days: int
    features_as_of: str | None
    predictions: list[Prediction]
    unknown_customer_ids: list[int]


# ---------- helpers ----------
def tier_for(percentile: float) -> str:
    return "high" if percentile >= 90 else "medium" if percentile >= 50 else "low"


class Stats:
    def __init__(self) -> None:
        self.started = time.time()
        self.lock = threading.Lock()
        self.requests = 0
        self.status = Counter()
        self.by_path = Counter()
        self.latencies: deque[float] = deque(maxlen=1000)
        self.predictions = 0

    def record(self, path: str, status: int, ms: float) -> None:
        with self.lock:
            self.requests += 1
            self.status[f"{status // 100}xx"] += 1
            self.by_path[path] += 1
            self.latencies.append(ms)

    def add_predictions(self, n: int) -> None:
        with self.lock:
            self.predictions += n

    def snapshot(self) -> dict:
        with self.lock:
            lat = np.array(self.latencies) if self.latencies else np.array([0.0])
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests_total": self.requests,
                "responses_by_class": dict(self.status),
                "requests_by_path": dict(self.by_path),
                "predictions_served": self.predictions,
                "latency_ms_last_1000": {
                    "p50": round(float(np.percentile(lat, 50)), 2),
                    "p95": round(float(np.percentile(lat, 95)), 2),
                    "max": round(float(lat.max()), 2),
                },
            }


# ---------- app ----------
def create_app(model_path: str | None = None, features_path: str | None = None,
               meta_path: str | None = None) -> FastAPI:
    model_path = model_path or os.getenv("MODEL_PATH", "models/model.joblib")
    features_path = features_path or os.getenv("FEATURES_PATH", "data/processed/serving_features.parquet")
    meta_path = meta_path or os.getenv("META_PATH", "data/processed/feature_meta.json")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for p in (model_path, features_path):
            if not Path(p).exists():
                raise RuntimeError(f"Required file not found: {p} (run build_features.py and train_model.py first)")
        art = joblib.load(model_path)
        feats = pd.read_parquet(features_path)
        feats.index = feats.index.astype(int)
        missing = set(art["feature_columns"]) - set(feats.columns)
        if missing:
            raise RuntimeError(f"Serving table is missing model features: {sorted(missing)}")
        meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        # score distribution of the serving population -> percentile / tier for any score
        scores = np.sort(predict_spend(art, feats))
        app.state.artifact, app.state.features, app.state.scores = art, feats, scores
        app.state.as_of = meta.get("serving_as_of")
        app.state.horizon = int(meta.get("horizon_days", 90))
        app.state.stats = Stats()
        log.info(json.dumps({"event": "startup", "model_version": art["model_version"],
                             "customers": len(feats), "features_as_of": app.state.as_of}))
        yield

    app = FastAPI(title="CVO scoring service", version="0.4.0", lifespan=lifespan,
                  description="Predicts a customer's spend over the next 90 days.")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # unhandled -> structured 500, never a stack trace to the client
            log.exception("unhandled error request_id=%s", rid)
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
        stats = getattr(request.app.state, "stats", None)
        if stats:
            stats.record(request.url.path, response.status_code, ms)
        log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method,
                             "path": request.url.path, "status": response.status_code,
                             "latency_ms": round(ms, 2)}))
        return response

    def _score(frame: pd.DataFrame) -> list[Prediction]:
        spend = predict_spend(app.state.artifact, frame)
        pct = np.searchsorted(app.state.scores, spend, side="right") / len(app.state.scores) * 100
        app.state.stats.add_predictions(len(spend))
        ids = frame.index if frame.index.name == "customer_id" else [None] * len(frame)
        return [Prediction(customer_id=None if i is None else int(i), predicted_spend_90d=round(float(s), 2),
                           percentile=round(float(p), 1), value_tier=tier_for(p))
                for i, s, p in zip(ids, spend, pct)]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        ok = hasattr(app.state, "artifact")
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/model")
    def model_info():
        a = app.state.artifact
        return {"model_version": a["model_version"], "trained_at": a["trained_at"], "params": a["params"],
                "calibration_factor": a["calibration_factor"], "feature_columns": a["feature_columns"],
                "test_metrics": a["test_metrics"], "features_as_of": app.state.as_of,
                "customers_in_feature_table": len(app.state.features), "horizon_days": app.state.horizon}

    @app.get("/stats")
    def stats():
        return app.state.stats.snapshot()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if req.customer_id not in app.state.features.index:
            raise HTTPException(404, f"customer_id {req.customer_id} not found in feature table")
        p = _score(app.state.features.loc[[req.customer_id]])[0]
        return PredictResponse(**p.model_dump(), model_version=app.state.artifact["model_version"],
                               horizon_days=app.state.horizon, features_as_of=app.state.as_of)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest):
        idx = app.state.features.index
        known = [i for i in req.customer_ids if i in idx]
        unknown = sorted({i for i in req.customer_ids if i not in idx})
        preds = _score(app.state.features.loc[known]) if known else []
        return BatchResponse(model_version=app.state.artifact["model_version"], horizon_days=app.state.horizon,
                             features_as_of=app.state.as_of, predictions=preds, unknown_customer_ids=unknown)

    @app.post("/predict/features", response_model=PredictResponse)
    def predict_from_features(req: FeaturesRequest):
        print(req.features)
        cols = app.state.artifact["feature_columns"]
        missing = [c for c in cols if c not in req.features]
        extra = [k for k in req.features if k not in cols]
        if missing or extra:
            raise HTTPException(422, {"missing_features": missing, "unknown_features": extra})
        row = pd.DataFrame([{c: (np.nan if req.features[c] is None else req.features[c]) for c in cols}],
                           dtype="float64")
        p = _score(row)[0]
        return PredictResponse(**p.model_dump(), model_version=app.state.artifact["model_version"],
                               horizon_days=app.state.horizon, features_as_of=None)

    return app


app = create_app()
