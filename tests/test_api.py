import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cvo_demo.modeling import predict_spend, run_training
from cvo_demo.service import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory, synthetic_ds, feats):
    d = tmp_path_factory.mktemp("serving")
    artifact, _ = run_training(synthetic_ds, feats)
    joblib.dump(artifact, d / "model.joblib")
    serving = synthetic_ds[synthetic_ds.split == "test"].set_index("customer_id")[feats]
    serving.to_parquet(d / "serving.parquet")
    app = create_app(str(d / "model.joblib"), str(d / "serving.parquet"), str(d / "no_meta.json"))
    with TestClient(app) as c:
        c.artifact, c.serving = artifact, serving
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"ready": True}


def test_model_info(client):
    j = client.get("/model").json()
    assert j["model_version"] == client.artifact["model_version"]
    assert j["feature_columns"] == client.artifact["feature_columns"]
    assert "auc_will_buy" in j["test_metrics"]


def test_predict_known_customer(client):
    r = client.post("/predict", json={"customer_id": 5})
    assert r.status_code == 200
    j = r.json()
    assert j["customer_id"] == 5 and j["predicted_spend_90d"] >= 0
    assert 0 <= j["percentile"] <= 100 and j["value_tier"] in {"high", "medium", "low"}
    assert j["model_version"] == client.artifact["model_version"] and j["horizon_days"] == 90
    # must equal offline scoring: no train/serve skew in the scoring path
    expected = predict_spend(client.artifact, client.serving.loc[[5]])[0]
    assert j["predicted_spend_90d"] == pytest.approx(expected, abs=0.01)


def test_predict_unknown_customer_404(client):
    assert client.post("/predict", json={"customer_id": 999999}).status_code == 404


@pytest.mark.parametrize("body", [{}, {"customer_id": "abc"}, {"customer_id": None}])
def test_predict_invalid_input_422(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_batch_mixed_known_unknown(client):
    r = client.post("/predict/batch", json={"customer_ids": [1, 2, 999999, 3, 1]})
    j = r.json()
    assert r.status_code == 200
    assert [p["customer_id"] for p in j["predictions"]] == [1, 2, 3, 1]  # order kept, duplicates allowed
    assert j["unknown_customer_ids"] == [999999]


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"customer_ids": []}).status_code == 422
    assert client.post("/predict/batch", json={"customer_ids": list(range(1001))}).status_code == 422


def test_predict_from_features_matches_lookup(client):
    row = client.serving.loc[7].where(client.serving.loc[7].notna(), None).to_dict()
    a = client.post("/predict/features", json={"features": row}).json()
    b = client.post("/predict", json={"customer_id": 7}).json()
    assert a["customer_id"] is None
    assert a["predicted_spend_90d"] == pytest.approx(b["predicted_spend_90d"], abs=0.01)


def test_predict_from_features_validates_contract(client):
    full = {c: 1.0 for c in client.artifact["feature_columns"]}
    bad_missing = {k: v for k, v in full.items() if k != "recency_days"}
    r = client.post("/predict/features", json={"features": bad_missing})
    assert r.status_code == 422 and r.json()["detail"]["missing_features"] == ["recency_days"]
    r = client.post("/predict/features", json={"features": {**full, "typo_feature": 1.0}})
    assert r.status_code == 422 and r.json()["detail"]["unknown_features"] == ["typo_feature"]


def test_tiers_are_ordered_by_score(client):
    ids = list(range(0, 600))
    preds = client.post("/predict/batch", json={"customer_ids": ids}).json()["predictions"]
    by_tier = {t: [p["predicted_spend_90d"] for p in preds if p["value_tier"] == t] for t in ("high", "medium", "low")}
    assert min(by_tier["high"]) >= max(by_tier["low"])
    assert 0.05 < len(by_tier["high"]) / len(preds) < 0.2


def test_observability_headers_and_stats(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123" and float(r.headers["X-Process-Time-ms"]) >= 0
    s = client.get("/stats").json()
    assert s["requests_total"] > 0 and s["predictions_served"] > 0
    assert s["latency_ms_last_1000"]["p95"] >= s["latency_ms_last_1000"]["p50"]


def test_missing_artifacts_fail_fast(tmp_path):
    app = create_app(str(tmp_path / "nope.joblib"), str(tmp_path / "nope.parquet"))
    with pytest.raises(RuntimeError, match="not found"):
        with TestClient(app):
            pass
