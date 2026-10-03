import numpy as np
import pandas as pd
import pytest

FEATS = ["recency_days", "frequency", "monetary_total", "spend_90d", "avg_gap_days"]


@pytest.fixture(scope="session")
def feats():
    return FEATS


@pytest.fixture(scope="session")
def synthetic_ds():
    rng = np.random.default_rng(0)
    parts = []
    for i, d in enumerate(pd.date_range("2010-09-01", periods=5, freq="90D")):
        n = 600
        df = pd.DataFrame({
            "snapshot_date": d, "customer_id": np.arange(n),
            "recency_days": rng.uniform(1, 300, n), "frequency": rng.integers(1, 30, n).astype(float),
            "monetary_total": rng.lognormal(6, 1, n), "spend_90d": rng.lognormal(4, 1.5, n),
            "avg_gap_days": np.where(rng.random(n) < .1, np.nan, rng.uniform(5, 100, n)),
        })
        p = 1 / (1 + np.exp((df.recency_days - 100) / 40))
        df["target_spend_90d"] = (rng.random(n) < p) * df.frequency * rng.lognormal(3, .4, n)
        df["split"] = "train" if i < 4 else "test"
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
