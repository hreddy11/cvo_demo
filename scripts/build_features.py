"""Build training snapshots + the serving feature table.

Test snapshot  = the last cutoff (default 2011-09-09).
Train snapshots = test cutoff minus k*horizon days (k=1..N), so each train target window ends
at or before the test cutoff: no future information leaks into training.

Outputs (data/processed/):
  features.parquet          snapshot_date, customer_id, <features>, target_spend_90d, target_will_buy, split
  serving_features.parquet  features as of the end of the data, indexed by customer_id (no target)
  feature_meta.json         feature list, cutoffs, horizon
"""
import argparse, json
from pathlib import Path

import pandas as pd

from cvo_demo.data import load_clean
from cvo_demo.features import FEATURE_COLUMNS, TARGET, build_features, build_snapshot

OUT = Path("data/processed")


def main(test_cutoff: str, horizon: int, n_train: int) -> None:
    clean = load_clean()
    test_cutoff = pd.Timestamp(test_cutoff)
    train_cutoffs = [test_cutoff - pd.Timedelta(days=horizon * k) for k in range(n_train, 0, -1)]

    parts = []
    for c in train_cutoffs:
        s = build_snapshot(clean, c, horizon); s["split"] = "train"; parts.append(s)
        print(f"train {c.date()}: {len(s):>5} customers, buy rate {s.target_will_buy.mean():.0%}")
    s = build_snapshot(clean, test_cutoff, horizon); s["split"] = "test"; parts.append(s)
    print(f"test  {test_cutoff.date()}: {len(s):>5} customers, buy rate {s.target_will_buy.mean():.0%}")

    ds = pd.concat(parts, ignore_index=True)
    ds.to_parquet(OUT / "features.parquet", index=False)

    serving_date = clean["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    serving = build_features(clean, serving_date)
    serving.to_parquet(OUT / "serving_features.parquet")
    print(f"serving table as of {serving_date.date()}: {len(serving)} customers")

    meta = {
        "feature_columns": FEATURE_COLUMNS, "target": TARGET, "horizon_days": horizon,
        "train_cutoffs": [str(c.date()) for c in train_cutoffs],
        "test_cutoff": str(test_cutoff.date()), "serving_as_of": str(serving_date.date()),
    }
    (OUT / "feature_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved features.parquet, serving_features.parquet, feature_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-cutoff", default="2011-09-09")
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--n-train", type=int, default=4)
    a = ap.parse_args()
    main(a.test_cutoff, a.horizon, a.n_train)
