"""Train baselines + GBM, save models/model.joblib, models/metrics.json, models/test_predictions.parquet."""
import json
from pathlib import Path

import joblib
import pandas as pd

from cvo_demo.modeling import run_training

DATA, OUT = Path("data/processed"), Path("models")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = pd.read_parquet(DATA / "features.parquet")
    meta = json.loads((DATA / "feature_meta.json").read_text())
    artifact, metrics = run_training(ds, meta["feature_columns"], meta["target"])
    preds = metrics.pop("_test_predictions")

    joblib.dump(artifact, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_parquet(OUT / "test_predictions.parquet", index=False)

    cols = ["mae", "rmse", "rmsle", "spearman", "auc_will_buy", "top_decile_lift", "calibration_ratio"]
    print(f"model_version: {artifact['model_version']}   best params: {metrics['best_params']}")
    print(f"\nTEST ({metrics['data']['test_cutoff']}, {metrics['data']['n_test_rows']} customers)")
    print(pd.DataFrame(metrics["test"]).T[cols].round(3).to_string())
    print("\nTop features (permutation importance, validation):")
    for k, v in metrics["permutation_importance_top10"].items():
        print(f"  {k:<26}{v:.4f}")


if __name__ == "__main__":
    main()
