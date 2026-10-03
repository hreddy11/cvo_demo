"""Training, evaluation and prediction for the next-90-day spend model.

Target handling: the target is zero-inflated and heavy-tailed, so models are fit on log1p(spend).
Back-transforming with expm1 gives a biased estimate of the *mean* (Jensen's inequality). We fix totals
with one scalar fit on the out-of-sample validation snapshot so predicted total spend = actual total spend:
    spend_hat = calibration_factor * expm1(pred_log)
(A Duan smearing factor was tried first and was unstable on this zero-inflated target.)
The factor rescales money predictions only; it does not change customer ranking.
"""
from __future__ import annotations

import itertools
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

SEED = 42
GRID = [
    dict(learning_rate=lr, max_leaf_nodes=leaves, max_iter=iters, min_samples_leaf=40, l2_regularization=1.0)
    for lr, leaves, iters in itertools.product([0.05, 0.1], [8, 16], [150, 300])
]


# ---------- metrics ----------
def evaluate(y_true, y_pred) -> dict:
    """Metrics on the original (money) scale. Ranking metrics matter as much as error:
    CVO decisions usually *rank* customers (who gets the offer)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    n_top = max(1, int(round(0.1 * len(y_true))))
    order = np.argsort(-y_pred, kind="stable")
    total = y_true.sum()
    capture = float(y_true[order[:n_top]].sum() / total) if total > 0 else float("nan")
    has_both = 0 < (y_true > 0).sum() < len(y_true)
    constant = np.ptp(y_pred) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float("nan") if constant else float(spearmanr(y_true, y_pred).statistic)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "rmsle": float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))),
        "spearman": rho,
        "auc_will_buy": float(roc_auc_score(y_true > 0, y_pred)) if (has_both and not constant) else float("nan"),
        "top_decile_capture": capture,
        "top_decile_lift": capture / 0.1 if capture == capture else float("nan"),
        "mean_pred": float(y_pred.mean()),
        "mean_actual": float(y_true.mean()),
        "calibration_ratio": float(y_pred.mean() / y_true.mean()) if y_true.mean() > 0 else float("nan"),
    }


# ---------- model helpers ----------
def make_gbm(**params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=SEED, **params)  # handles NaN natively


def make_ridge():
    return make_pipeline(SimpleImputer(strategy="median"),
                         QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED),
                         Ridge(alpha=10.0))


def to_spend(pred_log, calibration_factor: float = 1.0) -> np.ndarray:
    return np.clip(calibration_factor * np.expm1(np.asarray(pred_log, float)), 0.0, None)


def fit_calibration(y_money, pred_log) -> float:
    """Scalar so that sum(predicted spend) == sum(actual spend) on held-out data."""
    raw = to_spend(pred_log, 1.0).sum()
    return float(np.sum(y_money) / raw) if raw > 0 else 1.0


def predict_spend(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Shared by offline evaluation and the API."""
    log_pred = artifact["model"].predict(X[artifact["feature_columns"]])
    return to_spend(log_pred, artifact["calibration_factor"])


# ---------- training ----------
def run_training(ds: pd.DataFrame, feature_columns: list[str], target: str = "target_spend_90d") -> tuple[dict, dict]:
    """Time-based protocol:
        fit = earlier train snapshots | validation = LAST train snapshot (tuning, calibration, importance)
        final model = refit on all train snapshots | test = held-out later snapshot (reported once)
    Returns (artifact, metrics)."""
    train = ds[ds["split"] == "train"]
    test = ds[ds["split"] == "test"]
    snaps = sorted(train["snapshot_date"].unique())
    if len(snaps) < 2 or test.empty:
        raise ValueError("Need >=2 train snapshots and a test split")
    val_snap = snaps[-1]
    fit, val = train[train["snapshot_date"] < val_snap], train[train["snapshot_date"] == val_snap]

    def xy(df):
        return df[feature_columns], np.log1p(df[target].to_numpy())

    Xf, yf = xy(fit); Xv, yv = xy(val); Xa, ya = xy(train); Xt, _ = xy(test)

    # --- baselines (validation, fit on `fit`) ---
    val_baselines = {
        "mean": evaluate(val[target], np.full(len(val), fit[target].mean())),
        "last_90d_spend": evaluate(val[target], val["spend_90d"]),
    }
    ridge = make_ridge().fit(Xf, yf)
    ridge_cal = fit_calibration(val[target], ridge.predict(Xv))
    val_baselines["ridge"] = evaluate(val[target], to_spend(ridge.predict(Xv), ridge_cal))

    # --- tune GBM on validation (log-scale RMSE) ---
    grid_results = []
    for params in GRID:
        m = make_gbm(**params).fit(Xf, yf)
        rmse_log = float(np.sqrt(np.mean((m.predict(Xv) - yv) ** 2)))
        grid_results.append({"params": params, "val_rmse_log": rmse_log})
    best = min(grid_results, key=lambda r: r["val_rmse_log"])["params"]

    tuned = make_gbm(**best).fit(Xf, yf)
    calibration = fit_calibration(val[target], tuned.predict(Xv))
    val_metrics = {**val_baselines, "gbm": evaluate(val[target], to_spend(tuned.predict(Xv), calibration))}
    imp = permutation_importance(tuned, Xv, yv, scoring="neg_root_mean_squared_error",
                                 n_repeats=5, random_state=SEED)
    importance = (pd.Series(imp.importances_mean, index=feature_columns).sort_values(ascending=False))

    # --- final fit on all train snapshots, evaluate once on test ---
    final = make_gbm(**best).fit(Xa, ya)
    ridge_all = make_ridge().fit(Xa, ya)
    yt = test[target]
    test_preds = {
        "mean": np.full(len(test), train[target].mean()),
        "last_90d_spend": test["spend_90d"].to_numpy(),
        "ridge": to_spend(ridge_all.predict(Xt), ridge_cal),
        "gbm": to_spend(final.predict(Xt), calibration),
    }
    test_metrics = {k: evaluate(yt, v) for k, v in test_preds.items()}

    version = "gbm-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    artifact = {
        "model": final, "feature_columns": feature_columns, "calibration_factor": calibration,
        "model_version": version, "target": target,
        "trained_at": datetime.now(timezone.utc).isoformat(), "params": best,
        "test_metrics": test_metrics["gbm"],
    }
    metrics = {
        "model_version": version,
        "data": {
            "train_cutoffs": [str(pd.Timestamp(s).date()) for s in snaps[:-1]],
            "validation_cutoff": str(pd.Timestamp(val_snap).date()),
            "test_cutoff": str(pd.Timestamp(test["snapshot_date"].iloc[0]).date()),
            "n_fit_rows": int(len(fit)), "n_val_rows": int(len(val)),
            "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
        },
        "best_params": best, "calibration_factor": calibration, "ridge_calibration_factor": ridge_cal,
        "grid_results": grid_results, "validation": val_metrics, "test": test_metrics,
        "permutation_importance_top10": importance.head(10).round(5).to_dict(),
    }
    preds = test[["customer_id", "recency_days", "frequency", "monetary_total"]].copy()
    preds["actual"] = yt.to_numpy()
    for k, v in test_preds.items():
        preds[f"pred_{k}"] = v
    metrics["_test_predictions"] = preds  # popped by the caller
    return artifact, metrics
