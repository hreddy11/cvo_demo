import numpy as np
import pandas as pd
import pytest

from cvo_demo.modeling import evaluate, predict_spend, run_training, to_spend


def test_evaluate_perfect_ranking():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10.0, 90.0] * 10)
    m = evaluate(y, y)
    # top 10% of customers (the ten 90s) hold 900 of 1000 total -> capture 0.9, lift 9x
    assert m["top_decile_capture"] == pytest.approx(0.9)
    assert m["top_decile_lift"] == pytest.approx(9.0)
    assert m["spearman"] == pytest.approx(1.0) and m["auc_will_buy"] == pytest.approx(1.0)
    assert m["mae"] == 0


def test_evaluate_constant_prediction_does_not_crash():
    m = evaluate(np.array([0, 1.0, 5.0, 0]), np.full(4, 2.0))
    assert np.isnan(m["spearman"]) and np.isnan(m["auc_will_buy"])


def test_calibration_matches_total():
    from cvo_demo.modeling import fit_calibration
    y = np.array([0, 0, 10.0, 30.0]); pl = np.log1p(np.array([1.0, 2.0, 5.0, 8.0]))
    c = fit_calibration(y, pl)
    assert to_spend(pl, c).sum() == pytest.approx(y.sum())


def test_to_spend_nonnegative():
    assert (to_spend(np.array([-5.0, 0.0, 3.0]), 1.1) >= 0).all()


def test_run_training_end_to_end(synthetic_ds, feats):
    artifact, metrics = run_training(synthetic_ds, feats)
    assert {"model", "feature_columns", "calibration_factor", "model_version"} <= artifact.keys()
    assert set(metrics["test"]) == {"mean", "last_90d_spend", "ridge", "gbm"}
    # model must beat the constant baseline on ranking when there is real signal
    assert metrics["test"]["gbm"]["auc_will_buy"] > 0.65
    # time protocol: validation snapshot is the last train snapshot, strictly before test
    assert metrics["data"]["validation_cutoff"] < metrics["data"]["test_cutoff"]
    test = synthetic_ds[synthetic_ds.split == "test"]
    out = predict_spend(artifact, test)
    assert out.shape == (len(test),) and (out >= 0).all() and np.isfinite(out).all()
