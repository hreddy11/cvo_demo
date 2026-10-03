import numpy as np
import pandas as pd
import pytest

from cvo_demo.data import clean_transactions
from cvo_demo.features import FEATURE_COLUMNS, TARGET, build_features, build_snapshot

CUTOFF = pd.Timestamp("2011-03-01")


@pytest.fixture(scope="module")
def clean():
    rng = np.random.default_rng(1)
    rows = []
    inv = 600000
    for cid in range(12000, 12300):
        n_orders = rng.integers(1, 15)
        for d in sorted(rng.integers(0, 600, n_orders)):
            inv += 1
            date = pd.Timestamp("2010-01-01") + pd.Timedelta(days=int(d), hours=10)
            for _ in range(rng.integers(1, 4)):
                rows.append((str(inv), str(rng.integers(1000, 1050)), "x", int(rng.integers(1, 10)), date,
                             float(rng.choice([1.5, 3.0, 8.0])), float(cid), "United Kingdom"))
            if rng.random() < 0.1:  # a return
                rows.append(("C" + str(inv), "1001", "x", -1, date + pd.Timedelta(days=2), 3.0, float(cid), "United Kingdom"))
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "description", "quantity",
                                     "invoice_date", "price", "customer_id", "country"])
    return clean_transactions(df)


def test_columns_and_population(clean):
    X = build_features(clean, CUTOFF)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.is_unique
    # only customers who bought before the cutoff
    first = clean[~clean.is_return].groupby("customer_id").invoice_date.min()
    assert set(X.index) == set(first[first < CUTOFF].index)


def test_no_future_leakage(clean):
    """Features must not change if we corrupt every row on/after the cutoff."""
    X1 = build_features(clean, CUTOFF)
    corrupted = clean.copy()
    fut = corrupted.invoice_date >= CUTOFF
    corrupted.loc[fut, "quantity"] *= 100
    corrupted.loc[fut, "line_value"] *= 100
    X2 = build_features(corrupted, CUTOFF)
    pd.testing.assert_frame_equal(X1, X2)


def test_target_only_uses_future_window(clean):
    snap = build_snapshot(clean, CUTOFF, 90).set_index("customer_id")
    end = CUTOFF + pd.Timedelta(days=90)
    fut = clean[(clean.invoice_date >= CUTOFF) & (clean.invoice_date < end) & ~clean.is_return & (clean.quantity > 0)]
    expected = fut.groupby("customer_id").line_value.sum()
    got = snap[TARGET][snap[TARGET] > 0]
    assert got.index.isin(expected.index).all()
    np.testing.assert_allclose(got.sort_index().values, expected.reindex(got.index).sort_index().values)


def test_sane_ranges(clean):
    X = build_features(clean, CUTOFF)
    assert (X["recency_days"] >= 0).all() and (X["tenure_days"] >= X["recency_days"]).all()
    assert X["return_rate"].between(0, 1).all()
    assert (X["frequency"] >= 1).all()
