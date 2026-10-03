"""Time-safe customer features.

Rule: every feature for a snapshot date `cutoff` is computed ONLY from rows with
invoice_date < cutoff. The target looks at [cutoff, cutoff + horizon). The same
`build_features` function is used offline (training) and to produce the serving table,
so there is no train/serve skew in the feature definitions.
"""
import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
WINDOWS = (30, 90, 180)

FEATURE_COLUMNS = [
    "recency_days", "tenure_days", "frequency", "monetary_total", "avg_order_value",
    "max_order_value", "std_order_value", "items_total", "avg_gap_days", "overdue_ratio",
    "orders_per_month", "n_unique_products",
    "orders_30d", "orders_90d", "orders_180d",
    "spend_30d", "spend_90d", "spend_180d",
    "spend_trend_90_vs_prior", "return_rate", "return_value_share", "is_uk",
]
TARGET = "target_spend_90d"


def _sales(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["is_return"] & (df["quantity"] > 0)]


def build_features(clean: pd.DataFrame, cutoff) -> pd.DataFrame:
    """One row per customer with >=1 sale before `cutoff`. Index: customer_id."""
    cutoff = pd.Timestamp(cutoff)
    hist = clean[clean["invoice_date"] < cutoff]
    sales = _sales(hist)
    returns = hist[hist["is_return"]]

    orders = (
        sales.groupby(["customer_id", "invoice"])
        .agg(order_date=("invoice_date", "min"), order_value=("line_value", "sum"),
             n_items=("quantity", "sum"))
        .reset_index()
    )
    g = orders.groupby("customer_id")
    f = g.agg(
        first_order=("order_date", "min"), last_order=("order_date", "max"),
        frequency=("invoice", "nunique"), monetary_total=("order_value", "sum"),
        avg_order_value=("order_value", "mean"), max_order_value=("order_value", "max"),
        std_order_value=("order_value", "std"), items_total=("n_items", "sum"),
    )
    f["recency_days"] = (cutoff - f["last_order"]) / DAY
    f["tenure_days"] = (cutoff - f["first_order"]) / DAY
    # NaN for one-time buyers (no gap defined); tree models handle NaN natively.
    f["avg_gap_days"] = ((f["last_order"] - f["first_order"]) / DAY) / (f["frequency"] - 1).replace(0, np.nan)
    f["overdue_ratio"] = f["recency_days"] / f["avg_gap_days"]
    f["orders_per_month"] = f["frequency"] / np.maximum(f["tenure_days"] / 30.0, 1.0)
    f["n_unique_products"] = sales.groupby("customer_id")["stock_code"].nunique()
    f["std_order_value"] = f["std_order_value"].fillna(0.0)

    for w in WINDOWS:
        recent = orders[orders["order_date"] >= cutoff - w * DAY]
        rg = recent.groupby("customer_id")
        f[f"orders_{w}d"] = rg["invoice"].nunique().reindex(f.index).fillna(0)
        f[f"spend_{w}d"] = rg["order_value"].sum().reindex(f.index).fillna(0.0)

    prior = orders[(orders["order_date"] < cutoff - 90 * DAY) & (orders["order_date"] >= cutoff - 180 * DAY)]
    prior_spend = prior.groupby("customer_id")["order_value"].sum().reindex(f.index).fillna(0.0)
    f["spend_trend_90_vs_prior"] = (f["spend_90d"] - prior_spend) / (prior_spend + 1.0)

    n_ret = returns.groupby("customer_id")["invoice"].nunique().reindex(f.index).fillna(0)
    ret_val = returns.groupby("customer_id")["line_value"].sum().abs().reindex(f.index).fillna(0.0)
    f["return_rate"] = n_ret / (n_ret + f["frequency"])
    f["return_value_share"] = ret_val / (ret_val + f["monetary_total"])

    last_country = sales.sort_values("invoice_date").groupby("customer_id")["country"].last()
    f["is_uk"] = (last_country.reindex(f.index) == "United Kingdom").astype(int)

    out = f[FEATURE_COLUMNS].astype("float64")
    out.index.name = "customer_id"
    return out


def build_target(clean: pd.DataFrame, cutoff, horizon_days: int = 90) -> pd.Series:
    """Gross sales value in [cutoff, cutoff + horizon). Customers with no purchase are absent (-> 0)."""
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + horizon_days * DAY
    fut = _sales(clean[(clean["invoice_date"] >= cutoff) & (clean["invoice_date"] < end)])
    return fut.groupby("customer_id")["line_value"].sum().rename(TARGET)


def build_snapshot(clean: pd.DataFrame, cutoff, horizon_days: int = 90) -> pd.DataFrame:
    """Features as of cutoff + target over the next `horizon_days`. One row per customer."""
    X = build_features(clean, cutoff)
    y = build_target(clean, cutoff, horizon_days).reindex(X.index).fillna(0.0)
    snap = X.join(y)
    snap["target_will_buy"] = (snap[TARGET] > 0).astype(int)
    snap.insert(0, "snapshot_date", pd.Timestamp(cutoff))
    return snap.reset_index()
