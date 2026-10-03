"""Loading and cleaning of the raw transactions (same rules as notebook 01)."""
from pathlib import Path

import pandas as pd

TRANSACTIONS_PATH = Path("data/processed/transactions.parquet")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows we cannot attribute to a customer or price; flag returns; add line_value."""
    d = df.dropna(subset=["customer_id"]).copy()
    d["customer_id"] = d["customer_id"].astype(int)
    d["invoice"] = d["invoice"].astype(str)
    d["is_return"] = d["invoice"].str.startswith("C")
    d = d[d["price"] > 0].copy()
    d["line_value"] = d["quantity"] * d["price"]
    return d


def load_clean(path: Path = TRANSACTIONS_PATH) -> pd.DataFrame:
    return clean_transactions(pd.read_parquet(path))
