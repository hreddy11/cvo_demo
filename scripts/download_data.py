"""Download UCI Online Retail II and convert to a local parquet file.

Usage:
    python scripts/download_data.py                        # full dataset (~1M rows, fine on a Mac)
    python scripts/download_data.py --max-customers 2000   # shrink to a random customer subset
"""
import argparse, io, zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW = Path("data/raw")
OUT = Path("data/processed/transactions.parquet")


def main(max_customers, seed):
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    xlsx = RAW / "online_retail_II.xlsx"

    if not xlsx.exists():
        print(f"Downloading {URL} ...")
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
            xlsx.write_bytes(z.read(name))
    print("Reading Excel (both sheets, takes ~1-2 min)...")
    sheets = pd.read_excel(xlsx, sheet_name=None)  # dict of DataFrames
    df = pd.concat(sheets.values(), ignore_index=True)

    df = df.rename(columns={
        "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
        "Customer ID": "customer_id", "Country": "country",
    })
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["description"] = df["description"].astype(str)

    if max_customers:
        ids = pd.Series(df["customer_id"].dropna().unique())
        keep = ids.sample(min(max_customers, len(ids)), random_state=seed)
        df = df[df["customer_id"].isin(keep)]

    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df):,} rows, {df['customer_id'].nunique():,} customers -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-customers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.max_customers, a.seed)
