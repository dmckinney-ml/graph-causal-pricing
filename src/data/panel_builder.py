"""
Build the store × product × week panel.

Observation unit: (STORE_ID, PRODUCT_ID, WEEK_NO)

Outcome  : units_sold = sum(QUANTITY); sold_any = (units_sold > 0)
Treatments:
  promo_any      – binary: display=1 OR mailer=1  (primary)
  display        – binary (follow-up)
  mailer         – binary (follow-up)
  discount_rate  – continuous RETAIL_DISC / (SALES_VALUE + RETAIL_DISC) (follow-up)

Zero-unit rows are INCLUDED (full Cartesian product of observed
store × product × week); missing sales are filled with 0.
"""

from __future__ import annotations

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _load_causal(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Treat anything non-zero / non-"0" as promotion active
    for col in ("display", "mailer"):
        df[col] = (df[col].astype(str).str.strip() != "0").astype(int)
    return df


# ── aggregation ──────────────────────────────────────────────────────────────

def _aggregate_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Sum sales per (STORE_ID, PRODUCT_ID, WEEK_NO)."""
    agg = (
        tx.groupby(["STORE_ID", "PRODUCT_ID", "WEEK_NO"], observed=True)
        .agg(
            units_sold=("QUANTITY", "sum"),
            sales_value=("SALES_VALUE", "sum"),
            retail_disc_total=("RETAIL_DISC", "sum"),   # stored as negatives
        )
        .reset_index()
    )
    # discount_rate: |retail_disc| / (sales_value + |retail_disc|)
    gross = agg["sales_value"] + agg["retail_disc_total"].abs()
    agg["discount_rate"] = agg["retail_disc_total"].abs() / gross.replace(0, pd.NA)
    agg["discount_rate"] = agg["discount_rate"].fillna(0.0)
    return agg


def _build_full_index(tx_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Cartesian product of (STORE_ID, PRODUCT_ID) pairs × all weeks where
    the product appears in any store.  This ensures zero-unit rows are
    included when a product was available but not purchased.
    """
    # Weeks where each product was ever sold (defines its 'active' window)
    product_week_range = (
        tx_agg.groupby("PRODUCT_ID", observed=True)["WEEK_NO"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "week_min", "max": "week_max"})
    )

    # Stores where each product was ever sold
    product_stores = (
        tx_agg[["PRODUCT_ID", "STORE_ID"]]
        .drop_duplicates()
    )

    # For each product-store pair, expand over every week in the product's range
    rows = []
    for _, row in product_week_range.iterrows():
        pid = row["PRODUCT_ID"]
        weeks = range(int(row["week_min"]), int(row["week_max"]) + 1)
        stores = product_stores.loc[
            product_stores["PRODUCT_ID"] == pid, "STORE_ID"
        ].values
        for s in stores:
            for w in weeks:
                rows.append((s, pid, w))

    index_df = pd.DataFrame(rows, columns=["STORE_ID", "PRODUCT_ID", "WEEK_NO"])
    return index_df


# ── public API ────────────────────────────────────────────────────────────────

def build_panel(
    transaction_path: str,
    causal_path: str,
    product_path: str,
) -> pd.DataFrame:
    """
    Build the full store × product × week panel.

    Parameters
    ----------
    transaction_path : path to transaction_data.csv
    causal_path      : path to causal_data.csv
    product_path     : path to product.csv

    Returns
    -------
    pd.DataFrame with one row per (STORE_ID, PRODUCT_ID, WEEK_NO).
    All zero-unit observations are retained; `sold_any` flags non-zero rows.
    """
    tx = _load_transactions(transaction_path)
    causal = _load_causal(causal_path)
    product = pd.read_csv(product_path)
    product.columns = product.columns.str.strip()

    # Aggregate transactions to store × product × week
    tx_agg = _aggregate_transactions(tx)

    # Full Cartesian index (includes zero-unit cells)
    index_df = _build_full_index(tx_agg)

    # Left-join actual sales onto the full index
    panel = index_df.merge(tx_agg, on=["STORE_ID", "PRODUCT_ID", "WEEK_NO"], how="left")
    panel["units_sold"] = panel["units_sold"].fillna(0.0)
    panel["sales_value"] = panel["sales_value"].fillna(0.0)
    panel["retail_disc_total"] = panel["retail_disc_total"].fillna(0.0)
    panel["discount_rate"] = panel["discount_rate"].fillna(0.0)
    panel["sold_any"] = (panel["units_sold"] > 0).astype(int)

    # Join causal (treatment) flags
    causal["promo_any"] = ((causal["display"] == 1) | (causal["mailer"] == 1)).astype(int)
    panel = panel.merge(
        causal[["STORE_ID", "PRODUCT_ID", "WEEK_NO", "display", "mailer", "promo_any"]],
        on=["STORE_ID", "PRODUCT_ID", "WEEK_NO"],
        how="left",
    )
    for col in ("display", "mailer", "promo_any"):
        panel[col] = panel[col].fillna(0).astype(int)

    # Join product hierarchy features
    hierarchy_cols = ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]
    panel = panel.merge(product[hierarchy_cols], on="PRODUCT_ID", how="left")

    # Tidy dtypes
    panel = panel.astype({"STORE_ID": int, "PRODUCT_ID": int, "WEEK_NO": int})
    panel = panel.sort_values(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).reset_index(drop=True)

    return panel
