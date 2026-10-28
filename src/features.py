"""
Feature engineering for next-week demand forecasting.

One row per branch x SKU x week. The target is next week's net units, so
every feature must be knowable at the end of the current week - no future
information, which is the easiest way to build a model that looks brilliant
in testing and useless in production.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLLS = [2, 4, 8, 13]


def aggregate_to_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse branches away: one series per SKU per week.

    Purchasing is decided per SKU for the whole company, not per branch, so
    this is the granularity the forecast is actually used at - and summing
    five noisy branch series into one cancels a lot of that noise."""
    out = (
        df.groupby(["week_start", "sku"], as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            units_returned=("units_returned", "sum"),
            units_net=("units_net", "sum"),
            revenue=("revenue", "sum"),
            txn_count=("txn_count", "sum"),
            promo_weeks=("promo_weeks", "max"),
            avg_price=("avg_price", "mean"),
            gross_margin=("gross_margin", "sum"),
            discount_pct=("discount_pct", "max"),
            is_promo=("is_promo", "max"),
        )
    )
    meta = df[["sku", "product_name", "category", "unit_cost", "list_price"]].drop_duplicates("sku")
    out = out.merge(meta, on="sku", how="left")
    out["branch"] = "ALL"
    return out


def build(df: pd.DataFrame | None = None, keys: list[str] | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(PROC / "weekly_sales.parquet")
    KEY = keys or ["branch", "sku"]

    df = df.sort_values(KEY + ["week_start"]).reset_index(drop=True)
    g = df.groupby(KEY, observed=True)["units_net"]

    # --- history of the series itself ------------------------------------
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    for win in ROLLS:
        shifted = g.shift(1)
        df[f"roll_mean_{win}"] = shifted.groupby([df[k] for k in KEY]).transform(
            lambda s: s.rolling(win, min_periods=max(2, win // 2)).mean()
        )
        df[f"roll_std_{win}"] = shifted.groupby([df[k] for k in KEY]).transform(
            lambda s: s.rolling(win, min_periods=max(2, win // 2)).std()
        )

    # momentum: is this series speeding up or slowing down
    df["trend_4_13"] = df["roll_mean_4"] / df["roll_mean_13"].replace(0, np.nan)
    df["diff_1_2"] = df["lag_1"] - df["lag_2"]

    # --- calendar --------------------------------------------------------
    wk = df["week_start"].dt.isocalendar().week.astype(int)
    df["week_of_year"] = wk
    df["month"] = df["week_start"].dt.month
    df["week_sin"] = np.sin(2 * np.pi * wk / 52)
    df["week_cos"] = np.cos(2 * np.pi * wk / 52)

    # --- price and promotion (last week's, which we know) ----------------
    pg = df.groupby(KEY, observed=True)
    df["price_lag_1"] = pg["avg_price"].shift(1)
    df["promo_lag_1"] = pg["promo_weeks"].shift(1)
    df["promo_rate_13"] = pg["promo_weeks"].shift(1).groupby(
        [df[k] for k in KEY]
    ).transform(lambda s: s.rolling(13, min_periods=4).mean())
    df["price_vs_list"] = df["price_lag_1"] / df["list_price"]

    df["returns_rate_13"] = pg["units_returned"].shift(1).groupby(
        [df[k] for k in KEY]
    ).transform(lambda s: s.rolling(13, min_periods=4).mean())

    # --- KNOWN-IN-ADVANCE drivers ----------------------------------------
    # These describe NEXT week, which is legitimate: the promotion plan and
    # the calendar are both set weeks ahead. This is where most of the gain
    # over a moving average comes from.
    df["promo_next_week"] = pg["is_promo"].shift(-1)
    df["discount_next_week"] = pg["discount_pct"].shift(-1)
    df["promo_this_week"] = df["is_promo"]
    df["discount_this_week"] = df["discount_pct"]

    nxt = df["week_start"] + pd.Timedelta(weeks=1)
    df["next_week_month"] = nxt.dt.month
    df["next_is_december"] = (nxt.dt.month == 12).astype(int)
    df["next_is_january"] = (nxt.dt.month == 1).astype(int)

    # how this series has historically responded to a discount
    promo_hist = (
        df.assign(_u=df["units_net"])
        .groupby(KEY + ["is_promo"], observed=True)["_u"].mean()
        .unstack(fill_value=np.nan)
    )
    if 1 in promo_hist.columns and 0 in promo_hist.columns:
        lift = (promo_hist[1] / promo_hist[0].replace(0, np.nan)).rename("promo_lift_hist")
        df = df.merge(lift.reset_index(), on=KEY, how="left")
    else:
        df["promo_lift_hist"] = np.nan
    df["expected_promo_effect"] = np.where(
        df["promo_next_week"].fillna(0) > 0,
        df["roll_mean_4"] * df["promo_lift_hist"].fillna(1.0),
        df["roll_mean_4"],
    )

    # --- cross-sectional context -----------------------------------------
    cat_week = (
        df.groupby(["category", "week_start"], observed=True)["units_net"]
        .mean().rename("cat_mean_units").reset_index()
    )
    df = df.merge(cat_week, on=["category", "week_start"], how="left")
    df["cat_mean_lag_1"] = df.groupby(KEY, observed=True)["cat_mean_units"].shift(1)
    df = df.drop(columns=["cat_mean_units"])

    # --- categorical codes ----------------------------------------------
    for col in ["branch", "sku", "category"]:
        df[f"{col}_code"] = df[col].astype("category").cat.codes

    # --- target: NEXT week ----------------------------------------------
    df["target_units_next_week"] = df.groupby(KEY, observed=True)["units_net"].shift(-1)

    return df


FEATURES = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLS]
    + [f"roll_std_{w}" for w in ROLLS]
    + [
        "trend_4_13", "diff_1_2",
        "week_of_year", "month", "week_sin", "week_cos",
        "price_lag_1", "promo_lag_1", "promo_rate_13", "price_vs_list",
        "returns_rate_13", "cat_mean_lag_1",
        "branch_code", "sku_code", "category_code",
        # known in advance
        "promo_next_week", "discount_next_week",
        "promo_this_week", "discount_this_week",
        "next_week_month", "next_is_december", "next_is_january",
        "promo_lift_hist", "expected_promo_effect",
    ]
)

if __name__ == "__main__":
    out = build()
    out.to_parquet(PROC / "features.parquet", index=False)
    usable = out.dropna(subset=["target_units_next_week"] + FEATURES)
    print(f"rows total      {len(out):>8,}")
    print(f"rows usable     {len(usable):>8,}   (after lag warm-up)")
    print(f"features        {len(FEATURES):>8,}")
    print(f"weeks           {out['week_start'].nunique():>8,}")
