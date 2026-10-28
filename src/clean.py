"""
Turn the raw transaction export into a trustworthy weekly fact table.

Five things happen here, in this order:

  1. parse       - three date formats into one real date column
  2. normalise   - branch names collapsed onto one canonical spelling
  3. match       - free-text product names resolved to catalogue SKUs
                   (rapidfuzz, with an auto-accept threshold and a review queue)
  4. repair      - duplicates removed, missing prices and revenue recomputed,
                   rows whose revenue disagrees with qty x price corrected,
                   returns split out instead of silently cancelling sales
  5. aggregate   - a clean weekly fact table at branch x SKU level

Everything it could not decide on its own is written to a review file rather
than guessed at, and every decision is counted in a data-quality report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

AUTO_ACCEPT = 88     # score at or above this is applied without a human
REVIEW_FLOOR = 72    # between the two, a human decides; below, unmatched

# abbreviations seen in the wild, expanded before matching
EXPANSIONS = {
    r"\bmin\.?\b": "mineral", r"\bsprkl\b": "sparkling", r"\bstl\b": "still",
    r"\bchoc\b": "chocolate", r"\bunslt\b": "unsalted", r"\bslt\b": "salted",
    r"\bshmp\b": "shampoo", r"\bt/paste\b": "toothpaste", r"\bdeo\b": "deodorant",
    r"\bsunfl\b": "sunflower", r"\blg\b": "long grain", r"\bspag\b": "spaghetti",
    r"\binst\b": "instant", r"\bclnr\b": "cleaner", r"\bppr twl\b": "paper towels",
    r"\blndry\b": "laundry", r"\bpwd\b": "powder", r"\bnorm\b": "normal hair",
    r"\bclsc\b": "classic", r"\bbtl\b": "bottled", r"\bhh\b": "household",
}


def norm_text(s: str) -> str:
    """Lowercase, expand abbreviations, unify units, strip punctuation."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    for pat, rep in EXPANSIONS.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"(\d)\s*(l|ml|g|kg|pcs|roll)\b", r"\1\2", s)   # "1.5 l" -> "1.5l"
    s = re.sub(r"[^a-z0-9.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_dates(col: pd.Series) -> pd.Series:
    """Parse ISO, dd/mm/yyyy and dd.mm.yyyy without silently mixing them up."""
    s = col.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    patterns = [
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{2}\.\d{2}\.\d{4}$", "%d.%m.%Y"),
    ]
    for pat, fmt in patterns:
        m = s.str.match(pat, na=False)
        out.loc[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    return out


def norm_branch(col: pd.Series) -> pd.Series:
    """'br.01', 'BRANCH-1', 'Branch1' -> 'Branch 01'."""
    digits = col.astype("string").str.extract(r"(\d+)", expand=False)
    return ("Branch " + digits.str.zfill(2)).astype("string")


def match_products(raw_names: pd.Series, master: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve each distinct raw product name to a catalogue SKU.

    Matching runs once per DISTINCT name, not once per row - 42k rows collapse
    to a few hundred distinct spellings, which is what makes this fast.
    """
    master = master.copy()
    master["norm"] = master["product_name"].map(norm_text)
    choices = master["norm"].tolist()

    distinct = (
        pd.DataFrame({"product_raw": raw_names.dropna().unique()})
        .assign(norm=lambda d: d["product_raw"].map(norm_text))
    )

    records = []
    for raw, nm in zip(distinct["product_raw"], distinct["norm"]):
        if not nm:
            records.append((raw, None, None, 0.0, "empty"))
            continue
        # token_set_ratio tolerates a dropped size token and reordered words
        hit = process.extractOne(nm, choices, scorer=fuzz.token_set_ratio)
        idx, score = hit[2], hit[1]
        row = master.iloc[idx]
        status = (
            "auto" if score >= AUTO_ACCEPT
            else "review" if score >= REVIEW_FLOOR
            else "unmatched"
        )
        records.append((raw, row["sku"], row["product_name"], float(score), status))

    return pd.DataFrame(
        records, columns=["product_raw", "sku", "matched_name", "match_score", "match_status"]
    )


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    txns = pd.read_csv(RAW / "sales_transactions.csv")
    master = pd.read_csv(RAW / "products_master.csv")
    report["rows_in"] = len(txns)

    # 1. duplicates --------------------------------------------------------
    before = len(txns)
    txns = txns.drop_duplicates(
        subset=["txn_date", "branch_raw", "product_raw", "qty", "unit_price", "revenue"]
    )
    report["duplicate_rows_removed"] = before - len(txns)

    # 2. dates -------------------------------------------------------------
    txns["txn_date"] = parse_dates(txns["txn_date"])
    report["unparseable_dates"] = int(txns["txn_date"].isna().sum())
    txns = txns.dropna(subset=["txn_date"])

    # 3. branches ----------------------------------------------------------
    txns["branch"] = norm_branch(txns["branch_raw"])
    report["branch_spellings_in"] = int(txns["branch_raw"].nunique())
    report["branches_out"] = int(txns["branch"].nunique())

    # 4. product matching --------------------------------------------------
    mapping = match_products(txns["product_raw"], master)
    report["distinct_product_spellings"] = int(len(mapping))
    counts = mapping["match_status"].value_counts().to_dict()
    report["name_match_auto"] = int(counts.get("auto", 0))
    report["name_match_review"] = int(counts.get("review", 0))
    report["name_match_unmatched"] = int(counts.get("unmatched", 0) + counts.get("empty", 0))

    txns = txns.merge(mapping, on="product_raw", how="left")

    matched = txns["match_status"].eq("auto")
    report["rows_auto_matched"] = int(matched.sum())
    report["row_auto_match_rate_pct"] = round(100 * matched.mean(), 1)

    # anything not auto-accepted leaves the pipeline and lands in a file a
    # human can work through - it is never guessed into the numbers
    review = txns.loc[~matched, ["txn_id", "product_raw", "matched_name",
                                 "match_score", "match_status"]]
    review.drop_duplicates(subset=["product_raw"]).sort_values(
        "match_score", ascending=False
    ).to_csv(OUT / "match_review_queue.csv", index=False)
    txns = txns[matched].copy()

    # 5. prices and revenue ------------------------------------------------
    txns = txns.merge(master[["sku", "list_price", "unit_cost", "category"]],
                      on="sku", how="left")

    no_price = txns["unit_price"].isna() & txns["revenue"].notna() & txns["qty"].ne(0)
    txns.loc[no_price, "unit_price"] = (
        txns.loc[no_price, "revenue"] / txns.loc[no_price, "qty"]
    ).round(2)
    report["prices_recovered_from_revenue"] = int(no_price.sum())

    still_no_price = txns["unit_price"].isna()
    txns.loc[still_no_price, "unit_price"] = txns.loc[still_no_price, "list_price"]
    report["prices_filled_from_catalogue"] = int(still_no_price.sum())

    expected = (txns["qty"] * txns["unit_price"]).round(2)
    no_rev = txns["revenue"].isna()
    txns.loc[no_rev, "revenue"] = expected[no_rev]
    report["revenue_recomputed_missing"] = int(no_rev.sum())

    # revenue that disagrees with qty x price by more than 1% is a typo in the
    # revenue field, not a price change - trust qty and price
    drift = (txns["revenue"] - expected).abs() > (expected.abs() * 0.01 + 0.02)
    txns.loc[drift, "revenue"] = expected[drift]
    report["revenue_corrected_mismatch"] = int(drift.sum())

    # 6. returns -----------------------------------------------------------
    txns["is_return"] = txns["qty"] < 0
    report["return_rows"] = int(txns["is_return"].sum())

    # 7. weekly fact table -------------------------------------------------
    txns["week_start"] = txns["txn_date"].dt.to_period("W-SUN").dt.start_time

    weekly = (
        txns.groupby(["week_start", "branch", "sku"], as_index=False)
        .agg(
            units_sold=("qty", lambda s: s[s > 0].sum()),
            units_returned=("qty", lambda s: -s[s < 0].sum()),
            revenue=("revenue", "sum"),
            txn_count=("txn_id", "count"),
            promo_weeks=("promo_flag", "max"),
            avg_price=("unit_price", "mean"),
        )
    )
    weekly["units_returned"] = weekly["units_returned"].fillna(0).astype(int)
    weekly["units_net"] = weekly["units_sold"] - weekly["units_returned"]
    weekly = weekly.merge(master[["sku", "product_name", "category", "unit_cost"]],
                          on="sku", how="left")
    weekly["gross_margin"] = (weekly["revenue"] - weekly["units_net"] * weekly["unit_cost"]).round(2)

    # fill the calendar so a week with no sales is a zero, not a missing row -
    # a forecast model has to see the zeros
    full = (
        pd.MultiIndex.from_product(
            [sorted(weekly["week_start"].unique()),
             sorted(weekly["branch"].unique()),
             sorted(weekly["sku"].unique())],
            names=["week_start", "branch", "sku"],
        ).to_frame(index=False)
    )
    weekly = full.merge(weekly, on=["week_start", "branch", "sku"], how="left")
    num = ["units_sold", "units_returned", "units_net", "revenue",
           "txn_count", "promo_weeks", "gross_margin"]
    weekly[num] = weekly[num].fillna(0)
    weekly = weekly.drop(columns=["product_name", "category", "unit_cost"]).merge(
        master[["sku", "product_name", "category", "unit_cost", "list_price"]],
        on="sku", how="left"
    )
    weekly["avg_price"] = weekly["avg_price"].fillna(weekly["list_price"])

    # the promotion plan is a separate, clean source - join it on so the
    # forecast can see next week's planned discount, not just last week's
    promo = pd.read_csv(RAW / "promo_calendar.csv")
    promo["week_start"] = pd.to_datetime(promo["week_start"])
    promo["week_start"] = promo["week_start"].dt.to_period("W-SUN").dt.start_time
    promo = promo.groupby(["week_start", "sku"], as_index=False)["discount_pct"].max()
    weekly = weekly.merge(promo, on=["week_start", "sku"], how="left")
    weekly["discount_pct"] = weekly["discount_pct"].fillna(0)
    weekly["is_promo"] = (weekly["discount_pct"] > 0).astype(int)
    report["promo_weeks_in_calendar"] = int((weekly["is_promo"] == 1).sum())

    report["weekly_rows_out"] = len(weekly)
    report["weeks"] = int(weekly["week_start"].nunique())
    report["skus"] = int(weekly["sku"].nunique())
    report["revenue_total"] = round(float(weekly["revenue"].sum()), 2)

    weekly.to_parquet(OUT / "weekly_sales.parquet", index=False)
    txns.to_parquet(OUT / "transactions_clean.parquet", index=False)
    mapping.to_csv(OUT / "product_name_mapping.csv", index=False)
    (OUT / "data_quality_report.json").write_text(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    r = run()
    width = max(len(k) for k in r)
    print("Data quality report")
    print("-" * (width + 16))
    for k, v in r.items():
        print(f"{k:<{width}}  {v:>12,}" if isinstance(v, (int, float)) else f"{k:<{width}}  {v}")
