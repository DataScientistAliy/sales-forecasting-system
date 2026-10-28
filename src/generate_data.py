"""
Generate a deliberately messy synthetic sales dataset.

Produces:
  data/raw/sales_transactions.csv  - dirty transaction log, as exported from a POS/ERP
  data/raw/products_master.csv     - the clean product catalogue
  data/raw/inventory_snapshot.csv  - current stock on hand per branch/SKU

The mess is intentional and mirrors what real exports look like:
inconsistent product spellings, three different date formats, missing
prices, exact duplicate rows, returns as negative quantities, and
branch names typed a different way every week.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng_np = np.random.default_rng(SEED)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
WEEKS = 156
START = pd.Timestamp("2023-10-02")  # a Monday

CATEGORIES = {
    "Beverages": [
        ("Mineral Water Still 1.5L", 6),
        ("Mineral Water Sparkling 1.5L", 6),
        ("Orange Juice 1L", 12),
        ("Apple Juice 1L", 12),
        ("Green Tea Bottled 0.5L", 24),
        ("Cola Classic 0.5L", 24),
        ("Energy Drink 0.25L", 24),
    ],
    "Dairy": [
        ("Whole Milk 1L", 12),
        ("Skimmed Milk 1L", 12),
        ("Greek Yogurt 400g", 8),
        ("Butter Unsalted 200g", 20),
        ("Cheese Gouda 250g", 10),
        ("Sour Cream 300g", 12),
    ],
    "Snacks": [
        ("Potato Chips Salted 150g", 20),
        ("Potato Chips Paprika 150g", 20),
        ("Salted Peanuts 200g", 16),
        ("Chocolate Bar Dark 100g", 24),
        ("Chocolate Bar Milk 100g", 24),
        ("Biscuits Butter 300g", 12),
        ("Crackers Sesame 200g", 16),
    ],
    "Household": [
        ("Dish Soap 500ml", 12),
        ("Laundry Powder 3kg", 4),
        ("Glass Cleaner 750ml", 12),
        ("Paper Towels 2-roll", 10),
        ("Trash Bags 30L 20pcs", 20),
        ("Floor Cleaner 1L", 12),
    ],
    "Personal Care": [
        ("Shampoo Normal Hair 400ml", 12),
        ("Shower Gel Fresh 400ml", 12),
        ("Toothpaste Mint 100ml", 24),
        ("Hand Soap 300ml", 12),
        ("Deodorant Spray 150ml", 18),
        ("Razor Blades 4pcs", 20),
    ],
    "Pantry": [
        ("Sunflower Oil 1L", 12),
        ("Rice Long Grain 1kg", 10),
        ("Pasta Spaghetti 500g", 20),
        ("Sugar White 1kg", 10),
        ("Salt Fine 1kg", 20),
        ("Tomato Paste 400g", 24),
        ("Black Tea Loose 200g", 16),
        ("Instant Coffee 100g", 12),
    ],
}

BRANCH_CANON = ["Branch 01", "Branch 02", "Branch 03", "Branch 04", "Branch 05"]
# how a human might type the same branch
BRANCH_VARIANTS = {
    "Branch 01": ["Branch 01", "branch 1", "BRANCH-01", "Br.01", "Branch1"],
    "Branch 02": ["Branch 02", "branch 2", "BRANCH-02", "Br.02", "Branch2"],
    "Branch 03": ["Branch 03", "branch 3", "BRANCH-03", "Br.03", "Branch3"],
    "Branch 04": ["Branch 04", "branch 4", "BRANCH-04", "Br.04", "Branch4"],
    "Branch 05": ["Branch 05", "branch 5", "BRANCH-05", "Br.05", "Branch5"],
}

ABBREV = {
    "Mineral": "Min.", "Sparkling": "Sprkl", "Still": "Stl",
    "Chocolate": "Choc", "Unsalted": "Unslt", "Salted": "Slt",
    "Shampoo": "Shmp", "Toothpaste": "T/paste", "Deodorant": "Deo",
    "Sunflower": "Sunfl", "Long Grain": "LG", "Spaghetti": "Spag",
    "Instant": "Inst", "Household": "HH", "Cleaner": "Clnr",
    "Paper Towels": "Ppr Twl", "Laundry": "Lndry", "Powder": "Pwd",
    "Normal Hair": "Norm", "Classic": "Clsc", "Bottled": "Btl",
}


def _mangle(name: str, rng: random.Random) -> str:
    """Return a plausible human-typed variant of a clean product name."""
    s = name
    style = rng.random()

    if style < 0.30:                                  # abbreviate a word
        for full, short in ABBREV.items():
            if full in s and rng.random() < 0.6:
                s = s.replace(full, short)
                break
    if style < 0.55:                                  # case noise
        s = rng.choice([s.upper(), s.lower(), s.title()])
    if rng.random() < 0.25:                           # drop the size token
        parts = s.split()
        if len(parts) > 2:
            s = " ".join(parts[:-1])
    if rng.random() < 0.20:                           # single-character typo
        i = rng.randrange(len(s))
        if s[i].isalpha():
            s = s[:i] + rng.choice("aeiorstl") + s[i + 1:]
    if rng.random() < 0.20:                           # punctuation noise
        s = s.replace(" ", rng.choice(["  ", " - ", "_"]), 1)
    if rng.random() < 0.15:                           # stray whitespace
        s = rng.choice(["  ", " "]) + s + rng.choice(["", " ", "   "])
    return s


def build_master() -> pd.DataFrame:
    rows = []
    sku_n = 1000
    for cat, items in CATEGORIES.items():
        for name, pack in items:
            sku_n += 7
            base = round(rng_np.uniform(1.2, 9.5), 2)
            rows.append(
                {
                    "sku": f"SKU-{sku_n}",
                    "product_name": name,
                    "category": cat,
                    "pack_size": pack,
                    "unit_cost": base,
                    "list_price": round(base * rng_np.uniform(1.25, 1.75), 2),
                }
            )
    return pd.DataFrame(rows)


def build_promo_calendar(master: pd.DataFrame) -> pd.DataFrame:
    """The promotion plan, as marketing sets it weeks ahead.

    This is the single most valuable input a demand forecast can have, and
    almost every client already keeps it in a spreadsheet.
    """
    rng = random.Random(SEED + 99)
    rows = []
    for w in range(WEEKS):
        week_start = START + pd.Timedelta(weeks=w)
        # a handful of SKUs are promoted each week
        for sku in rng.sample(list(master["sku"]), k=rng.randint(2, 6)):
            rows.append(
                {
                    "week_start": week_start.strftime("%Y-%m-%d"),
                    "sku": sku,
                    "discount_pct": rng.choice([10, 15, 20, 25, 30, 35]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["week_start", "sku"])


def build_transactions(master: pd.DataFrame, promo: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    rows = []
    promo_map = {
        (r.week_start, r.sku): r.discount_pct for r in promo.itertuples()
    }

    # per-SKU demand character
    prof = {}
    for _, p in master.iterrows():
        prof[p["sku"]] = {
            "base": rng_np.uniform(8, 70),
            "trend": rng_np.uniform(-0.0035, 0.0065),
            "season_amp": rng_np.uniform(0.05, 0.35),
            "season_shift": rng_np.uniform(0, 2 * np.pi),
            "noise": rng_np.uniform(0.08, 0.20),
            "elasticity": rng_np.uniform(2.5, 4.5),
        }
    branch_mult = {b: rng_np.uniform(0.55, 1.5) for b in BRANCH_CANON}

    for w in range(WEEKS):
        week_start = START + pd.Timedelta(weeks=w)
        for _, p in master.iterrows():
            sku, pr = p["sku"], prof[p["sku"]]
            season = 1 + pr["season_amp"] * np.sin(2 * np.pi * w / 52 + pr["season_shift"])
            level = pr["base"] * (1 + pr["trend"]) ** w * season

            # December trading peak - knowable from the calendar
            if week_start.month == 12:
                level *= 1.30
            elif week_start.month == 1:
                level *= 0.88

            disc = promo_map.get((week_start.strftime("%Y-%m-%d"), sku), 0)
            is_promo = disc > 0
            # price elasticity: a 25% cut roughly doubles the units
            uplift = 1 + (disc / 100.0) * pr["elasticity"] if is_promo else 1.0

            for b in BRANCH_CANON:
                mult = branch_mult[b] * uplift
                qty = max(0, int(rng_np.normal(level * mult, level * pr["noise"])))
                if qty == 0:
                    continue

                # split the week into 1-3 delivery days
                for _ in range(rng.randint(1, 3)):
                    q = max(1, int(qty / rng.randint(1, 3)))
                    day = week_start + pd.Timedelta(days=rng.randint(0, 6))
                    price = p["list_price"] * (
                        (1 - disc / 100.0) if is_promo else rng_np.uniform(0.97, 1.03)
                    )
                    rows.append(
                        {
                            "txn_date": day,
                            "branch_raw": rng.choice(BRANCH_VARIANTS[b]),
                            "product_raw": _mangle(p["product_name"], rng),
                            "qty": q,
                            "unit_price": round(price, 2),
                            "promo_flag": int(is_promo),
                        }
                    )

    df = pd.DataFrame(rows)

    # ---- inject the mess -------------------------------------------------
    n = len(df)
    df["revenue"] = (df["qty"] * df["unit_price"]).round(2)

    # 1. returns recorded as negative quantities
    ret = df.sample(frac=0.012, random_state=SEED).index
    df.loc[ret, "qty"] *= -1
    df.loc[ret, "revenue"] *= -1

    # 2. missing unit_price (revenue still present)
    miss_p = df.sample(frac=0.025, random_state=SEED + 1).index
    df.loc[miss_p, "unit_price"] = np.nan

    # 3. missing revenue (price still present)
    miss_r = df.sample(frac=0.018, random_state=SEED + 2).index
    df.loc[miss_r, "revenue"] = np.nan

    # 4. revenue that disagrees with qty * price (data-entry error)
    bad = df.sample(frac=0.009, random_state=SEED + 3).index
    df.loc[bad, "revenue"] = (df.loc[bad, "revenue"] * rng_np.uniform(1.5, 4.0, len(bad))).round(2)

    # 5. three different date formats, as text
    fmt = rng_np.choice(["iso", "dmy_slash", "dmy_dot"], size=n, p=[0.6, 0.25, 0.15])
    out = []
    for d, f in zip(df["txn_date"], fmt):
        if f == "iso":
            out.append(d.strftime("%Y-%m-%d"))
        elif f == "dmy_slash":
            out.append(d.strftime("%d/%m/%Y"))
        else:
            out.append(d.strftime("%d.%m.%Y"))
    df["txn_date"] = out

    # 6. exact duplicate rows
    dup = df.sample(frac=0.015, random_state=SEED + 4)
    df = pd.concat([df, dup], ignore_index=True)

    # 7. a few completely empty product names
    blank = df.sample(frac=0.004, random_state=SEED + 5).index
    df.loc[blank, "product_raw"] = rng_np.choice(["", " ", "N/A", "???"], size=len(blank))

    df["txn_id"] = [f"T{i:07d}" for i in range(len(df))]
    df = df.sample(frac=1.0, random_state=SEED + 6).reset_index(drop=True)
    return df[["txn_id", "txn_date", "branch_raw", "product_raw", "qty",
               "unit_price", "revenue", "promo_flag"]]


def build_inventory(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BRANCH_CANON:
        for _, p in master.iterrows():
            rows.append(
                {
                    "branch": b,
                    "sku": p["sku"],
                    "on_hand_units": int(max(0, rng_np.normal(140, 90))),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)

    master = build_master()
    promo = build_promo_calendar(master)
    txns = build_transactions(master, promo)
    inv = build_inventory(master)

    master.to_csv(RAW / "products_master.csv", index=False)
    promo.to_csv(RAW / "promo_calendar.csv", index=False)
    txns.to_csv(RAW / "sales_transactions.csv", index=False)
    inv.to_csv(RAW / "inventory_snapshot.csv", index=False)

    print(f"products_master.csv     {len(master):>7,} rows")
    print(f"promo_calendar.csv      {len(promo):>7,} rows")
    print(f"sales_transactions.csv  {len(txns):>7,} rows")
    print(f"inventory_snapshot.csv  {len(inv):>7,} rows")
    print()
    print("Sample of the raw transaction log:")
    print(txns.head(8).to_string(index=False))
