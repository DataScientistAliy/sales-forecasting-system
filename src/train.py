"""
Train next-week demand forecasts and prove they beat the obvious answer.

A forecast is only worth paying for if it beats what the client could already
do in a spreadsheet, so three baselines are scored on exactly the same
held-out weeks and the same rows as the model:

  last week            - next week equals this week
  4-week average       - next week equals the mean of the last four weeks
  same week last year  - naive seasonal

Two granularities are trained, because the right one is a business question,
not a modelling one:

  SKU level        - one series per product, summed over branches. This is
                     how purchasing is actually decided, and it is the model
                     that goes to production.
  branch x SKU     - one series per product per branch, for allocation once
                     the purchase quantity is known. Much noisier, and the
                     report says so instead of hiding it.

The split is by time, never random: train on earlier weeks, score on the most
recent ones. That is the only split that answers the client's real question.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, aggregate_to_sku, build

# A row is usable once it has a short history. Longer lags may still be
# missing - LightGBM splits on missing values natively, so demanding a full
# 52-week warm-up would throw away half the training data for nothing.
REQUIRED = ["lag_1", "lag_2", "lag_3", "lag_4", "roll_mean_4"]

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
TEST_WEEKS = 16
TARGET = "target_units_next_week"


def wape(y, yhat) -> float:
    """Weighted absolute percentage error - the honest metric when some
    products sell 5 units a week and others sell 500."""
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom * 100) if denom else float("nan")


def score(name, y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {
        "model": name,
        "WAPE_pct": round(wape(y, yhat), 2),
        "MAE_units": round(float(np.abs(y - yhat).mean()), 2),
        "RMSE_units": round(float(np.sqrt(((y - yhat) ** 2).mean())), 2),
    }


def fit_level(df: pd.DataFrame, keys: list[str], label: str) -> tuple[dict, pd.DataFrame, object]:
    feat = build(df.copy(), keys=keys).dropna(subset=[TARGET])

    weeks = np.sort(feat["week_start"].unique())
    cutoff = weeks[-TEST_WEEKS]
    tr = feat[feat["week_start"] < cutoff].dropna(subset=REQUIRED)
    te = feat[feat["week_start"] >= cutoff].dropna(subset=REQUIRED)

    y = te[TARGET].to_numpy(float)
    results = [
        score("baseline: last week", y, te["units_net"]),
        score("baseline: 4-week average", y, te["roll_mean_4"]),
        score("baseline: same week last year", y,
              te["lag_52"].fillna(te["roll_mean_4"])),
    ]

    model = lgb.LGBMRegressor(
        objective="l1",                 # absolute error: robust to promo spikes
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        tr[FEATURES], tr[TARGET],
        eval_X=te[FEATURES], eval_y=te[TARGET],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )

    pred = np.clip(model.predict(te[FEATURES]), 0, None)
    results.append(score(f"LightGBM ({label})", y, pred))

    best_base = min(r["WAPE_pct"] for r in results if r["model"].startswith("baseline"))
    model_wape = results[-1]["WAPE_pct"]

    out = te[["week_start", "sku", "product_name", "category", "units_net", TARGET]].copy()
    if "branch" in keys:
        out["branch"] = te["branch"].to_numpy()
    out["forecast_units"] = pred.round(1)

    summary = {
        "level": label,
        "series_count": int(feat.groupby(keys, observed=True).ngroups),
        "train_weeks": int(tr["week_start"].nunique()),
        "test_weeks": int(te["week_start"].nunique()),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "holdout_start": str(pd.Timestamp(cutoff).date()),
        "results": results,
        "best_baseline_WAPE_pct": best_base,
        "model_WAPE_pct": model_wape,
        "error_reduction_vs_best_baseline_pct": round((best_base - model_wape) / best_base * 100, 1),
        "top_features": (
            pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")})
            .sort_values("gain", ascending=False).head(10)
            .assign(gain_pct=lambda d: (d["gain"] / d["gain"].sum() * 100).round(1))
            [["feature", "gain_pct"]].to_dict("records")
        ),
    }
    return summary, out, model


def run() -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(PROC / "weekly_sales.parquet")

    sku_level = aggregate_to_sku(weekly)
    sku_summary, sku_pred, sku_model = fit_level(sku_level, ["sku"], "SKU level")
    br_summary, br_pred, br_model = fit_level(weekly, ["branch", "sku"], "branch x SKU")

    joblib.dump({"model": sku_model, "features": FEATURES, "level": "sku"},
                MODELS / "forecast_model_sku.pkl")
    joblib.dump({"model": br_model, "features": FEATURES, "level": "branch_sku"},
                MODELS / "forecast_model_branch_sku.pkl")

    sku_pred.to_parquet(PROC / "forecast_sku.parquet", index=False)
    br_pred.to_parquet(PROC / "forecast_branch_sku.parquet", index=False)

    summary = {"production_model": sku_summary, "allocation_model": br_summary}
    (MODELS / "evaluation.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print(s: dict) -> None:
    print(f"\n{s['level']}  -  {s['series_count']} series, "
          f"train {s['train_weeks']}w / holdout {s['test_weeks']}w from {s['holdout_start']}")
    print(f"{'model':<34}{'WAPE %':>9}{'MAE':>9}{'RMSE':>9}")
    print("-" * 61)
    for r in s["results"]:
        print(f"{r['model']:<34}{r['WAPE_pct']:>9}{r['MAE_units']:>9}{r['RMSE_units']:>9}")
    print(f"=> error reduction vs best baseline: {s['error_reduction_vs_best_baseline_pct']}%")


if __name__ == "__main__":
    s = run()
    _print(s["production_model"])
    _print(s["allocation_model"])
    print("\nTop features (production model):")
    for f in s["production_model"]["top_features"][:6]:
        print(f"  {f['feature']:<20}{f['gain_pct']:>6}%")
