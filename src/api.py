"""
Prediction service.

The model needs 40 features, most of them derived from the product's own
sales history - so the client should not have to supply them. This service
keeps the history, builds the features itself, and asks the caller only for
what it cannot know: which product, and what the promotion plan says about
next week.

Run locally:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs for a ready-made test page.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, aggregate_to_sku, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

app = FastAPI(
    title="Demand Forecast API",
    version="1.0.0",
    description="Next-week unit demand per SKU, with the planned discount taken into account.",
)

STATE: dict = {}


@app.on_event("startup")
def load() -> None:
    bundle = joblib.load(MODELS / "forecast_model_sku.pkl")
    weekly = pd.read_parquet(PROC / "weekly_sales.parquet")
    feats = build(aggregate_to_sku(weekly), keys=["sku"])

    # the most recent complete week per SKU is what we forecast forward from
    latest_week = feats["week_start"].max()
    latest = feats[feats["week_start"] == latest_week].set_index("sku")

    STATE.update(
        model=bundle["model"],
        features=bundle["features"],
        latest=latest,
        latest_week=latest_week,
        catalogue=weekly[["sku", "product_name", "category", "list_price"]]
        .drop_duplicates("sku").set_index("sku"),
    )


class ForecastRequest(BaseModel):
    sku: str = Field(..., examples=["SKU-1007"])
    planned_discount_pct: float = Field(
        0, ge=0, le=60,
        description="Discount planned for next week, from the promotion calendar. "
                    "0 means no promotion.",
    )


class ForecastResponse(BaseModel):
    sku: str
    product_name: str
    category: str
    forecast_week: str
    forecast_units: float
    recent_4w_average: float
    planned_discount_pct: float
    uplift_vs_recent_pct: float
    basis: Literal["model"]


def _predict_row(sku: str, discount: float) -> dict:
    latest: pd.DataFrame = STATE["latest"]
    if sku not in latest.index:
        raise HTTPException(404, f"Unknown SKU: {sku}")

    row = latest.loc[[sku]].copy()

    # override the known-in-advance drivers with what the caller tells us
    row["promo_next_week"] = 1 if discount > 0 else 0
    row["discount_next_week"] = discount
    lift = row["promo_lift_hist"].fillna(1.0)
    row["expected_promo_effect"] = np.where(
        discount > 0, row["roll_mean_4"] * lift, row["roll_mean_4"]
    )

    pred = float(np.clip(STATE["model"].predict(row[STATE["features"]])[0], 0, None))
    recent = float(row["roll_mean_4"].iloc[0])
    meta = STATE["catalogue"].loc[sku]
    week = (STATE["latest_week"] + pd.Timedelta(weeks=1)).date()

    return {
        "sku": sku,
        "product_name": str(meta["product_name"]),
        "category": str(meta["category"]),
        "forecast_week": str(week),
        "forecast_units": round(pred, 1),
        "recent_4w_average": round(recent, 1),
        "planned_discount_pct": discount,
        "uplift_vs_recent_pct": round((pred / recent - 1) * 100, 1) if recent else 0.0,
        "basis": "model",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "model" in STATE,
        "history_through": str(STATE.get("latest_week", "")),
        "skus": int(len(STATE.get("latest", []))),
    }


@app.get("/model")
def model_info() -> dict:
    import json
    ev = json.loads((MODELS / "evaluation.json").read_text())
    p = ev["production_model"]
    return {
        "level": p["level"],
        "trained_on_weeks": p["train_weeks"],
        "holdout_weeks": p["test_weeks"],
        "holdout_WAPE_pct": p["model_WAPE_pct"],
        "best_baseline_WAPE_pct": p["best_baseline_WAPE_pct"],
        "error_reduction_vs_baseline_pct": p["error_reduction_vs_best_baseline_pct"],
        "top_features": p["top_features"][:5],
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> dict:
    return _predict_row(req.sku, req.planned_discount_pct)


@app.post("/forecast/batch")
def forecast_batch(reqs: list[ForecastRequest]) -> dict:
    if len(reqs) > 500:
        raise HTTPException(413, "Send at most 500 SKUs per call.")
    return {"results": [_predict_row(r.sku, r.planned_discount_pct) for r in reqs]}


@app.get("/forecast/all")
def forecast_all(planned_discount_pct: float = 0) -> dict:
    out = [_predict_row(s, planned_discount_pct) for s in STATE["latest"].index]
    out.sort(key=lambda d: d["forecast_units"], reverse=True)
    return {"forecast_week": out[0]["forecast_week"] if out else None,
            "count": len(out), "results": out}


@app.get("/catalogue")
def catalogue() -> dict:
    c = STATE["catalogue"].reset_index()
    return {"count": len(c), "items": c.to_dict("records")}
