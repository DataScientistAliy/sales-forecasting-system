"""
Streamlit dashboard: the whole system in one page.

Run locally:
    streamlit run src/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, aggregate_to_sku, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# validated categorical palette, used in its documented order
C_ACTUAL = "#2a78d6"     # slot 1, blue
C_FORECAST = "#eb6834"   # slot 2, orange
C_MUTED = "#8a8982"
C_GRID = "rgba(138,137,130,0.22)"
INK = "#1c1c1b"
INK_2 = "#52514e"

st.set_page_config(page_title="Demand Forecast", page_icon="📦", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      h1 {font-size: 1.85rem !important; letter-spacing: -0.02em; margin-bottom: .15rem;}
      .sub {color: #52514e; font-size: .95rem; margin-bottom: 1.6rem;}
      [data-testid="stMetricValue"] {font-size: 1.75rem; letter-spacing: -0.02em;}
      [data-testid="stMetricLabel"] {color:#52514e; font-size:.78rem;
        text-transform:uppercase; letter-spacing:.06em;}
      .panel {border:1px solid rgba(138,137,130,.28); border-radius:10px;
        padding:1.1rem 1.25rem; background:rgba(252,252,251,.6);}
      .qrow {display:flex; justify-content:space-between; padding:.33rem 0;
        border-bottom:1px solid rgba(138,137,130,.16); font-size:.9rem;}
      .qrow:last-child {border-bottom:none;}
      .qk {color:#52514e;} .qv {font-variant-numeric:tabular-nums; font-weight:600;}
      .note {color:#52514e; font-size:.86rem; line-height:1.5;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    weekly = pd.read_parquet(PROC / "weekly_sales.parquet")
    fc_sku = pd.read_parquet(PROC / "forecast_sku.parquet")
    quality = json.loads((PROC / "data_quality_report.json").read_text())
    ev = json.loads((MODELS / "evaluation.json").read_text())
    return weekly, fc_sku, quality, ev


@st.cache_resource
def load_model():
    bundle = joblib.load(MODELS / "forecast_model_sku.pkl")
    weekly = pd.read_parquet(PROC / "weekly_sales.parquet")
    feats = build(aggregate_to_sku(weekly), keys=["sku"])
    latest = feats[feats["week_start"] == feats["week_start"].max()].set_index("sku")
    return bundle, latest


weekly, fc_sku, quality, ev = load()
prod = ev["production_model"]

st.markdown("# Weekly demand forecast")
st.markdown(
    '<div class="sub">40 SKUs · 5 branches · 3 years of history · '
    'rebuilt from a raw POS export every night</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- KPI row
last52 = weekly[weekly["week_start"] > weekly["week_start"].max() - pd.Timedelta(weeks=52)]
next_week_units = fc_sku[fc_sku["week_start"] == fc_sku["week_start"].max()]["forecast_units"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue, last 52 weeks", f"${last52['revenue'].sum()/1e6:.2f}M")
k2.metric("Forecast accuracy (WAPE)", f"{prod['model_WAPE_pct']:.1f}%",
          f"-{prod['error_reduction_vs_best_baseline_pct']:.1f}% error vs baseline",
          delta_color="inverse")
k3.metric("Units forecast, next week", f"{next_week_units:,.0f}")
k4.metric("Product names auto-resolved", f"{quality['row_auto_match_rate_pct']:.1f}%",
          f"{quality['distinct_product_spellings']:,} spellings → {quality['skus']} SKUs",
          delta_color="off")

st.markdown("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Forecast vs actual", "Does it beat a spreadsheet?", "Promotion simulator", "Data quality"]
)

# ------------------------------------------------------- forecast vs actual
with tab1:
    left, right = st.columns([1, 3])
    with left:
        names = fc_sku[["sku", "product_name"]].drop_duplicates().sort_values("product_name")
        choice = st.selectbox("Product", names["product_name"], index=0)
        sku = names.loc[names["product_name"] == choice, "sku"].iloc[0]
        st.markdown(
            '<div class="note">The line stops where the model stops being tested. '
            'Everything to the right of the split was never seen during training.</div>',
            unsafe_allow_html=True,
        )

    hist = (
        aggregate_to_sku(weekly).query("sku == @sku")
        .sort_values("week_start").tail(78)
    )
    f = fc_sku.query("sku == @sku").sort_values("week_start")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["week_start"], y=hist["units_net"], name="Actual",
        mode="lines", line=dict(color=C_ACTUAL, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Actual <b>%{y:,.0f}</b> units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=f["week_start"] + pd.Timedelta(weeks=1), y=f["forecast_units"], name="Forecast",
        mode="lines+markers", line=dict(color=C_FORECAST, width=2),
        marker=dict(size=8, color=C_FORECAST, line=dict(width=2, color="#fcfcfb")),
        hovertemplate="%{x|%d %b %Y}<br>Forecast <b>%{y:,.0f}</b> units<extra></extra>",
    ))
    if len(f):
        fig.add_vline(x=f["week_start"].min(), line_width=1, line_dash="dot", line_color=C_MUTED)
        fig.add_annotation(x=f["week_start"].min(), yref="paper", y=1.04,
                           text="holdout starts", showarrow=False,
                           font=dict(size=11, color=INK_2), xanchor="left", xshift=6)
    fig.update_layout(
        height=430, hovermode="x unified",
        margin=dict(l=8, r=8, t=34, b=8),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="units / week", gridcolor=C_GRID, zeroline=False,
                   title_font=dict(color=INK_2, size=11)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", showline=True, linecolor=C_GRID),
    )
    with right:
        st.plotly_chart(fig, use_container_width=True)

    if len(f):
        err = (f["forecast_units"] - f["target_units_next_week"]).abs().sum()
        den = f["target_units_next_week"].abs().sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("This product's WAPE", f"{err/den*100:.1f}%" if den else "n/a")
        c2.metric("Mean weekly units", f"{f['target_units_next_week'].mean():,.0f}")
        c3.metric("Weeks tested", f"{len(f)}")

# --------------------------------------------------------------- baselines
with tab2:
    st.markdown("#### Held-out error, model against the obvious alternatives")
    st.markdown(
        '<div class="note">Same 16 weeks, same rows, same metric for every row below. '
        'A forecast is only worth paying for if it beats what the client could already '
        'do in a spreadsheet — so that is what it is measured against.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    for level_key, label in [("production_model", "SKU level — used for purchasing"),
                             ("allocation_model", "Branch × SKU — used for allocation")]:
        s = ev[level_key]
        rows = pd.DataFrame(s["results"])
        rows["is_model"] = rows["model"].str.startswith("LightGBM")
        rows = rows.sort_values("WAPE_pct", ascending=True)

        fig = go.Figure(go.Bar(
            x=rows["WAPE_pct"], y=rows["model"], orientation="h",
            marker=dict(color=[C_ACTUAL if m else C_MUTED for m in rows["is_model"]],
                        cornerradius=4),
            text=[f"{v:.1f}%" for v in rows["WAPE_pct"]],
            textposition="outside", textfont=dict(color=INK, size=12),
            hovertemplate="%{y}<br>WAPE <b>%{x:.2f}%</b><extra></extra>",
        ))
        fig.update_layout(
            height=210, bargap=0.35,
            margin=dict(l=8, r=60, t=36, b=8),
            title=dict(text=f"{label}  ·  {s['series_count']} series",
                       font=dict(size=13, color=INK_2)),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK, size=12), showlegend=False,
            xaxis=dict(title="WAPE % (lower is better)", gridcolor=C_GRID, zeroline=False,
                       title_font=dict(color=INK_2, size=11)),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.markdown("#### What the model leans on")
    imp = pd.DataFrame(prod["top_features"]).head(8)
    fig = go.Figure(go.Bar(
        x=imp["gain_pct"], y=imp["feature"], orientation="h",
        marker=dict(color=C_ACTUAL, cornerradius=4),
        text=[f"{v:.1f}%" for v in imp["gain_pct"]], textposition="outside",
        textfont=dict(color=INK, size=12),
        hovertemplate="%{y}<br><b>%{x:.1f}%</b> of split gain<extra></extra>",
    ))
    fig.update_layout(
        height=300, bargap=0.35, margin=dict(l=8, r=60, t=8, b=8),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12), showlegend=False,
        xaxis=dict(title="share of split gain", gridcolor=C_GRID, zeroline=False,
                   title_font=dict(color=INK_2, size=11)),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="note"><b>The headline finding.</b> The single strongest input is the '
        'planned promotion effect — not the sales history. Feeding the promotion calendar '
        'into the forecast is what takes the error from 32% down to 25%. Most businesses '
        'already keep that calendar in a spreadsheet and never connect it to their '
        'forecasting.</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------- promo simulator
with tab3:
    st.markdown("#### What happens if we discount next week?")
    st.markdown(
        '<div class="note">This calls the same prediction code the API serves. '
        'Move the slider to re-forecast against a different promotion plan.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    bundle, latest = load_model()
    model, feat_names = bundle["model"], bundle["features"]

    cat = weekly[["sku", "product_name"]].drop_duplicates().sort_values("product_name")
    c1, c2 = st.columns([1, 1])
    pick = c1.selectbox("Product ", cat["product_name"], key="sim")
    disc = c2.slider("Planned discount next week (%)", 0, 40, 20, step=5)
    sku = cat.loc[cat["product_name"] == pick, "sku"].iloc[0]

    def predict(d: float) -> float:
        row = latest.loc[[sku]].copy()
        row["promo_next_week"] = 1 if d > 0 else 0
        row["discount_next_week"] = d
        lift = row["promo_lift_hist"].fillna(1.0)
        row["expected_promo_effect"] = np.where(d > 0, row["roll_mean_4"] * lift,
                                                row["roll_mean_4"])
        return float(np.clip(model.predict(row[feat_names])[0], 0, None))

    base, withp = predict(0), predict(disc)
    price = float(weekly.loc[weekly["sku"] == sku, "list_price"].iloc[0])
    cost = float(weekly.loc[weekly["sku"] == sku, "unit_cost"].iloc[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("No promotion", f"{base:,.0f} units")
    m2.metric(f"At {disc}% off", f"{withp:,.0f} units",
              f"{(withp/base-1)*100:+.0f}%" if base else "—")
    margin_b = base * (price - cost)
    margin_p = withp * (price * (1 - disc / 100) - cost)
    m3.metric("Gross margin effect", f"${margin_p - margin_b:+,.0f}",
              "promotion pays for itself" if margin_p > margin_b else "margin lost",
              delta_color="normal" if margin_p > margin_b else "inverse")

    grid = list(range(0, 45, 5))
    units = [predict(d) for d in grid]
    margins = [u * (price * (1 - d / 100) - cost) for u, d in zip(units, grid)]

    fig = go.Figure(go.Scatter(
        x=grid, y=margins, mode="lines+markers",
        line=dict(color=C_FORECAST, width=2),
        marker=dict(size=8, color=C_FORECAST, line=dict(width=2, color="#fcfcfb")),
        hovertemplate="%{x}% off<br>margin <b>$%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_hline(y=margin_b, line_width=1, line_dash="dot", line_color=C_MUTED)
    fig.add_annotation(xref="paper", x=0.01, y=margin_b, text="no promotion",
                       showarrow=False, yshift=12, font=dict(size=11, color=INK_2))
    fig.update_layout(
        height=330, margin=dict(l=8, r=8, t=30, b=8),
        title=dict(text="Forecast gross margin by discount depth",
                   font=dict(size=13, color=INK_2)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12), showlegend=False,
        xaxis=dict(title="discount %", gridcolor=C_GRID, zeroline=False,
                   title_font=dict(color=INK_2, size=11)),
        yaxis=dict(title="gross margin, next week ($)", gridcolor=C_GRID, zeroline=False,
                   title_font=dict(color=INK_2, size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ data quality
with tab4:
    st.markdown("#### What the raw export looked like, and what was done about it")
    st.markdown(
        '<div class="note">Every number below is produced by the pipeline itself, '
        'on each run. Nothing here was cleaned by hand.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    groups = {
        "Volume": [
            ("Rows in the raw export", "rows_in"),
            ("Weekly fact rows out", "weekly_rows_out"),
            ("Weeks of history", "weeks"),
        ],
        "Structural mess removed": [
            ("Exact duplicate rows dropped", "duplicate_rows_removed"),
            ("Branch spellings collapsed", "branch_spellings_in"),
            ("…onto this many branches", "branches_out"),
            ("Unparseable dates", "unparseable_dates"),
        ],
        "Product name resolution": [
            ("Distinct spellings found", "distinct_product_spellings"),
            ("Auto-resolved to a SKU", "name_match_auto"),
            ("Sent to the review queue", "name_match_review"),
            ("Left unmatched", "name_match_unmatched"),
            ("Row-level auto-match rate %", "row_auto_match_rate_pct"),
        ],
        "Numeric repairs": [
            ("Prices recovered from revenue", "prices_recovered_from_revenue"),
            ("Prices filled from catalogue", "prices_filled_from_catalogue"),
            ("Missing revenue recomputed", "revenue_recomputed_missing"),
            ("Revenue mismatches corrected", "revenue_corrected_mismatch"),
            ("Returns separated out", "return_rows"),
        ],
    }
    cols = st.columns(2)
    for i, (title, items) in enumerate(groups.items()):
        rows = "".join(
            f'<div class="qrow"><span class="qk">{label}</span>'
            f'<span class="qv">{quality.get(key, 0):,}</span></div>'
            for label, key in items
        )
        cols[i % 2].markdown(
            f'<div class="panel"><b>{title}</b><div style="height:.6rem"></div>{rows}</div>'
            '<div style="height:1rem"></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="note"><b>Why there is a review queue at all.</b> '
        f'{quality["name_match_review"]:,} spellings scored below the auto-accept '
        'threshold. Those rows leave the pipeline and land in a CSV a human works '
        'through — they are never guessed into the numbers. A matching engine that '
        'reports 100% is hiding its mistakes inside the totals.</div>',
        unsafe_allow_html=True,
    )
