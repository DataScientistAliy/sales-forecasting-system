"""
Telegram bot: the forecast where the buyer already is.

A dashboard only helps the people who remember to open it. The person who
actually places the purchase order checks their phone. This bot pushes a
Monday digest and answers questions on demand, off the same model the API
serves - no second source of truth.

    /start              what this bot does
    /forecast           next week's top movers
    /sku <name>         one product, with and without a promotion
    /risk               products forecast to outsell the stock on hand
    /digest             send the Monday digest now

Run:
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    export DIGEST_CHAT_ID="-1001234567890"     # optional, for the weekly push
    python src/bot.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import aggregate_to_sku, build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("bot")


# ---------------------------------------------------------------- the brain
class Forecaster:
    """Everything the bot needs to answer a question, loaded once."""

    def __init__(self) -> None:
        bundle = joblib.load(MODELS / "forecast_model_sku.pkl")
        self.model = bundle["model"]
        self.features = bundle["features"]

        weekly = pd.read_parquet(PROC / "weekly_sales.parquet")
        feats = build(aggregate_to_sku(weekly), keys=["sku"])
        self.latest_week = feats["week_start"].max()
        self.latest = feats[feats["week_start"] == self.latest_week].set_index("sku")

        self.catalogue = (
            weekly[["sku", "product_name", "category", "list_price", "unit_cost"]]
            .drop_duplicates("sku").set_index("sku")
        )
        inv = pd.read_csv(RAW / "inventory_snapshot.csv")
        self.stock = inv.groupby("sku")["on_hand_units"].sum()

    @property
    def target_week(self) -> str:
        return str((self.latest_week + pd.Timedelta(weeks=1)).date())

    def predict(self, sku: str, discount: float = 0.0) -> float:
        row = self.latest.loc[[sku]].copy()
        row["promo_next_week"] = 1 if discount > 0 else 0
        row["discount_next_week"] = discount
        lift = row["promo_lift_hist"].fillna(1.0)
        row["expected_promo_effect"] = np.where(
            discount > 0, row["roll_mean_4"] * lift, row["roll_mean_4"]
        )
        return float(np.clip(self.model.predict(row[self.features])[0], 0, None))

    def all_forecasts(self) -> pd.DataFrame:
        rows = []
        for sku in self.latest.index:
            pred = self.predict(sku)
            recent = float(self.latest.loc[sku, "roll_mean_4"])
            meta = self.catalogue.loc[sku]
            rows.append(
                {
                    "sku": sku,
                    "product_name": meta["product_name"],
                    "forecast": pred,
                    "recent_4w": recent,
                    "change_pct": (pred / recent - 1) * 100 if recent else 0.0,
                    "on_hand": int(self.stock.get(sku, 0)),
                    "revenue": pred * float(meta["list_price"]),
                }
            )
        return pd.DataFrame(rows)

    def find(self, text: str) -> str | None:
        """Loose product lookup, so the user can type 'milk' not 'SKU-1091'."""
        t = text.strip().lower()
        if t.upper() in self.catalogue.index:
            return t.upper()
        names = self.catalogue["product_name"].str.lower()
        hit = names[names.str.contains(t, regex=False)]
        return hit.index[0] if len(hit) else None


# ------------------------------------------------------------ message bodies
def fmt_digest(f: Forecaster) -> str:
    df = f.all_forecasts()
    movers = df.reindex(df["change_pct"].abs().sort_values(ascending=False).index).head(5)
    risk = df[df["forecast"] > df["on_hand"]].sort_values("forecast", ascending=False)

    lines = [
        f"<b>Week of {f.target_week}</b>",
        f"Forecast demand: <b>{df['forecast'].sum():,.0f} units</b> "
        f"· ${df['revenue'].sum():,.0f}",
        "",
        "<b>Biggest moves vs the last 4 weeks</b>",
    ]
    for r in movers.itertuples():
        arrow = "▲" if r.change_pct >= 0 else "▼"
        lines.append(
            f"{arrow} {r.product_name} — {r.forecast:,.0f} u ({r.change_pct:+.0f}%)"
        )
    if len(risk):
        lines += ["", f"<b>⚠ Stock risk — {len(risk)} product(s)</b>"]
        for r in risk.head(5).itertuples():
            lines.append(
                f"• {r.product_name}: need {r.forecast:,.0f}, have {r.on_hand:,} "
                f"(short {r.forecast - r.on_hand:,.0f})"
            )
    else:
        lines += ["", "✓ No stock shortfalls forecast."]
    lines += ["", "<i>/sku &lt;name&gt; for one product · /risk for the full list</i>"]
    return "\n".join(lines)


def fmt_sku(f: Forecaster, sku: str) -> str:
    meta = f.catalogue.loc[sku]
    base = f.predict(sku)
    recent = float(f.latest.loc[sku, "roll_mean_4"])
    on_hand = int(f.stock.get(sku, 0))
    price, cost = float(meta["list_price"]), float(meta["unit_cost"])

    lines = [
        f"<b>{meta['product_name']}</b>  <code>{sku}</code>",
        f"<i>{meta['category']}</i>",
        "",
        f"Next week ({f.target_week}): <b>{base:,.0f} units</b>",
        f"Last 4 weeks average: {recent:,.0f} units  ({base/recent-1:+.0%})"
        if recent else "",
        f"Stock on hand: {on_hand:,} units"
        + ("  ⚠ short" if base > on_hand else "  ✓ covered"),
        "",
        "<b>If you promote it next week</b>",
    ]
    for d in (10, 20, 30):
        u = f.predict(sku, d)
        margin = u * (price * (1 - d / 100) - cost) - base * (price - cost)
        lines.append(f"  {d}% off → {u:,.0f} u  ·  margin {margin:+,.0f}")
    return "\n".join(x for x in lines if x != "")


def fmt_risk(f: Forecaster) -> str:
    df = f.all_forecasts()
    risk = df[df["forecast"] > df["on_hand"]].sort_values(
        "forecast", ascending=False
    )
    if not len(risk):
        return "✓ Every product is covered by stock on hand for next week."
    lines = [f"<b>Stock shortfalls, week of {f.target_week}</b>", ""]
    for r in risk.itertuples():
        lines.append(
            f"• <b>{r.product_name}</b>\n"
            f"   need {r.forecast:,.0f} · have {r.on_hand:,} · "
            f"order ≥ {r.forecast - r.on_hand:,.0f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- wiring
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print(__doc__)
        print("TELEGRAM_BOT_TOKEN is not set.\n")
        print("Preview of the Monday digest this bot would send:\n")
        print(fmt_digest(Forecaster()).replace("<b>", "").replace("</b>", "")
              .replace("<i>", "").replace("</i>", ""))
        return

    from aiogram import Bot, Dispatcher, F, html
    from aiogram.filters import Command, CommandStart
    from aiogram.types import Message
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    f = Forecaster()
    bot = Bot(token, parse_mode="HTML")
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start(m: Message) -> None:
        await m.answer(
            "<b>Demand forecast bot</b>\n\n"
            "/forecast — next week's top movers\n"
            "/sku &lt;name&gt; — one product, with promotion scenarios\n"
            "/risk — products that will outsell their stock\n"
            "/digest — send the Monday digest now\n\n"
            f"<i>History through {f.latest_week.date()}.</i>"
        )

    @dp.message(Command("forecast", "digest"))
    async def digest(m: Message) -> None:
        await m.answer(fmt_digest(f))

    @dp.message(Command("risk"))
    async def risk(m: Message) -> None:
        await m.answer(fmt_risk(f))

    @dp.message(Command("sku"))
    async def sku(m: Message) -> None:
        q = (m.text or "").partition(" ")[2].strip()
        if not q:
            await m.answer("Give me a product, e.g. <code>/sku milk</code>")
            return
        found = f.find(q)
        if not found:
            await m.answer(f"Nothing matching {html.quote(q)}. Try part of the name.")
            return
        await m.answer(fmt_sku(f, found))

    @dp.message(F.text)
    async def freetext(m: Message) -> None:
        found = f.find(m.text or "")
        await m.answer(fmt_sku(f, found) if found else
                       "I did not recognise that product. /start for the commands.")

    chat_id = os.getenv("DIGEST_CHAT_ID")
    if chat_id:
        sched = AsyncIOScheduler(timezone=os.getenv("TZ", "UTC"))
        sched.add_job(lambda: asyncio.create_task(bot.send_message(chat_id, fmt_digest(f))),
                      "cron", day_of_week="mon", hour=7, minute=30)
        sched.start()
        log.info("Monday 07:30 digest scheduled for chat %s", chat_id)

    log.info("bot running")
    asyncio.run(dp.start_polling(bot))


if __name__ == "__main__":
    main()
