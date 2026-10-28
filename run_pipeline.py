"""
One command, whole system: generate -> clean -> features -> train.

This is what the scheduler calls. In production the generate step is
replaced by whatever pulls the real export - an API call, an SFTP drop,
a database query - and nothing downstream changes.

    python run_pipeline.py            full rebuild
    python run_pipeline.py --no-data  keep existing raw files
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def step(name: str, fn) -> None:
    print(f"\n[{name}]")
    t0 = time.time()
    fn()
    print(f"  done in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true",
                    help="skip regenerating the raw source files")
    args = ap.parse_args()

    if not args.no_data:
        import generate_data
        step("1/3  source data", lambda: __import__("runpy").run_path(
            str(Path(__file__).parent / "src" / "generate_data.py"), run_name="__main__"))

    import clean
    step("2/3  clean + match", lambda: print(
        "  auto-match rate: "
        f"{clean.run()['row_auto_match_rate_pct']}%"))

    import train
    def _train() -> None:
        s = train.run()
        p = s["production_model"]
        print(f"  holdout WAPE {p['model_WAPE_pct']}%  "
              f"(baseline {p['best_baseline_WAPE_pct']}%, "
              f"-{p['error_reduction_vs_best_baseline_pct']}% error)")
    step("3/3  train + evaluate", _train)

    print("\nReady.")
    print("  dashboard   streamlit run src/app.py")
    print("  api         uvicorn src.api:app --reload")
    print("  bot         TELEGRAM_BOT_TOKEN=... python src/bot.py")


if __name__ == "__main__":
    main()
