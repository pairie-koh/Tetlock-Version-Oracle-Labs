"""
Tetlock Oracle Labs -- run_cycle.py
Main orchestrator. Runs the full Tetlock forecasting pipeline:

  1. Data Fetch -> GDELT, Hyperliquid, Weather, Contracts
  2. Newswire   -> Gather and normalize news (Perplexity + Haiku)
  3. Forecast   -> Base rate -> Decompose -> Evidence -> Update -> Adversarial
  4. Evaluate   -> Score against market prices
  5. Lessons    -> Rebuild feedback cache

Usage:
  python run_cycle.py                 # full cycle
  python run_cycle.py --forecast-only # skip data fetch + newswire
  python run_cycle.py --evaluate-only # only score latest predictions
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from constants import BRIEFINGS_DIR


def fetch_data_sources():
    """Fetch all external data sources and save to disk.

    Each source is wrapped in try/except so failures don't break the cycle.
    """
    print("\n  Fetching GDELT news signals...")
    try:
        from gdelt import fetch_all_queries, save_results
        results = fetch_all_queries()
        save_results(results)
        total = sum(r.get("article_count", 0) for r in results.values())
        print(f"    GDELT: {len(results)} queries, {total} articles")
    except Exception as e:
        print(f"    GDELT: FAILED ({e})")

    print("  Fetching Hyperliquid perp prices...")
    try:
        from hyperliquid import fetch_all, save_prices
        prices = fetch_all()
        save_prices(prices)
        print(f"    Hyperliquid: {len(prices)} assets")
    except Exception as e:
        print(f"    Hyperliquid: FAILED ({e})")

    print("  Fetching NWS weather forecasts...")
    try:
        from weather import fetch_all as fetch_weather, save_forecasts
        forecasts = fetch_weather()
        save_forecasts(forecasts)
        print(f"    Weather: {len(forecasts)} cities")
    except Exception as e:
        print(f"    Weather: FAILED ({e})")

    print("  Fetching active contracts from Polymarket...")
    try:
        from contracts import fetch_active_markets, filter_and_rank_markets, format_contract
        markets = fetch_active_markets(limit=200)
        selected = filter_and_rank_markets(markets, target_count=30)
        print(f"    Contracts: {len(selected)} active markets")
    except Exception as e:
        print(f"    Contracts: FAILED ({e})")


def run_full_cycle(forecast_only=False, evaluate_only=False):
    """Run the complete Tetlock forecast cycle."""
    start = datetime.now(timezone.utc)
    print(f"{'='*60}")
    print(f"TETLOCK ORACLE LABS -- Forecast Cycle")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    if evaluate_only:
        print("\n[Mode: evaluate-only]")
        from market_data import fetch_all_prices
        from evaluate import evaluate_latest, append_scores, print_scorecard
        prices = fetch_all_prices()
        scores = evaluate_latest(prices)
        if scores:
            append_scores(scores)
            print_scorecard(scores)

        # Rebuild lessons cache after scoring
        try:
            from lessons import rebuild_lessons_cache
            print("\n  Rebuilding lessons cache...")
            rebuild_lessons_cache()
        except Exception as e:
            print(f"  Lessons rebuild failed: {e}")
        return

    # -- Stage 1: Data Fetch --
    if not forecast_only:
        print("\n[Stage 1/5] DATA FETCH")
        fetch_data_sources()

    # -- Stage 2: Newswire --
    if forecast_only:
        print("\n[Mode: forecast-only -- loading latest briefing]")
        briefing_path = os.path.join(BRIEFINGS_DIR, "latest.json")
        if not os.path.exists(briefing_path):
            print("ERROR: No briefing found. Run without --forecast-only first.")
            sys.exit(1)
        with open(briefing_path, "r") as f:
            briefing = json.load(f)
        print(f"  Loaded {len(briefing.get('facts', []))} facts from latest briefing")
    else:
        print("\n[Stage 2/5] NEWSWIRE")
        from newswire import run_newswire
        briefing = run_newswire()

    # -- Stage 3: Forecast --
    print("\n[Stage 3/5] TETLOCK FORECAST PIPELINE")
    from forecast import run_forecast
    predictions = run_forecast(briefing)

    # -- Stage 4: Evaluate --
    print("\n[Stage 4/5] EVALUATION")
    from market_data import fetch_all_prices
    from evaluate import evaluate_latest, append_scores, print_scorecard

    scores = evaluate_latest(fetch_all_prices())
    if scores:
        append_scores(scores)
        print_scorecard(scores)

    # -- Stage 5: Lessons --
    print("\n[Stage 5/5] LESSONS")
    try:
        from lessons import rebuild_lessons_cache
        rebuild_lessons_cache()
    except Exception as e:
        print(f"  Lessons rebuild failed: {e}")

    # -- Summary --
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"Cycle complete in {elapsed:.1f}s")
    print(f"Predictions: {len(predictions)}")
    for key, pred in predictions.items():
        div = pred['divergence_from_market']
        tier = pred.get('model_tier', '?')
        sources = pred.get('data_sources_used', [])
        print(f"  {key}: market={pred['market_price']:.3f} -> "
              f"ours={pred['final_probability']:.3f} "
              f"(div={div:+.3f}, tier={tier}, sources={len(sources)})")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetlock Oracle Labs -- Forecast Cycle")
    parser.add_argument("--forecast-only", action="store_true",
                        help="Skip data fetch + newswire, use latest briefing")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only evaluate latest predictions")
    args = parser.parse_args()

    run_full_cycle(
        forecast_only=args.forecast_only,
        evaluate_only=args.evaluate_only,
    )
