"""
Tetlock Oracle Labs — run_cycle.py
Main orchestrator. Runs the full Tetlock forecasting pipeline:

  1. Newswire  → Gather and normalize news
  2. Forecast  → Base rate → Decompose → Evidence → Update → Adversarial
  3. Evaluate  → Score against market prices

Usage:
  python run_cycle.py                 # full cycle
  python run_cycle.py --forecast-only # skip newswire, use latest briefing
  python run_cycle.py --evaluate-only # only score latest predictions
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from constants import BRIEFINGS_DIR


def run_full_cycle(forecast_only=False, evaluate_only=False):
    """Run the complete Tetlock forecast cycle."""
    start = datetime.now(timezone.utc)
    print(f"{'='*60}")
    print(f"TETLOCK ORACLE LABS — Forecast Cycle")
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
        return

    # ── Stage 1: Newswire ──
    if forecast_only:
        print("\n[Mode: forecast-only — loading latest briefing]")
        briefing_path = os.path.join(BRIEFINGS_DIR, "latest.json")
        if not os.path.exists(briefing_path):
            print("ERROR: No briefing found. Run without --forecast-only first.")
            sys.exit(1)
        with open(briefing_path, "r") as f:
            briefing = json.load(f)
        print(f"  Loaded {len(briefing.get('facts', []))} facts from latest briefing")
    else:
        print("\n[Stage 1/3] NEWSWIRE")
        from newswire import run_newswire
        briefing = run_newswire()

    # ── Stage 2: Forecast ──
    print("\n[Stage 2/3] TETLOCK FORECAST PIPELINE")
    from forecast import run_forecast
    predictions = run_forecast(briefing)

    # ── Stage 3: Evaluate ──
    print("\n[Stage 3/3] EVALUATION")
    from market_data import fetch_all_prices
    from evaluate import evaluate_latest, append_scores, print_scorecard

    # Slight delay — prices may have moved since forecast
    scores = evaluate_latest(fetch_all_prices())
    if scores:
        append_scores(scores)
        print_scorecard(scores)

    # ── Summary ──
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"Cycle complete in {elapsed:.1f}s")
    print(f"Predictions: {len(predictions)}")
    for key, pred in predictions.items():
        print(f"  {key}: market={pred['market_price']:.3f} → "
              f"ours={pred['final_probability']:.3f} "
              f"(div={pred['divergence_from_market']:+.3f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetlock Oracle Labs — Forecast Cycle")
    parser.add_argument("--forecast-only", action="store_true",
                        help="Skip newswire, use latest briefing")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only evaluate latest predictions")
    args = parser.parse_args()

    run_full_cycle(
        forecast_only=args.forecast_only,
        evaluate_only=args.evaluate_only,
    )
