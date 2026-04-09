"""
Tetlock Oracle Labs — evaluate.py
Scoring Engine

Evaluates forecasts against actual market prices. Designed for
direct comparison with Oracle Labs v2.

Metrics:
  - Squared Error (SE): (prediction - actual)^2
  - Brier Score: mean SE across all predictions
  - Calibration: are predictions at X% correct X% of the time?
  - Edge vs Market: does our divergence from market improve accuracy?
  - Per-component attribution: how much did each pipeline step help?
"""

import csv
import json
import os
from datetime import datetime, timezone

from constants import MARKETS, SCORES_DIR, PREDICTIONS_DIR


def score_prediction(prediction, actual_price):
    """Score a single prediction against actual market price.

    Returns:
        {
            "market": str,
            "timestamp": str,
            "our_prediction": float,
            "market_prediction": float,  # market price at time of forecast
            "actual_price": float,       # market price at resolution/evaluation
            "our_se": float,             # our squared error
            "market_se": float,          # market's squared error (baseline)
            "edge": float,              # market_se - our_se (positive = we beat market)
            "base_rate_se": float,       # SE if we just used base rate
            "divergence": float,         # how far we were from market
        }
    """
    our_pred = prediction["final_probability"]
    market_pred = prediction["market_price"]
    base_rate = prediction["base_rate"]

    our_se = (our_pred - actual_price) ** 2
    market_se = (market_pred - actual_price) ** 2
    base_rate_se = (base_rate - actual_price) ** 2

    return {
        "market": prediction["market"],
        "timestamp": prediction["timestamp"],
        "our_prediction": round(our_pred, 4),
        "market_prediction": round(market_pred, 4),
        "actual_price": round(actual_price, 4),
        "our_se": round(our_se, 6),
        "market_se": round(market_se, 6),
        "base_rate_se": round(base_rate_se, 6),
        "edge": round(market_se - our_se, 6),
        "divergence": round(our_pred - market_pred, 4),
        "cycle_number": prediction.get("cycle_number", 0),
        "evidence_count": prediction.get("evidence_count", 0),
        "adversarial_adjustment": prediction.get("adversarial_adjustment", 0),
    }


def evaluate_latest(current_prices):
    """Evaluate the most recent predictions against current prices.

    Args:
        current_prices: {market_key: float} current market prices

    Returns:
        list of score dicts
    """
    pred_path = os.path.join(PREDICTIONS_DIR, "latest.json")
    if not os.path.exists(pred_path):
        print("No predictions to evaluate.")
        return []

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    scores = []
    for market_key, prediction in predictions.items():
        actual = current_prices.get(market_key)
        if actual is None:
            continue
        score = score_prediction(prediction, actual)
        scores.append(score)

    return scores


def append_scores(scores):
    """Append scores to the scores CSV."""
    os.makedirs(SCORES_DIR, exist_ok=True)
    csv_path = os.path.join(SCORES_DIR, "scores_history.csv")

    fields = [
        "market", "timestamp", "our_prediction", "market_prediction",
        "actual_price", "our_se", "market_se", "base_rate_se", "edge",
        "divergence", "cycle_number", "evidence_count", "adversarial_adjustment",
    ]

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for score in scores:
            writer.writerow({k: score.get(k, "") for k in fields})


def compute_summary(scores):
    """Compute summary statistics from a list of scores.

    Returns:
        {
            "total_predictions": int,
            "mean_our_se": float,       # our Brier score
            "mean_market_se": float,    # market Brier score (baseline)
            "mean_base_rate_se": float, # base rate Brier score
            "mean_edge": float,         # average edge vs market
            "pct_beat_market": float,   # % of predictions that beat market
            "pct_beat_base_rate": float, # % that beat pure base rate
        }
    """
    if not scores:
        return {"total_predictions": 0}

    n = len(scores)
    mean_our = sum(s["our_se"] for s in scores) / n
    mean_market = sum(s["market_se"] for s in scores) / n
    mean_base = sum(s["base_rate_se"] for s in scores) / n
    mean_edge = sum(s["edge"] for s in scores) / n
    beat_market = sum(1 for s in scores if s["edge"] > 0) / n
    beat_base = sum(1 for s in scores if s["our_se"] < s["base_rate_se"]) / n

    return {
        "total_predictions": n,
        "mean_our_se": round(mean_our, 6),
        "mean_market_se": round(mean_market, 6),
        "mean_base_rate_se": round(mean_base, 6),
        "mean_edge": round(mean_edge, 6),
        "pct_beat_market": round(beat_market * 100, 1),
        "pct_beat_base_rate": round(beat_base * 100, 1),
    }


def print_scorecard(scores):
    """Print a formatted scorecard."""
    summary = compute_summary(scores)

    if summary["total_predictions"] == 0:
        print("No scores to display.")
        return

    print(f"\n{'='*60}")
    print(f"SCORECARD — {summary['total_predictions']} predictions")
    print(f"{'='*60}")
    print(f"  Our Brier Score:       {summary['mean_our_se']:.6f}")
    print(f"  Market Brier Score:    {summary['mean_market_se']:.6f}")
    print(f"  Base Rate Brier Score: {summary['mean_base_rate_se']:.6f}")
    print(f"  Mean Edge vs Market:   {summary['mean_edge']:+.6f}")
    print(f"  Beat Market:           {summary['pct_beat_market']:.1f}%")
    print(f"  Beat Base Rate:        {summary['pct_beat_base_rate']:.1f}%")

    verdict = "BEATING" if summary["mean_edge"] > 0 else "LOSING TO"
    print(f"\n  Verdict: {verdict} the market by {abs(summary['mean_edge']):.6f} SE")

    print(f"\n  Per-prediction breakdown:")
    for s in scores:
        edge_str = f"{s['edge']:+.6f}"
        marker = "✓" if s["edge"] > 0 else "✗"
        print(f"    {marker} {s['market']}: ours={s['our_prediction']:.3f} "
              f"market={s['market_prediction']:.3f} actual={s['actual_price']:.3f} "
              f"edge={edge_str}")

    # Save scorecard
    os.makedirs(SCORES_DIR, exist_ok=True)
    scorecard_path = os.path.join(SCORES_DIR, "latest_scorecard.json")
    with open(scorecard_path, "w") as f:
        json.dump({"summary": summary, "scores": scores}, f, indent=2)


def load_all_scores():
    """Load all historical scores from CSV."""
    csv_path = os.path.join(SCORES_DIR, "scores_history.csv")
    if not os.path.exists(csv_path):
        return []

    scores = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["our_se", "market_se", "base_rate_se", "edge",
                        "our_prediction", "market_prediction", "actual_price",
                        "divergence", "adversarial_adjustment"]:
                if key in row and row[key]:
                    row[key] = float(row[key])
            for key in ["cycle_number", "evidence_count"]:
                if key in row and row[key]:
                    row[key] = int(row[key])
            scores.append(row)
    return scores


if __name__ == "__main__":
    from market_data import fetch_all_prices

    print("=== Evaluation ===")
    prices = fetch_all_prices()

    if not prices:
        print("Could not fetch prices for evaluation.")
    else:
        scores = evaluate_latest(prices)
        if scores:
            append_scores(scores)
            print_scorecard(scores)

            # Also show all-time stats
            all_scores = load_all_scores()
            if len(all_scores) > len(scores):
                print(f"\n{'='*60}")
                print(f"ALL-TIME ({len(all_scores)} predictions)")
                all_summary = compute_summary(all_scores)
                print(f"  Mean Edge: {all_summary['mean_edge']:+.6f}")
                print(f"  Beat Market: {all_summary['pct_beat_market']:.1f}%")
        else:
            print("No predictions to evaluate against current prices.")
