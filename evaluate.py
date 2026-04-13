"""
Tetlock Oracle Labs -- evaluate.py
Scoring Engine (Proper Tetlock Methodology)

Evaluates forecasts using the metrics from Tetlock's Good Judgment Project:

PRIMARY METRICS (require resolution -- did the event actually happen?):
  - Brier Score: (prediction - outcome)^2, where outcome is 0 or 1
    Perfect = 0.0, coin flip = 0.25, maximally wrong = 1.0
  - Calibration: when you say 70%, events should happen ~70% of the time
  - Resolution: ability to make decisive forecasts (not always hovering near 50%)

SECONDARY METRICS (available before resolution):
  - Divergence from market: how much we disagree with the crowd
  - Market comparison SE: snapshot-vs-snapshot error (NOT a Brier score)
  - Component attribution: base rate vs evidence vs adversarial contribution

Two modes:
  1. Pre-resolution: track predictions and market divergence (no Brier score possible)
  2. Post-resolution: once contracts resolve, compute real Brier scores + calibration

References:
  - Tetlock & Gardner (2015), "Superforecasting", Ch. 4-5 on Brier scoring
  - Good Judgment Project scoring rules
  - Brier (1950), "Verification of Forecasts Expressed in Terms of Probability"
"""

import csv
import json
import os
from datetime import datetime, timezone

from constants import MARKETS, SCORES_DIR, PREDICTIONS_DIR


# -- Resolution Tracking -----------------------------------------------------

RESOLUTIONS_FILE = os.path.join(SCORES_DIR, "resolutions.json")


def load_resolutions():
    """Load resolution outcomes for contracts that have resolved.

    Returns:
        dict of {market_key: {"outcome": 0 or 1, "resolved_at": str, "source": str}}
    """
    if not os.path.exists(RESOLUTIONS_FILE):
        return {}
    with open(RESOLUTIONS_FILE, "r") as f:
        return json.load(f)


def save_resolutions(resolutions):
    """Save resolution outcomes."""
    os.makedirs(SCORES_DIR, exist_ok=True)
    with open(RESOLUTIONS_FILE, "w") as f:
        json.dump(resolutions, f, indent=2)


def record_resolution(market_key, outcome, source="manual"):
    """Record that a contract has resolved.

    Args:
        market_key: which market resolved
        outcome: 1 if YES happened, 0 if NO
        source: how we know ("manual", "polymarket", "auto")
    """
    assert outcome in (0, 1), "Outcome must be 0 (NO) or 1 (YES)"
    resolutions = load_resolutions()
    resolutions[market_key] = {
        "outcome": outcome,
        "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
    }
    save_resolutions(resolutions)
    print(f"  Recorded resolution: {market_key} -> {'YES' if outcome == 1 else 'NO'} (source: {source})")


# -- Brier Score (Real) ------------------------------------------------------

def brier_score(prediction, outcome):
    """Compute Brier score for a single prediction.

    Brier score = (prediction - outcome)^2
    where outcome is 0 or 1 (binary resolution).

    Perfect score = 0.0 (predicted 1.0 for something that happened)
    Coin flip = 0.25 (always predict 0.5)
    Maximally wrong = 1.0 (predicted 1.0 for something that didn't happen)

    A Brier score below 0.25 means you're doing better than always guessing 50%.
    Tetlock's superforecasters averaged ~0.15 in the GJP.
    """
    return (prediction - outcome) ** 2


# -- Calibration Analysis ----------------------------------------------------

def compute_calibration(predictions_with_outcomes, n_bins=5):
    """Compute calibration curve: are predictions at X% correct X% of the time?

    Tetlock's superforecasters are well-calibrated: when they say 70%, events
    happen about 70% of the time. This is the single most important quality.

    Args:
        predictions_with_outcomes: list of (prediction_prob, outcome_0_or_1)
        n_bins: number of bins for calibration curve

    Returns:
        list of {
            "bin_center": float,  # e.g. 0.3
            "bin_range": str,     # e.g. "0.2-0.4"
            "mean_prediction": float,
            "mean_outcome": float,  # fraction that actually happened
            "count": int,
            "calibration_error": float,  # |mean_prediction - mean_outcome|
        }
    """
    if not predictions_with_outcomes:
        return []

    bin_width = 1.0 / n_bins
    bins = [[] for _ in range(n_bins)]

    for pred, outcome in predictions_with_outcomes:
        idx = min(int(pred / bin_width), n_bins - 1)
        bins[idx].append((pred, outcome))

    result = []
    for i, bin_data in enumerate(bins):
        if not bin_data:
            continue
        lo = i * bin_width
        hi = (i + 1) * bin_width
        center = (lo + hi) / 2
        mean_pred = sum(p for p, _ in bin_data) / len(bin_data)
        mean_out = sum(o for _, o in bin_data) / len(bin_data)
        result.append({
            "bin_center": round(center, 2),
            "bin_range": f"{lo:.1f}-{hi:.1f}",
            "mean_prediction": round(mean_pred, 4),
            "mean_outcome": round(mean_out, 4),
            "count": len(bin_data),
            "calibration_error": round(abs(mean_pred - mean_out), 4),
        })

    return result


def calibration_score(predictions_with_outcomes, n_bins=5):
    """Compute overall calibration score (mean absolute calibration error).

    Lower is better. 0.0 = perfectly calibrated.
    """
    bins = compute_calibration(predictions_with_outcomes, n_bins)
    if not bins:
        return None
    total_count = sum(b["count"] for b in bins)
    weighted_error = sum(b["calibration_error"] * b["count"] for b in bins) / total_count
    return round(weighted_error, 4)


# -- Resolution Metric -------------------------------------------------------

def resolution_score(predictions):
    """Compute resolution: how decisive are the forecasts?

    Resolution measures the variance of predictions. A forecaster who always
    says 50% has zero resolution. One who makes bold (and correct) predictions
    has high resolution.

    Score = variance of predictions around the base rate.
    Higher is better (you're making differentiated predictions).
    """
    if not predictions:
        return 0.0
    mean_pred = sum(predictions) / len(predictions)
    variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
    return round(variance, 6)


# -- Pre-Resolution Scoring (Market Comparison) ------------------------------

def score_prediction_vs_market(prediction, later_market_price):
    """Score a prediction against a later market price snapshot.

    NOTE: This is NOT a Brier score. It's a market-comparison metric only.
    It measures whether our divergence from the market moved us closer to
    where the market eventually went. Useful as a leading indicator before
    resolution, but should not be confused with forecasting accuracy.

    Args:
        prediction: prediction dict from forecast.py
        later_market_price: market price at evaluation time

    Returns:
        dict with market comparison metrics
    """
    our_pred = prediction["final_probability"]
    market_at_forecast = prediction["market_price"]
    base_rate = prediction["base_rate"]

    # Squared errors against the later market price (NOT Brier scores)
    our_se = (our_pred - later_market_price) ** 2
    market_se = (market_at_forecast - later_market_price) ** 2
    base_rate_se = (base_rate - later_market_price) ** 2

    return {
        "market": prediction["market"],
        "timestamp": prediction["timestamp"],
        "our_prediction": round(our_pred, 4),
        "market_at_forecast": round(market_at_forecast, 4),
        "market_at_eval": round(later_market_price, 4),
        "our_se_vs_market": round(our_se, 6),
        "market_drift_se": round(market_se, 6),
        "base_rate_se": round(base_rate_se, 6),
        "market_comparison_edge": round(market_se - our_se, 6),
        "divergence_at_forecast": round(our_pred - market_at_forecast, 4),
        "cycle_number": prediction.get("cycle_number", 0),
        "evidence_count": prediction.get("evidence_count", 0),
        "adversarial_adjustment": prediction.get("adversarial_adjustment", 0),
    }


# -- Post-Resolution Scoring (Real Brier) ------------------------------------

def score_prediction_resolved(prediction, outcome):
    """Score a prediction against actual binary outcome. THIS is the real score.

    Args:
        prediction: prediction dict from forecast.py
        outcome: 0 or 1 (did the event happen?)

    Returns:
        dict with real Brier score and component attribution
    """
    our_pred = prediction["final_probability"]
    market_pred = prediction["market_price"]
    base_rate = prediction["base_rate"]
    pre_adversarial = prediction.get("pre_adversarial", our_pred)

    # Real Brier scores
    our_brier = brier_score(our_pred, outcome)
    market_brier = brier_score(market_pred, outcome)
    base_rate_brier = brier_score(base_rate, outcome)
    coin_flip_brier = brier_score(0.5, outcome)  # 0.25 always

    # Component attribution: how much did each step help?
    # base_rate -> pre_adversarial -> final (after adversarial adjustment)
    base_to_updated = brier_score(base_rate, outcome) - brier_score(pre_adversarial, outcome)
    adversarial_help = brier_score(pre_adversarial, outcome) - brier_score(our_pred, outcome)

    return {
        "market": prediction["market"],
        "timestamp": prediction["timestamp"],
        "outcome": outcome,
        "our_prediction": round(our_pred, 4),
        "market_prediction": round(market_pred, 4),
        "base_rate": round(base_rate, 4),
        # Real Brier scores
        "our_brier": round(our_brier, 6),
        "market_brier": round(market_brier, 6),
        "base_rate_brier": round(base_rate_brier, 6),
        "coin_flip_brier": 0.25,
        # Edge = market_brier - our_brier (positive = we beat market)
        "edge_vs_market": round(market_brier - our_brier, 6),
        "edge_vs_base_rate": round(base_rate_brier - our_brier, 6),
        "edge_vs_coin_flip": round(coin_flip_brier - our_brier, 6),
        # Component attribution
        "evidence_contribution": round(base_to_updated, 6),
        "adversarial_contribution": round(adversarial_help, 6),
        # Metadata
        "cycle_number": prediction.get("cycle_number", 0),
        "evidence_count": prediction.get("evidence_count", 0),
        "model_tier": prediction.get("model_tier", "unknown"),
    }


# -- Auto-Resolution Detection -----------------------------------------------

RESOLUTION_PRICE_THRESHOLD = 0.02  # price below this = resolved NO
RESOLUTION_PRICE_UPPER = 0.98     # price above this = resolved YES


def check_auto_resolutions(current_prices):
    """Auto-detect resolved contracts from Polymarket prices.

    When a contract resolves on Polymarket, the YES token price goes to
    ~1.00 (resolved YES) or ~0.00 (resolved NO). We detect this and
    auto-record the resolution.

    Also checks the Polymarket Gamma API for explicit resolution status.

    Args:
        current_prices: {market_key: float} from market_data.fetch_all_prices()

    Returns:
        list of newly resolved market_keys
    """
    resolutions = load_resolutions()
    newly_resolved = []

    for market_key, price in current_prices.items():
        if market_key in resolutions:
            continue  # already resolved

        # Price-based detection
        if price <= RESOLUTION_PRICE_THRESHOLD:
            record_resolution(market_key, outcome=0, source="auto_price")
            newly_resolved.append(market_key)
            print(f"  AUTO-RESOLVED: {market_key} -> NO (price={price:.4f})")
        elif price >= RESOLUTION_PRICE_UPPER:
            record_resolution(market_key, outcome=1, source="auto_price")
            newly_resolved.append(market_key)
            print(f"  AUTO-RESOLVED: {market_key} -> YES (price={price:.4f})")

    # Also check Gamma API for explicit resolution
    try:
        _check_gamma_resolutions(resolutions, newly_resolved)
    except Exception as e:
        print(f"  Gamma resolution check failed: {e}")

    return newly_resolved


def _check_gamma_resolutions(existing_resolutions, already_resolved):
    """Check Polymarket Gamma API for explicitly resolved markets."""
    import requests
    from constants import POLYMARKET_GAMMA_URL

    for market_key, market in MARKETS.items():
        if market_key in existing_resolutions:
            continue
        if market_key in already_resolved:
            continue

        condition_id = market.get("condition_id", "")
        if not condition_id:
            continue

        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/markets",
                params={"condition_id": condition_id},
                timeout=10,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            if isinstance(data, list) and data:
                market_data = data[0]
            elif isinstance(data, dict):
                market_data = data
            else:
                continue

            # Check for resolution flags
            if market_data.get("closed") or market_data.get("resolved"):
                outcome_prices = market_data.get("outcomePrices", "")
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)

                if outcome_prices and len(outcome_prices) >= 2:
                    yes_price = float(outcome_prices[0])
                    if yes_price >= 0.95:
                        record_resolution(market_key, outcome=1, source="gamma_api")
                        already_resolved.append(market_key)
                        print(f"  GAMMA-RESOLVED: {market_key} -> YES")
                    elif yes_price <= 0.05:
                        record_resolution(market_key, outcome=0, source="gamma_api")
                        already_resolved.append(market_key)
                        print(f"  GAMMA-RESOLVED: {market_key} -> NO")

        except Exception:
            continue


# -- Main Evaluation Functions -----------------------------------------------

def evaluate_latest(current_prices):
    """Evaluate latest predictions.

    First checks for auto-resolutions (contracts that settled on Polymarket).
    If a contract has resolved, use real Brier scoring.
    If not, use market-comparison scoring (clearly labeled as pre-resolution).

    Args:
        current_prices: {market_key: float} current market prices

    Returns:
        list of score dicts (mixed resolved + pre-resolution)
    """
    pred_path = os.path.join(PREDICTIONS_DIR, "latest.json")
    if not os.path.exists(pred_path):
        print("  No predictions to evaluate.")
        return []

    # Auto-detect resolutions from current prices
    newly_resolved = check_auto_resolutions(current_prices)
    if newly_resolved:
        print(f"  {len(newly_resolved)} new resolution(s) detected!")

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    resolutions = load_resolutions()
    scores = []

    for market_key, prediction in predictions.items():
        if market_key in resolutions:
            # Contract resolved -- real Brier score
            outcome = resolutions[market_key]["outcome"]
            score = score_prediction_resolved(prediction, outcome)
            score["scoring_type"] = "resolved"
            scores.append(score)
        else:
            # Not resolved -- market comparison only
            actual = current_prices.get(market_key)
            if actual is None:
                continue
            score = score_prediction_vs_market(prediction, actual)
            score["scoring_type"] = "pre_resolution"
            scores.append(score)

    return scores


def evaluate_all_resolved():
    """Evaluate ALL historical predictions for resolved contracts.

    Scans predictions/ for all prediction files, matches against
    resolutions, computes Brier scores + calibration.

    Returns:
        {
            "resolved_scores": list of score dicts,
            "brier_score": float (mean),
            "calibration": list of bin dicts,
            "calibration_score": float,
            "resolution_score": float,
            "n_resolved": int,
            "comparison_to_market": float (mean market Brier),
        }
    """
    resolutions = load_resolutions()
    if not resolutions:
        print("  No resolutions recorded. Use `python evaluate.py resolve <market> <0|1>` to record outcomes.")
        return None

    # Gather all predictions for resolved markets
    pred_files = []
    if os.path.isdir(PREDICTIONS_DIR):
        for fname in os.listdir(PREDICTIONS_DIR):
            if fname.endswith(".json") and fname != "latest.json":
                pred_files.append(os.path.join(PREDICTIONS_DIR, fname))

    all_scores = []
    all_pred_outcome_pairs = []

    for pf in sorted(pred_files):
        with open(pf, "r") as f:
            predictions = json.load(f)

        for market_key, prediction in predictions.items():
            if market_key in resolutions:
                outcome = resolutions[market_key]["outcome"]
                score = score_prediction_resolved(prediction, outcome)
                all_scores.append(score)
                all_pred_outcome_pairs.append(
                    (prediction["final_probability"], outcome)
                )

    if not all_scores:
        print("  No resolved predictions found.")
        return None

    # Compute aggregate metrics
    mean_brier = sum(s["our_brier"] for s in all_scores) / len(all_scores)
    mean_market_brier = sum(s["market_brier"] for s in all_scores) / len(all_scores)
    preds_only = [s["our_prediction"] for s in all_scores]
    cal_bins = compute_calibration(all_pred_outcome_pairs)
    cal_score = calibration_score(all_pred_outcome_pairs)
    res_score = resolution_score(preds_only)

    return {
        "resolved_scores": all_scores,
        "brier_score": round(mean_brier, 6),
        "market_brier_score": round(mean_market_brier, 6),
        "calibration": cal_bins,
        "calibration_score": cal_score,
        "resolution_score": res_score,
        "n_resolved": len(all_scores),
        "edge_vs_market": round(mean_market_brier - mean_brier, 6),
    }


# -- Append to History -------------------------------------------------------

def append_scores(scores):
    """Append scores to the scores CSV."""
    os.makedirs(SCORES_DIR, exist_ok=True)
    csv_path = os.path.join(SCORES_DIR, "scores_history.csv")

    fields = [
        "market", "timestamp", "scoring_type", "our_prediction",
        "market_prediction", "market_at_forecast", "market_at_eval",
        "outcome", "our_brier", "market_brier",
        "our_se_vs_market", "market_drift_se", "base_rate_se",
        "edge_vs_market", "market_comparison_edge",
        "divergence_at_forecast",
        "cycle_number", "evidence_count", "model_tier",
    ]

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for score in scores:
            writer.writerow({k: score.get(k, "") for k in fields})


# -- Display -----------------------------------------------------------------

def print_scorecard(scores):
    """Print a formatted scorecard with proper Tetlock framing."""
    if not scores:
        print("  No scores to display.")
        return

    resolved = [s for s in scores if s.get("scoring_type") == "resolved"]
    pre_res = [s for s in scores if s.get("scoring_type") == "pre_resolution"]

    print(f"\n{'='*60}")
    print(f"SCORECARD -- {len(scores)} predictions")
    print(f"{'='*60}")

    # -- Resolved contracts (REAL scores) --
    if resolved:
        mean_brier = sum(s["our_brier"] for s in resolved) / len(resolved)
        mean_mkt = sum(s["market_brier"] for s in resolved) / len(resolved)
        edge = mean_mkt - mean_brier

        print(f"\n  RESOLVED CONTRACTS ({len(resolved)} predictions):")
        print(f"  {'─'*50}")
        print(f"  Our Brier Score:    {mean_brier:.4f}  {'(better than coin flip)' if mean_brier < 0.25 else '(worse than coin flip)'}")
        print(f"  Market Brier Score: {mean_mkt:.4f}")
        print(f"  Coin Flip Brier:    0.2500")
        print(f"  Edge vs Market:     {edge:+.4f}  {'(WE WIN)' if edge > 0 else '(MARKET WINS)'}")

        # Calibration
        pairs = [(s["our_prediction"], s["outcome"]) for s in resolved]
        cal = compute_calibration(pairs)
        if cal:
            print(f"\n  Calibration (Predicted vs Actual):")
            for b in cal:
                bar = "#" * int(b["mean_outcome"] * 20)
                print(f"    {b['bin_range']}: predicted={b['mean_prediction']:.2f} "
                      f"actual={b['mean_outcome']:.2f} (n={b['count']}) {bar}")
            cs = calibration_score(pairs)
            print(f"  Calibration Score: {cs:.4f} (0=perfect)")

        print(f"\n  Per-prediction:")
        for s in resolved:
            outcome_str = "YES" if s["outcome"] == 1 else "NO"
            edge_s = s["edge_vs_market"]
            marker = "+" if edge_s > 0 else "-"
            print(f"    [{marker}] {s['market']}: predicted={s['our_prediction']:.3f} "
                  f"outcome={outcome_str} brier={s['our_brier']:.4f} "
                  f"edge={edge_s:+.4f}")

    # -- Pre-resolution contracts (market comparison only) --
    if pre_res:
        print(f"\n  PRE-RESOLUTION ({len(pre_res)} predictions):")
        print(f"  {'─'*50}")
        print(f"  (No Brier scores possible -- contracts haven't resolved)")
        print(f"  Market comparison (informational only, NOT accuracy):")

        for s in pre_res:
            div = s["divergence_at_forecast"]
            mce = s["market_comparison_edge"]
            marker = ">" if mce > 0 else "<"
            print(f"    [{marker}] {s['market']}: ours={s['our_prediction']:.3f} "
                  f"mkt_then={s['market_at_forecast']:.3f} "
                  f"mkt_now={s['market_at_eval']:.3f} "
                  f"div={div:+.3f}")

    # -- Reference points --
    print(f"\n  Reference Brier Scores:")
    print(f"    0.000 = Perfect (impossible in practice)")
    print(f"    0.150 = GJP superforecasters (top 2%)")
    print(f"    0.200 = Good forecaster")
    print(f"    0.250 = Coin flip (always predict 50%)")
    print(f"    1.000 = Maximally wrong")

    # Save scorecard
    os.makedirs(SCORES_DIR, exist_ok=True)
    scorecard_path = os.path.join(SCORES_DIR, "latest_scorecard.json")
    with open(scorecard_path, "w") as f:
        json.dump({"scores": scores}, f, indent=2)


def load_all_scores():
    """Load all historical scores from CSV."""
    csv_path = os.path.join(SCORES_DIR, "scores_history.csv")
    if not os.path.exists(csv_path):
        return []

    scores = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["our_brier", "market_brier", "our_se_vs_market",
                        "market_drift_se", "base_rate_se", "edge_vs_market",
                        "market_comparison_edge", "our_prediction",
                        "market_prediction", "market_at_forecast",
                        "market_at_eval", "divergence_at_forecast"]:
                if key in row and row[key]:
                    row[key] = float(row[key])
            for key in ["outcome"]:
                if key in row and row[key]:
                    row[key] = int(row[key])
            for key in ["cycle_number", "evidence_count"]:
                if key in row and row[key]:
                    row[key] = int(row[key])
            scores.append(row)
    return scores


# -- CLI Entry Point ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "resolve":
        # Record a resolution: python evaluate.py resolve <market_key> <0|1>
        if len(sys.argv) < 4:
            print("Usage: python evaluate.py resolve <market_key> <0|1>")
            print("  Example: python evaluate.py resolve regime_fall 0")
            print(f"\nAvailable markets: {', '.join(MARKETS.keys())}")
            sys.exit(1)

        mkt = sys.argv[2]
        outcome = int(sys.argv[3])
        if mkt not in MARKETS:
            print(f"Unknown market: {mkt}")
            print(f"Available: {', '.join(MARKETS.keys())}")
            sys.exit(1)
        record_resolution(mkt, outcome)

    elif len(sys.argv) >= 2 and sys.argv[1] == "resolved":
        # Show all resolved scores: python evaluate.py resolved
        result = evaluate_all_resolved()
        if result:
            print(f"\n{'='*60}")
            print(f"ALL RESOLVED PREDICTIONS ({result['n_resolved']})")
            print(f"{'='*60}")
            print(f"  Brier Score (ours): {result['brier_score']:.4f}")
            print(f"  Brier Score (market): {result['market_brier_score']:.4f}")
            print(f"  Edge vs Market: {result['edge_vs_market']:+.4f}")
            print(f"  Calibration Score: {result['calibration_score']:.4f}")
            print(f"  Resolution Score: {result['resolution_score']:.6f}")

            if result["calibration"]:
                print(f"\n  Calibration Curve:")
                for b in result["calibration"]:
                    print(f"    {b['bin_range']}: predicted={b['mean_prediction']:.2f} "
                          f"actual={b['mean_outcome']:.2f} (n={b['count']})")

    elif len(sys.argv) >= 2 and sys.argv[1] == "resolutions":
        # Show current resolutions
        resolutions = load_resolutions()
        if resolutions:
            print("Recorded resolutions:")
            for mkt, res in resolutions.items():
                outcome_str = "YES" if res["outcome"] == 1 else "NO"
                print(f"  {mkt}: {outcome_str} (resolved {res['resolved_at']}, source: {res['source']})")
        else:
            print("No resolutions recorded yet.")
            print(f"Use: python evaluate.py resolve <market_key> <0|1>")

    else:
        # Default: evaluate latest predictions
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

                # Show all-time resolved stats if available
                result = evaluate_all_resolved()
                if result:
                    print(f"\n{'='*60}")
                    print(f"ALL-TIME RESOLVED ({result['n_resolved']} predictions)")
                    print(f"  Brier: {result['brier_score']:.4f} | "
                          f"Market: {result['market_brier_score']:.4f} | "
                          f"Edge: {result['edge_vs_market']:+.4f}")
            else:
                print("No predictions to evaluate against current prices.")
