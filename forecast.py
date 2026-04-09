"""
Tetlock Oracle Labs — forecast.py
Main Forecasting Pipeline

This ties together the full Tetlock/BPT methodology:

  1. BASE RATE  → Outside view anchor (base_rates.py)
  2. DECOMPOSE  → Break into sub-questions (decompose.py)
  3. EVIDENCE   → Gather and classify news (newswire.py + evidence.py)
  4. UPDATE     → Bayesian update with likelihood ratios (updater.py)
  5. ADVERSARIAL → Argue against own position, then adjust
  6. OUTPUT     → Final calibrated probability with full audit trail

Each step produces structured output that feeds the next.
The entire reasoning chain is logged for post-hoc analysis.
"""

import json
import os
from datetime import datetime, timezone

from constants import MARKETS, SONNET_MODEL, SHRINKAGE_KEEP, PREDICTIONS_DIR
from market_data import fetch_all_prices
from newswire import call_openrouter, parse_json_response
from base_rates import estimate_base_rate
from decompose import decompose_question, combine_sub_probabilities
from evidence import classify_evidence, summarize_evidence
from updater import (
    load_belief_state, initialize_belief, bayesian_update,
    format_update_summary,
)


def adversarial_review(market_key, current_prob, evidence_summary, reasoning_so_far):
    """Tetlock's adversarial thinking: argue against your own position.

    Force the model to construct the strongest case AGAINST the current estimate,
    then decide if an adjustment is warranted.

    Returns:
        {
            "adjustment": float,     # how much to adjust (-0.15 to +0.15)
            "counter_argument": str, # the strongest argument against current position
            "response": str,         # why the adjustment is or isn't warranted
        }
    """
    market = MARKETS[market_key]

    prompt = f"""You are a superforecaster performing an adversarial review.

QUESTION: {market['name']}
CURRENT ESTIMATE: {current_prob:.3f} ({current_prob*100:.1f}%)

EVIDENCE CONSIDERED:
{evidence_summary}

REASONING SO FAR:
{reasoning_so_far}

YOUR TASK: Construct the STRONGEST possible argument that this estimate is WRONG.

1. If the estimate seems too HIGH, argue for why it should be lower.
2. If the estimate seems too LOW, argue for why it should be higher.
3. Consider: What evidence might we be missing? What assumptions are we making?
   What would a smart person who disagrees with us say?

After making your counter-argument, decide: does this warrant an adjustment?

Return a JSON object:
{{
    "counter_argument": "The strongest 2-3 sentence argument against the current estimate",
    "adjustment": a float between -0.15 and +0.15 (0.0 means no change warranted),
    "response": "1-2 sentences explaining why you did or didn't adjust"
}}

IMPORTANT:
- Most of the time, the adjustment should be small or zero.
- Only make large adjustments (>0.05) if you found a genuine blind spot.
- This is about intellectual honesty, not contrarianism for its own sake.

Return ONLY the JSON object."""

    raw = call_openrouter(prompt, SONNET_MODEL, max_tokens=800)

    try:
        result = parse_json_response(raw)
        adj = float(result.get("adjustment", 0.0))
        result["adjustment"] = max(-0.15, min(0.15, adj))
        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {
            "adjustment": 0.0,
            "counter_argument": f"Failed to generate: {e}",
            "response": "No adjustment due to parse failure.",
        }


def run_forecast(briefing):
    """Run the full Tetlock forecasting pipeline for all markets.

    Args:
        briefing: output from newswire.run_newswire() containing facts

    Returns:
        dict of {market_key: prediction_record}
    """
    facts = briefing.get("facts", [])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("\n=== Tetlock Forecast Pipeline ===")
    print(f"Timestamp: {timestamp}")
    print(f"Facts available: {len(facts)}")

    # Fetch current market prices
    print("\n[0/5] Fetching market prices...")
    prices = fetch_all_prices()

    predictions = {}

    for market_key in MARKETS:
        market = MARKETS[market_key]
        market_price = prices.get(market_key)

        if market_price is None:
            print(f"\n  SKIP {market_key}: no price available")
            continue

        print(f"\n{'='*60}")
        print(f"Market: {market['name']}")
        print(f"Current market price: {market_price:.4f}")

        # ── Step 1: Base Rate ──
        print(f"\n[1/5] Establishing base rate (outside view)...")
        belief_state = load_belief_state(market_key)

        if belief_state is None:
            # First cycle — establish base rate
            base_rate_info = estimate_base_rate(market_key)
            base_rate = base_rate_info["base_rate"]
            belief_state = initialize_belief(market_key, base_rate, market_price)
            print(f"  Base rate: {base_rate:.3f}")
            print(f"  Reference class: {base_rate_info['reference_class'][:80]}")
        else:
            # Subsequent cycles — use existing belief as prior
            base_rate_info = {
                "base_rate": belief_state["base_rate"],
                "reference_class": "loaded from prior cycle",
            }
            print(f"  Loaded prior belief: {belief_state['probability']:.3f}")
            print(f"  (cycle #{belief_state['cycle_count'] + 1})")

        # ── Step 2: Decompose ──
        print(f"\n[2/5] Decomposing question...")
        decomposition = decompose_question(market_key, base_rate_info)
        sub_questions = decomposition.get("sub_questions", [])
        print(f"  {len(sub_questions)} sub-questions, logic: {decomposition.get('combination_logic')}")
        for i, sq in enumerate(sub_questions):
            print(f"    SQ{i}: [{sq['relationship']}] {sq['question'][:70]}")

        # ── Step 3: Classify Evidence ──
        print(f"\n[3/5] Evaluating evidence (BPT)...")
        evidence_evals = classify_evidence(facts, market_key, sub_questions)
        print(f"  {len(evidence_evals)} evidence items evaluated")
        ev_summary = summarize_evidence(evidence_evals)
        if evidence_evals:
            print(ev_summary)

        # ── Step 4: Bayesian Update ──
        print(f"\n[4/5] Bayesian update...")
        belief_state = bayesian_update(belief_state, evidence_evals, market_price)
        print(format_update_summary(belief_state))

        current_prob = belief_state["probability"]

        # ── Step 5: Adversarial Review ──
        print(f"\n[5/5] Adversarial review...")
        latest_update = belief_state["update_history"][-1] if belief_state["update_history"] else {}
        reasoning = (
            f"Base rate: {base_rate_info['base_rate']:.3f}. "
            f"After {len(evidence_evals)} evidence items, "
            f"combined LR={latest_update.get('total_combined_lr', 1.0):.3f}. "
            f"Raw posterior: {latest_update.get('raw_posterior', current_prob):.3f}. "
            f"After shrinkage toward market ({market_price:.3f}): {current_prob:.3f}."
        )

        adversarial = adversarial_review(market_key, current_prob, ev_summary, reasoning)
        adj = adversarial["adjustment"]

        final_prob = max(0.02, min(0.98, current_prob + adj))
        print(f"  Counter-argument: {adversarial['counter_argument'][:80]}...")
        print(f"  Adjustment: {adj:+.3f}")
        print(f"  Final probability: {final_prob:.3f}")

        # Save adversarial adjustment to belief state
        if abs(adj) > 0.001:
            belief_state["probability"] = round(final_prob, 4)
            from updater import save_belief_state
            save_belief_state(belief_state)

        # ── Build Prediction Record ──
        divergence = final_prob - market_price

        prediction = {
            "market": market_key,
            "question": market["name"],
            "timestamp": timestamp,
            "market_price": round(market_price, 4),
            "base_rate": round(base_rate_info["base_rate"], 4),
            "pre_adversarial": round(current_prob, 4),
            "adversarial_adjustment": round(adj, 4),
            "final_probability": round(final_prob, 4),
            "divergence_from_market": round(divergence, 4),
            "evidence_count": len(evidence_evals),
            "sub_question_count": len(sub_questions),
            "combination_logic": decomposition.get("combination_logic"),
            "cycle_number": belief_state["cycle_count"],
            "shrinkage_keep": SHRINKAGE_KEEP,
            "reasoning_chain": {
                "base_rate": base_rate_info,
                "decomposition": decomposition,
                "evidence_summary": [
                    {k: v for k, v in ev.items() if k != "reasoning"}
                    for ev in evidence_evals
                ],
                "adversarial": adversarial,
            },
        }

        predictions[market_key] = prediction

        print(f"\n  SUMMARY: {market_price:.3f} (market) → {final_prob:.3f} (ours) | "
              f"divergence: {divergence:+.3f}")

    # Save predictions
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pred_path = os.path.join(
        PREDICTIONS_DIR,
        f"{timestamp.replace(':', '').replace('-', '')}.json"
    )
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    with open(os.path.join(PREDICTIONS_DIR, "latest.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\n=== Forecasts saved to {pred_path} ===")
    return predictions


if __name__ == "__main__":
    # Run standalone with latest briefing
    import sys

    briefing_path = os.path.join("briefings", "latest.json")
    if len(sys.argv) > 1:
        briefing_path = sys.argv[1]

    if os.path.exists(briefing_path):
        with open(briefing_path, "r") as f:
            briefing = json.load(f)
    else:
        print(f"No briefing at {briefing_path}. Running newswire first...")
        from newswire import run_newswire
        briefing = run_newswire()

    predictions = run_forecast(briefing)

    for key, pred in predictions.items():
        print(f"\n{'='*40}")
        print(f"{pred['question']}")
        print(f"  Market: {pred['market_price']:.3f}")
        print(f"  Base rate: {pred['base_rate']:.3f}")
        print(f"  Our estimate: {pred['final_probability']:.3f}")
        print(f"  Divergence: {pred['divergence_from_market']:+.3f}")
