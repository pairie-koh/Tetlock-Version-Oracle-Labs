"""
Tetlock Oracle Labs -- forecast.py
Main Forecasting Pipeline

This ties together the full Tetlock/BPT methodology:

  0. DATA GATHER -> Collect context from all sources
  1. BASE RATE   -> Outside view anchor (base_rates.py)
  2. DECOMPOSE   -> Break into sub-questions (decompose.py)
  3. EVIDENCE    -> Gather and classify news (newswire.py + evidence.py)
  4. UPDATE      -> Bayesian update with likelihood ratios (updater.py)
  5. ADVERSARIAL -> Argue against own position, then adjust
  6. OUTPUT      -> Final calibrated probability with full audit trail

Data sources integrated:
  - Polymarket CLOB prices (market_data.py)
  - Perplexity + Haiku newswire (newswire.py)
  - GDELT global news monitoring (gdelt.py)
  - Hyperliquid 24/7 perp prices (hyperliquid.py)
  - NWS weather forecasts (weather.py)
  - pmxt order flow signals (orderflow.py)
  - Smart money detector (smart_money.py)
  - Lessons from past predictions (lessons.py)
"""

import json
import os
from datetime import datetime, timezone

from constants import (
    MARKETS, SONNET_MODEL, SHRINKAGE_KEEP, PREDICTIONS_DIR, OPUS_MODEL,
    SONNET_THRESHOLD, OPUS_THRESHOLD, EXTREMIZING_D,
)
from market_data import fetch_all_prices
from newswire import call_openrouter, parse_json_response
from base_rates import estimate_base_rate
from decompose import decompose_question, combine_sub_probabilities
from evidence import classify_evidence, summarize_evidence, select_model_tier
from updater import (
    load_belief_state, initialize_belief, bayesian_update,
    format_update_summary,
)


# -- Data Source Gathering ----------------------------------------------------

def _fetch_smart_money(market_key, market):
    """Fetch smart money signals and format as a text block."""
    from smart_money import analyze_market
    sm_result = analyze_market(market_key, market)
    if sm_result and sm_result.get("signal_strength", 0) > 0:
        sm_lines = [
            f"SMART MONEY SIGNAL ({market_key}):",
            f"  Strength: {sm_result['signal_strength']:.2f}/1.00",
            f"  Direction: {sm_result['direction']}",
        ] + [f"  - {ev}" for ev in sm_result.get("evidence", [])[:5]]
        print(f"    Smart money: signal={sm_result['signal_strength']:.2f} {sm_result['direction']}")
        return "\n".join(sm_lines)
    print("    Smart money: no signal")
    return ""


def gather_supplementary_data(market_key):
    """Gather all supplementary data sources, returning text blocks.

    Each source is wrapped in a try/except so a single failure doesn't
    break the pipeline. Returns a dict of {source_name: text_block}.
    """
    context = {}
    market = MARKETS.get(market_key, {})

    # Sources that return a text block directly via format_for_prompt()
    simple_sources = [
        ("gdelt",       "GDELT",       lambda: __import__("gdelt").format_for_prompt(contract_slug=market.get("slug"))),
        ("hyperliquid", "Hyperliquid", lambda: __import__("hyperliquid").format_for_prompt()),
        ("weather",     "Weather",     lambda: __import__("weather").format_for_prompt()),
        ("orderflow",   "Order flow",  lambda: __import__("orderflow").format_for_prompt(contract_key=market_key)),
        ("smart_money", "Smart money", lambda: _fetch_smart_money(market_key, market)),
        ("lessons",     "Lessons",     lambda: __import__("lessons").build_lessons_block(market_key, domain=market.get("domain", ""))),
    ]

    for source_key, label, fetcher in simple_sources:
        try:
            block = fetcher()
            if block:
                context[source_key] = block
                if source_key not in ("smart_money",):  # smart_money prints its own status
                    print(f"    {label}: loaded")
        except Exception as e:
            print(f"    {label}: unavailable ({e})")

    return context


def build_context_block(supplementary_data):
    """Merge all supplementary data into a single context block for prompts."""
    return "\n\n".join(supplementary_data.values()) if supplementary_data else ""


# -- Extremizing --------------------------------------------------------------

def extremize(probability, d=None):
    """Push probability away from 50% (Tetlock's extremizing transform).

    Tetlock's GJP found that aggregated forecasts are systematically too
    close to 50%. The extremizing transform corrects this by pushing
    predictions toward 0 or 1 based on how far they are from 50%.

    Formula: extremized = 0.5 + (1 + d) * (p - 0.5)

    Where d is the extremizing constant:
      d = 0.0: no change
      d = 0.3: moderate push (GJP typical for individual forecasters)
      d = 0.5: strong push (GJP typical for team aggregates)

    Only apply when we have evidence-backed divergence from 50%.
    Don't extremize if we're already very close to 50% (not enough signal).

    References:
      Baron et al. (2014), "Two Reasons to Make Aggregated Probability
      Forecasts More Extreme"
    """
    if d is None:
        d = EXTREMIZING_D

    if d == 0:
        return probability

    extremized = 0.5 + (1.0 + d) * (probability - 0.5)
    return max(0.02, min(0.98, extremized))


# -- Adversarial Review -------------------------------------------------------

def adversarial_review(market_key, current_prob, evidence_summary,
                       reasoning_so_far, context_block=""):
    """Tetlock's adversarial thinking: argue against your own position.

    Force the model to construct the strongest case AGAINST the current estimate,
    then decide if an adjustment is warranted.

    Uses Opus for high-divergence contracts, Sonnet otherwise.
    """
    market = MARKETS[market_key]

    # Use higher-tier model for adversarial review
    model = SONNET_MODEL

    extra_context = ""
    if context_block:
        extra_context = f"\nADDITIONAL CONTEXT:\n{context_block}\n"

    prompt = f"""You are a superforecaster performing an adversarial review.

QUESTION: {market['name']}
CURRENT ESTIMATE: {current_prob:.3f} ({current_prob*100:.1f}%)

EVIDENCE CONSIDERED:
{evidence_summary}

REASONING SO FAR:
{reasoning_so_far}
{extra_context}
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

    raw = call_openrouter(prompt, model, max_tokens=800)

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


# -- Main Pipeline ------------------------------------------------------------

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
    print("\n[0/6] Fetching market prices...")
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

        # -- Step 0: Gather supplementary data --
        print(f"\n[0/6] Gathering supplementary data...")
        supp_data = gather_supplementary_data(market_key)
        context_block = build_context_block(supp_data)
        print(f"  {len(supp_data)} data sources loaded")

        # -- Step 1: Base Rate --
        print(f"\n[1/6] Establishing base rate (outside view)...")
        belief_state = load_belief_state(market_key)

        if belief_state is None:
            base_rate_info = estimate_base_rate(market_key)
            base_rate = base_rate_info["base_rate"]
            belief_state = initialize_belief(market_key, base_rate, market_price)
            print(f"  Base rate: {base_rate:.3f}")
            print(f"  Reference class: {base_rate_info['reference_class'][:80]}")
        else:
            base_rate_info = {
                "base_rate": belief_state["base_rate"],
                "reference_class": "loaded from prior cycle",
            }
            print(f"  Loaded prior belief: {belief_state['probability']:.3f}")
            print(f"  (cycle #{belief_state['cycle_count'] + 1})")

        # -- Step 2: Decompose --
        print(f"\n[2/6] Decomposing question...")
        decomposition = decompose_question(market_key, base_rate_info)
        sub_questions = decomposition.get("sub_questions", [])
        print(f"  {len(sub_questions)} sub-questions, logic: {decomposition.get('combination_logic')}")
        for i, sq in enumerate(sub_questions):
            print(f"    SQ{i}: [{sq['relationship']}] {sq['question'][:70]}")

        # -- Step 3: Classify Evidence (with three-tier LLM) --
        # Use prior divergence if available, else None (defaults to Sonnet)
        prior_div = None
        if belief_state.get("probability") is not None:
            prior_div = belief_state["probability"] - market_price

        print(f"\n[3/6] Evaluating evidence (BPT)...")
        evidence_evals = classify_evidence(
            facts, market_key, sub_questions, divergence=prior_div
        )
        print(f"  {len(evidence_evals)} evidence items evaluated")
        ev_summary = summarize_evidence(evidence_evals)
        if evidence_evals:
            print(ev_summary)

        # -- Step 4: Bayesian Update --
        print(f"\n[4/6] Bayesian update...")
        belief_state = bayesian_update(belief_state, evidence_evals, market_price)
        print(format_update_summary(belief_state))

        current_prob = belief_state["probability"]

        # -- Step 5: Adversarial Review --
        print(f"\n[5/6] Adversarial review...")
        latest_update = belief_state["update_history"][-1] if belief_state["update_history"] else {}
        reasoning = (
            f"Base rate: {base_rate_info['base_rate']:.3f}. "
            f"After {len(evidence_evals)} evidence items, "
            f"combined LR={latest_update.get('total_combined_lr', 1.0):.3f}. "
            f"Raw posterior: {latest_update.get('raw_posterior', current_prob):.3f}. "
            f"After shrinkage toward market ({market_price:.3f}): {current_prob:.3f}."
        )

        adversarial = adversarial_review(
            market_key, current_prob, ev_summary, reasoning,
            context_block=context_block,
        )
        adj = adversarial["adjustment"]

        post_adversarial = max(0.02, min(0.98, current_prob + adj))
        print(f"  Counter-argument: {adversarial['counter_argument'][:80]}...")
        print(f"  Adjustment: {adj:+.3f}")
        print(f"  Post-adversarial: {post_adversarial:.3f}")

        # -- Step 5b: Extremizing (Tetlock GJP transform) --
        # Push away from 50% -- corrects for the systematic tendency of
        # aggregated/shrunk forecasts to be too close to the center.
        # Only extremize if we have meaningful evidence (not just base rate).
        pre_extremize = post_adversarial
        if len(evidence_evals) > 0:
            final_prob = extremize(post_adversarial)
            if abs(final_prob - pre_extremize) > 0.001:
                print(f"  Extremized: {pre_extremize:.3f} -> {final_prob:.3f} "
                      f"(d={EXTREMIZING_D})")
        else:
            final_prob = post_adversarial
            print(f"  Extremizing skipped (no evidence)")

        # Save to belief state
        if abs(final_prob - current_prob) > 0.001:
            belief_state["probability"] = round(final_prob, 4)
            from updater import save_belief_state
            save_belief_state(belief_state)

        print(f"  Final probability: {final_prob:.3f}")

        # -- Step 6: Build Prediction Record --
        divergence = final_prob - market_price
        model_used, tier = select_model_tier(prior_div)

        prediction = {
            "market": market_key,
            "question": market["name"],
            "timestamp": timestamp,
            "market_price": round(market_price, 4),
            "base_rate": round(base_rate_info["base_rate"], 4),
            "pre_adversarial": round(current_prob, 4),
            "adversarial_adjustment": round(adj, 4),
            "pre_extremize": round(pre_extremize, 4),
            "extremizing_d": EXTREMIZING_D,
            "final_probability": round(final_prob, 4),
            "divergence_from_market": round(divergence, 4),
            "evidence_count": len(evidence_evals),
            "sub_question_count": len(sub_questions),
            "combination_logic": decomposition.get("combination_logic"),
            "cycle_number": belief_state["cycle_count"],
            "model_tier": tier,
            "shrinkage_keep": SHRINKAGE_KEEP,
            "data_sources_used": list(supp_data.keys()),
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

        print(f"\n  SUMMARY: {market_price:.3f} (market) -> {final_prob:.3f} (ours) | "
              f"divergence: {divergence:+.3f} | tier: {tier}")

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
        print(f"  Model tier: {pred['model_tier']}")
        print(f"  Data sources: {pred['data_sources_used']}")
