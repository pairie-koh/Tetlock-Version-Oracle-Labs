"""
Tetlock Oracle Labs — base_rates.py
Outside View / Reference Class Forecasting

Tetlock's first principle: before analyzing the specifics, ask
"how often do events like this happen?" That's your anchor.

This module asks the LLM to:
1. Identify the reference class for a given question
2. Estimate the base rate from historical frequency
3. Return a prior probability grounded in the outside view

This is Step 1 of the Tetlock pipeline — the anchor before any
inside-view evidence adjusts the estimate.
"""

import json
import os
from datetime import datetime, timezone

from constants import MARKETS, SONNET_MODEL
from newswire import call_openrouter, parse_json_response


def estimate_base_rate(market_key):
    """Ask the LLM to establish a base rate using reference class forecasting.

    Returns:
        {
            "market": str,
            "reference_class": str,           # what class of events this belongs to
            "historical_frequency": str,       # how often events in this class resolve YES
            "base_rate": float,                # the prior probability (0.0-1.0)
            "reasoning": str,                  # why this reference class was chosen
            "caveats": str,                    # what makes this case unusual
        }
    """
    market = MARKETS[market_key]
    question = market["name"]
    end_date = market.get("end_date", "unknown")
    domain = market.get("domain", "unknown")

    now = datetime.now(timezone.utc)
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_left = (end_dt - now).days
    except (ValueError, TypeError):
        days_left = "unknown"

    prompt = f"""You are a superforecaster using reference class forecasting (the "outside view").

QUESTION: {question}
DOMAIN: {domain}
RESOLUTION DATE: {end_date}
DAYS REMAINING: {days_left}

Your task is to establish a BASE RATE — the prior probability BEFORE looking at any current news or evidence.

Follow this process:
1. IDENTIFY THE REFERENCE CLASS: What broader category of events does this belong to?
   Be specific. "Regime change in authoritarian states" is better than "political events."
   Consider the time horizon — what's the annual rate, then adjust for the specific window.

2. ESTIMATE HISTORICAL FREQUENCY: How often have events in this reference class occurred?
   Use concrete numbers where possible. "3 out of ~50 authoritarian regimes have fallen
   to popular uprising in the last 20 years" is better than "rarely."

3. ADJUST FOR TIME WINDOW: The question asks about a specific period ({days_left} days).
   Convert annual or decadal rates to this window.

4. NOTE CAVEATS: What makes this specific case different from the reference class?
   These are NOT adjustments yet — just flags for the inside-view analysis later.

Return a JSON object with these exact fields:
- "reference_class": description of the reference class (1-2 sentences)
- "historical_frequency": concrete historical data supporting the rate (2-3 sentences)
- "base_rate": a float between 0.0 and 1.0
- "reasoning": your reasoning chain (3-5 sentences)
- "caveats": what makes this case potentially different (2-3 sentences)

IMPORTANT:
- Be humble. The outside view exists to prevent overconfidence.
- Most dramatic-sounding events have low base rates.
- If unsure, err toward the base rate being low for dramatic events.
- Do NOT incorporate current news. This is ONLY about historical frequency.

Return ONLY the JSON object. No markdown, no explanation outside the JSON."""

    raw = call_openrouter(prompt, SONNET_MODEL, max_tokens=1500)

    try:
        result = parse_json_response(raw)
        result["market"] = market_key
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Validate base_rate is in range
        br = float(result.get("base_rate", 0.5))
        result["base_rate"] = max(0.01, min(0.99, br))

        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  WARN: Failed to parse base rate response: {e}")
        return {
            "market": market_key,
            "reference_class": "unknown",
            "historical_frequency": "could not determine",
            "base_rate": 0.5,
            "reasoning": f"Failed to parse LLM response: {e}",
            "caveats": "Using uninformative prior due to parse failure",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_all_base_rates():
    """Estimate base rates for all markets."""
    print("=== Base Rate Estimation (Outside View) ===")
    results = {}
    for key in MARKETS:
        print(f"  Estimating base rate for {key}...")
        result = estimate_base_rate(key)
        results[key] = result
        print(f"    Reference class: {result['reference_class'][:80]}...")
        print(f"    Base rate: {result['base_rate']:.3f}")
    return results


if __name__ == "__main__":
    rates = get_all_base_rates()
    for key, r in rates.items():
        print(f"\n{'='*60}")
        print(f"Market: {key}")
        print(f"Reference class: {r['reference_class']}")
        print(f"Historical frequency: {r['historical_frequency']}")
        print(f"Base rate: {r['base_rate']:.3f}")
        print(f"Reasoning: {r['reasoning']}")
        print(f"Caveats: {r['caveats']}")
