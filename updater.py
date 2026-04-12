"""
Tetlock Oracle Labs — updater.py
Bayesian Belief Updater

This module maintains persistent belief states per contract and updates
them using likelihood ratios from the evidence evaluator.

Core formula (Bayes' rule via odds form):
    posterior_odds = prior_odds × likelihood_ratio_1 × likelihood_ratio_2 × ...
    posterior_prob = posterior_odds / (1 + posterior_odds)

This is how superforecasters actually work: they don't start fresh each cycle.
They maintain a running estimate and make incremental adjustments based on
new evidence — turning a dial, not flipping a switch.

State persistence: beliefs are saved to state/{market_key}.json between cycles.
"""

import json
import os
import math
from datetime import datetime, timezone

from constants import MARKETS, STATE_DIR, SHRINKAGE_KEEP, MAX_LR_PER_SQ, MAX_COMBINED_LR


def prob_to_odds(p):
    """Convert probability to odds. P=0.75 → odds=3.0 (3:1 in favor)."""
    p = max(0.001, min(0.999, p))
    return p / (1.0 - p)


def odds_to_prob(odds):
    """Convert odds back to probability. odds=3.0 → P=0.75."""
    return odds / (1.0 + odds)


def load_belief_state(market_key):
    """Load persistent belief state for a market.

    Returns:
        {
            "market": str,
            "probability": float,       # current belief
            "base_rate": float,          # original outside-view anchor
            "update_history": [          # log of all updates
                {
                    "timestamp": str,
                    "prior": float,
                    "posterior": float,
                    "evidence_count": int,
                    "total_lr": float,
                }
            ],
            "cycle_count": int,
        }
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f"{market_key}.json")

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    # No prior state — return None so caller knows to initialize
    return None


def save_belief_state(state):
    """Save belief state to disk."""
    os.makedirs(STATE_DIR, exist_ok=True)
    market_key = state["market"]
    path = os.path.join(STATE_DIR, f"{market_key}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def initialize_belief(market_key, base_rate, market_price):
    """Initialize belief state for a new market.

    Tetlock principle: blend the outside view (base rate) with the market price.
    The market is a strong prior -- it aggregates thousands of participants.
    We use SHRINKAGE_KEEP to weight our base rate vs the crowd.

    initial_prob = market_price + SHRINKAGE_KEEP * (base_rate - market_price)

    With SHRINKAGE_KEEP=0.5, this splits the difference between our base rate
    and the market. This is more principled than ignoring the market at init.
    """
    # Blend base rate with market price using shrinkage
    initial_prob = market_price + SHRINKAGE_KEEP * (base_rate - market_price)
    initial_prob = max(0.02, min(0.98, initial_prob))

    state = {
        "market": market_key,
        "probability": round(initial_prob, 4),
        "base_rate": base_rate,
        "initial_blend": round(initial_prob, 4),
        "market_price_at_init": market_price,
        "update_history": [],
        "cycle_count": 0,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    save_belief_state(state)
    return state


def bayesian_update(belief_state, evidence_evaluations, market_price):
    """Update belief using Bayesian updating with likelihood ratios.

    Process:
    1. Start with current belief (prior for this cycle)
    2. Convert to odds
    3. Multiply by each evidence likelihood ratio
    4. Convert back to probability
    5. Apply overconfidence shrinkage toward market price
    6. Save updated state

    Args:
        belief_state: current state dict
        evidence_evaluations: list from evidence.classify_evidence()
        market_price: current Polymarket price for shrinkage

    Returns:
        updated belief_state dict
    """
    prior = belief_state["probability"]
    prior_odds = prob_to_odds(prior)

    # Group evidence by sub-question to avoid over-updating
    # from multiple pieces of evidence about the same thing
    sq_lrs = {}  # {sq_index: [list of likelihood ratios]}
    for ev in evidence_evaluations:
        sq_lrs.setdefault(ev.get("sub_question_index", 0), []).append(
            ev.get("likelihood_ratio", 1.0)
        )

    # For each sub-question, take the geometric mean of its evidence LRs.
    # This dampens the effect of multiple weak signals about the same thing.
    # Then cap per-SQ LR to prevent runaway updating (Tetlock: small incremental
    # updates, not dramatic swings).
    combined_lr = 1.0
    lr_details = []
    for sq_idx, lrs in sq_lrs.items():
        if not lrs:
            continue
        # Filter out zero/negative LRs that would crash log
        valid_lrs = [lr for lr in lrs if lr > 0]
        if not valid_lrs:
            continue
        # Geometric mean
        log_mean = sum(math.log(lr) for lr in valid_lrs) / len(valid_lrs)
        sq_lr = math.exp(log_mean)
        # Cap per-sub-question LR
        sq_lr = max(1.0 / MAX_LR_PER_SQ, min(MAX_LR_PER_SQ, sq_lr))
        combined_lr *= sq_lr
        lr_details.append({
            "sub_question_index": sq_idx,
            "evidence_count": len(valid_lrs),
            "individual_lrs": [round(lr, 4) for lr in valid_lrs],
            "combined_lr": round(sq_lr, 4),
            "capped": sq_lr == MAX_LR_PER_SQ or sq_lr == 1.0 / MAX_LR_PER_SQ,
        })

    # Cap total combined LR across all sub-questions
    combined_lr = max(1.0 / MAX_COMBINED_LR, min(MAX_COMBINED_LR, combined_lr))

    # Apply Bayesian update
    posterior_odds = prior_odds * combined_lr
    raw_posterior = odds_to_prob(posterior_odds)

    # Clamp to avoid extreme probabilities (Tetlock: stay away from 0 and 1)
    raw_posterior = max(0.02, min(0.98, raw_posterior))

    # Overconfidence shrinkage: pull toward market price
    # adjusted = market + SHRINKAGE_KEEP * (estimate - market)
    shrunk = market_price + SHRINKAGE_KEEP * (raw_posterior - market_price)
    shrunk = max(0.02, min(0.98, shrunk))

    # Update state
    belief_state["probability"] = round(shrunk, 4)
    belief_state["cycle_count"] = belief_state.get("cycle_count", 0) + 1
    belief_state["last_updated"] = datetime.now(timezone.utc).isoformat()
    belief_state["update_history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prior": round(prior, 4),
        "raw_posterior": round(raw_posterior, 4),
        "shrunk_posterior": round(shrunk, 4),
        "market_price": round(market_price, 4),
        "evidence_count": len(evidence_evaluations),
        "total_combined_lr": round(combined_lr, 4),
        "lr_details": lr_details,
    })

    # Keep only last 50 updates to prevent unbounded growth
    if len(belief_state["update_history"]) > 50:
        belief_state["update_history"] = belief_state["update_history"][-50:]

    save_belief_state(belief_state)
    return belief_state


def format_update_summary(belief_state):
    """Format the latest update for display."""
    if not belief_state["update_history"]:
        return f"  {belief_state['market']}: {belief_state['probability']:.3f} (no updates)"

    latest = belief_state["update_history"][-1]
    prior = latest["prior"]
    posterior = latest["shrunk_posterior"]
    lr = latest["total_combined_lr"]
    n = latest["evidence_count"]
    market = latest["market_price"]

    delta = posterior - prior
    arrow = "↑" if delta > 0.005 else "↓" if delta < -0.005 else "→"

    return (
        f"  {belief_state['market']}: {prior:.3f} {arrow} {posterior:.3f} "
        f"(LR={lr:.3f}, {n} evidence, market={market:.3f})"
    )


if __name__ == "__main__":
    # Demo: simulate a Bayesian update
    print("=== Bayesian Updater Demo ===\n")

    state = initialize_belief("regime_fall", base_rate=0.05, market_price=0.08)
    print(f"Initial belief: {state['probability']:.3f}")

    # Simulate some evidence
    fake_evidence = [
        {"sub_question_index": 0, "likelihood_ratio": 1.2, "fact_summary": "Carrier group deployed"},
        {"sub_question_index": 1, "likelihood_ratio": 1.1, "fact_summary": "Rial depreciation"},
        {"sub_question_index": 2, "likelihood_ratio": 0.9, "fact_summary": "Protests contained"},
    ]

    state = bayesian_update(state, fake_evidence, market_price=0.08)
    print(format_update_summary(state))

    # Second cycle with different evidence
    fake_evidence_2 = [
        {"sub_question_index": 0, "likelihood_ratio": 0.8, "fact_summary": "Diplomatic talks resume"},
        {"sub_question_index": 2, "likelihood_ratio": 1.3, "fact_summary": "Large protests in Tehran"},
    ]

    state = bayesian_update(state, fake_evidence_2, market_price=0.07)
    print(format_update_summary(state))
