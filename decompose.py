"""
Tetlock Oracle Labs — decompose.py
Question Decomposition into Sub-Questions

Tetlock's key insight: "How superforecasters work — they break questions
into sub-questions and estimate components separately."

Instead of asking "What's the probability of X?", this module breaks X
into necessary conditions and estimates each one. The final probability
is the product of the sub-question probabilities (for AND conditions)
or uses appropriate combination logic.

This is Step 2 of the Tetlock pipeline.
"""

import json
from datetime import datetime, timezone

from constants import MARKETS, DECOMPOSITION_HINTS, SONNET_MODEL, MAX_SUB_QUESTIONS
from newswire import call_openrouter, parse_json_response


def decompose_question(market_key, base_rate_info):
    """Break a market question into sub-questions.

    Args:
        market_key: key into MARKETS dict
        base_rate_info: output from base_rates.estimate_base_rate()

    Returns:
        {
            "market": str,
            "main_question": str,
            "sub_questions": [
                {
                    "question": str,
                    "relationship": "necessary" | "sufficient" | "contributing",
                    "weight": float,  # importance weight 0-1
                }
            ],
            "combination_logic": str,  # how sub-questions combine
            "reasoning": str,
        }
    """
    market = MARKETS[market_key]
    question = market["name"]
    hints = DECOMPOSITION_HINTS.get(market_key, [])

    hints_block = ""
    if hints:
        hints_block = "\nHere are some potentially relevant sub-questions to consider:\n"
        for h in hints:
            hints_block += f"  - {h}\n"
        hints_block += "\nYou may use these, modify them, or propose entirely different ones.\n"

    prompt = f"""You are a superforecaster decomposing a complex question into sub-questions.

MAIN QUESTION: {question}

BASE RATE (outside view): {base_rate_info.get('base_rate', 'unknown')}
REFERENCE CLASS: {base_rate_info.get('reference_class', 'unknown')}
{hints_block}
TASK: Break this question into {MAX_SUB_QUESTIONS} or fewer sub-questions that, together,
capture the key factors determining whether this event occurs.

For each sub-question:
1. State the question clearly and specifically
2. Classify the relationship to the main question:
   - "necessary": this MUST be true for the main event to occur
   - "sufficient": this alone WOULD cause the main event
   - "contributing": this makes the main event more likely but isn't required
3. Assign a weight (0.0-1.0) reflecting how important this factor is

Also specify the combination logic:
- "all_necessary": main event requires ALL necessary conditions (multiply probabilities)
- "any_sufficient": main event occurs if ANY sufficient condition is met
- "weighted_average": factors contribute independently (weighted sum)
- "mixed": combination of the above (explain in reasoning)

IMPORTANT:
- Sub-questions should be as independent as possible (avoid double-counting)
- Each should be something we can evaluate with evidence
- Prefer concrete, observable conditions over abstract ones
- The decomposition should explain MOST of the variance in the outcome

Return a JSON object:
{{
    "sub_questions": [
        {{
            "question": "...",
            "relationship": "necessary" | "sufficient" | "contributing",
            "weight": 0.0-1.0
        }}
    ],
    "combination_logic": "all_necessary" | "any_sufficient" | "weighted_average" | "mixed",
    "reasoning": "2-3 sentences explaining why you decomposed it this way"
}}

Return ONLY the JSON object."""

    raw = call_openrouter(prompt, SONNET_MODEL, max_tokens=2000)

    try:
        result = parse_json_response(raw)
        result["market"] = market_key
        result["main_question"] = question
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Validate sub_questions
        sqs = result.get("sub_questions", [])
        for sq in sqs:
            sq["weight"] = max(0.0, min(1.0, float(sq.get("weight", 0.5))))
            if sq.get("relationship") not in ("necessary", "sufficient", "contributing"):
                sq["relationship"] = "contributing"

        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  WARN: Failed to parse decomposition: {e}")
        # Fallback: use hints as sub-questions
        fallback_sqs = [
            {"question": h, "relationship": "contributing", "weight": 1.0 / len(hints)}
            for h in (hints or ["Is this event likely to occur?"])
        ]
        return {
            "market": market_key,
            "main_question": question,
            "sub_questions": fallback_sqs,
            "combination_logic": "weighted_average",
            "reasoning": f"Fallback decomposition due to parse failure: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def combine_sub_probabilities(decomposition, sub_probs):
    """Combine sub-question probabilities into a main question probability.

    Args:
        decomposition: output from decompose_question()
        sub_probs: list of floats, one per sub-question

    Returns:
        float: combined probability
    """
    logic = decomposition.get("combination_logic", "weighted_average")
    sqs = decomposition.get("sub_questions", [])

    if not sub_probs:
        return 0.5

    if logic == "all_necessary":
        # P(main) = product of P(each necessary condition)
        prob = 1.0
        for p in sub_probs:
            prob *= p
        return prob

    elif logic == "any_sufficient":
        # P(main) = 1 - product of (1 - P(each sufficient condition))
        prob = 1.0
        for p in sub_probs:
            prob *= (1.0 - p)
        return 1.0 - prob

    elif logic == "weighted_average":
        # P(main) = weighted average of sub-probabilities
        weights = [sq.get("weight", 1.0) for sq in sqs]
        total_weight = sum(weights)
        if total_weight == 0:
            return sum(sub_probs) / len(sub_probs)
        return sum(p * w for p, w in zip(sub_probs, weights)) / total_weight

    else:
        # Mixed or unknown: use weighted average as default
        weights = [sq.get("weight", 1.0) for sq in sqs]
        total_weight = sum(weights)
        if total_weight == 0:
            return sum(sub_probs) / len(sub_probs)
        return sum(p * w for p, w in zip(sub_probs, weights)) / total_weight


if __name__ == "__main__":
    from base_rates import estimate_base_rate

    for key in MARKETS:
        print(f"\n{'='*60}")
        print(f"Decomposing: {MARKETS[key]['name']}")

        br = estimate_base_rate(key)
        print(f"Base rate: {br['base_rate']:.3f}")

        decomp = decompose_question(key, br)
        print(f"\nCombination logic: {decomp['combination_logic']}")
        print(f"Reasoning: {decomp['reasoning']}")
        print(f"\nSub-questions:")
        for i, sq in enumerate(decomp["sub_questions"], 1):
            print(f"  {i}. [{sq['relationship']}] (w={sq['weight']:.2f}) {sq['question']}")
