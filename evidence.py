"""
Tetlock Oracle Labs — evidence.py
Bayesian Process Tracing (BPT) Evidence Evaluation

This is the core methodological innovation. Instead of asking the LLM
"what's the probability?", we force it through structured diagnostic tests
from political science's Bayesian Process Tracing framework:

  - Straw-in-the-wind: weak evidence, slightly adjusts beliefs
  - Hoop test: necessary but not sufficient (failing it eliminates a hypothesis)
  - Smoking gun: sufficient but not necessary (passing it confirms a hypothesis)
  - Doubly decisive: both necessary and sufficient (rare in practice)

For each piece of evidence, the LLM estimates:
  P(evidence | H_yes) — likelihood if the event WILL happen
  P(evidence | H_no)  — likelihood if the event WON'T happen

The ratio P(E|H_yes) / P(E|H_no) is the likelihood ratio — the Bayesian
update factor. This is structurally different from asking for P(H|E) directly,
and forces the LLM to reason about evidence diagnosticity rather than just
evidence volume.

References:
  - Fairfield & Charman (2017), "Explicit Bayesian Analysis for Process Tracing"
  - Bennett (2015), "Disciplining Our Conjectures"
  - Tetlock & Gardner (2015), "Superforecasting"
"""

import json
from datetime import datetime, timezone

from constants import (
    MARKETS, HAIKU_MODEL, SONNET_MODEL, OPUS_MODEL,
    DIAGNOSTIC_TESTS, SONNET_THRESHOLD, OPUS_THRESHOLD,
)
from newswire import call_openrouter, parse_json_response


def select_model_tier(divergence_from_market):
    """Select LLM tier based on how far our estimate diverges from market.

    Three-tier system from Oracle Labs v2:
      - Haiku: low divergence (<5%), routine triage
      - Sonnet: moderate divergence (5-15%), deeper analysis
      - Opus: high divergence (>15%), highest-stakes reasoning

    Returns (model_id, tier_name).
    """
    abs_div = abs(divergence_from_market) if divergence_from_market else 0

    if abs_div >= OPUS_THRESHOLD:
        return OPUS_MODEL, "opus"
    elif abs_div >= SONNET_THRESHOLD:
        return SONNET_MODEL, "sonnet"
    else:
        return HAIKU_MODEL, "haiku"


def classify_evidence(facts, market_key, sub_questions, divergence=None):
    """Classify each fact by its diagnostic type and compute likelihood ratios.

    For each fact, relative to each sub-question, the LLM estimates:
      - diagnostic_type: straw_in_the_wind | hoop | smoking_gun | doubly_decisive
      - p_evidence_given_yes: P(this evidence | sub-question answer is YES)
      - p_evidence_given_no:  P(this evidence | sub-question answer is NO)
      - direction: "supports_yes" | "supports_no" | "ambiguous"

    Args:
        facts: list of fact dicts from newswire
        market_key: which market these facts relate to
        sub_questions: list of sub-question dicts from decompose
        divergence: optional float, current divergence from market price.
            Used to select Haiku/Sonnet/Opus tier. Defaults to Sonnet.

    Returns:
        list of evidence evaluations, each containing:
        {
            "fact": str,
            "sub_question": str,
            "sub_question_index": int,
            "diagnostic_type": str,
            "p_evidence_given_yes": float,
            "p_evidence_given_no": float,
            "likelihood_ratio": float,
            "direction": str,
            "reasoning": str,
            "model_tier": str,
        }
    """
    # Select model tier based on divergence
    if divergence is not None:
        model, tier = select_model_tier(divergence)
    else:
        model, tier = SONNET_MODEL, "sonnet"

    market = MARKETS[market_key]
    question = market["name"]

    # Filter facts relevant to this market
    market_facts = [f for f in facts if f.get("market") == market_key]
    if not market_facts:
        market_facts = facts  # if no market tag, use all

    if not market_facts or not sub_questions:
        return []

    # Format facts for the prompt
    facts_block = ""
    for i, fact in enumerate(market_facts[:20], 1):  # cap at 20 facts
        claim = fact.get("claim", str(fact))
        source = fact.get("source", "unknown")
        conf = fact.get("confidence", "medium")
        facts_block += f"  FACT {i}: [{source}, {conf}] {claim}\n"

    # Format sub-questions
    sq_block = ""
    for i, sq in enumerate(sub_questions):
        sq_block += f"  SQ{i}: {sq['question']}\n"

    prompt = f"""You are a Bayesian process tracer evaluating evidence diagnosticity.

MAIN QUESTION: {question}

SUB-QUESTIONS:
{sq_block}

EVIDENCE:
{facts_block}

For each piece of evidence, evaluate it against the MOST RELEVANT sub-question.
For each fact, provide:

1. DIAGNOSTIC TYPE — classify the evidence:
   - "straw_in_the_wind": Weak evidence. Could be consistent with multiple hypotheses.
     Slightly adjusts beliefs. (Likelihood ratio close to 1.0, e.g., 1.1-1.5 or 0.7-0.9)
   - "hoop": The hypothesis MUST pass this test to remain viable, but passing doesn't
     confirm it. Failing it is very damaging. (Failing: ratio < 0.3. Passing: ratio ~1.0-1.5)
   - "smoking_gun": Finding this evidence strongly confirms the hypothesis, but not
     finding it doesn't rule it out. (Finding: ratio > 3.0. Not finding: ratio ~0.7-1.0)
   - "doubly_decisive": Both necessary and sufficient. Rare. (ratio > 5.0 or < 0.2)

2. LIKELIHOOD ESTIMATES:
   - P(evidence | YES): How likely would we see this evidence if the sub-question answer is YES?
   - P(evidence | NO): How likely would we see this evidence if the sub-question answer is NO?
   Both should be between 0.05 and 0.95.

3. DIRECTION: Does this evidence support YES, NO, or is it ambiguous?

CRITICAL RULES:
- Think about DIAGNOSTICITY, not just direction. A piece of evidence can point toward YES
  but be completely undiagnostic (equally likely under both hypotheses).
- Routine news that would occur regardless of outcome has likelihood ratio ≈ 1.0.
- Be conservative with likelihood ratios. Most evidence is straw-in-the-wind.
- Smoking guns and doubly decisive evidence are RARE. Don't over-classify.

Return a JSON array of objects:
[
  {{
    "fact_index": 1,
    "fact_summary": "brief summary",
    "sub_question_index": 0,
    "diagnostic_type": "straw_in_the_wind",
    "p_evidence_given_yes": 0.6,
    "p_evidence_given_no": 0.5,
    "direction": "supports_yes",
    "reasoning": "1-2 sentences"
  }}
]

Return ONLY the JSON array."""

    print(f"  Using {tier} tier ({model}) for evidence evaluation")
    raw = call_openrouter(prompt, model, max_tokens=4000)

    try:
        evaluations = parse_json_response(raw)
        if not isinstance(evaluations, list):
            return []

        # Compute likelihood ratios and validate
        processed = []
        for ev in evaluations:
            p_yes = float(ev.get("p_evidence_given_yes", 0.5))
            p_no = float(ev.get("p_evidence_given_no", 0.5))

            # Clamp to avoid division by zero and extreme ratios
            p_yes = max(0.05, min(0.95, p_yes))
            p_no = max(0.05, min(0.95, p_no))

            lr = p_yes / p_no

            # Validate diagnostic type against likelihood ratio
            dtype = ev.get("diagnostic_type", "straw_in_the_wind")
            if dtype not in DIAGNOSTIC_TESTS:
                dtype = "straw_in_the_wind"

            sq_idx = int(ev.get("sub_question_index", 0))
            sq_text = sub_questions[sq_idx]["question"] if sq_idx < len(sub_questions) else "unknown"

            processed.append({
                "fact_index": ev.get("fact_index", 0),
                "fact_summary": ev.get("fact_summary", ""),
                "sub_question": sq_text,
                "sub_question_index": sq_idx,
                "diagnostic_type": dtype,
                "p_evidence_given_yes": p_yes,
                "p_evidence_given_no": p_no,
                "likelihood_ratio": round(lr, 4),
                "direction": ev.get("direction", "ambiguous"),
                "reasoning": ev.get("reasoning", ""),
                "model_tier": tier,
            })

        return processed

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  WARN: Failed to parse evidence evaluations: {e}")
        return []


def summarize_evidence(evaluations):
    """Summarize evidence evaluations for logging/display."""
    if not evaluations:
        return "No evidence evaluated."

    lines = []
    for ev in evaluations:
        lr = ev["likelihood_ratio"]
        direction = "→" if ev["direction"] == "supports_yes" else "←" if ev["direction"] == "supports_no" else "?"
        lines.append(
            f"  {direction} [{ev['diagnostic_type']}] LR={lr:.2f} | "
            f"SQ{ev['sub_question_index']}: {ev['fact_summary'][:60]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with dummy facts
    dummy_facts = [
        {"claim": "US deployed additional carrier group to Persian Gulf", "source": "Reuters", "market": "regime_fall", "confidence": "high"},
        {"claim": "Iranian rial fell 5% against dollar this week", "source": "FT", "market": "regime_fall", "confidence": "high"},
        {"claim": "Protests reported in Isfahan and Tabriz", "source": "BBC", "market": "regime_fall", "confidence": "medium"},
    ]
    dummy_sqs = [
        {"question": "Is there a credible external military intervention?", "relationship": "sufficient", "weight": 0.3},
        {"question": "Are economic conditions severe enough to trigger collapse?", "relationship": "contributing", "weight": 0.25},
        {"question": "Is there a mass popular uprising?", "relationship": "necessary", "weight": 0.3},
    ]

    print("=== Evidence Evaluation Demo ===")
    evals = classify_evidence(dummy_facts, "regime_fall", dummy_sqs)
    print(summarize_evidence(evals))
