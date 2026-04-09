# Tetlock Oracle Labs

Can structured superforecasting methodology give LLMs an edge over prediction markets?

[Oracle Labs v2](https://github.com/pairie-koh/Oracle-Labs-V2) discovered that unstructured LLM forecasting has **negative edge** vs. liquid prediction markets — the system improved 54% by learning to copy the market and suppress its own signal.

This repo tests a different approach: instead of asking an LLM "what's the probability?", we force it through **Philip Tetlock's superforecasting methodology** combined with **Bayesian Process Tracing** from political science.

## The Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  BASE RATE   │───▶│  DECOMPOSE   │───▶│   EVIDENCE    │
│ Outside view │    │ Sub-questions │    │  BPT classify │
│ Reference    │    │ Necessary vs  │    │  Likelihood   │
│ class anchor │    │ sufficient    │    │  ratios       │
└─────────────┘    └──────────────┘    └───────┬───────┘
                                               │
┌─────────────┐    ┌──────────────┐           │
│   OUTPUT     │◀──│ ADVERSARIAL  │◀──────────┘
│ Calibrated   │    │ Argue against│    ┌───────────────┐
│ probability  │    │ own position │◀───│   BAYESIAN    │
│ + audit trail│    └──────────────┘    │   UPDATE      │
└─────────────┘                         │ Prior × LR    │
                                        └───────────────┘
```

**Step 1 — Base Rate:** Before looking at any news, establish the outside-view probability using reference class forecasting.

**Step 2 — Decompose:** Break the question into sub-questions. Estimate components separately.

**Step 3 — Evidence (BPT):** For each piece of news, classify its diagnostic power: straw-in-the-wind, hoop test, smoking gun, or doubly decisive. Compute likelihood ratios P(E|Yes)/P(E|No).

**Step 4 — Bayesian Update:** Multiply prior odds by likelihood ratios. Persist beliefs across cycles.

**Step 5 — Adversarial Review:** Argue against your own position. Adjust if a genuine blind spot is found.

## Why This Might Work

Oracle Labs v2 failed because LLMs process evidence by **volume** (lots of news = high probability) rather than **diagnosticity** (does this news actually distinguish between outcomes?).

Tetlock's superforecasters beat intelligence analysts by doing the opposite: starting with base rates, decomposing questions, and evaluating each piece of evidence for its actual diagnostic value.

## Quick Start

```bash
# Set API keys
export OPENROUTER_API_KEY=...
export PERPLEXITY_API_KEY=...

# Install dependencies
pip install -r requirements.txt

# Run a full forecast cycle
python run_cycle.py
```

## Comparison with Oracle Labs v2

Same markets. Same data sources. Different methodology. The results will show whether structured reasoning gives LLMs positive edge.

| | Oracle Labs v2 | Tetlock Version |
|---|---|---|
| Prior | None | Reference class base rate |
| Evidence | Volume-based | Diagnosticity-based |
| Memory | Fresh each cycle | Persistent Bayesian state |
| Self-correction | Mechanical shrinkage | Adversarial review |

## References

- Tetlock & Gardner (2015), *Superforecasting: The Art and Science of Prediction*
- Fairfield & Charman (2017), "Explicit Bayesian Analysis for Process Tracing"
- Bennett (2015), "Disciplining Our Conjectures: Systematizing Process Tracing"
- Inspired by [mgaldino/PPC_BPT](https://github.com/mgaldino/PPC_BPT) (Bayesian Process Tracing)
