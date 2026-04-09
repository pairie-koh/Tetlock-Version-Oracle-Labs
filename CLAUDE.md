# CLAUDE.md

## What This Project Is

A Tetlock-inspired AI forecasting system that predicts Polymarket contract prices using structured superforecasting methodology and Bayesian Process Tracing (BPT). Built as a direct comparison to Oracle Labs v2, which used brute-force LLM forecasting and learned that unstructured LLM judgment adds negative value vs. prediction markets.

**Core hypothesis:** Structured epistemic reasoning (Tetlock + BPT) gives LLMs positive edge over markets, where unstructured reasoning does not.

**Comparison target:** Oracle Labs v2 (`oracle-lab-v2-fresh/`) — same markets, same data sources, different methodology.

## Methodology

The pipeline implements five sequential steps, each producing structured output:

### 1. Base Rate (Outside View) — `base_rates.py`
Before looking at any news, establish: "How often do events like this happen?"
Uses reference class forecasting to anchor the prior probability.

### 2. Decomposition — `decompose.py`
Break the main question into 3-5 sub-questions. Estimate each component separately.
This is how Tetlock's superforecasters work — they don't answer monolithic questions.

### 3. Evidence Evaluation (BPT) — `evidence.py`
For each piece of news, classify it using diagnostic tests from Bayesian Process Tracing:
- **Straw-in-the-wind**: Weak evidence, barely moves the needle (LR ≈ 1.0-1.5)
- **Hoop test**: Hypothesis must pass this to survive, but passing doesn't confirm (LR failing < 0.3)
- **Smoking gun**: Finding this strongly confirms the hypothesis (LR > 3.0)
- **Doubly decisive**: Both necessary and sufficient (rare, LR > 5.0)

For each piece of evidence, estimate P(evidence | YES) and P(evidence | NO).
The ratio is the likelihood ratio — the Bayesian update factor.

### 4. Bayesian Updating — `updater.py`
Convert prior probability to odds, multiply by likelihood ratios, convert back.
Apply overconfidence shrinkage toward market price.
Persist belief state across cycles (incremental updates, not fresh starts).

### 5. Adversarial Review — `forecast.py`
Force the model to argue against its own position. Adjust if a genuine blind spot is found.

## Key Differences from Oracle Labs v2

| Dimension | Oracle Labs v2 | Tetlock Version |
|---|---|---|
| Prior | None (starts from scratch) | Reference class base rate |
| Reasoning | "What's the probability?" | Decompose → classify evidence → update |
| Evidence handling | Volume-based | Diagnosticity-based (likelihood ratios) |
| Cross-cycle memory | None (fresh each cycle) | Persistent Bayesian belief state |
| Self-correction | Brute-force shrinkage | Adversarial review + structured updating |
| Shrinkage | 0.75 (trust LLM more) | 0.50 (more conservative) |

## Project Layout

```
├── constants.py       # Markets, APIs, Tetlock config
├── market_data.py     # Polymarket CLOB price fetching
├── newswire.py        # News gathering (Perplexity + Haiku)
├── base_rates.py      # Step 1: Outside view / reference class
├── decompose.py       # Step 2: Question decomposition
├── evidence.py        # Step 3: BPT evidence evaluation
├── updater.py         # Step 4: Bayesian belief updating
├── forecast.py        # Step 5: Main pipeline + adversarial review
├── evaluate.py        # Scoring engine
├── run_cycle.py       # Orchestrator
├── state/             # Persistent belief states (per market)
├── briefings/         # News briefings
├── predictions/       # Forecast outputs with full audit trails
└── scores/            # Evaluation results
```

## Commands

```bash
# Full cycle: news → forecast → evaluate
python run_cycle.py

# Forecast only (reuse latest news)
python run_cycle.py --forecast-only

# Evaluate only (score latest predictions)
python run_cycle.py --evaluate-only

# Individual modules
python base_rates.py      # just base rates
python decompose.py       # just decomposition
python evaluate.py        # just scoring
```

## Environment Variables

```
OPENROUTER_API_KEY=...    # Claude via OpenRouter
PERPLEXITY_API_KEY=...    # News gathering
```

## Workflow Requirements

1. **Never claim something works without testing it.** Run the code, show the output.
2. **Work modularly.** Complete one module at a time.
3. **Be explicit about unknowns.** "I don't know" is acceptable.
4. **Every prediction has a full audit trail.** Base rate → decomposition → evidence → update → adversarial.
