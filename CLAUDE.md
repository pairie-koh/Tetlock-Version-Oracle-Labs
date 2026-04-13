# CLAUDE.md

## What This Project Is

A Tetlock-inspired AI forecasting system that predicts Polymarket contract prices using structured superforecasting methodology and Bayesian Process Tracing (BPT). Built as a direct comparison to Oracle Labs v2, which used brute-force LLM forecasting and learned that unstructured LLM judgment adds negative value vs. prediction markets.

**Core hypothesis:** Structured epistemic reasoning (Tetlock + BPT) gives LLMs positive edge over markets, where unstructured reasoning does not.

**Comparison target:** Oracle Labs v2 (`oracle-lab-v2-fresh/`) -- same markets, same data sources, different methodology.

## Methodology

The pipeline implements six sequential steps, each producing structured output:

### 0. Data Gather -- `run_cycle.py` + individual fetchers
Collect context from all external sources before forecasting:
- GDELT global news monitoring (65 languages, 15-min updates)
- Hyperliquid 24/7 perp prices (BTC, ETH, oil, S&P, gold)
- NWS weather forecasts (for temperature contracts)
- Polymarket contract discovery (~30 diverse active contracts)
- pmxt order book archive (millisecond-precision snapshots)

### 1. Base Rate (Outside View) -- `base_rates.py`
Before looking at any news, establish: "How often do events like this happen?"
Uses reference class forecasting to anchor the prior probability.

### 2. Decomposition -- `decompose.py`
Break the main question into 3-5 sub-questions. Estimate each component separately.
This is how Tetlock's superforecasters work -- they don't answer monolithic questions.

### 3. Evidence Evaluation (BPT) -- `evidence.py`
For each piece of news, classify it using diagnostic tests from Bayesian Process Tracing:
- **Straw-in-the-wind**: Weak evidence, barely moves the needle (LR ~1.0-1.5)
- **Hoop test**: Hypothesis must pass this to survive, but passing doesn't confirm (LR failing < 0.3)
- **Smoking gun**: Finding this strongly confirms the hypothesis (LR > 3.0)
- **Doubly decisive**: Both necessary and sufficient (rare, LR > 5.0)

Uses three-tier LLM escalation based on divergence from market:
- Haiku: <5% divergence (routine triage)
- Sonnet: 5-15% divergence (deeper analysis)
- Opus: >15% divergence (highest-stakes reasoning)

### 4. Bayesian Updating -- `updater.py`
Convert prior probability to odds, multiply by likelihood ratios, convert back.
Apply overconfidence shrinkage toward market price.
Persist belief state across cycles (incremental updates, not fresh starts).
- Initial belief blends base rate with market price (not raw base rate alone)
- LRs capped per sub-question (max 5:1) and overall (max 10:1)
- Prevents runaway updating from compounding weak signals

### 5. Adversarial Review + Extremizing -- `forecast.py`
Force the model to argue against its own position. Adjust if a genuine blind spot is found.
Then apply Tetlock's extremizing transform (d=0.3) to push forecasts away from 50%.
Supplementary data (GDELT, Hyperliquid, weather, order flow, smart money, lessons) is
injected as additional context for the adversarial reviewer.

### 6. Lessons Feedback -- `lessons.py`
After scoring, rebuild per-contract and per-domain performance stats.
Lessons are computed against ACTUAL OUTCOMES when available (not market prices).
Inject bias warnings ("you tend to predict TOO HIGH vs outcomes") into future prompts.
Falls back to market-divergence only when contracts haven't resolved yet.

## Key Differences from Oracle Labs v2

| Dimension | Oracle Labs v2 | Tetlock Version |
|---|---|---|
| Prior | None (starts from scratch) | Reference class base rate |
| Reasoning | "What's the probability?" | Decompose -> classify evidence -> update |
| Evidence handling | Volume-based | Diagnosticity-based (likelihood ratios) |
| Cross-cycle memory | None (fresh each cycle) | Persistent Bayesian belief state |
| Self-correction | Brute-force shrinkage | Adversarial review + structured updating |
| Shrinkage | 0.75 (trust LLM more) | 0.50 (more conservative) |
| LLM tiering | Haiku only at forecast | Three-tier: Haiku/Sonnet/Opus by divergence |
| Feedback loop | None | Per-contract + per-domain bias tracking (vs outcomes) |
| Extremizing | None | GJP extremizing transform (d=0.3) |
| LR safety | None | Per-SQ cap (5:1), total cap (10:1) |
| Auto-resolution | None | Detects settled contracts from Polymarket prices + Gamma API |
| Scoring | SE vs market price (mislabeled "Brier") | Real Brier scores vs binary outcomes (0/1) |
| Calibration tracking | None | Calibration curves + resolution metric |

## Project Layout

```
├── constants.py           # Markets, APIs, models, Tetlock config
├── market_data.py         # Polymarket CLOB price fetching
├── newswire.py            # News gathering (Perplexity + Haiku)
├── base_rates.py          # Step 1: Outside view / reference class
├── decompose.py           # Step 2: Question decomposition
├── evidence.py            # Step 3: BPT evidence eval (three-tier LLM)
├── updater.py             # Step 4: Bayesian belief updating
├── forecast.py            # Step 5: Main pipeline + adversarial review
├── evaluate.py            # Tetlock scoring (Brier, calibration, resolution)
├── lessons.py             # Step 6: Feedback loop (bias tracking)
├── run_cycle.py           # Orchestrator (5-stage pipeline)
│
├── gdelt.py               # GDELT DOC 2.0 news signals (free, no key)
├── hyperliquid.py         # 24/7 perp prices (BTC, ETH, oil, S&P, gold)
├── weather.py             # NWS weather forecasts (NYC, Miami)
├── orderflow.py           # pmxt archive order book analysis (polars)
├── smart_money.py         # Quantitative trading pattern detector
├── contracts.py           # Active contract discovery (~30 markets)
├── rolling_contracts.py   # Daily rolling contracts (BTC, oil, temp)
│
├── state/                 # Persistent belief states (per market)
├── briefings/             # News briefings
├── predictions/           # Forecast outputs with full audit trails
├── scores/                # Brier scores, calibration, resolutions
├── contracts/             # Discovered contract data
├── data/                  # GDELT results, weather, lessons cache
└── price_history/         # Historical price CSVs
```

## Commands

```bash
# Full cycle: data fetch -> news -> forecast -> evaluate -> lessons
python run_cycle.py

# Forecast only (reuse latest news + data)
python run_cycle.py --forecast-only

# Evaluate only (score latest predictions + rebuild lessons)
python run_cycle.py --evaluate-only

# Individual data sources
python gdelt.py                  # fetch GDELT news
python hyperliquid.py            # fetch perp prices
python weather.py                # fetch NWS forecasts
python contracts.py              # discover active contracts
python rolling_contracts.py      # fetch daily rolling contracts
python orderflow.py              # fetch pmxt order book data

# Individual pipeline stages
python base_rates.py             # just base rates
python decompose.py              # just decomposition
python evaluate.py               # evaluate latest (pre-resolution + resolved)
python lessons.py rebuild        # rebuild lessons cache
python lessons.py show           # show all lessons

# Resolution tracking (record actual outcomes for real Brier scores)
python evaluate.py resolve regime_fall 0   # record: Iranian regime did NOT fall
python evaluate.py resolve regime_fall 1   # record: Iranian regime DID fall
python evaluate.py resolutions             # show all recorded resolutions
python evaluate.py resolved                # show all-time Brier scores + calibration
```

## Environment Variables

```
OPENROUTER_API_KEY=...    # Claude via OpenRouter (Haiku/Sonnet/Opus)
PERPLEXITY_API_KEY=...    # News gathering
```

## Data Sources

| Source | Module | Key/Auth | Rate Limit | What It Provides |
|---|---|---|---|---|
| Polymarket CLOB | market_data.py | None | None | Prediction market prices |
| Polymarket Gamma | contracts.py | None | None | Market discovery, metadata |
| Perplexity | newswire.py | API key | Paid | Structured news summaries |
| GDELT DOC 2.0 | gdelt.py | None | 1 req/5s | Global news article counts + headlines |
| Hyperliquid | hyperliquid.py | None | None | 24/7 BTC, ETH, oil, S&P, gold prices |
| NWS | weather.py | None (User-Agent) | Polite use | Temperature forecasts (NYC, Miami) |
| pmxt Archive | orderflow.py | None | None | Millisecond order book snapshots |

## Scoring Methodology

The evaluation system uses proper Tetlock scoring from the GJP:

**Primary metrics** (require resolution -- did the event actually happen?):
- **Brier Score**: `(prediction - outcome)^2` where outcome is 0 or 1. Perfect=0.0, coin flip=0.25, maximally wrong=1.0. GJP superforecasters averaged ~0.15.
- **Calibration**: When you say 70%, events should happen ~70% of the time. Measured via calibration curves.
- **Resolution**: Ability to make decisive forecasts (not always hovering near 50%).

**Secondary metrics** (available before resolution):
- Market comparison SE: does our divergence from market move us closer to where the market eventually goes? Informational only, NOT a Brier score.

**Two modes**:
1. Pre-resolution: predictions are tracked with market comparison (clearly labeled as not-yet-scored)
2. Post-resolution: once a contract resolves (auto-detected from Polymarket or manually via `evaluate.py resolve <market> <0|1>`), real Brier scores + calibration are computed

**Auto-resolution**: When a Polymarket contract settles, its price goes to ~0.00 or ~1.00. The system auto-detects this during evaluation and records the resolution. Also checks the Gamma API for explicit resolution status.

## Workflow Requirements

1. **Never claim something works without testing it.** Run the code, show the output.
2. **Work modularly.** Complete one module at a time.
3. **Be explicit about unknowns.** "I don't know" is acceptable.
4. **Every prediction has a full audit trail.** Base rate -> decomposition -> evidence -> update -> adversarial.
5. **Data source failures are non-fatal.** Each source is wrapped in try/except. The pipeline runs with whatever is available.
