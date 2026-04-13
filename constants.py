"""
Tetlock Oracle Labs -- Constants
Markets, API endpoints, models, and Tetlock-specific configuration.
"""

import json
import os
from datetime import datetime, timezone

# -- Contract Loader ----------------------------------------------------------
# Loads markets from contracts/active_contracts.json (same contracts as Oracle
# Labs v2). Binary contracts become one MARKETS entry each. Multi-outcome
# contracts are flattened: each outcome becomes a separate binary entry.

CONTRACTS_DIR = "contracts"
CONTRACTS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), CONTRACTS_DIR, "active_contracts.json"
)

# Maximum outcomes to include per multi-outcome contract (v2 uses 15)
MAX_OUTCOMES_PER_EVENT = 15


def load_markets_from_contracts():
    """Load markets from active_contracts.json, flattening multi-outcome
    contracts into individual binary entries.

    Returns dict: {market_key: {name, slug, condition_id, yes_token_id,
                                end_date, domain, parent_event}}
    """
    if not os.path.exists(CONTRACTS_JSON):
        # Fallback if JSON doesn't exist yet
        return {}

    with open(CONTRACTS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    now = datetime.now(timezone.utc)
    markets = {}

    for c in data.get("contracts", []):
        contract_type = c.get("contract_type", "binary")
        domain = c.get("domain", c.get("category", ""))

        # Parse end date, skip expired contracts
        end_date_str = c.get("end_date", "")
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt < now:
                    continue
            except (ValueError, AttributeError):
                pass

        if contract_type == "binary":
            slug = c.get("slug", "")
            if not slug or not c.get("yes_token_id"):
                continue
            markets[slug] = {
                "name": c.get("question", c.get("contract_name", "")),
                "slug": slug,
                "condition_id": c.get("condition_id", ""),
                "yes_token_id": c.get("yes_token_id", ""),
                "end_date": end_date_str[:10] if end_date_str else "",
                "domain": domain,
                "parent_event": None,
            }

        elif contract_type == "multi-outcome":
            outcomes = c.get("outcomes", [])
            if not outcomes:
                continue

            # Limit to top outcomes by price (same as v2)
            if len(outcomes) > MAX_OUTCOMES_PER_EVENT:
                outcomes = sorted(
                    outcomes,
                    key=lambda o: o.get("yes_price", 0),
                    reverse=True,
                )[:MAX_OUTCOMES_PER_EVENT]

            parent_slug = c.get("slug", c.get("event_slug", ""))

            for outcome in outcomes:
                o_slug = outcome.get("market_slug", "")
                yes_token = outcome.get("yes_token_id", "")
                if not o_slug or not yes_token:
                    continue

                markets[o_slug] = {
                    "name": outcome.get("question", ""),
                    "slug": o_slug,
                    "condition_id": outcome.get("condition_id", ""),
                    "yes_token_id": yes_token,
                    "end_date": end_date_str[:10] if end_date_str else "",
                    "domain": domain,
                    "parent_event": parent_slug,
                }

    return markets


MARKETS = load_markets_from_contracts()

# -- API Endpoints ------------------------------------------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"

# -- Models -------------------------------------------------------------------

PERPLEXITY_MODEL = "sonar-pro"
HAIKU_MODEL = "anthropic/claude-haiku-4.5"
SONNET_MODEL = "anthropic/claude-sonnet-4"
OPUS_MODEL = "anthropic/claude-opus-4"

# Tiered escalation thresholds (divergence from market to trigger next tier)
SONNET_THRESHOLD = 0.05   # >5% divergence -> use Sonnet
OPUS_THRESHOLD = 0.15     # >15% divergence -> use Opus

# -- Tetlock-Specific Config --------------------------------------------------

# Shrinkage: how much to trust LLM vs market. 0.0 = market only, 1.0 = LLM only.
# We start at 0.5 (split the difference) -- more conservative than Oracle Labs' 0.75.
SHRINKAGE_KEEP = 0.50

# Extremizing constant (Tetlock GJP): push forecasts away from 50%.
# d=0.0 means no extremizing. d=0.3 is GJP default for individual forecasters.
# Baron et al. (2014) found d=0.3 to 0.5 optimal for aggregated forecasts.
EXTREMIZING_D = 0.30

# Bayesian prior: when no prior belief exists, start at market price.
DEFAULT_PRIOR_SOURCE = "market"

# Maximum sub-questions per decomposition
MAX_SUB_QUESTIONS = 5

# Evidence diagnostic test types (from Bayesian Process Tracing)
DIAGNOSTIC_TESTS = ["straw_in_the_wind", "hoop", "smoking_gun", "doubly_decisive"]

# Maximum likelihood ratio per sub-question group. Prevents runaway updating
# from compounding multiple weak signals. Tetlock: superforecasters make
# small incremental updates, not dramatic swings.
MAX_LR_PER_SQ = 5.0   # no single sub-question can push odds more than 5:1
MAX_COMBINED_LR = 10.0 # total combined LR across all sub-questions

# -- Perplexity Query Templates -----------------------------------------------
# Per-market custom query templates. Markets without entries here use the
# generic template in newswire.py (which auto-generates queries from the
# market question text).
QUERY_TEMPLATES = {}

# -- Decomposition Hints ------------------------------------------------------
# Per-market hints for the decomposer -- what sub-questions matter.
# Markets without entries here get LLM-generated sub-questions.
DECOMPOSITION_HINTS = {}

# -- Paths --------------------------------------------------------------------

STATE_DIR = "state"
BRIEFINGS_DIR = "briefings"
PREDICTIONS_DIR = "predictions"
SCORES_DIR = "scores"
PRICE_HISTORY_DIR = "price_history"
DATA_DIR = "data"
PRICE_CSV = os.path.join(PRICE_HISTORY_DIR, "prices.csv")
