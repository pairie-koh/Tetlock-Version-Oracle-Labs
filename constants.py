"""
Tetlock Oracle Labs — Constants
Markets, API endpoints, models, and Tetlock-specific configuration.
"""

import os

# ── Markets ──────────────────────────────────────────────────────────────────
# Same contracts as Oracle Labs v2 for direct comparison.
# To add a market: add an entry here + QUERY_TEMPLATES + DECOMPOSITION_HINTS.

MARKETS = {
    "regime_fall": {
        "name": "Will the Iranian regime fall by June 30?",
        "slug": "regime-fall",
        "condition_id": "0x9352c559e9648ab4cab236087b64ca85c5b7123a4c7d9d7d4efde4a39c18056f",
        "yes_token_id": "38397507750621893057346880033441136112987238933685677349709401910643842844855",
        "no_token_id": "95949957895141858444199258452803633110472396604599808168788254125381075552218",
        "end_date": "2025-06-30",
        "domain": "geopolitics",
    },
}

# ── API Endpoints ────────────────────────────────────────────────────────────

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"

# ── Models ───────────────────────────────────────────────────────────────────

PERPLEXITY_MODEL = "sonar-pro"
HAIKU_MODEL = "anthropic/claude-haiku-4.5"
SONNET_MODEL = "anthropic/claude-sonnet-4"

# ── Tetlock-Specific Config ─────────────────────────────────────────────────

# Shrinkage: how much to trust LLM vs market. 0.0 = market only, 1.0 = LLM only.
# We start at 0.5 (split the difference) — more conservative than Oracle Labs' 0.75.
SHRINKAGE_KEEP = 0.50

# Bayesian prior: when no prior belief exists, start at market price.
DEFAULT_PRIOR_SOURCE = "market"

# Maximum sub-questions per decomposition
MAX_SUB_QUESTIONS = 5

# Evidence diagnostic test types (from Bayesian Process Tracing)
DIAGNOSTIC_TESTS = ["straw_in_the_wind", "hoop", "smoking_gun", "doubly_decisive"]

# ── Perplexity Query Templates ───────────────────────────────────────────────

QUERY_TEMPLATES = {
    "regime_fall": [
        (
            "military_security",
            "What are the latest developments in the past 24 hours regarding "
            "military pressure on Iran, including US military deployments, "
            "Israeli operations or threats, IRGC activities, strikes or "
            "confrontations, and internal security actions against protesters? "
            "Include specific sources and timestamps."
        ),
        (
            "political_diplomatic_economic",
            "What are the latest developments in the past 24 hours regarding "
            "Iranian regime stability, including diplomatic negotiations, "
            "internal political dynamics, economic conditions and sanctions, "
            "protest movements, and international statements or UN actions? "
            "Include specific sources and timestamps."
        ),
    ],
}

# ── Decomposition Hints ─────────────────────────────────────────────────────
# Per-market hints for the decomposer — what sub-questions matter.

DECOMPOSITION_HINTS = {
    "regime_fall": [
        "Is the military/IRGC likely to fracture or defect?",
        "Is there a mass popular uprising underway or imminent?",
        "Is there a credible external military intervention planned?",
        "Are economic conditions severe enough to trigger regime collapse?",
        "Is there a succession crisis or internal power struggle?",
    ],
}

# ── Paths ────────────────────────────────────────────────────────────────────

STATE_DIR = "state"
BRIEFINGS_DIR = "briefings"
PREDICTIONS_DIR = "predictions"
SCORES_DIR = "scores"
PRICE_HISTORY_DIR = "price_history"
PRICE_CSV = os.path.join(PRICE_HISTORY_DIR, "prices.csv")
