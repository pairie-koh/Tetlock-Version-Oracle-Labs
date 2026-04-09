"""
Tetlock Oracle Labs -- lessons.py
Feedback loop: scan past predictions, compute per-contract and per-domain
performance stats, inject lessons into future prompts.

Adapted from Oracle Labs v2 for the Tetlock prediction format.

Two entry points:
  rebuild_lessons_cache()  -- called after scoring (evaluate.py)
  build_lessons_block()    -- called during prompt building (forecast.py)
"""

import glob
import json
import os
from datetime import datetime, timezone

from constants import DATA_DIR, PREDICTIONS_DIR, CONTRACTS_DIR

CACHE_PATH = os.path.join(DATA_DIR, "lessons_cache.json")
CONTRACTS_PATH = os.path.join(CONTRACTS_DIR, "active_contracts.json")

MAX_HISTORY = 5       # last N predictions to show in prompt
MIN_CONTRACT = 2      # min predictions before showing per-contract lesson
MIN_DOMAIN = 5        # min predictions before showing per-domain lesson
BIAS_THRESHOLD = 0.02 # |avg_error| above this = bias

_cache = None  # module-level singleton


# -- Cache Building -----------------------------------------------------------

def _load_domain_lookup():
    """Build market_key -> domain mapping from MARKETS + active_contracts.json."""
    lookup = {}

    # From constants.py MARKETS dict
    try:
        from constants import MARKETS
        for key, market in MARKETS.items():
            domain = market.get("domain", "")
            if domain:
                lookup[key] = domain
    except ImportError:
        pass

    # From dynamically discovered contracts
    if os.path.exists(CONTRACTS_PATH):
        try:
            with open(CONTRACTS_PATH) as f:
                data = json.load(f)
            for c in data.get("contracts", []):
                slug = c.get("slug", "")
                cats = c.get("categories", [])
                if slug and cats:
                    lookup[slug] = cats[0]  # use first category as domain
        except Exception:
            pass

    return lookup


def rebuild_lessons_cache():
    """Scan all prediction files, compute stats, write cache."""
    domain_lookup = _load_domain_lookup()

    # Collect all predictions grouped by market key
    by_contract = {}  # key -> list of entries

    pattern = os.path.join(PREDICTIONS_DIR, "*.json")
    files = sorted(glob.glob(pattern))

    # Skip latest.json (it's a symlink/copy of the most recent)
    files = [f for f in files if not f.endswith("latest.json")]

    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        # Tetlock format: {market_key: prediction_record}
        for key, pred in data.items():
            if not isinstance(pred, dict):
                continue

            market_price = pred.get("market_price")
            final_prob = pred.get("final_probability")
            if market_price is None or final_prob is None:
                continue

            entry = {
                "timestamp": pred.get("timestamp", ""),
                "prediction": final_prob,
                "market_price": market_price,
                "base_rate": pred.get("base_rate"),
                "divergence": pred.get("divergence_from_market", 0),
                "signed_error": final_prob - market_price,
                "evidence_count": pred.get("evidence_count", 0),
                "cycle": pred.get("cycle_number", 0),
            }

            by_contract.setdefault(key, []).append(entry)

    # Sort each contract's history by timestamp
    for key in by_contract:
        by_contract[key].sort(key=lambda e: e["timestamp"])

    # Build per-contract stats
    per_contract = {}
    for key, entries in by_contract.items():
        errors = [e["signed_error"] for e in entries]
        avg_err = sum(errors) / len(errors)
        avg_abs = sum(abs(e) for e in errors) / len(errors)

        bias = "neutral"
        if avg_err < -BIAS_THRESHOLD:
            bias = "TOO LOW"
        elif avg_err > BIAS_THRESHOLD:
            bias = "TOO HIGH"

        per_contract[key] = {
            "n_predictions": len(entries),
            "avg_signed_error": round(avg_err, 3),
            "avg_abs_error": round(avg_abs, 3),
            "bias": bias,
            "history": entries[-MAX_HISTORY:],
        }

    # Build per-domain stats
    per_domain = {}
    for key, entries in by_contract.items():
        domain = domain_lookup.get(key, "")
        if not domain:
            continue

        if domain not in per_domain:
            per_domain[domain] = {"errors": [], "abs_errors": []}

        for e in entries:
            per_domain[domain]["errors"].append(e["signed_error"])
            per_domain[domain]["abs_errors"].append(abs(e["signed_error"]))

    # Finalize domain stats
    domain_stats = {}
    for domain_key, raw in per_domain.items():
        n = len(raw["errors"])
        if n == 0:
            continue

        avg_err = sum(raw["errors"]) / n
        avg_abs = sum(raw["abs_errors"]) / n

        bias = "neutral"
        if avg_err < -BIAS_THRESHOLD:
            bias = "TOO LOW"
        elif avg_err > BIAS_THRESHOLD:
            bias = "TOO HIGH"

        domain_stats[domain_key] = {
            "n_predictions": n,
            "avg_signed_error": round(avg_err, 3),
            "avg_abs_error": round(avg_abs, 3),
            "bias": bias,
        }

    # Write cache
    cache = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_files_scanned": len(files),
        "per_contract": per_contract,
        "per_domain": domain_stats,
    }

    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        print(f"  WARNING: Could not write lessons cache: {e}")

    n_contracts = len(per_contract)
    n_domains = len(domain_stats)
    print(f"  Lessons cache: {n_contracts} contracts, {n_domains} domains "
          f"(from {len(files)} prediction files)")
    return cache


# -- Cache Loading ------------------------------------------------------------

def load_lessons_cache():
    """Load cache from disk with in-memory singleton."""
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH) as f:
            _cache = json.load(f)
        return _cache
    except Exception:
        return {}


def reset_cache():
    """Clear the in-memory cache (useful between test runs)."""
    global _cache
    _cache = None


# -- Prompt Block Building ----------------------------------------------------

def build_lessons_block(contract_key, domain=None):
    """Build a text block for prompt injection.

    Returns a formatted string showing past performance on this contract
    and/or domain, or "" if insufficient data.
    """
    cache = load_lessons_cache()
    if not cache:
        return ""

    lines = []

    # Per-contract lesson
    contract_data = cache.get("per_contract", {}).get(contract_key, {})
    n_contract = contract_data.get("n_predictions", 0)

    if n_contract >= MIN_CONTRACT:
        history = contract_data.get("history", [])

        preds = [f"{h['prediction']:.2f}" for h in history[-MAX_HISTORY:]]
        mkts = [f"{h['market_price']:.2f}" for h in history[-MAX_HISTORY:]]
        lines.append(
            f"PAST PERFORMANCE ON THIS CONTRACT ({n_contract} predictions):"
        )
        lines.append(f"  You: {', '.join(preds)} | Market: {', '.join(mkts)}")

        bias = contract_data.get("bias", "")
        avg_err = contract_data.get("avg_signed_error", 0)
        if bias and bias != "neutral":
            lines.append(
                f"  Bias: consistently {bias} vs market (avg error: {avg_err:+.3f})"
            )

    # Per-domain lesson (as fallback or supplement)
    domain_data = cache.get("per_domain", {}).get(domain or "", {})
    n_domain = domain_data.get("n_predictions", 0)

    if n_domain >= MIN_DOMAIN and n_contract < 3:
        bias = domain_data.get("bias", "")
        avg_abs = domain_data.get("avg_abs_error", 0)

        domain_line = f"DOMAIN PATTERN ({domain}, {n_domain} predictions): "
        if bias and bias != "neutral":
            domain_line += f"you tend to predict {bias} (avg abs error: {avg_abs:.3f})"
        else:
            domain_line += f"avg abs error: {avg_abs:.3f}"

        lines.append(domain_line)

    if not lines:
        return ""

    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_lessons_cache()
    elif len(sys.argv) > 1 and sys.argv[1] == "show":
        rebuild_lessons_cache()
        reset_cache()
        cache = load_lessons_cache()
        domain_lookup = _load_domain_lookup()
        for key in sorted(cache.get("per_contract", {}).keys()):
            domain = domain_lookup.get(key, "")
            block = build_lessons_block(key, domain=domain)
            if block:
                print(f"\n--- {key} ---")
                print(block)
        print(f"\n--- Domain summaries ---")
        for dom, stats in cache.get("per_domain", {}).items():
            print(f"  {dom}: {stats}")
    else:
        print("Usage: python lessons.py rebuild   -- rebuild cache from prediction files")
        print("       python lessons.py show      -- rebuild + show all lessons")
