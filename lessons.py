"""
Tetlock Oracle Labs -- lessons.py
Feedback loop: scan past predictions, compute per-contract and per-domain
performance stats, inject lessons into future prompts.

KEY PRINCIPLE (from Tetlock's Superforecasting):
  Lessons must be computed against ACTUAL OUTCOMES (0 or 1), not market prices.
  "Did I predict too high for events that didn't happen?" is the right question.
  "Did I diverge from the market?" is NOT -- the market could also be wrong.

Two modes:
  - With resolutions: real Brier-based lessons (preferred)
  - Without resolutions: divergence-from-market as weak proxy (clearly labeled)

Two entry points:
  rebuild_lessons_cache()  -- called after scoring (evaluate.py)
  build_lessons_block()    -- called during prompt building (forecast.py)
"""

import glob
import json
import os
from datetime import datetime, timezone

from constants import DATA_DIR, PREDICTIONS_DIR, CONTRACTS_DIR, SCORES_DIR

CACHE_PATH = os.path.join(DATA_DIR, "lessons_cache.json")
CONTRACTS_PATH = os.path.join(CONTRACTS_DIR, "active_contracts.json")
RESOLUTIONS_PATH = os.path.join(SCORES_DIR, "resolutions.json")

MAX_HISTORY = 20      # last N predictions to show in prompt (was 5, too small)
MIN_CONTRACT = 2      # min predictions before showing per-contract lesson
MIN_DOMAIN = 5        # min predictions before showing per-domain lesson
BIAS_THRESHOLD = 0.02 # |avg_error| above this = bias

_cache = None  # module-level singleton


def _classify_bias(avg_err, resolved=True):
    """Classify bias direction from average signed error."""
    if avg_err < -BIAS_THRESHOLD:
        return "TOO LOW vs outcomes" if resolved else "BELOW market"
    elif avg_err > BIAS_THRESHOLD:
        return "TOO HIGH vs outcomes" if resolved else "ABOVE market"
    return "neutral"


# -- Resolution Loading -------------------------------------------------------

def _load_resolutions():
    """Load resolution outcomes from evaluate.py's resolutions file."""
    if not os.path.exists(RESOLUTIONS_PATH):
        return {}
    try:
        with open(RESOLUTIONS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


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
                    lookup[slug] = cats[0]
        except Exception:
            pass

    return lookup


def rebuild_lessons_cache():
    """Scan all prediction files, compute stats, write cache.

    Uses resolutions (actual outcomes) when available for real Brier-based
    lessons. Falls back to market-divergence for unresolved contracts.
    """
    domain_lookup = _load_domain_lookup()
    resolutions = _load_resolutions()

    # Collect all predictions grouped by market key
    by_contract = {}  # key -> list of entries

    pattern = os.path.join(PREDICTIONS_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not f.endswith("latest.json")]

    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

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
                "divergence_from_market": pred.get("divergence_from_market", 0),
                "evidence_count": pred.get("evidence_count", 0),
                "cycle": pred.get("cycle_number", 0),
            }

            # If this contract has resolved, compute REAL error
            if key in resolutions:
                outcome = resolutions[key]["outcome"]
                entry["outcome"] = outcome
                entry["brier_score"] = (final_prob - outcome) ** 2
                entry["signed_error"] = final_prob - outcome
                # positive = predicted too high, negative = predicted too low
                entry["scoring_type"] = "resolved"
            else:
                # No resolution -- use market divergence as weak proxy
                entry["signed_error"] = final_prob - market_price
                entry["scoring_type"] = "pre_resolution"

            by_contract.setdefault(key, []).append(entry)

    # Sort each contract's history by timestamp
    for key in by_contract:
        by_contract[key].sort(key=lambda e: e["timestamp"])

    # Build per-contract stats
    per_contract = {}
    for key, entries in by_contract.items():
        resolved_entries = [e for e in entries if e.get("scoring_type") == "resolved"]
        has_resolution = len(resolved_entries) > 0

        if has_resolution:
            # USE REAL ERRORS (prediction - outcome)
            errors = [e["signed_error"] for e in resolved_entries]
            briers = [e["brier_score"] for e in resolved_entries]
            avg_err = sum(errors) / len(errors)
            avg_abs = sum(abs(e) for e in errors) / len(errors)
            avg_brier = sum(briers) / len(briers)

            bias = _classify_bias(avg_err, resolved=True)

            per_contract[key] = {
                "n_predictions": len(entries),
                "n_resolved": len(resolved_entries),
                "scoring_type": "resolved",
                "avg_signed_error": round(avg_err, 4),
                "avg_abs_error": round(avg_abs, 4),
                "avg_brier": round(avg_brier, 4),
                "bias": bias,
                "history": entries[-MAX_HISTORY:],
            }
        else:
            # Pre-resolution: use market divergence (clearly labeled)
            errors = [e["signed_error"] for e in entries]
            avg_err = sum(errors) / len(errors)
            avg_abs = sum(abs(e) for e in errors) / len(errors)

            bias = _classify_bias(avg_err, resolved=False)

            per_contract[key] = {
                "n_predictions": len(entries),
                "n_resolved": 0,
                "scoring_type": "pre_resolution",
                "avg_divergence": round(avg_err, 4),
                "avg_abs_divergence": round(avg_abs, 4),
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
            per_domain[domain] = {
                "resolved_errors": [],
                "resolved_briers": [],
                "unresolved_divergences": [],
            }

        for e in entries:
            if e.get("scoring_type") == "resolved":
                per_domain[domain]["resolved_errors"].append(e["signed_error"])
                per_domain[domain]["resolved_briers"].append(e["brier_score"])
            else:
                per_domain[domain]["unresolved_divergences"].append(e["signed_error"])

    # Finalize domain stats
    domain_stats = {}
    for domain_key, raw in per_domain.items():
        resolved = raw["resolved_errors"]
        unresolved = raw["unresolved_divergences"]

        if resolved:
            # Prefer resolved stats
            avg_err = sum(resolved) / len(resolved)
            avg_brier = sum(raw["resolved_briers"]) / len(raw["resolved_briers"])

            bias = _classify_bias(avg_err, resolved=True)

            domain_stats[domain_key] = {
                "n_resolved": len(resolved),
                "n_unresolved": len(unresolved),
                "scoring_type": "resolved",
                "avg_signed_error": round(avg_err, 4),
                "avg_brier": round(avg_brier, 4),
                "bias": bias,
            }
        elif unresolved:
            avg_div = sum(unresolved) / len(unresolved)
            bias = _classify_bias(avg_div, resolved=False)

            domain_stats[domain_key] = {
                "n_resolved": 0,
                "n_unresolved": len(unresolved),
                "scoring_type": "pre_resolution",
                "avg_divergence": round(avg_div, 4),
                "bias": bias,
            }

    # Write cache
    cache = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_files_scanned": len(files),
        "n_resolutions": len(resolutions),
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
    n_resolved_contracts = sum(1 for c in per_contract.values() if c.get("scoring_type") == "resolved")
    print(f"  Lessons cache: {n_contracts} contracts ({n_resolved_contracts} resolved), "
          f"{n_domains} domains (from {len(files)} prediction files)")
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

    Tetlock principle: lessons should reference ACTUAL OUTCOMES when available,
    not just market divergence. "You predicted 70% but the event didn't happen"
    is a real lesson. "You predicted 70% but the market was at 60%" is not.
    """
    cache = load_lessons_cache()
    if not cache:
        return ""

    lines = []

    # Per-contract lesson
    contract_data = cache.get("per_contract", {}).get(contract_key, {})
    n_contract = contract_data.get("n_predictions", 0)

    if n_contract >= MIN_CONTRACT:
        scoring = contract_data.get("scoring_type", "pre_resolution")

        if scoring == "resolved":
            # REAL lessons from actual outcomes
            n_res = contract_data.get("n_resolved", 0)
            avg_brier = contract_data.get("avg_brier", 0)
            bias = contract_data.get("bias", "neutral")
            avg_err = contract_data.get("avg_signed_error", 0)

            lines.append(f"PAST PERFORMANCE ON THIS CONTRACT ({n_res} resolved predictions):")
            lines.append(f"  Your Brier score: {avg_brier:.4f} "
                         f"{'(better than coin flip)' if avg_brier < 0.25 else '(worse than coin flip)'}")

            if bias != "neutral":
                lines.append(f"  WARNING: You tend to predict {bias} (avg error: {avg_err:+.4f})")
                if "TOO HIGH" in bias:
                    lines.append("  -> Correct for this by nudging your estimate DOWN")
                elif "TOO LOW" in bias:
                    lines.append("  -> Correct for this by nudging your estimate UP")

            # Show recent prediction history
            history = contract_data.get("history", [])
            recent = [h for h in history[-5:] if h.get("scoring_type") == "resolved"]
            if recent:
                for h in recent:
                    outcome_str = "YES" if h.get("outcome") == 1 else "NO"
                    lines.append(f"  Cycle {h.get('cycle', '?')}: "
                                 f"predicted {h['prediction']:.2f}, outcome={outcome_str}, "
                                 f"brier={h.get('brier_score', 0):.4f}")
        else:
            # Pre-resolution: weaker signal, clearly labeled
            history = contract_data.get("history", [])
            preds = [f"{h['prediction']:.2f}" for h in history[-5:]]
            mkts = [f"{h['market_price']:.2f}" for h in history[-5:]]
            lines.append(f"TRACKING THIS CONTRACT ({n_contract} predictions, NOT YET RESOLVED):")
            lines.append(f"  Your predictions: {', '.join(preds)}")
            lines.append(f"  Market prices:    {', '.join(mkts)}")

            bias = contract_data.get("bias", "neutral")
            if bias != "neutral":
                avg_div = contract_data.get("avg_divergence", 0)
                lines.append(f"  Note: You are consistently {bias} (avg divergence: {avg_div:+.3f})")
                lines.append(f"  (This is vs market only -- not necessarily wrong)")

    # Per-domain lesson (as fallback or supplement)
    domain_data = cache.get("per_domain", {}).get(domain or "", {})
    n_domain_resolved = domain_data.get("n_resolved", 0)

    if n_domain_resolved >= MIN_DOMAIN:
        # Domain lesson from actual outcomes
        bias = domain_data.get("bias", "neutral")
        avg_brier = domain_data.get("avg_brier", 0)

        domain_line = f"DOMAIN PATTERN ({domain}, {n_domain_resolved} resolved predictions): "
        if bias != "neutral":
            domain_line += f"you tend to predict {bias} (Brier: {avg_brier:.4f})"
        else:
            domain_line += f"well-calibrated in this domain (Brier: {avg_brier:.4f})"
        lines.append(domain_line)

    elif domain_data.get("n_unresolved", 0) >= MIN_DOMAIN and n_contract < 3:
        # Weak proxy: domain divergence from market
        bias = domain_data.get("bias", "neutral")
        if bias != "neutral":
            lines.append(f"DOMAIN NOTE ({domain}): "
                         f"you tend to predict {bias} vs market (pre-resolution data only)")

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
