"""
Oracle Lab — rolling_contracts.py
Fetches today's rolling daily contracts from Polymarket.

Each rolling contract has a predictable event slug pattern.
This script constructs today's slugs, fetches event data from the
Gamma API, and saves the result to contracts/rolling_today.json.

Usage: python rolling_contracts.py [--date YYYY-MM-DD]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

from constants import POLYMARKET_GAMMA_URL


# Slug patterns: {month} = lowercase full month, {day} = day without zero-pad, {year} = 4-digit
ROLLING_CONTRACTS = {
    "bitcoin_daily": {
        "name": "Bitcoin Up or Down",
        "slug_template": "bitcoin-up-or-down-on-{month}-{day}-{year}",
        "category": "economics",
        "type": "binary",
    },
    "oil_daily": {
        "name": "Crude Oil (CL) Up or Down",
        "slug_template": "cl-up-or-down-on-{month}-{day}-{year}",
        "category": "economics",
        "type": "binary",
    },
    "nyc_temp": {
        "name": "Highest temperature in NYC",
        "slug_template": "highest-temperature-in-nyc-on-{month}-{day}-{year}",
        "category": "weather",
        "type": "multi-outcome",
    },
    "miami_temp": {
        "name": "Highest temperature in Miami",
        "slug_template": "highest-temperature-in-miami-on-{month}-{day}-{year}",
        "category": "weather",
        "type": "multi-outcome",
    },
}


def build_slug(template, date):
    """Build event slug from template and date."""
    month = date.strftime("%B").lower()  # e.g. "march"
    day = str(date.day)                  # e.g. "23" (no zero-pad)
    year = str(date.year)                # e.g. "2026"
    return template.format(month=month, day=day, year=year)


def fetch_event(slug):
    """Fetch event data from Gamma API by slug."""
    url = f"{POLYMARKET_GAMMA_URL}/events"
    params = {"slug": slug}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        if events and len(events) > 0:
            return events[0]
    except requests.exceptions.RequestException as e:
        print(f"  ERROR fetching event: {e}")

    return None


def _parse_json_field(raw, transform=None):
    """Parse a field that may be a JSON string or already a list."""
    if isinstance(raw, list):
        return [transform(x) for x in raw] if transform else raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return [transform(x) for x in parsed] if transform else parsed
        except (json.JSONDecodeError, ValueError):
            return [raw] if not transform else []
    return []


def parse_market(market):
    """Extract useful fields from a market object."""
    return {
        "question": market.get("question", ""),
        "slug": market.get("slug", ""),
        "condition_id": market.get("conditionId", ""),
        "token_ids": _parse_json_field(market.get("clobTokenIds", "")),
        "outcomes": _parse_json_field(market.get("outcomes", "")),
        "prices": _parse_json_field(market.get("outcomePrices", ""), transform=float),
        "volume": float(market.get("volumeNum", 0)),
        "liquidity": float(market.get("liquidityNum", 0)),
        "end_date": market.get("endDate", ""),
    }


def fetch_rolling_contracts(target_date):
    """Fetch all rolling contracts for a given date."""
    results = {}

    for key, config in ROLLING_CONTRACTS.items():
        slug = build_slug(config["slug_template"], target_date)
        print(f"  {config['name']}: {slug}")

        event = fetch_event(slug)
        if not event:
            print(f"    NOT FOUND")
            results[key] = {
                "name": config["name"],
                "slug": slug,
                "category": config["category"],
                "type": config["type"],
                "status": "not_found",
                "markets": [],
            }
            continue

        raw_markets = event.get("markets", [])
        markets = [parse_market(m) for m in raw_markets]

        print(f"    FOUND — {len(markets)} market(s)")
        for m in markets:
            price_str = ", ".join(f"{o}={p:.3f}" for o, p in zip(m["outcomes"], m["prices"]))
            print(f"      {m['question'][:60]} | {price_str}")

        results[key] = {
            "name": config["name"],
            "slug": slug,
            "event_id": event.get("id", ""),
            "category": config["category"],
            "type": config["type"],
            "status": "active",
            "markets": markets,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch today's rolling contracts from Polymarket")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), defaults to today")
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        target_date = datetime.now(timezone.utc)

    date_str = target_date.strftime("%Y-%m-%d")
    print(f"=== Rolling Contracts for {date_str} ===\n")

    results = fetch_rolling_contracts(target_date)

    # Summary
    found = sum(1 for r in results.values() if r["status"] == "active")
    total = len(results)
    print(f"\n=== Found {found}/{total} rolling contracts ===")

    # Save
    output_dir = "contracts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "rolling_today.json")

    output_data = {
        "date": date_str,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contracts": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
