"""
Oracle Lab — contracts.py
Pulls ~30 active, high-volume contracts from Polymarket across diverse domains.
Saves to contracts/active_contracts.json.

Usage: python contracts.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict

import requests

from constants import POLYMARKET_GAMMA_URL


def fetch_active_markets(limit=200):
    """Fetch active markets from Polymarket Gamma API."""
    url = f"{POLYMARKET_GAMMA_URL}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "archived": "false",
        "limit": limit,
    }

    print(f"Fetching markets from {url}...")
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        markets = resp.json()
        print(f"  Fetched {len(markets)} active markets")
        return markets
    except requests.exceptions.RequestException as e:
        print(f"  ERROR fetching markets: {e}")
        return []


def extract_categories(market):
    """Extract categories/tags from market data."""
    categories = set()

    # Direct category field
    if market.get("category"):
        categories.add(market["category"])

    # Event-level categories
    for event in market.get("events", []):
        if event.get("category"):
            categories.add(event["category"])

    # Infer from question text (simple heuristics)
    question = market.get("question", "").lower()
    if any(word in question for word in ["trump", "biden", "election", "senate", "congress", "president"]):
        categories.add("politics")
    if any(word in question for word in ["war", "military", "conflict", "ceasefire", "invasion"]):
        categories.add("geopolitics")
    if any(word in question for word in ["btc", "bitcoin", "eth", "ethereum", "crypto", "price"]):
        categories.add("crypto")
    if any(word in question for word in ["nfl", "nba", "super bowl", "championship", "fifa", "world cup", "finals"]):
        categories.add("sports")
    if any(word in question for word in ["movie", "film", "box office", "album", "song", "artist"]):
        categories.add("entertainment")
    if any(word in question for word in ["economy", "gdp", "inflation", "jobs", "unemployment", "recession"]):
        categories.add("economics")
    if any(word in question for word in ["tech", "ai", "technology", "release", "apple", "google"]):
        categories.add("tech")

    return list(categories)


def parse_token_ids(market):
    """Parse YES and NO token IDs from clobTokenIds field."""
    clob_token_ids = market.get("clobTokenIds", "")

    if not clob_token_ids:
        return None, None

    try:
        # API returns: "[\"123...\", \"456...\"]"
        token_ids = json.loads(clob_token_ids)
        if len(token_ids) >= 2:
            # First is YES, second is NO (based on outcomes order)
            return token_ids[0], token_ids[1]
    except (json.JSONDecodeError, IndexError):
        pass

    return None, None


def filter_and_rank_markets(markets, target_count=30):
    """
    Filter markets to those with good volume and diversity.
    Returns top N markets ranked by volume within diverse categories.
    """
    print(f"\nFiltering {len(markets)} markets...")

    # Filter out markets without required data
    valid_markets = []
    for m in markets:
        # Must have token IDs
        yes_token, no_token = parse_token_ids(m)
        if not yes_token or not no_token:
            continue

        # Must have reasonable volume (at least $1,000)
        volume = float(m.get("volumeNum", 0))
        if volume < 1000:
            continue

        # Must have end date in the future
        end_date = m.get("endDate")
        if not end_date:
            continue

        try:
            end_datetime = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            if end_datetime < datetime.now(timezone.utc):
                continue
        except (ValueError, AttributeError):
            continue

        valid_markets.append(m)

    print(f"  {len(valid_markets)} markets passed basic filters")

    # Sort by volume
    valid_markets.sort(key=lambda m: float(m.get("volumeNum", 0)), reverse=True)

    # Diversify by category
    category_counts = defaultdict(int)
    selected_markets = []

    # First pass: take top markets ensuring diversity (max 8 per category)
    max_per_category = 8
    for m in valid_markets:
        if len(selected_markets) >= target_count:
            break

        categories = extract_categories(m)

        # Check if we can add this market without over-concentrating
        can_add = False
        for cat in categories:
            if category_counts[cat] < max_per_category:
                can_add = True
                break

        if can_add or not categories:  # Allow if no categories identified
            selected_markets.append(m)
            for cat in categories:
                category_counts[cat] += 1

    # If we don't have enough, take more from top volume markets
    if len(selected_markets) < target_count:
        for m in valid_markets:
            if m not in selected_markets:
                selected_markets.append(m)
                if len(selected_markets) >= target_count:
                    break

    print(f"  Selected {len(selected_markets)} diverse markets")
    print(f"  Category distribution: {dict(category_counts)}")

    return selected_markets


def format_contract(market):
    """Format market data into contract format matching MARKETS dict structure."""
    yes_token, no_token = parse_token_ids(market)

    # Parse outcomePrices whether it's a list or JSON string
    prices_raw = market.get("outcomePrices", [])
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except (json.JSONDecodeError, ValueError):
            prices_raw = []
    try:
        current_prices = {"yes": float(prices_raw[0]), "no": float(prices_raw[1])}
    except (IndexError, TypeError, ValueError):
        current_prices = {"yes": 0, "no": 0}

    return {
        "question": market.get("question", ""),
        "slug": market.get("slug", ""),
        "condition_id": market.get("conditionId", ""),
        "yes_token_id": yes_token,
        "no_token_id": no_token,
        "end_date": market.get("endDateIso", market.get("endDate", "")[:10] if market.get("endDate") else ""),
        "description": market.get("description", "")[:500],
        "categories": extract_categories(market),
        "volume": float(market.get("volumeNum", 0)),
        "liquidity": float(market.get("liquidityNum", 0)),
        "current_prices": current_prices,
        "resolution_source": market.get("resolutionSource", ""),
        "polymarket_id": market.get("id", ""),
    }


def test_clob_price_lookup(token_id):
    """Test that we can fetch price for a token ID from CLOB API."""
    from constants import POLYMARKET_CLOB_URL

    url = f"{POLYMARKET_CLOB_URL}/midpoint"
    params = {"token_id": token_id}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price = float(data.get("mid", 0))
        return price
    except Exception as e:
        print(f"    WARNING: CLOB price lookup failed: {e}")
        return None


def main():
    """Main execution: fetch, filter, save contracts."""
    print("=== Polymarket Contract Puller ===\n")

    # Fetch active markets
    markets = fetch_active_markets(limit=200)

    if not markets:
        print("ERROR: No markets fetched. Exiting.")
        sys.exit(1)

    # Filter and rank to get ~30 diverse contracts
    selected = filter_and_rank_markets(markets, target_count=30)

    if not selected:
        print("ERROR: No markets passed filtering. Exiting.")
        sys.exit(1)

    # Format contracts
    print("\nFormatting contracts...")
    contracts = []
    for m in selected:
        contract = format_contract(m)
        contracts.append(contract)

    # Print summary
    print(f"\n=== Selected {len(contracts)} Contracts ===")
    for i, c in enumerate(contracts, 1):
        print(f"{i:2}. {c['question'][:70]:70} | ${c['volume']:>12,.0f} | {', '.join(c['categories'][:3])}")

    # Test CLOB API with first contract
    print("\n=== Testing CLOB API ===")
    if contracts:
        test_contract = contracts[0]
        print(f"Testing price lookup for: {test_contract['question'][:60]}")
        print(f"  YES token: {test_contract['yes_token_id'][:40]}...")

        price = test_clob_price_lookup(test_contract['yes_token_id'])
        if price is not None:
            print(f"  CLOB midpoint price: {price:.4f}")
            print(f"  Gamma API price: {test_contract['current_prices']['yes']:.4f}")
            print("  CLOB API lookup: OK")
        else:
            print("  CLOB API lookup: FAILED (but continuing anyway)")

    # Save to contracts/active_contracts.json
    output_dir = "contracts"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "active_contracts.json")

    output_data = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_contracts": len(contracts),
        "contracts": contracts,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n=== Saved {len(contracts)} contracts to {output_file} ===")

    # Summary statistics
    total_volume = sum(c["volume"] for c in contracts)
    avg_volume = total_volume / len(contracts) if contracts else 0

    print(f"\nSummary:")
    print(f"  Total volume: ${total_volume:,.0f}")
    print(f"  Average volume: ${avg_volume:,.0f}")
    print(f"  Contracts saved: {len(contracts)}")

    # Category breakdown
    category_counts = defaultdict(int)
    for c in contracts:
        for cat in c["categories"]:
            category_counts[cat] += 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
