"""
Tetlock Oracle Labs — market_data.py
Fetches Polymarket prices and manages price history.
No LLM calls — pure CLOB API + local I/O.
"""

import csv
import json
import os
import time
from datetime import datetime, timezone

import requests

from constants import MARKETS, POLYMARKET_CLOB_URL, PRICE_HISTORY_DIR, PRICE_CSV


def fetch_midpoint(token_id):
    """Fetch current midpoint price for a token from Polymarket CLOB."""
    url = f"{POLYMARKET_CLOB_URL}/midpoint"
    resp = requests.get(url, params={"token_id": token_id}, timeout=15)
    resp.raise_for_status()
    return float(resp.json()["mid"])


def fetch_all_prices():
    """Fetch current YES prices for all markets. Returns {market_key: float}."""
    prices = {}
    for key, market in MARKETS.items():
        try:
            price = fetch_midpoint(market["yes_token_id"])
            prices[key] = price
            print(f"  {key}: {price:.4f}")
        except Exception as e:
            print(f"  {key}: FAILED ({e})")
    return prices


def append_price_row(prices):
    """Append a row to prices.csv."""
    os.makedirs(PRICE_HISTORY_DIR, exist_ok=True)
    market_keys = list(MARKETS.keys())
    ts = int(time.time())

    file_exists = os.path.exists(PRICE_CSV) and os.path.getsize(PRICE_CSV) > 0

    with open(PRICE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp"] + market_keys)
        row = [ts] + [prices.get(k, "") for k in market_keys]
        writer.writerow(row)


def get_price_history(market_key, hours=168):
    """Read recent price history from CSV. Returns [(timestamp, price), ...]."""
    if not os.path.exists(PRICE_CSV):
        return []

    cutoff = time.time() - (hours * 3600)
    history = []

    with open(PRICE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["timestamp"])
                val = row.get(market_key, "")
                if val == "" or ts < cutoff:
                    continue
                history.append((int(ts), float(val)))
            except (ValueError, KeyError):
                continue

    return history


if __name__ == "__main__":
    print("=== Fetching Polymarket Prices ===")
    prices = fetch_all_prices()
    if prices:
        append_price_row(prices)
        print(f"\nAppended to {PRICE_CSV}")
    else:
        print("No prices fetched.")
