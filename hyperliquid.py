"""
Oracle Lab — hyperliquid.py
Fetches 24/7 perp prices from Hyperliquid for real-time market context.

Hyperliquid perps trade around the clock, so we get live BTC, ETH, oil,
S&P 500, and gold prices even when traditional markets are closed.
This gives the LLM forecaster price context that Polymarket CLOB alone
can't provide (CLOB only has prediction market prices, not asset prices).

Two classes of perps:
  1. Native perps (BTC, ETH, SOL) — fetched via allMids endpoint
  2. Builder perps (xyz:CL, xyz:SP500, xyz:GOLD) — fetched via candleSnapshot

API is free, no auth needed for info endpoints.

Usage: python hyperliquid.py
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"

# Assets we care about, grouped by relevance to our forecast contracts
ASSETS = {
    # Native perps — available via allMids
    "BTC": {
        "coin": "BTC",
        "label": "Bitcoin",
        "source": "native",
        "relevant_to": ["bitcoin_daily", "what-price-will-bitcoin-hit-before-2027"],
    },
    "ETH": {
        "coin": "ETH",
        "label": "Ethereum",
        "source": "native",
        "relevant_to": [],
    },
    "SOL": {
        "coin": "SOL",
        "label": "Solana",
        "source": "native",
        "relevant_to": [],
    },
    # Builder perps — need candleSnapshot
    "OIL": {
        "coin": "xyz:CL",
        "label": "WTI Crude Oil",
        "source": "builder",
        "relevant_to": ["oil_daily", "will-gas-hit-by-end-of-march"],
    },
    "SP500": {
        "coin": "xyz:SP500",
        "label": "S&P 500",
        "source": "builder",
        "relevant_to": ["us-recession-by-end-of-2026"],
    },
    "GOLD": {
        "coin": "xyz:GOLD",
        "label": "Gold",
        "source": "builder",
        "relevant_to": [],
    },
}


def fetch_all_mids():
    """Fetch midpoint prices for all native perps via allMids."""
    resp = requests.post(
        HYPERLIQUID_API,
        json={"type": "allMids"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_daily_open(coin="BTC"):
    """Fetch today's daily candle open price for a native perp.

    Uses candleSnapshot with 1d interval. The open price of the current
    daily candle tells us where BTC started the day, which is what the
    "Bitcoin Up or Down" contract resolves against.
    """
    now_ms = int(time.time() * 1000)
    # Ask for last 48h to ensure we capture today's daily candle
    start_ms = now_ms - 48 * 60 * 60 * 1000

    resp = requests.post(
        HYPERLIQUID_API,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": now_ms,
            },
        },
        timeout=10,
    )
    resp.raise_for_status()
    candles = resp.json()

    if not candles:
        return None

    # The last candle is today's daily candle
    latest = candles[-1]
    return {
        "open": float(latest["o"]),
        "high": float(latest["h"]),
        "low": float(latest["l"]),
        "close": float(latest["c"]),
        "timestamp": latest.get("t", 0),
    }


def fetch_builder_price(coin):
    """Fetch latest price for a builder perp via candleSnapshot.

    Builder perps (prefixed like xyz:CL) aren't in allMids,
    but candleSnapshot works with 1-minute candles.
    """
    now_ms = int(time.time() * 1000)
    # Ask for a 5-minute window ending now
    start_ms = now_ms - 5 * 60 * 1000

    resp = requests.post(
        HYPERLIQUID_API,
        json={
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": now_ms,
            },
        },
        timeout=10,
    )
    resp.raise_for_status()
    candles = resp.json()

    if not candles:
        return None

    # Use the close price of the most recent candle
    latest = candles[-1]
    return float(latest["c"])


def fetch_prices():
    """Fetch prices for all configured assets.

    Returns dict like:
    {
        "BTC": {"price": 68174.5, "label": "Bitcoin", "coin": "BTC"},
        "OIL": {"price": 99.1, "label": "WTI Crude Oil", "coin": "xyz:CL"},
        ...
    }
    """
    results = {}

    # Batch-fetch native perps
    try:
        all_mids = fetch_all_mids()
    except Exception as e:
        print(f"  WARNING: allMids failed: {e}")
        all_mids = {}

    for key, asset in ASSETS.items():
        coin = asset["coin"]

        if asset["source"] == "native":
            # Look up in allMids response
            price_str = all_mids.get(coin)
            if price_str is not None:
                try:
                    price = float(price_str)
                    results[key] = {
                        "price": price,
                        "label": asset["label"],
                        "coin": coin,
                        "relevant_to": asset["relevant_to"],
                    }
                    print(f"  {asset['label']} ({coin}): ${price:,.2f}")
                except (ValueError, TypeError):
                    print(f"  WARNING: Could not parse price for {coin}: {price_str}")
            else:
                print(f"  WARNING: {coin} not found in allMids")

        elif asset["source"] == "builder":
            try:
                price = fetch_builder_price(coin)
                if price is not None:
                    results[key] = {
                        "price": price,
                        "label": asset["label"],
                        "coin": coin,
                        "relevant_to": asset["relevant_to"],
                    }
                    print(f"  {asset['label']} ({coin}): ${price:,.2f}")
                else:
                    print(f"  WARNING: No candle data for {coin}")
            except Exception as e:
                print(f"  WARNING: candleSnapshot failed for {coin}: {e}")

            time.sleep(0.1)  # Small delay between builder perp calls

    # Fetch BTC daily candle for intraday context
    if "BTC" in results:
        try:
            btc_daily = fetch_daily_open("BTC")
            if btc_daily:
                results["BTC"]["daily_open"] = btc_daily["open"]
                results["BTC"]["daily_high"] = btc_daily["high"]
                results["BTC"]["daily_low"] = btc_daily["low"]
                pct_change = ((results["BTC"]["price"] - btc_daily["open"]) / btc_daily["open"]) * 100
                results["BTC"]["intraday_pct_change"] = round(pct_change, 2)
                results["BTC"]["intraday_direction"] = "UP" if pct_change >= 0 else "DOWN"
                print(f"  BTC daily open: ${btc_daily['open']:,.2f} | change: {pct_change:+.2f}%")
        except Exception as e:
            print(f"  WARNING: BTC daily candle failed: {e}")

    return results


def save_prices(prices):
    """Save fetched prices to data/hyperliquid_prices.json."""
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Hyperliquid perpetual futures (24/7)",
        "prices": prices,
    }

    out_path = os.path.join(output_dir, "hyperliquid_prices.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(prices)} prices to {out_path}")
    return out_path


def load_prices():
    """Load previously saved Hyperliquid prices. Returns dict or None."""
    path = os.path.join("data", "hyperliquid_prices.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def format_for_prompt():
    """Format Hyperliquid prices as a text block for LLM prompts.

    Returns a string like:
      REAL-TIME ASSET PRICES (Hyperliquid 24/7 perps, as of 2026-03-23T15:00:00Z):
        Bitcoin (BTC): $68,174.50
        WTI Crude Oil (xyz:CL): $99.10
        S&P 500 (xyz:SP500): $6,478.70
        Gold (xyz:GOLD): $4,363.10
        Ethereum (ETH): $2,055.95
        Solana (SOL): $86.65
    """
    data = load_prices()
    if not data or not data.get("prices"):
        return ""

    fetched_at = data.get("fetched_at", "unknown")
    lines = [f"REAL-TIME ASSET PRICES (Hyperliquid 24/7 perps, as of {fetched_at}):"]

    for key, info in data["prices"].items():
        price = info["price"]
        label = info["label"]
        coin = info["coin"]
        lines.append(f"  {label} ({coin}): ${price:,.2f}")

    return "\n".join(lines)


def format_btc_intraday():
    """Format BTC intraday context for the bitcoin_daily contract prompt.

    Returns text block with open price, current price, % change, and direction.
    Returns empty string if data not available.
    """
    data = load_prices()
    if not data or not data.get("prices"):
        return ""

    btc = data["prices"].get("BTC", {})
    if "daily_open" not in btc:
        return ""

    fetched_at = data.get("fetched_at", "unknown")
    current = btc["price"]
    open_price = btc["daily_open"]
    pct = btc.get("intraday_pct_change", 0)
    direction = btc.get("intraday_direction", "FLAT")
    high = btc.get("daily_high", current)
    low = btc.get("daily_low", current)

    return f"""BITCOIN INTRADAY DATA (Hyperliquid, as of {fetched_at}):
  Today's open: ${open_price:,.2f}
  Current price: ${current:,.2f}
  Intraday change: {pct:+.2f}% ({direction})
  Today's high: ${high:,.2f}
  Today's low: ${low:,.2f}
  NOTE: The "Bitcoin Up or Down" contract resolves YES if BTC closes HIGHER than it opened."""


def main():
    print("=== Hyperliquid Price Fetch ===\n")

    prices = fetch_prices()

    if not prices:
        print("\nERROR: No prices fetched")
        return

    save_prices(prices)

    # Show the prompt block
    print(f"\n--- Prompt block ---")
    print(format_for_prompt())
    print(f"--- End prompt block ---")


if __name__ == "__main__":
    main()
