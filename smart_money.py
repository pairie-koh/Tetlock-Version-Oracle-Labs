"""
Oracle Lab — Smart Money Detector
A non-LLM quantitative agent that monitors Polymarket trading patterns to detect
informed trading activity.

Monitors:
- Volume spikes (sudden increases compared to recent average)
- Price movements (rapid changes in short windows)
- Bid-ask spread changes (spread widening/narrowing)
- Order book imbalance (heavy buying vs selling pressure)

Outputs a JSON signal for each contract with:
- signal_strength: 0 (no unusual activity) to 1 (very unusual activity)
- direction: "bullish", "bearish", or "neutral"
- evidence: list of what triggered the signal

Usage:
    python smart_money.py

Output:
    - Prints analysis to stdout
    - Saves JSON to smart_money_signals.json in script directory

Configuration:
    Adjust thresholds at top of file:
    - VOLUME_SPIKE_THRESHOLD: Volume ratio to trigger signal (default: 2.0x)
    - PRICE_MOVE_THRESHOLD: Price change % to trigger signal (default: 0.05 = 5%)
    - SPREAD_CHANGE_THRESHOLD: Spread change % to trigger signal (default: 0.5 = 50%)
    - IMBALANCE_THRESHOLD: Bid/ask ratio to trigger signal (default: 2.0:1)

Error Handling:
    - Returns neutral signal with error message if API fails
    - Falls back to volatility proxy if volume data unavailable
    - Handles missing historical data gracefully
"""

import json
import os
import sys
import csv
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import requests

from constants import MARKETS, POLYMARKET_CLOB_URL, POLYMARKET_GAMMA_URL, PRICE_HISTORY_DIR
from orderflow import format_for_smart_money

# Get absolute path to price history
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_CSV = os.path.join(SCRIPT_DIR, PRICE_HISTORY_DIR, "prices.csv")


# ── Configuration ────────────────────────────────────────────────────────────

# Volume spike threshold: how many times above average triggers a signal
VOLUME_SPIKE_THRESHOLD = 2.0  # 2x average

# Price movement threshold: percentage move that triggers a signal
PRICE_MOVE_THRESHOLD = 0.05  # 5%

# Spread change threshold: percentage change in spread that's significant
SPREAD_CHANGE_THRESHOLD = 0.5  # 50% change in spread

# Order book imbalance threshold: ratio of buy to sell pressure
IMBALANCE_THRESHOLD = 2.0  # 2:1 ratio


# ── API Functions ────────────────────────────────────────────────────────────

def fetch_market_data(condition_id: str) -> Optional[Dict]:
    """
    Fetch market metadata and volume data from Gamma API.
    Falls back to estimated data if API fails.
    """
    try:
        # Try to find market in Gamma API by searching all events
        url = f"{POLYMARKET_GAMMA_URL}/events"
        resp = requests.get(url, params={"active": "true"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Search for our market by condition_id
        for event in data:
            for market in event.get("markets", []):
                if market.get("conditionId") == condition_id:
                    return {
                        "volume24hr": event.get("volume24hr", 0),
                        "volume1wk": event.get("volume1wk", 0),
                        "liquidity": event.get("liquidity", 0),
                    }

        # Market not found - return minimal data
        print(f"  WARNING: Market not found in Gamma API, using fallback")
        return {
            "volume24hr": 0,
            "volume1wk": 0,
            "liquidity": 0,
        }
    except Exception as e:
        print(f"  WARNING: Gamma API error ({e}), using fallback")
        return {
            "volume24hr": 0,
            "volume1wk": 0,
            "liquidity": 0,
        }


def fetch_order_book(token_id: str) -> Optional[Dict]:
    """Fetch current order book for a token from CLOB API."""
    try:
        url = f"{POLYMARKET_CLOB_URL}/book"
        params = {"token_id": token_id}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ERROR fetching order book for token {token_id[:20]}...: {e}")
        return None


def fetch_midpoint(token_id: str) -> Optional[float]:
    """Fetch current midpoint price for a token from CLOB API."""
    try:
        url = f"{POLYMARKET_CLOB_URL}/midpoint"
        params = {"token_id": token_id}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return float(data["mid"])
    except Exception as e:
        print(f"  ERROR fetching midpoint for token {token_id[:20]}...: {e}")
        return None


# ── Historical Data Functions ────────────────────────────────────────────────

def load_price_history(market_key: str, hours: int = 24) -> List[Dict]:
    """
    Load price history for a market from prices.csv.
    Returns list of {t: timestamp, p: price} for last N hours.
    """
    if not os.path.exists(PRICE_CSV):
        return []

    now = time.time()
    cutoff = now - (hours * 3600)
    history = []

    try:
        with open(PRICE_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row["timestamp"])
                    if ts < cutoff:
                        continue
                    val = row.get(market_key, "")
                    if val == "":
                        continue
                    price = float(val)
                    history.append({"t": int(ts), "p": price})
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"  WARNING: Could not load price history: {e}")

    return history


def calculate_baselines(history: List[Dict]) -> Dict:
    """
    Calculate baseline statistics from price history.
    Returns: avg_price, avg_volume (if available), volatility.
    """
    if not history:
        return {"avg_price": None, "volatility": None}

    prices = [h["p"] for h in history]
    avg_price = sum(prices) / len(prices)

    # Calculate simple volatility (standard deviation of returns)
    if len(prices) > 1:
        returns = [(prices[i] - prices[i-1]) / prices[i-1]
                   for i in range(1, len(prices)) if prices[i-1] != 0]
        if returns:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
    else:
        volatility = 0.0

    return {
        "avg_price": avg_price,
        "volatility": volatility,
    }


# ── Signal Detection Functions ───────────────────────────────────────────────

def detect_volume_spike(current_volume_24h: float, avg_volume: float, volatility: Optional[float] = None) -> Tuple[bool, float, str]:
    """
    Detect if current 24h volume is significantly above average.
    If volume data unavailable, use volatility as a proxy.
    Returns: (is_spike, ratio, description)
    """
    # If we have volume data, use it
    if avg_volume > 0 and current_volume_24h > 0:
        ratio = current_volume_24h / avg_volume
        if ratio >= VOLUME_SPIKE_THRESHOLD:
            return True, ratio, f"Volume {ratio:.1f}x above average"
        return False, ratio, ""

    # Fallback: use volatility as proxy for volume
    # High volatility often indicates high trading activity
    if volatility is not None and volatility > 0.03:  # 3% daily volatility threshold
        return True, volatility * 10, f"High volatility detected ({volatility*100:.1f}%)"

    return False, 0.0, ""


def detect_price_movement(current_price: float, history: List[Dict], window_hours: int = 2) -> Tuple[bool, float, str]:
    """
    Detect rapid price movement in a short window.
    Returns: (is_significant, change_pct, description)
    """
    if not history or current_price is None:
        return False, 0.0, ""

    now = time.time()
    cutoff = now - (window_hours * 3600)

    # Find price at start of window
    past_prices = [h["p"] for h in history if h["t"] >= cutoff]
    if not past_prices:
        return False, 0.0, ""

    past_price = past_prices[0]
    if past_price == 0:
        return False, 0.0, ""

    change_pct = (current_price - past_price) / past_price

    if abs(change_pct) >= PRICE_MOVE_THRESHOLD:
        direction = "up" if change_pct > 0 else "down"
        return True, change_pct, f"Price moved {abs(change_pct)*100:.1f}% {direction} in {window_hours}h"

    return False, change_pct, ""


def detect_spread_change(book: Dict, baseline_spread: Optional[float] = None) -> Tuple[bool, float, str]:
    """
    Detect significant bid-ask spread changes.
    Returns: (is_significant, current_spread, description)
    """
    if not book or "bids" not in book or "asks" not in book:
        return False, 0.0, ""

    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return False, 0.0, ""

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    current_spread = best_ask - best_bid

    if baseline_spread is None or baseline_spread == 0:
        # No baseline - just report current spread
        return False, current_spread, f"Current spread: {current_spread:.4f}"

    spread_change = (current_spread - baseline_spread) / baseline_spread

    if abs(spread_change) >= SPREAD_CHANGE_THRESHOLD:
        direction = "widened" if spread_change > 0 else "narrowed"
        return True, current_spread, f"Spread {direction} {abs(spread_change)*100:.1f}%"

    return False, current_spread, ""


def detect_order_book_imbalance(book: Dict, depth: int = 5) -> Tuple[bool, float, str]:
    """
    Detect heavy buying or selling pressure in order book.
    Returns: (is_imbalanced, ratio, description)
    """
    if not book or "bids" not in book or "asks" not in book:
        return False, 0.0, ""

    bids = book.get("bids", [])[:depth]
    asks = book.get("asks", [])[:depth]

    if not bids or not asks:
        return False, 0.0, ""

    # Sum up volume on bid and ask sides
    bid_volume = sum(float(b["size"]) for b in bids)
    ask_volume = sum(float(a["size"]) for a in asks)

    if ask_volume == 0:
        return False, 0.0, ""

    ratio = bid_volume / ask_volume

    if ratio >= IMBALANCE_THRESHOLD:
        return True, ratio, f"Heavy buying pressure (bid/ask ratio: {ratio:.1f}:1)"
    elif ratio <= (1.0 / IMBALANCE_THRESHOLD):
        return True, ratio, f"Heavy selling pressure (bid/ask ratio: {ratio:.1f}:1)"

    return False, ratio, ""


# ── Main Analysis Function ───────────────────────────────────────────────────

def analyze_market(market_key: str, market_config: Dict) -> Dict:
    """
    Analyze a single market for smart money signals.
    Returns signal dict with strength, direction, and evidence.
    """
    print(f"\nAnalyzing {market_key}...")

    # Fetch current market data
    market_data = fetch_market_data(market_config["condition_id"])

    # Fetch current price and order book
    token_id = market_config["yes_token_id"]
    current_price = fetch_midpoint(token_id)
    order_book = fetch_order_book(token_id)

    if current_price is None:
        return {
            "market": market_key,
            "signal_strength": 0.0,
            "direction": "neutral",
            "evidence": ["ERROR: Could not fetch current price"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    print(f"  Current price: {current_price:.4f}")
    print(f"  24h volume: ${market_data.get('volume24hr', 0):,.2f}")

    # Load historical data
    history_24h = load_price_history(market_key, hours=24)
    baselines = calculate_baselines(history_24h)

    print(f"  Historical data points (24h): {len(history_24h)}")
    if baselines["avg_price"]:
        print(f"  Average price (24h): {baselines['avg_price']:.4f}")
        print(f"  Volatility (24h): {baselines['volatility']:.4f}")

    # Detect signals
    evidence = []
    signal_components = []

    # 1. Volume spike (or volatility spike if volume unavailable)
    current_volume = market_data.get("volume24hr", 0)
    avg_volume = market_data.get("volume1wk", 0) / 7 if market_data.get("volume1wk", 0) > 0 else 0
    volume_spike, vol_ratio, vol_desc = detect_volume_spike(
        current_volume, avg_volume, baselines.get("volatility")
    )
    if volume_spike:
        evidence.append(vol_desc)
        signal_components.append(min(vol_ratio / VOLUME_SPIKE_THRESHOLD, 2.0) * 0.3)  # Cap at 2x threshold, 30% weight

    # 2. Price movement
    price_move, change_pct, move_desc = detect_price_movement(current_price, history_24h, window_hours=2)
    if price_move:
        evidence.append(move_desc)
        signal_components.append(min(abs(change_pct) / PRICE_MOVE_THRESHOLD, 2.0) * 0.4)  # Cap at 2x threshold, 40% weight

    # 3. Spread analysis (look at orders near midpoint, not just best bid/ask)
    if order_book and current_price:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        # Find closest bid and ask to current price (within 10%)
        price_range = current_price * 0.1
        close_bids = [b for b in bids if abs(float(b["price"]) - current_price) <= price_range]
        close_asks = [a for a in asks if abs(float(a["price"]) - current_price) <= price_range]

        if close_bids and close_asks:
            best_close_bid = max(close_bids, key=lambda x: float(x["price"]))
            best_close_ask = min(close_asks, key=lambda x: float(x["price"]))
            tight_spread = float(best_close_ask["price"]) - float(best_close_bid["price"])
            spread_pct = (tight_spread / current_price * 100) if current_price > 0 else 0

            # Flag if spread near midpoint is > 5% of price
            if spread_pct > 5:
                evidence.append(f"Wide spread near midpoint ({spread_pct:.1f}%)")
                signal_components.append(0.15)
        elif not close_bids or not close_asks:
            # No orders near midpoint - sign of thin liquidity
            evidence.append("Thin liquidity (no orders near midpoint)")
            signal_components.append(0.2)

    # 4. Order book imbalance
    if order_book:
        imbalance_sig, ratio, imb_desc = detect_order_book_imbalance(order_book, depth=5)
        if imbalance_sig:
            evidence.append(imb_desc)
            signal_components.append(min(max(ratio, 1.0/ratio) / IMBALANCE_THRESHOLD, 2.0) * 0.15)  # Cap at 2x threshold, 15% weight

    # 5. Historical order flow from pmxt archive (if available)
    pmxt_signals = format_for_smart_money(market_key)
    if pmxt_signals:
        flow = pmxt_signals.get("flow")
        book = pmxt_signals.get("book")

        if flow and flow.get("flow_direction") != "neutral":
            pct = flow.get("buy_pct", 50)
            skew = abs(pct - 50) / 50  # 0 = balanced, 1 = fully one-sided
            if skew > 0.15:
                dir_label = "buy" if pct > 50 else "sell"
                evidence.append(f"Order flow {dir_label} pressure ({pct:.0f}% buys over {flow.get('num_events', 0)} events)")
                signal_components.append(min(skew, 0.5) * 0.3)  # 30% weight, capped

        if flow and flow.get("large_orders"):
            n_large = len(flow["large_orders"])
            largest = flow["large_orders"][0]
            evidence.append(f"{n_large} large order(s) detected (biggest: {largest['side']} {largest['size']:,} @ {largest['price']})")
            signal_components.append(min(n_large / 5, 1.0) * 0.15)  # 15% weight

        if book and abs(book.get("imbalance_trend", 0)) > 0.15:
            trend_dir = "bullish" if book["imbalance_trend"] > 0 else "bearish"
            evidence.append(f"Order book imbalance trending {trend_dir} (Δ{book['imbalance_trend']:+.2f})")
            signal_components.append(0.1)

    # Calculate overall signal strength (0 to 1)
    signal_strength = min(sum(signal_components), 1.0) if signal_components else 0.0

    # Determine direction
    direction = "neutral"
    if price_move:
        direction = "bullish" if change_pct > 0 else "bearish"
    elif order_book and imbalance_sig:
        # Use order book imbalance if no price movement
        imbalance_ratio = ratio if ratio >= 1.0 else 1.0 / ratio
        if imbalance_ratio >= IMBALANCE_THRESHOLD:
            direction = "bullish" if ratio >= IMBALANCE_THRESHOLD else "bearish"

    if not evidence:
        evidence.append("No unusual activity detected")

    print(f"  Signal strength: {signal_strength:.2f}")
    print(f"  Direction: {direction}")
    print(f"  Evidence: {'; '.join(evidence)}")

    return {
        "market": market_key,
        "signal_strength": round(signal_strength, 3),
        "direction": direction,
        "evidence": evidence,
        "current_price": current_price,
        "volume_24h": current_volume,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    """Main entry point - analyze all markets and output signals."""
    print("=== Smart Money Detector ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Markets to analyze: {list(MARKETS.keys())}")

    signals = []

    for market_key, market_config in MARKETS.items():
        signal = analyze_market(market_key, market_config)
        signals.append(signal)

    # Output results as JSON
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": signals,
        "config": {
            "volume_spike_threshold": VOLUME_SPIKE_THRESHOLD,
            "price_move_threshold": PRICE_MOVE_THRESHOLD,
            "spread_change_threshold": SPREAD_CHANGE_THRESHOLD,
            "imbalance_threshold": IMBALANCE_THRESHOLD,
        }
    }

    print("\n=== Smart Money Signals (JSON) ===")
    print(json.dumps(output, indent=2))

    # Save to file in script directory
    output_file = os.path.join(SCRIPT_DIR, "smart_money_signals.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved signals to {output_file}")

    return output


if __name__ == "__main__":
    main()
