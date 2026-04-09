"""
Oracle Lab — orderflow.py
Extracts order flow signals from pmxt Polymarket order book archive.

The pmxt archive (archive.pmxt.dev) provides free hourly Parquet snapshots
of every order book update on Polymarket at millisecond precision. This module
downloads recent snapshots, filters to tracked contracts, and computes
leading indicators that the LLM can use to detect informed trading
*before* it shows up in the midpoint price.

Signals computed per contract:
  - Bid/ask imbalance ratio (depth-weighted)
  - Spread trajectory (widening = uncertainty, narrowing = consensus)
  - Net order flow direction (buy vs sell pressure from price_change events)
  - Volume concentration (large orders as % of total)

Data format (pmxt Parquet schema):
  timestamp_received: UTC timestamp
  timestamp_created_at: UTC timestamp
  market_id: hex condition ID (matches Polymarket condition_id)
  update_type: 'book_snapshot' | 'price_change'
  data: JSON string with order book or trade details

Usage:
    python orderflow.py                    # Fetch latest + compute signals
    python orderflow.py --hours-back 12    # Fetch last 12 hours
    python orderflow.py --list-markets     # Show which tracked markets have data
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta

import polars as pl
import requests

# ── Configuration ────────────────────────────────────────────────────────────

PMXT_BASE_URL = "https://r2.pmxt.dev"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "orderflow_cache")
SIGNALS_FILE = os.path.join(DATA_DIR, "orderflow_signals.json")

# How many hourly snapshots to fetch (each is 200-750MB, filtered in-memory)
DEFAULT_HOURS_BACK = 6

# Large order threshold: orders >= this size are flagged
LARGE_ORDER_SIZE = 5000  # contracts

# Imbalance threshold: ratio above this is considered significant
IMBALANCE_SIGNIFICANT = 1.5


# ── Contract Registry ────────────────────────────────────────────────────────

def load_tracked_markets():
    """Build a lookup of condition_id -> contract info from active + rolling contracts.

    Returns dict: {condition_id: {"key": str, "name": str, "token_ids": [yes, no]}}
    """
    markets = {}

    # Static contracts from active_contracts.json
    active_path = os.path.join(SCRIPT_DIR, "contracts", "active_contracts.json")
    if os.path.exists(active_path):
        with open(active_path) as f:
            data = json.load(f)
        for c in data.get("contracts", []):
            cid = c.get("condition_id", "")
            if not cid:
                continue
            markets[cid] = {
                "key": c.get("slug", ""),
                "name": c.get("question", c.get("contract_name", "")),
                "yes_token": c.get("yes_token_id", ""),
                "no_token": c.get("no_token_id", ""),
                "category": c.get("category", ""),
            }

    # Rolling contracts from rolling_today.json
    rolling_path = os.path.join(SCRIPT_DIR, "contracts", "rolling_today.json")
    if os.path.exists(rolling_path):
        with open(rolling_path) as f:
            data = json.load(f)
        for key, contract in data.get("contracts", {}).items():
            for m in contract.get("markets", []):
                cid = m.get("condition_id", "")
                if not cid:
                    continue
                token_ids = m.get("token_ids", [])
                markets[cid] = {
                    "key": key,
                    "name": m.get("question", contract.get("name", "")),
                    "yes_token": token_ids[0] if len(token_ids) > 0 else "",
                    "no_token": token_ids[1] if len(token_ids) > 1 else "",
                    "category": contract.get("category", ""),
                }

    # Legacy single market from constants.py
    try:
        from constants import MARKETS
        for key, mkt in MARKETS.items():
            cid = mkt.get("condition_id", "")
            if cid and cid not in markets:
                markets[cid] = {
                    "key": key,
                    "name": mkt.get("name", ""),
                    "yes_token": mkt.get("yes_token_id", ""),
                    "no_token": mkt.get("no_token_id", ""),
                    "category": "",
                }
    except ImportError:
        pass

    return markets


# ── Parquet Download ─────────────────────────────────────────────────────────

def build_snapshot_url(dt):
    """Build pmxt archive URL for a given datetime (rounded to hour)."""
    hour_str = dt.strftime("%Y-%m-%dT%H")
    return f"{PMXT_BASE_URL}/polymarket_orderbook_{hour_str}.parquet"


def check_snapshot_exists(url):
    """HEAD request to check if a snapshot file exists."""
    try:
        resp = requests.head(url, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def get_snapshot_hours(hours_back=DEFAULT_HOURS_BACK):
    """Return list of hourly timestamps to fetch, most recent first."""
    now = datetime.now(timezone.utc)
    # Start from 2 hours ago (latest snapshot might not be ready yet)
    start = now - timedelta(hours=2)
    hours = []
    for i in range(hours_back):
        dt = start - timedelta(hours=i)
        hours.append(dt.replace(minute=0, second=0, microsecond=0))
    return hours


def load_snapshot_filtered(url, condition_ids):
    """Load a pmxt Parquet snapshot, filtering to only tracked markets.

    Uses polars lazy scan with predicate pushdown so we never load the
    full 500MB file into memory — only rows matching our condition IDs
    are materialized.

    Returns polars DataFrame or None on failure.
    """
    try:
        df = (
            pl.scan_parquet(url)
            .filter(pl.col("market_id").is_in(condition_ids))
            .collect()
        )
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"    WARNING: Failed to read {url}: {e}")
        return None


# ── Signal Extraction ────────────────────────────────────────────────────────

def extract_book_signals(df, condition_id, yes_token):
    """Extract order book signals from book_snapshot events for one market.

    Returns dict with:
      - bid_depth_near: total bid size within 10% of mid
      - ask_depth_near: total ask size within 10% of mid
      - imbalance_ratio: bid_depth / ask_depth (>1 = bullish)
      - spread: best_ask - best_bid
      - spread_pct: spread as % of midpoint
      - num_snapshots: how many snapshots contributed
      - best_bid / best_ask: latest values
    """
    books = df.filter(
        (pl.col("market_id") == condition_id)
        & (pl.col("update_type") == "book_snapshot")
    )

    if len(books) == 0:
        return None

    # Process each snapshot, aggregate
    all_imbalances = []
    all_spreads = []
    latest_best_bid = None
    latest_best_ask = None
    latest_ts = 0

    for row in books.iter_rows(named=True):
        data = json.loads(row["data"])
        token_id = data.get("token_id", "")

        # We want the YES side order book for consistent directionality
        # If this snapshot is for the NO token, the bid/ask semantics are inverted
        is_yes_side = (token_id == yes_token)

        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        best_bid = float(data.get("best_bid", 0))
        best_ask = float(data.get("best_ask", 1))
        ts = data.get("timestamp", 0)

        if not is_yes_side:
            # Flip: NO-side bid = YES-side ask (at 1-price)
            continue  # Skip NO-side snapshots, YES-side is sufficient

        if ts > latest_ts:
            latest_ts = ts
            latest_best_bid = best_bid
            latest_best_ask = best_ask

        mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.5
        spread = best_ask - best_bid

        # Compute depth near midpoint (within 10% of mid, or 0.05 absolute)
        depth_range = max(mid * 0.1, 0.05)
        bid_depth = sum(
            float(b[1]) for b in bids_raw
            if abs(float(b[0]) - mid) <= depth_range
        )
        ask_depth = sum(
            float(a[1]) for a in asks_raw
            if abs(float(a[0]) - mid) <= depth_range
        )

        if ask_depth > 0:
            all_imbalances.append(bid_depth / ask_depth)
        all_spreads.append(spread)

    if not all_imbalances:
        return None

    avg_imbalance = sum(all_imbalances) / len(all_imbalances)
    avg_spread = sum(all_spreads) / len(all_spreads)
    mid = ((latest_best_bid or 0) + (latest_best_ask or 1)) / 2

    # Trend: compare first half vs second half of imbalances
    if len(all_imbalances) >= 4:
        half = len(all_imbalances) // 2
        early_imb = sum(all_imbalances[:half]) / half
        late_imb = sum(all_imbalances[half:]) / (len(all_imbalances) - half)
        imbalance_trend = late_imb - early_imb  # positive = becoming more bullish
    else:
        imbalance_trend = 0.0

    # Spread trend
    if len(all_spreads) >= 4:
        half = len(all_spreads) // 2
        early_spread = sum(all_spreads[:half]) / half
        late_spread = sum(all_spreads[half:]) / (len(all_spreads) - half)
        spread_trend = late_spread - early_spread  # positive = widening
    else:
        spread_trend = 0.0

    return {
        "imbalance_ratio": round(avg_imbalance, 3),
        "imbalance_trend": round(imbalance_trend, 3),
        "spread": round(avg_spread, 4),
        "spread_pct": round((avg_spread / mid * 100) if mid > 0 else 0, 2),
        "spread_trend": round(spread_trend, 4),
        "best_bid": latest_best_bid,
        "best_ask": latest_best_ask,
        "num_snapshots": len(all_imbalances),
    }


def extract_flow_signals(df, condition_id, yes_token):
    """Extract order flow signals from price_change events for one market.

    Returns dict with:
      - net_buy_volume: total BUY size - total SELL size (YES side)
      - total_volume: total size of all changes
      - buy_count / sell_count: number of buy vs sell events
      - large_orders: list of orders >= LARGE_ORDER_SIZE
      - flow_direction: 'bullish' / 'bearish' / 'neutral'
      - avg_trade_size: mean order size
    """
    changes = df.filter(
        (pl.col("market_id") == condition_id)
        & (pl.col("update_type") == "price_change")
    )

    if len(changes) == 0:
        return None

    buy_volume = 0
    sell_volume = 0
    buy_count = 0
    sell_count = 0
    large_orders = []
    all_sizes = []

    for row in changes.iter_rows(named=True):
        data = json.loads(row["data"])
        token_id = data.get("token_id", "")
        side = data.get("change_side", "")
        size = float(data.get("change_size", 0))

        if size == 0:
            continue

        # Normalize to YES-side directionality
        is_yes_side = (token_id == yes_token)

        # A BUY on YES side = bullish. A BUY on NO side = bearish.
        if is_yes_side:
            effective_side = side
        else:
            effective_side = "SELL" if side == "BUY" else "BUY"

        if effective_side == "BUY":
            buy_volume += size
            buy_count += 1
        else:
            sell_volume += size
            sell_count += 1

        all_sizes.append(size)

        if size >= LARGE_ORDER_SIZE:
            large_orders.append({
                "side": effective_side,
                "size": int(size),
                "price": data.get("change_price", "?"),
                "timestamp": data.get("timestamp", 0),
            })

    total_volume = buy_volume + sell_volume
    net = buy_volume - sell_volume

    if total_volume == 0:
        return None

    # Flow direction based on net volume
    net_ratio = net / total_volume
    if net_ratio > 0.1:
        flow_direction = "bullish"
    elif net_ratio < -0.1:
        flow_direction = "bearish"
    else:
        flow_direction = "neutral"

    return {
        "net_buy_volume": int(net),
        "total_volume": int(total_volume),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_pct": round(buy_volume / total_volume * 100, 1),
        "flow_direction": flow_direction,
        "avg_trade_size": round(sum(all_sizes) / len(all_sizes), 1) if all_sizes else 0,
        "large_orders": sorted(large_orders, key=lambda x: x["size"], reverse=True)[:10],
        "num_events": len(all_sizes),
    }


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_orderflow(hours_back=DEFAULT_HOURS_BACK):
    """Download recent pmxt snapshots, compute signals for all tracked markets.

    Returns dict of {contract_key: {book_signals, flow_signals, ...}}.
    """
    print(f"=== Order Flow Analysis (last {hours_back}h) ===\n")

    # Load tracked markets
    tracked = load_tracked_markets()
    if not tracked:
        print("  WARNING: No tracked markets found. Run rolling_contracts.py first.")
        return {}

    condition_ids = list(tracked.keys())
    print(f"  Tracking {len(condition_ids)} markets")

    # Determine which hourly snapshots to fetch
    snapshot_hours = get_snapshot_hours(hours_back)
    print(f"  Checking {len(snapshot_hours)} hourly snapshots...\n")

    # Load and merge data from each snapshot
    all_frames = []
    loaded_count = 0

    for dt in snapshot_hours:
        url = build_snapshot_url(dt)
        hour_str = dt.strftime("%Y-%m-%d %H:00")
        print(f"  [{hour_str}] ", end="", flush=True)

        # Check availability
        if not check_snapshot_exists(url):
            print("not available")
            continue

        # Load filtered data
        df = load_snapshot_filtered(url, condition_ids)
        if df is not None and len(df) > 0:
            all_frames.append(df)
            loaded_count += 1
            matched_markets = df["market_id"].n_unique()
            print(f"{len(df):,} events across {matched_markets} markets")
        else:
            print("no matching events")

    if not all_frames:
        print("\n  WARNING: No order flow data found for tracked markets.")
        return {}

    # Merge all frames
    combined = pl.concat(all_frames)
    print(f"\n  Total: {len(combined):,} events from {loaded_count} snapshots")

    # Compute signals per market
    print(f"\n  Computing signals...\n")
    signals = {}

    for cid, info in tracked.items():
        market_df = combined.filter(pl.col("market_id") == cid)
        if len(market_df) == 0:
            continue

        key = info["key"]
        yes_token = info["yes_token"]

        book = extract_book_signals(market_df, cid, yes_token)
        flow = extract_flow_signals(market_df, cid, yes_token)

        if book is None and flow is None:
            continue

        signals[key] = {
            "name": info["name"],
            "category": info["category"],
            "condition_id": cid,
            "total_events": len(market_df),
            "book": book,
            "flow": flow,
        }

        # Print summary
        direction = "?"
        if flow:
            direction = flow["flow_direction"]
        imb = book["imbalance_ratio"] if book else "?"
        vol = flow["total_volume"] if flow else 0
        n_large = len(flow["large_orders"]) if flow else 0
        print(f"  {key}: flow={direction}, imbalance={imb}, "
              f"volume={vol:,}, large_orders={n_large}")

    return signals


def save_signals(signals):
    """Save computed signals to data/orderflow_signals.json."""
    os.makedirs(DATA_DIR, exist_ok=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "pmxt Polymarket order book archive (archive.pmxt.dev)",
        "num_markets": len(signals),
        "signals": signals,
    }

    with open(SIGNALS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved signals for {len(signals)} markets to {SIGNALS_FILE}")


def load_signals():
    """Load previously saved orderflow signals. Returns dict or None."""
    if not os.path.exists(SIGNALS_FILE):
        return None
    try:
        with open(SIGNALS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ── Prompt Formatting ────────────────────────────────────────────────────────

def format_for_prompt(contract_key=None):
    """Format order flow signals as a text block for LLM prompts.

    If contract_key is provided, only includes signals for that contract.
    Returns a string block or empty string if no data.
    """
    data = load_signals()
    if not data or not data.get("signals"):
        return ""

    fetched_at = data.get("fetched_at", "unknown")
    signals = data["signals"]

    # Filter to specific contract if requested
    if contract_key and contract_key in signals:
        signals = {contract_key: signals[contract_key]}
    elif contract_key:
        return ""  # Requested contract not in signals

    lines = [f"ORDER FLOW SIGNALS (Polymarket order book, as of {fetched_at}):"]

    for key, sig in signals.items():
        book = sig.get("book")
        flow = sig.get("flow")

        if not book and not flow:
            continue

        lines.append(f"\n  {sig['name']}:")

        if book:
            # Imbalance interpretation
            imb = book["imbalance_ratio"]
            if imb > IMBALANCE_SIGNIFICANT:
                imb_label = f"BULLISH ({imb:.2f}:1 bid/ask)"
            elif imb < 1 / IMBALANCE_SIGNIFICANT:
                imb_label = f"BEARISH (1:{1/imb:.2f} bid/ask)"
            else:
                imb_label = f"balanced ({imb:.2f}:1)"

            # Imbalance trend
            trend = book["imbalance_trend"]
            if abs(trend) > 0.1:
                trend_dir = "rising" if trend > 0 else "falling"
                imb_label += f", {trend_dir}"

            lines.append(f"    Bid/Ask imbalance: {imb_label}")

            # Spread
            spread_trend = book["spread_trend"]
            if abs(spread_trend) > 0.002:
                spread_dir = "widening (uncertainty increasing)" if spread_trend > 0 else "narrowing (consensus forming)"
            else:
                spread_dir = "stable"
            lines.append(f"    Spread: {book['spread']:.4f} ({book['spread_pct']:.1f}%), {spread_dir}")

        if flow:
            # Flow summary
            lines.append(f"    Net flow: {flow['flow_direction']} "
                        f"(buy {flow['buy_pct']:.0f}%, {flow['total_volume']:,} contracts)")

            # Large orders
            large = flow.get("large_orders", [])
            if large:
                buys = sum(1 for o in large if o["side"] == "BUY")
                sells = len(large) - buys
                largest = large[0]
                lines.append(f"    Large orders (>={LARGE_ORDER_SIZE:,}): "
                           f"{len(large)} detected ({buys} buys, {sells} sells), "
                           f"largest: {largest['side']} {largest['size']:,} @ {largest['price']}")

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


def format_for_smart_money(contract_key):
    """Return raw signal dict for a specific contract, for smart_money.py consumption.

    Returns dict with book + flow signals, or None.
    """
    data = load_signals()
    if not data or not data.get("signals"):
        return None
    return data["signals"].get(contract_key)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch and analyze Polymarket order flow from pmxt archive")
    parser.add_argument("--hours-back", type=int, default=DEFAULT_HOURS_BACK,
                        help=f"How many hours of data to fetch (default: {DEFAULT_HOURS_BACK})")
    parser.add_argument("--list-markets", action="store_true",
                        help="Just list tracked markets and exit")
    args = parser.parse_args()

    if args.list_markets:
        tracked = load_tracked_markets()
        print(f"Tracked markets ({len(tracked)}):")
        for cid, info in tracked.items():
            print(f"  {info['key']}: {info['name'][:60]}")
            print(f"    condition_id: {cid[:40]}...")
        return

    signals = analyze_orderflow(hours_back=args.hours_back)

    if signals:
        save_signals(signals)

        # Show prompt block
        print(f"\n--- Prompt block ---")
        print(format_for_prompt())
        print(f"--- End prompt block ---")
    else:
        print("\nNo signals computed.")


if __name__ == "__main__":
    main()
