"""
Oracle Lab — gdelt.py
Fetches structured news signals from GDELT DOC 2.0 API.

GDELT monitors news worldwide, updated every 15 minutes. We query it for
article counts and top headlines relevant to each contract category.
This gives the LLM forecaster systematic news coverage that Perplexity
summaries might miss.

API: https://api.gdeltproject.org/api/v2/doc/doc
- Free, no API key needed
- Rate limit: 1 request per 5 seconds
- Covers 65 languages, we filter to English

Usage: python gdelt.py [--timespan 2d]
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Queries mapped to contract categories/slugs.
# Each query targets keywords the LLM needs to reason about that contract.
# "relevant_to" links queries to contract slugs in active_contracts.json.
QUERIES = [
    # GEOPOLITICS — Iran / Strait of Hormuz cluster
    {
        "label": "Iran nuclear deal / diplomacy",
        "query": "(iran nuclear deal OR iran negotiations OR iran diplomacy OR iran sanctions) sourcelang:english",
        "relevant_to": [
            "us-iran-nuclear-deal-before-2027",
            "us-iran-nuclear-deal-by-april-30",
        ],
    },
    {
        "label": "Strait of Hormuz / shipping",
        "query": "(strait of hormuz OR hormuz shipping OR hormuz blockade OR iran oil tanker) sourcelang:english",
        "relevant_to": [
            "strait-of-hormuz-traffic-returns-to-normal-by-april-30",
            "will-iran-close-the-strait-of-hormuz-by-2027",
        ],
    },
    {
        "label": "World leaders / regime change",
        "query": "(leader resign OR president impeach OR prime minister ousted OR coup OR regime change) sourcelang:english",
        "relevant_to": ["next-leader-out-of-power-before-2027-795"],
    },
    {
        "label": "SAVE Act / election law",
        "query": "(SAVE Act OR election integrity bill OR voter citizenship) sourcelang:english",
        "relevant_to": ["save-act-signed-into-law-in-2026"],
    },
    # ECONOMICS
    {
        "label": "US recession signals",
        "query": "(US recession OR economic downturn OR GDP contraction OR unemployment rising) sourcelang:english",
        "relevant_to": ["us-recession-by-end-of-2026"],
    },
    {
        "label": "Federal Reserve / interest rates",
        "query": "(federal reserve rate OR fed funds rate OR FOMC decision OR jerome powell rate) sourcelang:english",
        "relevant_to": [
            "fed-decision-in-april",
            "jerome-powell-out-as-fed-chair-by",
            "who-will-be-confirmed-as-fed-chair",
        ],
    },
    {
        "label": "US inflation / CPI",
        "query": "(CPI inflation OR consumer price index OR inflation rate US) sourcelang:english",
        "relevant_to": ["march-inflation-us-annual"],
    },
    {
        "label": "Bitcoin / crypto markets",
        "query": "(bitcoin price OR BTC rally OR crypto market OR bitcoin ETF) sourcelang:english",
        "relevant_to": ["what-price-will-bitcoin-hit-before-2027"],
    },
    {
        "label": "Gas prices / oil prices",
        "query": "(gas prices OR gasoline prices OR oil prices OR crude oil WTI) sourcelang:english",
        "relevant_to": ["will-gas-hit-by-end-of-march"],
    },
    # POLITICS
    {
        "label": "2026 midterm elections",
        "query": "(2026 midterm OR midterm election OR house race 2026 OR senate race 2026) sourcelang:english",
        "relevant_to": [
            "which-party-will-win-the-house-in-2026",
            "balance-of-power-2026-midterms",
        ],
    },
    # SPORTS
    {
        "label": "FIFA World Cup 2026",
        "query": "(FIFA World Cup 2026 OR world cup qualifier OR world cup odds) sourcelang:english",
        "relevant_to": ["2026-fifa-world-cup-winner-595"],
    },
    {
        "label": "NBA Finals 2026",
        "query": "(NBA playoffs 2026 OR NBA finals OR NBA championship odds) sourcelang:english",
        "relevant_to": ["2026-nba-champion"],
    },
    # TECH / AI
    {
        "label": "AI model rankings / benchmarks",
        "query": "(AI model benchmark OR best AI model OR Claude OR GPT OR Gemini benchmark) sourcelang:english",
        "relevant_to": [
            "which-company-has-the-best-ai-model-end-of-march-751",
            "which-company-will-have-the-best-ai-model-for-coding-on-march-31",
        ],
    },
    # ENTERTAINMENT
    {
        "label": "GTA 6 release",
        "query": "(GTA 6 OR grand theft auto 6 OR GTA VI release OR rockstar games) sourcelang:english",
        "relevant_to": ["gta-6-launch-postponed-again"],
    },
]

# How many seconds to wait between requests (GDELT rate limit: 1 per 5s, use 8 for safety)
REQUEST_DELAY = 8


def fetch_articles(query, timespan="2d", max_records=10):
    """Fetch recent articles from GDELT DOC API.

    Returns list of article dicts with: title, domain, seendate, url, sourcecountry.
    Returns empty list on failure.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "json",
        "timespan": timespan,
    }

    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
        if resp.status_code == 429:
            # Rate limited — wait and retry once
            time.sleep(12)
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)

        resp.raise_for_status()

        # Handle BOM and empty responses
        text = resp.text.replace("\ufeff", "").strip()
        if not text:
            return []

        data = json.loads(text)
        return data.get("articles", [])

    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def fetch_article_count(query, timespan="2d"):
    """Fetch article count from GDELT for a query over a timespan.

    Uses TimelineVolInfo CSV mode and sums the volume values.
    Returns (total_articles, num_hours_with_coverage).
    """
    params = {
        "query": query,
        "mode": "TimelineVolInfo",
        "format": "csv",
        "timespan": timespan,
    }

    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(10)
            resp = requests.get(GDELT_DOC_API, params=params, timeout=15)

        resp.raise_for_status()
        text = resp.text.replace("\ufeff", "").strip()
        if not text:
            return 0, 0

        lines = text.split("\n")
        if len(lines) < 2:
            return 0, 0

        # Parse CSV: Date, Series, Value, ...
        total_vol = 0.0
        hours_with_coverage = 0
        for line in lines[1:]:  # skip header
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    vol = float(parts[2])
                    total_vol += vol
                    if vol > 0:
                        hours_with_coverage += 1
                except ValueError:
                    pass

        return total_vol, hours_with_coverage

    except Exception as e:
        print(f"    ERROR (count): {e}")
        return 0, 0


def run_queries(timespan="2d"):
    """Run all GDELT queries and collect results.

    For each query category:
    - Fetch top 10 article headlines (ArtList)
    - Wait for rate limit between requests

    Returns dict mapping query labels to results.
    """
    results = {}

    for i, q in enumerate(QUERIES):
        label = q["label"]
        query = q["query"]
        print(f"  [{i+1}/{len(QUERIES)}] {label}...")

        articles = fetch_articles(query, timespan=timespan, max_records=10)

        # Extract just what we need for the LLM
        headlines = [
            {
                "title": art.get("title", "").strip(),
                "domain": art.get("domain", ""),
                "date": art.get("seendate", ""),
                "country": art.get("sourcecountry", ""),
            }
            for art in articles
        ]
        article_count = len(headlines)
        print(f"    {article_count} articles found")

        results[label] = {
            "query": query.replace(" sourcelang:english", ""),
            "article_count": article_count,
            "relevant_to": q["relevant_to"],
            "headlines": headlines,
        }

        # Respect rate limit
        if i < len(QUERIES) - 1:
            time.sleep(REQUEST_DELAY)

    return results


def save_results(results, timespan):
    """Save GDELT results to data/gdelt_context.json."""
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timespan": timespan,
        "source": "GDELT DOC 2.0 API (global news monitoring)",
        "num_queries": len(results),
        "queries": results,
    }

    out_path = os.path.join(output_dir, "gdelt_context.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} query results to {out_path}")
    return out_path


def load_context():
    """Load previously saved GDELT context. Returns dict or None."""
    path = os.path.join("data", "gdelt_context.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_for_prompt(contract_slug=None):
    """Format GDELT context as a text block for LLM prompts.

    If contract_slug is provided, only includes queries relevant to that contract.
    Otherwise includes all queries as a general news summary.

    Returns a string block or empty string if no data.
    """
    data = load_context()
    if not data or not data.get("queries"):
        return ""

    fetched_at = data.get("fetched_at", "unknown")
    timespan = data.get("timespan", "2d")

    lines = [f"NEWS SIGNALS (GDELT global news monitoring, last {timespan}, as of {fetched_at}):"]

    for label, info in data["queries"].items():
        # Filter by contract slug if specified
        if contract_slug:
            relevant = info.get("relevant_to", [])
            if contract_slug not in relevant:
                continue

        count = info.get("article_count", 0)
        headlines = info.get("headlines", [])

        lines.append(f"\n  {label} ({count} recent articles):")

        # Show top 5 headlines
        for h in headlines[:5]:
            title = h.get("title", "")
            domain = h.get("domain", "")
            if title:
                lines.append(f"    - [{domain}] {title}")

    # If filtering produced no matches, return empty
    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


def format_general_summary():
    """Format a compact general news summary across all queries.

    This is used when no specific contract slug is available.
    Shows article counts per category as a quick signal.
    """
    data = load_context()
    if not data or not data.get("queries"):
        return ""

    fetched_at = data.get("fetched_at", "unknown")
    timespan = data.get("timespan", "2d")

    lines = [f"NEWS VOLUME (GDELT, last {timespan}, as of {fetched_at}):"]

    for label, info in data["queries"].items():
        count = info.get("article_count", 0)
        # Show top headline as a one-liner
        headlines = info.get("headlines", [])
        top = ""
        if headlines:
            top = f' — top: "{headlines[0].get("title", "")[:70]}"'
        lines.append(f"  {label}: {count} articles{top}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Fetch GDELT news context for Oracle Lab contracts")
    parser.add_argument("--timespan", type=str, default="2d", help="GDELT timespan (e.g. 1d, 2d, 7d)")
    args = parser.parse_args()

    print(f"=== GDELT News Fetch (timespan={args.timespan}) ===\n")
    print(f"Running {len(QUERIES)} queries with {REQUEST_DELAY}s delay between each...")
    print(f"Estimated time: ~{len(QUERIES) * REQUEST_DELAY}s\n")

    results = run_queries(timespan=args.timespan)

    if not results:
        print("\nERROR: No results fetched")
        return

    save_results(results, args.timespan)

    # Show summary
    total_articles = sum(r["article_count"] for r in results.values())
    print(f"\nTotal articles across all queries: {total_articles}")

    # Show the general summary block
    print(f"\n--- General summary block ---")
    print(format_general_summary())
    print(f"--- End summary block ---")


if __name__ == "__main__":
    main()
