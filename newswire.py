"""
Tetlock Oracle Labs — newswire.py
Two-stage news pipeline (same as Oracle Labs v2):
  1. Perplexity: broad news sweeps per market
  2. Haiku: raw text → structured JSON facts

The output feeds into the BPT evidence evaluator, not directly into forecasting.
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

from constants import (
    MARKETS, OPENROUTER_API_URL, PERPLEXITY_API_URL,
    PERPLEXITY_MODEL, HAIKU_MODEL, QUERY_TEMPLATES,
    BRIEFINGS_DIR,
)


def call_perplexity(prompt):
    """Call Perplexity API. Returns raw text."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")

    resp = requests.post(
        PERPLEXITY_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": PERPLEXITY_MODEL, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt, model, max_tokens=4096):
    """Call OpenRouter API. Returns raw text."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resp = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tetlock-oracle-labs",
            "X-Title": "Tetlock Oracle Labs",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(raw):
    """Extract JSON from an LLM response, stripping markdown fences."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    return json.loads(cleaned)


def gather_raw_news():
    """Run Perplexity queries. Returns [(market_key, focus, raw_text), ...]."""
    results = []
    total = sum(len(QUERY_TEMPLATES[k]) for k in MARKETS if k in QUERY_TEMPLATES)
    n = 0

    for market_key in MARKETS:
        if market_key not in QUERY_TEMPLATES:
            continue
        for focus, prompt in QUERY_TEMPLATES[market_key]:
            n += 1
            print(f"  [{n}/{total}] {market_key} / {focus}...")
            try:
                raw = call_perplexity(prompt)
                results.append((market_key, focus, raw))
                print(f"    Got {len(raw)} chars")
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append((market_key, focus, f"[Error: {e}]"))

    return results


def normalize_to_facts(raw_results):
    """Use Haiku to extract structured facts from raw news."""
    raw_block = ""
    for market_key, focus, text in raw_results:
        if text.startswith("[Error"):
            continue
        raw_block += f"\n=== Market: {market_key} | Focus: {focus} ===\n{text}\n"

    if not raw_block.strip():
        return []

    prompt = f"""Extract structured news facts from the following raw news reports.

For each distinct factual claim, create a JSON object with:
- "claim": A single, clear factual statement (1-2 sentences)
- "source": The news source that reported it
- "market": The market key (one of: {json.dumps(list(MARKETS.keys()))})
- "time": ISO8601 timestamp (best estimate)
- "confidence": "high", "medium", or "low"

Rules:
- Extract 10-30 facts
- Only verifiable factual claims — no analysis or speculation
- Deduplicate across sources
- If no real news, return an empty array

Return ONLY a JSON array. No markdown, no explanation.

Raw reports:
{raw_block}"""

    raw_response = call_openrouter(prompt, HAIKU_MODEL)

    try:
        facts = parse_json_response(raw_response)
        if not isinstance(facts, list):
            return []
        print(f"  Extracted {len(facts)} facts")
        return facts
    except json.JSONDecodeError as e:
        print(f"  WARN: Failed to parse facts JSON: {e}")
        return []


def run_newswire():
    """Full pipeline: gather news → normalize to facts → save briefing."""
    print("=== Newswire Pipeline ===")

    print("[1/2] Gathering raw news...")
    raw_results = gather_raw_news()

    useful = [r for r in raw_results if not r[2].startswith("[Error")]
    if not useful:
        print("ERROR: All news calls failed.")
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "facts": [], "raw_count": 0}

    print(f"\n[2/2] Normalizing {len(useful)} reports with Haiku...")
    facts = normalize_to_facts(raw_results)

    # Save briefing
    os.makedirs(BRIEFINGS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    briefing = {
        "timestamp": timestamp,
        "facts": facts,
        "raw_count": len(useful),
    }

    briefing_path = os.path.join(BRIEFINGS_DIR, f"{timestamp.replace(':', '').replace('-', '')}.json")
    with open(briefing_path, "w") as f:
        json.dump(briefing, f, indent=2)

    with open(os.path.join(BRIEFINGS_DIR, "latest.json"), "w") as f:
        json.dump(briefing, f, indent=2)

    print(f"\n=== Newswire complete: {len(facts)} facts ===")
    return briefing


if __name__ == "__main__":
    run_newswire()
