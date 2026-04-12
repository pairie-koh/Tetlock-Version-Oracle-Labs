"""
Oracle Lab -- weather.py
Fetches weather forecasts from the National Weather Service (NWS) API.

NWS API is free, no key needed, just requires a User-Agent header.
Provides today's forecast high/low, conditions, and hourly breakdown
for NYC and Miami to support temperature prediction contracts.

Usage: python weather.py
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

NWS_USER_AGENT = "OracleLab/1.0 (hello.pairie@gmail.com)"
NWS_HEADERS = {
    "User-Agent": NWS_USER_AGENT,
    "Accept": "application/geo+json",
}

CITIES = {
    "nyc": {
        "label": "New York City",
        "forecast_url": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
        "hourly_url": "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly",
        "relevant_to": ["nyc_temp"],
    },
    "miami": {
        "label": "Miami",
        "forecast_url": "https://api.weather.gov/gridpoints/MFL/75,67/forecast",
        "hourly_url": "https://api.weather.gov/gridpoints/MFL/75,67/forecast/hourly",
        "relevant_to": ["miami_temp"],
    },
}


def _fetch_nws(city_key, url_key):
    """Fetch forecast periods from NWS for a city. Returns periods list or None."""
    url = CITIES[city_key][url_key]
    try:
        resp = requests.get(url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json().get("properties", {}).get("periods", [])
    except Exception as e:
        label = "forecast" if "hourly" not in url_key else "hourly"
        print(f"  WARNING: NWS {label} failed for {city_key}: {e}")
        return None


def fetch_forecast(city_key):
    """Fetch daily forecast from NWS for a city. Returns periods list or None."""
    return _fetch_nws(city_key, "forecast_url")


def fetch_hourly(city_key):
    """Fetch hourly forecast from NWS for a city. Returns periods list or None."""
    return _fetch_nws(city_key, "hourly_url")


def parse_today_forecast(periods):
    """Extract today's daytime and tonight forecasts from daily periods.

    NWS daily forecast alternates: daytime, nighttime, daytime, nighttime...
    The first period with isDaytime=True is today's daytime forecast.
    """
    if not periods:
        return None

    today = None
    tonight = None
    for p in periods[:4]:  # Only look at first 2 days
        if p.get("isDaytime") and today is None:
            today = p
        elif not p.get("isDaytime") and tonight is None and today is not None:
            tonight = p

    if not today:
        return None

    result = {
        "high": today.get("temperature"),
        "high_unit": today.get("temperatureUnit", "F"),
        "forecast": today.get("shortForecast", ""),
        "detail": today.get("detailedForecast", ""),
        "wind": today.get("windSpeed", ""),
        "wind_dir": today.get("windDirection", ""),
    }

    if tonight:
        result["low"] = tonight.get("temperature")
        result["tonight_forecast"] = tonight.get("shortForecast", "")

    return result


def parse_today_hourly(hourly_periods):
    """Extract today's hourly temperatures from hourly forecast.

    Filters to only hours for today's date (UTC date of first period).
    Returns list of {hour, temp, short_forecast} dicts.
    """
    if not hourly_periods:
        return []

    # Determine today's date from the first period
    first_start = hourly_periods[0].get("startTime", "")
    today_date = first_start[:10]  # "2026-03-30"

    hourly = []
    for p in hourly_periods:
        start = p.get("startTime", "")
        if not start.startswith(today_date):
            # Past today — stop
            if hourly:
                break
            continue

        hourly.append({
            "hour": start,
            "temp": p.get("temperature"),
            "unit": p.get("temperatureUnit", "F"),
            "short": p.get("shortForecast", ""),
        })

    return hourly


def fetch_all():
    """Fetch forecasts for all configured cities. Returns dict."""
    results = {}

    for city_key, city_info in CITIES.items():
        label = city_info["label"]
        print(f"  Fetching {label}...")

        forecast_periods = fetch_forecast(city_key)
        time.sleep(0.5)  # Be polite to NWS API
        hourly_periods = fetch_hourly(city_key)
        time.sleep(0.5)

        if forecast_periods is None and hourly_periods is None:
            print(f"  WARNING: No data for {label}")
            continue

        city_result = {
            "label": label,
            "relevant_to": city_info["relevant_to"],
        }

        # Parse daily forecast
        today = parse_today_forecast(forecast_periods)
        if today:
            city_result["today_high"] = today["high"]
            city_result["today_high_unit"] = today["high_unit"]
            city_result["today_forecast"] = today["forecast"]
            city_result["today_detail"] = today["detail"]
            city_result["wind"] = today.get("wind", "")
            city_result["wind_direction"] = today.get("wind_dir", "")
            if "low" in today:
                city_result["tonight_low"] = today["low"]
                city_result["tonight_forecast"] = today.get("tonight_forecast", "")
            print(f"    Forecast high: {today['high']}°{today['high_unit']} — {today['forecast']}")

        # Parse hourly
        hourly = parse_today_hourly(hourly_periods)
        if hourly:
            city_result["hourly"] = hourly
            # Find peak temperature hour
            peak = max(hourly, key=lambda h: h["temp"])
            city_result["peak_temp"] = peak["temp"]
            city_result["peak_hour"] = peak["hour"]
            print(f"    Hourly: {len(hourly)} hours, peak {peak['temp']}°F")

        results[city_key] = city_result

    return results


def save_forecasts(forecasts):
    """Save fetched forecasts to data/weather_forecast.json."""
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "National Weather Service API (weather.gov)",
        "cities": forecasts,
    }

    out_path = os.path.join(output_dir, "weather_forecast.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved forecasts for {len(forecasts)} cities to {out_path}")
    return out_path


def load_forecasts():
    """Load previously saved weather forecasts. Returns dict or None."""
    path = os.path.join("data", "weather_forecast.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _format_hour_label(iso_str):
    """Convert ISO timestamp to short hour label like '8am', '1pm'."""
    try:
        # Parse the hour from the ISO string (e.g. "2026-03-30T08:00:00-04:00")
        hour_str = iso_str[11:13]
        hour = int(hour_str)
        if hour == 0:
            return "12am"
        elif hour < 12:
            return f"{hour}am"
        elif hour == 12:
            return "12pm"
        else:
            return f"{hour - 12}pm"
    except (ValueError, IndexError):
        return iso_str[11:16]


def format_for_prompt(city=None):
    """Format weather forecast as a text block for LLM prompts.

    Args:
        city: Optional city key ("nyc" or "miami"). If None, returns all cities.

    Returns prompt-ready text block, or empty string if no data.
    """
    data = load_forecasts()
    if not data or not data.get("cities"):
        return ""

    cities = data["cities"]
    fetched_at = data.get("fetched_at", "unknown")

    # Filter to requested city
    if city and city in cities:
        city_subset = {city: cities[city]}
    elif city:
        return ""  # Requested city not available
    else:
        city_subset = cities

    lines = [f"WEATHER FORECAST (National Weather Service, as of {fetched_at}):"]

    for key, info in city_subset.items():
        label = info.get("label", key.upper())
        lines.append(f"  {label}:")

        high = info.get("today_high")
        unit = info.get("today_high_unit", "F")
        if high is not None:
            lines.append(f"    Today's forecast high: {high} deg {unit}")

        low = info.get("tonight_low")
        if low is not None:
            lines.append(f"    Tonight's forecast low: {low} deg {unit}")

        forecast = info.get("today_forecast")
        if forecast:
            lines.append(f"    Conditions: {forecast}")

        detail = info.get("today_detail")
        if detail and detail != forecast:
            lines.append(f"    Detail: {detail}")

        # Hourly progression (compact format)
        hourly = info.get("hourly", [])
        if hourly:
            hourly_parts = [
                f"{h['temp']}F({_format_hour_label(h['hour'])})"
                for h in hourly
            ]
            lines.append(f"    Hourly temps today: {', '.join(hourly_parts)}")

        # Peak
        peak_temp = info.get("peak_temp")
        peak_hour = info.get("peak_hour")
        if peak_temp is not None and peak_hour:
            lines.append(f"    Peak expected: {peak_temp} deg F around {_format_hour_label(peak_hour)}")

    return "\n".join(lines)


def main():
    print("=== NWS Weather Forecast Fetch ===\n")

    forecasts = fetch_all()

    if not forecasts:
        print("\nERROR: No forecasts fetched")
        return

    save_forecasts(forecasts)

    # Show the prompt blocks
    print(f"\n--- Full prompt block ---")
    print(format_for_prompt())
    print(f"--- End prompt block ---")

    for city_key in CITIES:
        if city_key in forecasts:
            print(f"\n--- {city_key} prompt block ---")
            print(format_for_prompt(city=city_key))
            print(f"--- End {city_key} ---")


if __name__ == "__main__":
    main()
