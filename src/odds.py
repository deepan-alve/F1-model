"""
Betting odds fetching from The Odds API.

Fetches pre-race winner odds for F1 races. Used for market comparison
in predict.py output ("model disagrees with market by +3 positions").

Free tier: 500 requests/month (~20 per race for 24 races).
For backtesting: use historical odds datasets from Kaggle.

Graceful degradation: returns None when API unavailable.
"""

import os
from typing import Optional

import pandas as pd
import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "motorsport_formula_one"


def get_api_key() -> Optional[str]:
    """Get The Odds API key from environment variable."""
    return os.environ.get("ODDS_API_KEY")


def fetch_race_odds(api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch current F1 race winner odds.

    Returns a DataFrame with columns:
    - driver_name: Full driver name
    - abbreviation: 3-letter code (best-effort mapping)
    - implied_probability: Probability implied by odds (0-1)
    - market_position: Implied finishing position based on odds ranking

    Returns None if API is unavailable or key is missing.
    """
    if api_key is None:
        api_key = get_api_key()

    if not api_key:
        return None

    try:
        response = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
            params={
                "apiKey": api_key,
                "regions": "eu",
                "markets": "outrights",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )

        if response.status_code == 429:
            print("Warning: Odds API rate limit reached.")
            return None

        if response.status_code != 200:
            print(f"Warning: Odds API returned status {response.status_code}")
            return None

        data = response.json()
        if not data:
            return None

        return _parse_odds_response(data)

    except requests.RequestException as e:
        print(f"Warning: Could not fetch odds: {e}")
        return None


def _parse_odds_response(data: list[dict]) -> Optional[pd.DataFrame]:
    """Parse The Odds API response into a clean DataFrame."""
    if not data:
        return None

    # Aggregate odds across bookmakers
    driver_odds: dict[str, list[float]] = {}

    for event in data:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome["name"]
                    price = outcome.get("price", 0)
                    if price > 0:
                        driver_odds.setdefault(name, []).append(price)

    if not driver_odds:
        return None

    rows = []
    for name, prices in driver_odds.items():
        avg_price = sum(prices) / len(prices)
        implied_prob = 1.0 / avg_price if avg_price > 0 else 0
        rows.append({
            "driver_name": name,
            "implied_probability": implied_prob,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("implied_probability", ascending=False).reset_index(drop=True)
    df["market_position"] = range(1, len(df) + 1)

    # Best-effort abbreviation mapping
    df["abbreviation"] = df["driver_name"].map(_name_to_abbreviation)

    return df


# Common name to abbreviation mapping for current F1 grid
_DRIVER_NAME_MAP = {
    "max verstappen": "VER",
    "verstappen": "VER",
    "lando norris": "NOR",
    "norris": "NOR",
    "oscar piastri": "PIA",
    "piastri": "PIA",
    "charles leclerc": "LEC",
    "leclerc": "LEC",
    "lewis hamilton": "HAM",
    "hamilton": "HAM",
    "carlos sainz": "SAI",
    "sainz": "SAI",
    "george russell": "RUS",
    "russell": "RUS",
    "fernando alonso": "ALO",
    "alonso": "ALO",
    "lance stroll": "STR",
    "stroll": "STR",
    "pierre gasly": "GAS",
    "gasly": "GAS",
    "esteban ocon": "OCO",
    "ocon": "OCO",
    "alex albon": "ALB",
    "albon": "ALB",
    "yuki tsunoda": "TSU",
    "tsunoda": "TSU",
    "daniel ricciardo": "RIC",
    "ricciardo": "RIC",
    "valtteri bottas": "BOT",
    "bottas": "BOT",
    "kevin magnussen": "MAG",
    "magnussen": "MAG",
    "nico hulkenberg": "HUL",
    "hulkenberg": "HUL",
    "zhou guanyu": "ZHO",
    "guanyu zhou": "ZHO",
    "logan sargeant": "SAR",
    "sargeant": "SAR",
    "oliver bearman": "BEA",
    "bearman": "BEA",
    "liam lawson": "LAW",
    "lawson": "LAW",
    "jack doohan": "DOO",
    "doohan": "DOO",
    "isack hadjar": "HAD",
    "hadjar": "HAD",
    "andrea kimi antonelli": "ANT",
    "antonelli": "ANT",
    "gabriel bortoleto": "BOR",
    "bortoleto": "BOR",
}


def _name_to_abbreviation(name: str) -> Optional[str]:
    """Map a driver's full name to their 3-letter abbreviation."""
    lower = name.lower().strip()
    return _DRIVER_NAME_MAP.get(lower)


def compute_market_delta(
    predictions: pd.DataFrame,
    odds: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Add market comparison to predictions.

    Computes the difference between model's predicted position and
    the market-implied position from betting odds.

    Positive MarketDelta = model thinks driver will finish HIGHER than market expects.

    Args:
        predictions: Model predictions with Abbreviation and PredictedPosition
        odds: Betting odds DataFrame (or None if unavailable)

    Returns:
        predictions with MarketDelta column added
    """
    predictions = predictions.copy()

    if odds is None or odds.empty:
        predictions["MarketDelta"] = float("nan")
        predictions["MarketPosition"] = float("nan")
        return predictions

    # Merge on abbreviation
    odds_map = odds.set_index("abbreviation")["market_position"].to_dict()

    predictions["MarketPosition"] = predictions["Abbreviation"].map(odds_map)
    predictions["MarketDelta"] = predictions["MarketPosition"] - predictions["PredictedPosition"]

    return predictions
