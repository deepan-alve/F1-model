"""Tests for the betting odds module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.odds import (
    _name_to_abbreviation,
    _parse_odds_response,
    compute_market_delta,
    fetch_race_odds,
)


class TestNameMapping:
    def test_known_driver(self):
        assert _name_to_abbreviation("Max Verstappen") == "VER"
        assert _name_to_abbreviation("Lando Norris") == "NOR"

    def test_unknown_driver(self):
        assert _name_to_abbreviation("John Smith") is None

    def test_case_insensitive(self):
        assert _name_to_abbreviation("MAX VERSTAPPEN") == "VER"


class TestFetchOdds:
    @patch("src.odds.requests.get")
    def test_returns_none_on_rate_limit(self, mock_get):
        """Returns None on 429 rate limit."""
        mock_get.return_value = MagicMock(status_code=429)
        result = fetch_race_odds(api_key="test_key")
        assert result is None

    @patch("src.odds.requests.get")
    def test_returns_none_on_timeout(self, mock_get):
        """Returns None on request timeout."""
        import requests
        mock_get.side_effect = requests.Timeout("Connection timed out")
        result = fetch_race_odds(api_key="test_key")
        assert result is None

    def test_returns_none_without_api_key(self):
        """Returns None when no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            result = fetch_race_odds(api_key=None)
            assert result is None


class TestParseOddsResponse:
    def test_parses_valid_response(self):
        data = [{
            "bookmakers": [{
                "markets": [{
                    "key": "outrights",
                    "outcomes": [
                        {"name": "Max Verstappen", "price": 2.5},
                        {"name": "Lando Norris", "price": 4.0},
                        {"name": "Charles Leclerc", "price": 6.0},
                    ]
                }]
            }]
        }]

        result = _parse_odds_response(data)
        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]["market_position"] == 1
        # Verstappen has lowest odds = highest probability
        assert result.iloc[0]["driver_name"] == "Max Verstappen"

    def test_returns_none_for_empty(self):
        assert _parse_odds_response([]) is None
        assert _parse_odds_response(None) is None


class TestMarketDelta:
    def test_computes_delta(self):
        predictions = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "LEC"],
            "PredictedPosition": [1, 2, 3],
        })
        odds = pd.DataFrame({
            "abbreviation": ["VER", "NOR", "LEC"],
            "market_position": [1, 3, 2],
        })

        result = compute_market_delta(predictions, odds)
        assert "MarketDelta" in result.columns
        # NOR: market says P3, model says P2 -> delta = 3-2 = +1
        nor_delta = result[result["Abbreviation"] == "NOR"]["MarketDelta"].values[0]
        assert nor_delta == 1

    def test_handles_none_odds(self):
        predictions = pd.DataFrame({
            "Abbreviation": ["VER"],
            "PredictedPosition": [1],
        })
        result = compute_market_delta(predictions, None)
        assert pd.isna(result.iloc[0]["MarketDelta"])
