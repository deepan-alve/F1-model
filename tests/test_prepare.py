"""Tests for the autoresearch prepare.py (immutable scorer)."""

import numpy as np
import pandas as pd
import pytest

# Import the evaluate function directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "autoresearch"))

from prepare import evaluate


class TestEvaluate:
    def test_perfect_predictions(self):
        """Perfect predictions give Spearman of 1.0."""
        predictions = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "PIA", "HAM", "LEC"],
            "Year": [2024] * 5,
            "RoundNumber": [1] * 5,
            "PredictedPosition": [1, 2, 3, 4, 5],
        })
        actuals = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "PIA", "HAM", "LEC"],
            "Year": [2024] * 5,
            "RoundNumber": [1] * 5,
            "FinishPosition": [1, 2, 3, 4, 5],
        })

        score = evaluate(predictions, actuals)
        assert abs(score - 1.0) < 0.01

    def test_empty_predictions(self):
        """Empty predictions return 0.0."""
        predictions = pd.DataFrame(columns=["Abbreviation", "Year", "RoundNumber", "PredictedPosition"])
        actuals = pd.DataFrame(columns=["Abbreviation", "Year", "RoundNumber", "FinishPosition"])

        score = evaluate(predictions, actuals)
        assert score == 0.0

    def test_averages_across_races(self):
        """Score is averaged across multiple races."""
        # Race 1: perfect prediction
        # Race 2: inverse prediction
        predictions = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "PIA", "VER", "NOR", "PIA"],
            "Year": [2024] * 6,
            "RoundNumber": [1, 1, 1, 2, 2, 2],
            "PredictedPosition": [1, 2, 3, 1, 2, 3],
        })
        actuals = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "PIA", "VER", "NOR", "PIA"],
            "Year": [2024] * 6,
            "RoundNumber": [1, 1, 1, 2, 2, 2],
            "FinishPosition": [1, 2, 3, 3, 2, 1],  # Race 2 is reversed
        })

        score = evaluate(predictions, actuals)
        # Average of 1.0 and -1.0 = 0.0
        assert abs(score) < 0.1
