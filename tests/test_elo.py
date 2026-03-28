"""Tests for the adaptive Elo rating system."""

import numpy as np
import pandas as pd
import pytest

from src.elo import (
    DEFAULT_RATING,
    compute_elo_ratings,
    expected_score,
    get_k_factor,
    update_ratings_for_race,
)


class TestKFactor:
    def test_standard_year(self):
        """Non-regulation years use standard K."""
        assert get_k_factor(2023, 5) == 16

    def test_regulation_year_early(self):
        """First 3 races of regulation year use high K."""
        assert get_k_factor(2026, 1) == 48
        assert get_k_factor(2026, 3) == 48

    def test_regulation_year_mid(self):
        """Races 4-8 of regulation year use medium K."""
        assert get_k_factor(2026, 4) == 32
        assert get_k_factor(2026, 8) == 32

    def test_regulation_year_late(self):
        """Races 9+ of regulation year use standard K."""
        assert get_k_factor(2026, 9) == 16
        assert get_k_factor(2026, 20) == 16

    def test_sprint_half_weight(self):
        """Sprint races get half K-factor."""
        assert get_k_factor(2024, 5, is_sprint=True) == 8
        assert get_k_factor(2026, 1, is_sprint=True) == 24  # 48 * 0.5


class TestExpectedScore:
    def test_equal_ratings(self):
        """Equal ratings give 50% expected score."""
        assert abs(expected_score(1500, 1500) - 0.5) < 0.001

    def test_higher_rated_favored(self):
        """Higher rated player has > 50% expected score."""
        assert expected_score(1600, 1400) > 0.5

    def test_symmetric(self):
        """Expected scores sum to 1.0."""
        a_vs_b = expected_score(1600, 1400)
        b_vs_a = expected_score(1400, 1600)
        assert abs(a_vs_b + b_vs_a - 1.0) < 0.001


class TestUpdateRatings:
    def test_winner_gains_rating(self):
        """Race winner gains Elo points."""
        ratings = {"VER": 1500, "NOR": 1500, "PIA": 1500}
        results = [("VER", 1), ("NOR", 2), ("PIA", 3)]

        new_ratings = update_ratings_for_race(ratings, results, k_factor=16)

        assert new_ratings["VER"] > 1500
        assert new_ratings["PIA"] < 1500

    def test_last_place_loses_rating(self):
        """Last place finisher loses Elo points."""
        ratings = {"VER": 1500, "NOR": 1500, "PIA": 1500}
        results = [("VER", 1), ("NOR", 2), ("PIA", 3)]

        new_ratings = update_ratings_for_race(ratings, results, k_factor=16)
        assert new_ratings["PIA"] < 1500

    def test_ratings_conserved(self):
        """Total Elo points are approximately conserved."""
        ratings = {"VER": 1500, "NOR": 1500, "PIA": 1500}
        results = [("VER", 1), ("NOR", 2), ("PIA", 3)]

        new_ratings = update_ratings_for_race(ratings, results, k_factor=16)
        total_before = sum(ratings.values())
        total_after = sum(new_ratings.values())

        # Pairwise Elo has small conservation drift due to asymmetric updates
        assert abs(total_after - total_before) < 1.0

    def test_new_driver_gets_default(self):
        """New driver not in ratings dict gets default rating."""
        ratings = {"VER": 1600}
        results = [("VER", 1), ("NEW", 2)]

        new_ratings = update_ratings_for_race(ratings, results, k_factor=16)
        assert "NEW" in new_ratings


class TestComputeEloRatings:
    def test_adds_elo_columns(self):
        """compute_elo_ratings adds driver_elo and constructor_elo columns."""
        df = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "VER", "NOR"],
            "TeamName": ["Red Bull", "McLaren", "Red Bull", "McLaren"],
            "Year": [2024, 2024, 2024, 2024],
            "RoundNumber": [1, 1, 2, 2],
            "FinishPosition": [1, 2, 2, 1],
        })

        result = compute_elo_ratings(df)

        assert "driver_elo" in result.columns
        assert "constructor_elo" in result.columns

    def test_first_race_uses_default_rating(self):
        """First race starts with default Elo."""
        df = pd.DataFrame({
            "Abbreviation": ["VER", "NOR"],
            "TeamName": ["Red Bull", "McLaren"],
            "Year": [2024, 2024],
            "RoundNumber": [1, 1],
            "FinishPosition": [1, 2],
        })

        result = compute_elo_ratings(df)
        assert result.iloc[0]["driver_elo"] == DEFAULT_RATING
        assert result.iloc[1]["driver_elo"] == DEFAULT_RATING

    def test_consistent_winner_gains_over_time(self):
        """Driver who consistently wins accumulates higher Elo."""
        rows = []
        for r in range(1, 11):
            rows.append({"Abbreviation": "VER", "TeamName": "RB", "Year": 2024,
                        "RoundNumber": r, "FinishPosition": 1})
            rows.append({"Abbreviation": "NOR", "TeamName": "MC", "Year": 2024,
                        "RoundNumber": r, "FinishPosition": 2})

        df = pd.DataFrame(rows)
        result = compute_elo_ratings(df)

        # VER's Elo should increase over time
        ver_elos = result[result["Abbreviation"] == "VER"]["driver_elo"].values
        assert ver_elos[-1] > ver_elos[0]

        # NOR's Elo should decrease
        nor_elos = result[result["Abbreviation"] == "NOR"]["driver_elo"].values
        assert nor_elos[-1] < nor_elos[0]
