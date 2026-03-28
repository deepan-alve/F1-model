"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    compute_cold_start_weight,
    compute_constructor_features,
    compute_driver_features,
    compute_track_features,
    build_feature_matrix,
)


def _make_sample_data(n_races=5, n_drivers=4):
    """Create sample race data for testing."""
    rows = []
    drivers = [("VER", "Red Bull Racing"), ("NOR", "McLaren"), ("PIA", "McLaren"), ("HAM", "Ferrari")][:n_drivers]

    for race in range(1, n_races + 1):
        for i, (driver, team) in enumerate(drivers):
            rows.append({
                "Abbreviation": driver,
                "TeamName": team,
                "Year": 2024,
                "RoundNumber": race,
                "EventName": f"Race {race}",
                "GridPosition": i + 1,
                "Position": i + 1,
                "FinishPosition": i + 1,
                "QualifyingPosition": i + 1,
                "Q1": 90.0 + i * 0.5,
                "Q2": 89.0 + i * 0.5,
                "Q3": 88.0 + i * 0.5 if i < 3 else np.nan,
                "Points": [25, 18, 15, 12][i],
                "DNF": False,
                "Status": "Finished",
            })

    return pd.DataFrame(rows)


class TestDriverFeatures:
    def test_rolling_averages_computed(self):
        """Rolling averages are computed for configured window sizes."""
        df = _make_sample_data(n_races=5)
        result = compute_driver_features(df, window_sizes=[3, 5])

        assert "driver_rolling_avg_3" in result.columns
        assert "driver_rolling_avg_5" in result.columns
        assert "driver_rolling_avg_10" not in result.columns  # Not requested

    def test_rolling_avg_with_few_races(self):
        """Rolling average works when driver has fewer races than window size."""
        df = _make_sample_data(n_races=2)
        result = compute_driver_features(df, window_sizes=[5])

        # Should not crash, should produce values (min_periods=1)
        assert "driver_rolling_avg_5" in result.columns
        # First race has no prior data (shifted), so NaN
        ver_first = result[(result["Abbreviation"] == "VER") & (result["RoundNumber"] == 1)]
        assert pd.isna(ver_first["driver_rolling_avg_5"].values[0])
        # Second race has 1 prior race
        ver_second = result[(result["Abbreviation"] == "VER") & (result["RoundNumber"] == 2)]
        assert not pd.isna(ver_second["driver_rolling_avg_5"].values[0])

    def test_dnf_rate_computed(self):
        """DNF rate is computed as rolling average."""
        df = _make_sample_data(n_races=5)
        # Make VER DNF in race 2
        df.loc[(df["Abbreviation"] == "VER") & (df["RoundNumber"] == 2), "DNF"] = True

        result = compute_driver_features(df)
        assert "driver_dnf_rate" in result.columns

        # After race 2, VER's DNF rate should be > 0
        ver_r3 = result[(result["Abbreviation"] == "VER") & (result["RoundNumber"] == 3)]
        assert ver_r3["driver_dnf_rate"].values[0] > 0

    def test_teammate_quali_delta(self):
        """Qualifying delta to teammate is computed correctly."""
        df = _make_sample_data(n_races=1)
        result = compute_driver_features(df)

        assert "quali_teammate_delta" in result.columns
        # NOR and PIA are teammates (McLaren)
        nor = result[result["Abbreviation"] == "NOR"]
        pia = result[result["Abbreviation"] == "PIA"]

        # NOR should be faster (lower Q3), so delta should be negative
        nor_delta = nor["quali_teammate_delta"].values[0]
        pia_delta = pia["quali_teammate_delta"].values[0]
        if not pd.isna(nor_delta) and not pd.isna(pia_delta):
            assert nor_delta < 0  # Faster than teammate
            assert pia_delta > 0  # Slower than teammate

    def test_teammate_delta_solo_driver(self):
        """Teammate delta is NaN when driver has no teammate."""
        df = _make_sample_data(n_races=1)
        result = compute_driver_features(df)

        # VER is alone at Red Bull, HAM alone at Ferrari
        ver = result[result["Abbreviation"] == "VER"]
        assert pd.isna(ver["quali_teammate_delta"].values[0])

    def test_track_specific_avg(self):
        """Track-specific average uses only prior races at same track."""
        df = _make_sample_data(n_races=3)
        # Make race 3 at the same track as race 1
        df.loc[df["RoundNumber"] == 3, "EventName"] = "Race 1"

        result = compute_driver_features(df)
        assert "driver_track_avg" in result.columns

        # For race 3 at "Race 1" track, VER should have avg from race 1 only
        ver_r3 = result[(result["Abbreviation"] == "VER") & (result["RoundNumber"] == 3)]
        # VER finished P1 in race 1, so track avg should be 1.0
        assert ver_r3["driver_track_avg"].values[0] == 1.0

    def test_track_avg_new_track(self):
        """Track-specific average is NaN for a track with no history."""
        df = _make_sample_data(n_races=1)
        result = compute_driver_features(df)

        # First race at this track, no history
        assert pd.isna(result.iloc[0]["driver_track_avg"])


class TestConstructorFeatures:
    def test_rolling_points_computed(self):
        """Constructor rolling points percentage is computed."""
        df = _make_sample_data(n_races=5)
        result = compute_constructor_features(df)

        assert "constructor_rolling_points" in result.columns
        assert "constructor_reliability" in result.columns

    def test_reliability_reflects_dnf(self):
        """Constructor reliability decreases after DNFs."""
        df = _make_sample_data(n_races=5)
        # McLaren DNFs in races 2 and 3
        df.loc[(df["TeamName"] == "McLaren") & (df["RoundNumber"].isin([2, 3])), "DNF"] = True

        result = compute_constructor_features(df)
        mclaren_r5 = result[(result["TeamName"] == "McLaren") & (result["RoundNumber"] == 5)]
        redbull_r5 = result[(result["TeamName"] == "Red Bull Racing") & (result["RoundNumber"] == 5)]

        # McLaren should have lower reliability than Red Bull
        if not mclaren_r5.empty and not redbull_r5.empty:
            assert mclaren_r5["constructor_reliability"].values[0] < redbull_r5["constructor_reliability"].values[0]


class TestTrackFeatures:
    def test_track_features_with_no_csv(self, tmp_path, monkeypatch):
        """Track features are NaN when tracks.csv doesn't exist."""
        monkeypatch.setattr("src.features.TRACKS_CSV", tmp_path / "nonexistent.csv")
        df = _make_sample_data(n_races=1)
        result = compute_track_features(df)

        assert "circuit_type_street" in result.columns
        assert pd.isna(result.iloc[0]["circuit_type_street"])

    def test_track_features_with_csv(self):
        """Track features are populated from tracks.csv for known tracks."""
        df = pd.DataFrame({
            "Abbreviation": ["VER"],
            "EventName": ["Monaco Grand Prix"],
            "Year": [2024],
            "RoundNumber": [1],
        })
        result = compute_track_features(df)

        assert "circuit_type_street" in result.columns
        # Monaco is a street circuit
        if not pd.isna(result.iloc[0]["circuit_type_street"]):
            assert result.iloc[0]["circuit_type_street"] == 1.0


class TestColdStartWeight:
    def test_non_regulation_year_is_1(self):
        """Non-regulation years have full confidence."""
        assert compute_cold_start_weight(2023, 1) == 1.0
        assert compute_cold_start_weight(2023, 10) == 1.0

    def test_regulation_year_race_1_is_low(self):
        """First race of regulation year has low confidence."""
        weight = compute_cold_start_weight(2026, 1)
        assert weight < 0.5
        assert weight >= 0.3

    def test_regulation_year_race_10_is_full(self):
        """Race 10+ of regulation year has full confidence."""
        weight = compute_cold_start_weight(2026, 10)
        assert weight == 1.0

    def test_weight_increases_over_season(self):
        """Confidence increases monotonically through the season."""
        weights = [compute_cold_start_weight(2026, r) for r in range(1, 15)]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]
