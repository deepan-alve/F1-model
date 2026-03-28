"""Tests for the data pipeline module."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    EXPECTED_COLUMNS,
    enable_cache,
    fetch_race_results,
    fetch_qualifying_results,
    fetch_season_data,
    refresh_current_season,
    save_to_parquet,
    load_from_parquet,
)


def _make_mock_race_results():
    """Create a mock FastF1 results DataFrame for race sessions."""
    return pd.DataFrame(
        {
            "DriverNumber": ["1", "4", "81"],
            "BroadcastName": ["M VERSTAPPEN", "L NORRIS", "O PIASTRI"],
            "Abbreviation": ["VER", "NOR", "PIA"],
            "TeamName": ["Red Bull Racing", "McLaren", "McLaren"],
            "GridPosition": [1, 2, 3],
            "Position": [1, 2, 3],
            "Status": ["Finished", "Finished", "+1 Lap"],
            "Points": [25.0, 18.0, 15.0],
        }
    )


def _make_mock_quali_results():
    """Create a mock FastF1 results DataFrame for qualifying sessions."""
    return pd.DataFrame(
        {
            "Abbreviation": ["VER", "NOR", "PIA"],
            "TeamName": ["Red Bull Racing", "McLaren", "McLaren"],
            "Position": [1, 2, 3],
            "Q1": [pd.Timedelta(seconds=90.123), pd.Timedelta(seconds=90.456), pd.Timedelta(seconds=90.789)],
            "Q2": [pd.Timedelta(seconds=89.123), pd.Timedelta(seconds=89.456), pd.Timedelta(seconds=89.789)],
            "Q3": [pd.Timedelta(seconds=88.123), pd.Timedelta(seconds=88.456), pd.NaT],
        }
    )


class TestCacheSetup:
    def test_enable_cache_creates_directory(self, tmp_path, monkeypatch):
        """Cache directory is created if it doesn't exist."""
        monkeypatch.setattr("src.data_pipeline.CACHE_DIR", tmp_path / "cache")
        with patch("fastf1.Cache.enable_cache") as mock_cache:
            enable_cache()
            assert (tmp_path / "cache").exists()
            mock_cache.assert_called_once()


class TestFetchRaceResults:
    @patch("src.data_pipeline.enable_cache")
    @patch("fastf1.get_session")
    def test_returns_dataframe_with_expected_columns(self, mock_get_session, mock_cache):
        """Race results contain required columns."""
        mock_session = MagicMock()
        mock_session.results = _make_mock_race_results()
        mock_session.event = {"EventName": "Bahrain Grand Prix"}
        mock_get_session.return_value = mock_session

        result = fetch_race_results(2024, 1)

        assert result is not None
        assert "Abbreviation" in result.columns
        assert "Position" in result.columns
        assert "GridPosition" in result.columns
        assert "Year" in result.columns
        assert "RoundNumber" in result.columns
        assert len(result) == 3

    @patch("src.data_pipeline.enable_cache")
    @patch("fastf1.get_session")
    def test_returns_none_on_error(self, mock_get_session, mock_cache):
        """Gracefully returns None when session can't be loaded."""
        mock_get_session.side_effect = Exception("Network error")
        result = fetch_race_results(2024, 1)
        assert result is None


class TestFetchQualifyingResults:
    @patch("src.data_pipeline.enable_cache")
    @patch("fastf1.get_session")
    def test_converts_times_to_seconds(self, mock_get_session, mock_cache):
        """Qualifying times are converted from timedelta to float seconds."""
        mock_session = MagicMock()
        mock_session.results = _make_mock_quali_results()
        mock_get_session.return_value = mock_session

        result = fetch_qualifying_results(2024, 1)

        assert result is not None
        assert result["Q1"].dtype == float
        assert abs(result.iloc[0]["Q1"] - 90.123) < 0.01
        # Q3 NaT should become NaN
        assert pd.isna(result.iloc[2]["Q3"])


class TestFetchSeasonData:
    @patch("src.data_pipeline.fetch_qualifying_results")
    @patch("src.data_pipeline.fetch_race_results")
    @patch("src.data_pipeline.get_season_schedule")
    def test_merges_race_and_quali_data(self, mock_schedule, mock_race, mock_quali):
        """Season data merges race results with qualifying data."""
        mock_schedule.return_value = pd.DataFrame(
            {
                "RoundNumber": [1],
                "EventName": ["Bahrain Grand Prix"],
                "EventFormat": ["conventional"],
            }
        )

        race_df = _make_mock_race_results()
        race_df["Year"] = 2024
        race_df["RoundNumber"] = 1
        race_df["EventName"] = "Bahrain Grand Prix"
        mock_race.return_value = race_df

        quali_df = pd.DataFrame(
            {
                "Abbreviation": ["VER", "NOR", "PIA"],
                "QualifyingPosition": [1, 2, 3],
                "Q1": [90.1, 90.4, 90.7],
                "Q2": [89.1, 89.4, 89.7],
                "Q3": [88.1, 88.4, float("nan")],
            }
        )
        mock_quali.return_value = quali_df

        result = fetch_season_data(2024)

        assert not result.empty
        assert "QualifyingPosition" in result.columns
        assert "DNF" in result.columns
        assert "FinishPosition" in result.columns
        # All three finished, so no DNFs
        assert result["DNF"].sum() == 0
        # FinishPosition should match Position for finishers
        assert (result["FinishPosition"] == result["Position"]).all()

    @patch("src.data_pipeline.fetch_qualifying_results")
    @patch("src.data_pipeline.fetch_race_results")
    @patch("src.data_pipeline.get_season_schedule")
    def test_handles_missing_qualifying(self, mock_schedule, mock_race, mock_quali):
        """Season data handles missing qualifying session gracefully."""
        mock_schedule.return_value = pd.DataFrame(
            {
                "RoundNumber": [1],
                "EventName": ["Bahrain Grand Prix"],
                "EventFormat": ["conventional"],
            }
        )

        race_df = _make_mock_race_results()
        race_df["Year"] = 2024
        race_df["RoundNumber"] = 1
        race_df["EventName"] = "Bahrain Grand Prix"
        mock_race.return_value = race_df
        mock_quali.return_value = None  # Qualifying unavailable

        result = fetch_season_data(2024)

        assert not result.empty
        # Should fall back to GridPosition for QualifyingPosition
        assert "QualifyingPosition" in result.columns


class TestParquetIO:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Data survives parquet save/load roundtrip."""
        monkeypatch.setattr("src.data_pipeline.PROCESSED_DIR", tmp_path)

        df = pd.DataFrame(
            {
                "Abbreviation": ["VER", "NOR"],
                "Position": [1, 2],
                "Year": [2024, 2024],
            }
        )

        save_to_parquet(df, "test.parquet")
        loaded = load_from_parquet("test.parquet")

        assert loaded is not None
        assert len(loaded) == 2
        assert list(loaded["Abbreviation"]) == ["VER", "NOR"]

    def test_load_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        """Returns None when parquet file doesn't exist."""
        monkeypatch.setattr("src.data_pipeline.PROCESSED_DIR", tmp_path)
        result = load_from_parquet("nonexistent.parquet")
        assert result is None
