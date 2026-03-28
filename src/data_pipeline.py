"""
F1 data ingestion pipeline.

Fetches race results, qualifying, and session data from FastF1.
Caches to data/raw/ for fast subsequent loads.
Produces per-race DataFrames with driver results and session info.
"""

import os
from pathlib import Path

import fastf1
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def enable_cache():
    """Enable FastF1 disk cache for fast subsequent loads."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_season_schedule(year: int) -> pd.DataFrame:
    """Get the race schedule for a given season."""
    enable_cache()
    schedule = fastf1.get_event_schedule(year)
    # Filter to actual race events (exclude testing)
    races = schedule[schedule["EventFormat"].isin(["conventional", "sprint_shootout", "sprint_qualifying", "sprint"])]
    return races


def fetch_race_results(year: int, round_number: int) -> pd.DataFrame | None:
    """
    Fetch race results for a specific race.

    Returns a DataFrame with one row per driver:
    - DriverId, Abbreviation, TeamName
    - GridPosition, Position (finish), Status
    - Points
    """
    enable_cache()
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results

        if results is None or results.empty:
            return None

        df = results[
            [
                "DriverNumber",
                "BroadcastName",
                "Abbreviation",
                "TeamName",
                "GridPosition",
                "Position",
                "Status",
                "Points",
            ]
        ].copy()

        df["Year"] = year
        df["RoundNumber"] = round_number
        df["EventName"] = session.event["EventName"]

        # Clean position data
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Warning: Could not load race {year} R{round_number}: {e}")
        return None


def fetch_qualifying_results(year: int, round_number: int) -> pd.DataFrame | None:
    """
    Fetch qualifying results for a specific race.

    Returns a DataFrame with qualifying positions and times.
    """
    enable_cache()
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results

        if results is None or results.empty:
            return None

        df = results[
            [
                "Abbreviation",
                "TeamName",
                "Position",
                "Q1",
                "Q2",
                "Q3",
            ]
        ].copy()

        df["Year"] = year
        df["RoundNumber"] = round_number

        # Convert qualifying times to seconds
        for col in ["Q1", "Q2", "Q3"]:
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

        df.rename(columns={"Position": "QualifyingPosition"}, inplace=True)
        df["QualifyingPosition"] = pd.to_numeric(df["QualifyingPosition"], errors="coerce")

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Warning: Could not load qualifying {year} R{round_number}: {e}")
        return None


def fetch_season_data(year: int) -> pd.DataFrame:
    """
    Fetch all race + qualifying results for a full season.

    Returns a merged DataFrame with one row per driver per race,
    containing both race results and qualifying data.
    """
    schedule = get_season_schedule(year)
    all_results = []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]
        print(f"  Fetching {year} R{round_num}: {event_name}...")

        race = fetch_race_results(year, round_num)
        quali = fetch_qualifying_results(year, round_num)

        if race is None:
            continue

        if quali is not None:
            # Merge qualifying data into race results
            merged = race.merge(
                quali[["Abbreviation", "QualifyingPosition", "Q1", "Q2", "Q3"]],
                on="Abbreviation",
                how="left",
            )
        else:
            merged = race.copy()
            merged["QualifyingPosition"] = merged["GridPosition"]
            merged["Q1"] = None
            merged["Q2"] = None
            merged["Q3"] = None

        # Add DNF flag
        finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps"}
        merged["DNF"] = ~merged["Status"].isin(finished_statuses)

        # Encode finish position: DNFs get position 21
        merged["FinishPosition"] = merged["Position"].copy()
        merged.loc[merged["DNF"], "FinishPosition"] = 21

        all_results.append(merged)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def fetch_historical_data(
    start_year: int = 2019, end_year: int = 2025
) -> pd.DataFrame:
    """
    Fetch all historical race data from start_year to end_year.

    This is the slow one-time operation. Subsequent calls use FastF1's
    built-in cache and return near-instantly.
    """
    all_seasons = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year} season...")
        season_df = fetch_season_data(year)
        if not season_df.empty:
            all_seasons.append(season_df)

    if not all_seasons:
        return pd.DataFrame()

    return pd.concat(all_seasons, ignore_index=True)


def refresh_current_season(year: int = 2026, existing_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Incremental refresh: only fetch races not already in existing_data.

    This is the fast pre-race operation for predict.py.
    """
    schedule = get_season_schedule(year)

    if existing_data is not None:
        existing_rounds = set(
            existing_data[existing_data["Year"] == year]["RoundNumber"].unique()
        )
    else:
        existing_rounds = set()

    new_results = []
    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        if round_num in existing_rounds:
            continue

        race = fetch_race_results(year, round_num)
        if race is None:
            # Race hasn't happened yet, stop here
            break

        quali = fetch_qualifying_results(year, round_num)
        if quali is not None:
            merged = race.merge(
                quali[["Abbreviation", "QualifyingPosition", "Q1", "Q2", "Q3"]],
                on="Abbreviation",
                how="left",
            )
        else:
            merged = race.copy()
            merged["QualifyingPosition"] = merged["GridPosition"]
            merged["Q1"] = None
            merged["Q2"] = None
            merged["Q3"] = None

        finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps"}
        merged["DNF"] = ~merged["Status"].isin(finished_statuses)
        merged["FinishPosition"] = merged["Position"].copy()
        merged.loc[merged["DNF"], "FinishPosition"] = 21

        new_results.append(merged)

    if not new_results:
        return existing_data if existing_data is not None else pd.DataFrame()

    new_df = pd.concat(new_results, ignore_index=True)

    if existing_data is not None:
        return pd.concat([existing_data, new_df], ignore_index=True)
    return new_df


def save_to_parquet(df: pd.DataFrame, filename: str = "all_races.parquet"):
    """Save DataFrame to parquet in the processed directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DIR / filename
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")


def load_from_parquet(filename: str = "all_races.parquet") -> pd.DataFrame | None:
    """Load DataFrame from parquet if it exists."""
    filepath = PROCESSED_DIR / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


# Expected columns in the output DataFrame
EXPECTED_COLUMNS = [
    "DriverNumber",
    "BroadcastName",
    "Abbreviation",
    "TeamName",
    "GridPosition",
    "Position",
    "Status",
    "Points",
    "Year",
    "RoundNumber",
    "EventName",
    "QualifyingPosition",
    "Q1",
    "Q2",
    "Q3",
    "DNF",
    "FinishPosition",
]
