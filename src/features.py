"""
Feature engineering for F1 prediction model.

All feature computation functions live here. Each function takes raw race
data and returns additional feature columns. Features are organized by category:
driver, constructor, tire, and track.
"""

import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
TRACKS_CSV = PROJECT_ROOT / "data" / "tracks.csv"


def compute_driver_features(df: pd.DataFrame, window_sizes: list[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Compute driver-level features from historical race data.

    Features:
    - Rolling average finish position (last N races)
    - Qualifying delta to teammate
    - Track-specific historical performance
    - DNF rate (rolling window)

    Args:
        df: Race data sorted by Year, RoundNumber. One row per driver per race.
        window_sizes: Rolling window sizes for averaging.

    Returns:
        DataFrame with additional driver feature columns.
    """
    df = df.copy()
    df = df.sort_values(["Year", "RoundNumber", "Position"]).reset_index(drop=True)

    # Rolling average finish position per driver
    for window in window_sizes:
        col_name = f"driver_rolling_avg_{window}"
        df[col_name] = (
            df.groupby("Abbreviation")["FinishPosition"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Rolling DNF rate per driver
    df["driver_dnf_rate"] = (
        df.groupby("Abbreviation")["DNF"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Qualifying delta to teammate (same team, same race)
    df["quali_teammate_delta"] = _compute_teammate_quali_delta(df)

    # Track-specific average finish (historical performance at this circuit)
    df["driver_track_avg"] = _compute_track_specific_avg(df)

    return df


def _compute_teammate_quali_delta(df: pd.DataFrame) -> pd.Series:
    """
    Compute the qualifying time delta between a driver and their teammate.

    Positive = slower than teammate, negative = faster.
    Uses the best qualifying time (Q3 if available, else Q2, else Q1).
    """
    delta = pd.Series(np.nan, index=df.index)

    # Best qualifying time: Q3 > Q2 > Q1
    df_temp = df.copy()
    df_temp["best_quali"] = df_temp["Q3"].fillna(df_temp["Q2"]).fillna(df_temp["Q1"])

    for (year, rnd, team), group in df_temp.groupby(["Year", "RoundNumber", "TeamName"]):
        if len(group) < 2:
            continue

        times = group["best_quali"].values
        indices = group.index.values

        if len(group) == 2:
            if not np.isnan(times[0]) and not np.isnan(times[1]):
                delta.iloc[indices[0]] = times[0] - times[1]
                delta.iloc[indices[1]] = times[1] - times[0]

    return delta


def _compute_track_specific_avg(df: pd.DataFrame) -> pd.Series:
    """
    Compute driver's historical average finish at the current track.

    Uses EventName to identify the track. Only uses data from before
    the current race (no leakage).
    """
    result = pd.Series(np.nan, index=df.index)

    for idx, row in df.iterrows():
        driver = row["Abbreviation"]
        track = row["EventName"]
        year = row["Year"]
        rnd = row["RoundNumber"]

        # Prior races at this track by this driver
        mask = (
            (df["Abbreviation"] == driver)
            & (df["EventName"] == track)
            & ((df["Year"] < year) | ((df["Year"] == year) & (df["RoundNumber"] < rnd)))
        )
        prior = df.loc[mask, "FinishPosition"]

        if len(prior) > 0:
            result.iloc[idx] = prior.mean()

    return result


def compute_constructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute constructor-level features.

    Features:
    - Rolling constructor points percentage (share of total points)
    - Constructor reliability score (team DNF rate)
    - Constructor points trajectory (recent form)
    """
    df = df.copy()

    # Rolling constructor points (sum of team points per race, rolling window)
    team_race_points = (
        df.groupby(["Year", "RoundNumber", "TeamName"])["Points"]
        .sum()
        .reset_index()
        .rename(columns={"Points": "team_race_points"})
    )

    # Total points per round for normalization
    round_total = (
        df.groupby(["Year", "RoundNumber"])["Points"]
        .sum()
        .reset_index()
        .rename(columns={"Points": "round_total_points"})
    )

    team_race_points = team_race_points.merge(
        round_total, on=["Year", "RoundNumber"], how="left"
    )
    team_race_points["team_points_pct"] = np.where(
        team_race_points["round_total_points"] > 0,
        team_race_points["team_race_points"] / team_race_points["round_total_points"],
        0,
    )

    # Rolling average of points percentage
    team_race_points = team_race_points.sort_values(["TeamName", "Year", "RoundNumber"])
    team_race_points["constructor_rolling_points"] = (
        team_race_points.groupby("TeamName")["team_points_pct"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Constructor DNF rate
    team_dnf = (
        df.groupby(["Year", "RoundNumber", "TeamName"])["DNF"]
        .mean()
        .reset_index()
        .rename(columns={"DNF": "team_dnf_rate_race"})
    )
    team_dnf = team_dnf.sort_values(["TeamName", "Year", "RoundNumber"])
    team_dnf["constructor_reliability"] = (
        team_dnf.groupby("TeamName")["team_dnf_rate_race"]
        .transform(lambda x: 1 - x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Merge back to main DataFrame
    df = df.merge(
        team_race_points[["Year", "RoundNumber", "TeamName", "constructor_rolling_points"]],
        on=["Year", "RoundNumber", "TeamName"],
        how="left",
    )
    df = df.merge(
        team_dnf[["Year", "RoundNumber", "TeamName", "constructor_reliability"]],
        on=["Year", "RoundNumber", "TeamName"],
        how="left",
    )

    return df


def compute_tire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tire strategy features.

    For now, these are track-level historical averages. Full stint-level
    tire data from FastF1 will be integrated in Phase 4.

    Features:
    - Historical average pit stops per race at this track
    """
    df = df.copy()
    # Placeholder: will be enriched with FastF1 stint data in Phase 4
    df["tire_historical_stops"] = np.nan
    return df


def compute_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute track characteristics features from tracks.csv.

    Features:
    - circuit_type (street=1, permanent=0)
    - track_length_km
    - overtaking_difficulty (0-1 scale)
    - safety_car_rate (historical proportion)
    - elevation_high (meters)
    - drs_zones (count)
    """
    df = df.copy()

    if not TRACKS_CSV.exists():
        # No track data available, return nulls
        for col in ["circuit_type_street", "track_length_km", "overtaking_difficulty",
                     "safety_car_rate", "elevation_high", "drs_zones"]:
            df[col] = np.nan
        return df

    tracks = pd.read_csv(TRACKS_CSV)

    # Build a mapping from EventName keywords to circuit_id
    # This is fuzzy matching since EventName format varies
    track_map = _build_track_mapping(df, tracks)

    for col in ["circuit_type", "track_length_km", "overtaking_difficulty",
                 "safety_car_rate", "elevation_high", "drs_zones"]:
        df[col] = df["EventName"].map(
            lambda name, c=col: track_map.get(name, {}).get(c, np.nan)
        )

    # Encode circuit_type as binary
    df["circuit_type_street"] = (df["circuit_type"] == "street").astype(float)
    df.drop(columns=["circuit_type"], inplace=True, errors="ignore")

    return df


def _build_track_mapping(df: pd.DataFrame, tracks: pd.DataFrame) -> dict:
    """
    Build a mapping from EventName to track characteristics.

    Uses fuzzy matching on circuit name keywords.
    """
    mapping = {}
    track_keywords = {
        row["circuit_id"]: row.to_dict()
        for _, row in tracks.iterrows()
    }

    # Common event name to circuit_id mappings
    name_hints = {
        "bahrain": "bahrain",
        "saudi": "jeddah",
        "jeddah": "jeddah",
        "australia": "melbourne",
        "melbourne": "melbourne",
        "japan": "suzuka",
        "suzuka": "suzuka",
        "china": "shanghai",
        "shanghai": "shanghai",
        "miami": "miami",
        "emilia": "imola",
        "imola": "imola",
        "monaco": "monaco",
        "spain": "barcelona",
        "barcelona": "barcelona",
        "canada": "montreal",
        "montreal": "montreal",
        "austria": "spielberg",
        "spielberg": "spielberg",
        "britain": "silverstone",
        "british": "silverstone",
        "silverstone": "silverstone",
        "belgium": "spa",
        "spa": "spa",
        "hungary": "budapest",
        "budapest": "budapest",
        "dutch": "zandvoort",
        "netherlands": "zandvoort",
        "zandvoort": "zandvoort",
        "italy": "monza",
        "monza": "monza",
        "azerbaijan": "baku",
        "baku": "baku",
        "singapore": "marina_bay",
        "united states": "austin",
        "austin": "austin",
        "mexico": "mexico_city",
        "brazil": "interlagos",
        "são paulo": "interlagos",
        "sao paulo": "interlagos",
        "las vegas": "las_vegas",
        "qatar": "lusail",
        "abu dhabi": "yas_marina",
    }

    for event_name in df["EventName"].unique():
        lower_name = event_name.lower()
        matched_id = None
        for keyword, circuit_id in name_hints.items():
            if keyword in lower_name:
                matched_id = circuit_id
                break

        if matched_id and matched_id in track_keywords:
            mapping[event_name] = track_keywords[matched_id]

    return mapping


def compute_cold_start_weight(year: int, round_number: int, regulation_years: set[int] = {2014, 2022, 2026}) -> float:
    """
    Compute the regulation confidence weight for cold-start decay.

    Returns a value between 0.0 and 1.0. Low values mean "don't trust
    historical features much" (early in a regulation change). High values
    mean "historical features are reliable."

    Args:
        year: Race year
        round_number: Race number within the season
        regulation_years: Years when major regulation changes occurred

    Returns:
        Weight between 0.0 (no confidence in historical features) and 1.0 (full confidence)
    """
    if year not in regulation_years:
        return 1.0

    # Linear ramp from 0.3 to 1.0 over the first 10 races
    min_weight = 0.3
    full_confidence_race = 10
    weight = min_weight + (1.0 - min_weight) * min(round_number / full_confidence_race, 1.0)
    return weight


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete feature matrix from raw race data.

    Runs all feature computation functions and returns a DataFrame
    ready for model training.
    """
    df = compute_driver_features(df)
    df = compute_constructor_features(df)
    df = compute_tire_features(df)
    df = compute_track_features(df)

    # Add cold-start weights
    df["regulation_confidence"] = df.apply(
        lambda row: compute_cold_start_weight(row["Year"], row["RoundNumber"]),
        axis=1,
    )

    return df
