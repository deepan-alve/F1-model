"""
Adaptive Elo rating system for F1 drivers and constructors.

K-factor schedule adapts for regulation resets:
- Default: K = 16
- First 3 races of regulation change: K = 48 (3x, fast learning)
- Races 4-8 of regulation change: K = 32 (2x, stabilizing)
- Race 9+: K = 16 (back to normal)

Sprint races use half-weight updates (K * 0.5).
"""

import numpy as np
import pandas as pd


DEFAULT_RATING = 1500.0
DEFAULT_K = 16
REGULATION_CHANGE_YEARS = {2014, 2022, 2026}


def get_k_factor(
    year: int,
    round_number: int,
    is_sprint: bool = False,
    regulation_years: set[int] = REGULATION_CHANGE_YEARS,
    k_standard: float = DEFAULT_K,
    k_high: float = 48.0,
    k_medium: float = 32.0,
) -> float:
    """
    Get the K-factor for a given race based on regulation schedule.

    Args:
        year: Race year
        round_number: Race number within the season
        is_sprint: Whether this is a sprint race (half weight)
        regulation_years: Years with major regulation changes
        k_standard: Standard K-factor
        k_high: K-factor for first 3 races of regulation change
        k_medium: K-factor for races 4-8 of regulation change

    Returns:
        K-factor for Elo updates
    """
    if year in regulation_years:
        if round_number <= 3:
            k = k_high
        elif round_number <= 8:
            k = k_medium
        else:
            k = k_standard
    else:
        k = k_standard

    if is_sprint:
        k *= 0.5

    return k


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings_for_race(
    ratings: dict[str, float],
    race_results: list[tuple[str, int]],
    k_factor: float,
) -> dict[str, float]:
    """
    Update Elo ratings based on a single race result.

    Uses a pairwise comparison approach: each pair of drivers in the race
    generates an Elo update. A driver who finishes ahead "wins" the pairwise
    comparison.

    Args:
        ratings: Current {driver_id: rating} dict
        race_results: List of (driver_id, finish_position) sorted by position
        k_factor: K-factor for this race

    Returns:
        Updated ratings dict
    """
    new_ratings = dict(ratings)
    n = len(race_results)

    if n < 2:
        return new_ratings

    # Normalize K by number of comparisons per driver
    # Each driver is compared against (n-1) others
    k_per_pair = k_factor / (n - 1)

    for i in range(n):
        driver_a, pos_a = race_results[i]
        rating_a = new_ratings.get(driver_a, DEFAULT_RATING)
        total_update = 0.0

        for j in range(n):
            if i == j:
                continue

            driver_b, pos_b = race_results[j]
            rating_b = new_ratings.get(driver_b, DEFAULT_RATING)

            expected = expected_score(rating_a, rating_b)

            # Actual score: 1 if A finished ahead, 0 if behind, 0.5 if tied
            if pos_a < pos_b:
                actual = 1.0
            elif pos_a > pos_b:
                actual = 0.0
            else:
                actual = 0.5

            total_update += k_per_pair * (actual - expected)

        new_ratings[driver_a] = rating_a + total_update

    return new_ratings


def compute_elo_ratings(
    df: pd.DataFrame,
    k_standard: float = DEFAULT_K,
    k_high: float = 48.0,
    k_medium: float = 32.0,
    regulation_years: set[int] = REGULATION_CHANGE_YEARS,
) -> pd.DataFrame:
    """
    Compute Elo ratings for all drivers across the dataset.

    Ratings are computed BEFORE each race (using only prior races),
    so there's no data leakage.

    Args:
        df: Race data with Abbreviation, Year, RoundNumber, FinishPosition
        k_standard: Standard K-factor
        k_high: K-factor for early regulation change races
        k_medium: K-factor for mid regulation change races
        regulation_years: Set of years with major regulation changes

    Returns:
        DataFrame with driver_elo and constructor_elo columns added
    """
    df = df.copy()
    df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)

    driver_ratings: dict[str, float] = {}
    constructor_ratings: dict[str, float] = {}

    driver_elo_col = []
    constructor_elo_col = []

    # Process race by race
    races = df.groupby(["Year", "RoundNumber"])

    for (year, rnd), race_group in races:
        # Record PRE-RACE ratings (before this race's update)
        for _, row in race_group.iterrows():
            driver = row["Abbreviation"]
            team = row["TeamName"]
            driver_elo_col.append(driver_ratings.get(driver, DEFAULT_RATING))
            constructor_elo_col.append(constructor_ratings.get(team, DEFAULT_RATING))

        # Now update ratings based on this race's results
        k = get_k_factor(year, rnd, regulation_years=regulation_years,
                         k_standard=k_standard, k_high=k_high, k_medium=k_medium)

        # Driver Elo update
        results = [
            (row["Abbreviation"], row["FinishPosition"])
            for _, row in race_group.iterrows()
        ]
        results.sort(key=lambda x: x[1])
        driver_ratings = update_ratings_for_race(driver_ratings, results, k)

        # Constructor Elo update (use best finisher per team)
        team_best = {}
        for _, row in race_group.iterrows():
            team = row["TeamName"]
            pos = row["FinishPosition"]
            if team not in team_best or pos < team_best[team]:
                team_best[team] = pos

        constructor_results = sorted(team_best.items(), key=lambda x: x[1])
        constructor_ratings = update_ratings_for_race(
            constructor_ratings, constructor_results, k
        )

    df["driver_elo"] = driver_elo_col
    df["constructor_elo"] = constructor_elo_col

    return df
