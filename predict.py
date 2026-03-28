"""
F1 Race Prediction Script.

Run after qualifying to predict the race finishing order.
Outputs a confidence-annotated grid formatted for group chat copy-paste.

Usage:
    python predict.py --race "Bahrain Grand Prix" --year 2026
    python predict.py --test  # Run on 2025 holdout data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline import (
    enable_cache,
    fetch_historical_data,
    load_from_parquet,
    refresh_current_season,
    save_to_parquet,
)
from src.elo import compute_elo_ratings
from src.features import build_feature_matrix
from src.model import (
    RANKING_FEATURES,
    evaluate_spearman,
    predict_race_order,
    predict_with_confidence,
    prepare_ranking_data,
    train_dnf_classifier,
    train_ranker,
)
from src.odds import compute_market_delta, fetch_race_odds
from accuracy_tracker import log_prediction


def format_confidence_bar(confidence: float, width: int = 10) -> str:
    """Format confidence as a Unicode bar."""
    filled = round(confidence / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def format_prediction_output(
    predictions: pd.DataFrame,
    race_name: str,
    year: int,
    data_quality: str = "All tiers available",
) -> str:
    """
    Format predictions as a clean terminal output for group chat.

    Args:
        predictions: DataFrame with PredictedPosition, Abbreviation, Confidence
        race_name: Name of the race
        year: Race year
        data_quality: Description of which data tiers were available

    Returns:
        Formatted string for copy-paste
    """
    lines = []
    lines.append(f"{race_name} {year} - Model Predictions")
    lines.append("\u2501" * 50)

    for _, row in predictions.iterrows():
        pos = int(row["PredictedPosition"])
        driver = row["Abbreviation"]
        conf = row.get("Confidence", 0)
        bar = format_confidence_bar(conf)

        line = f"P{pos:<3} {driver:<4} (conf: {bar} {conf:.0f}%)"

        # Add market comparison if available
        if "MarketDelta" in row and not pd.isna(row.get("MarketDelta", np.nan)):
            delta = int(row["MarketDelta"])
            if delta > 0:
                line += f"  \u2191 +{delta} vs market"
            elif delta < 0:
                line += f"  \u2193 {delta} vs market"

        # Add DNF risk if available
        if "DNFProbability" in row and row.get("DNFProbability", 0) > 0.15:
            line += f"  \u26a0 DNF risk: {row['DNFProbability']:.0%}"

        lines.append(line)

    lines.append("\u2501" * 50)
    lines.append(f"Data quality: {data_quality}")

    return "\n".join(lines)


def run_prediction(race_name: str, year: int = 2026, use_bootstrap: bool = True):
    """Run the full prediction pipeline for a specific race."""
    enable_cache()

    # Load or fetch historical data
    print("Loading data...")
    all_data = load_from_parquet("all_races.parquet")
    if all_data is None:
        print("No cached data found. Fetching historical data (this may take a while)...")
        all_data = fetch_historical_data(2019, 2025)
        save_to_parquet(all_data, "all_races.parquet")

    # Refresh current season
    if year >= 2026:
        print(f"Refreshing {year} season data...")
        all_data = refresh_current_season(year, all_data)
        save_to_parquet(all_data, "all_races.parquet")

    # Build features
    print("Computing features...")
    featured_data = build_feature_matrix(all_data)

    # Compute Elo ratings
    print("Computing Elo ratings...")
    featured_data = compute_elo_ratings(featured_data)

    # Split: training data vs the race to predict
    race_mask = (
        (featured_data["EventName"].str.contains(race_name, case=False, na=False))
        & (featured_data["Year"] == year)
    )

    if race_mask.sum() == 0:
        print(f"Error: Race '{race_name}' not found in {year} data.")
        print("Available races:")
        for name in featured_data[featured_data["Year"] == year]["EventName"].unique():
            print(f"  - {name}")
        return

    race_data = featured_data[race_mask]
    train_data = featured_data[~race_mask & (featured_data["Year"] < year)]

    if len(train_data) < 100:
        print(f"Warning: Only {len(train_data)} training samples. Results may be unreliable.")

    # Train and predict
    if use_bootstrap:
        print("Training bootstrap ensemble (50 models)...")
        predictions = predict_with_confidence(train_data, race_data)
    else:
        print("Training single model...")
        X_train, y_train, groups_train = prepare_ranking_data(train_data)
        model = train_ranker(X_train, y_train, groups_train)
        predictions = predict_race_order(model, race_data)

    # DNF classifier
    print("Training DNF classifier...")
    dnf_model = train_dnf_classifier(train_data)
    dnf_features = [f for f in ["GridPosition", "driver_elo", "driver_dnf_rate", "constructor_reliability"]
                    if f in race_data.columns]
    X_dnf = race_data[dnf_features].fillna(0)
    predictions["DNFProbability"] = dnf_model.predict_proba(X_dnf.values)[:, 1]

    # Betting odds comparison
    data_quality_parts = ["Core: OK"]
    print("Fetching betting odds...")
    odds = fetch_race_odds()
    predictions = compute_market_delta(predictions, odds)
    if odds is not None:
        data_quality_parts.append("Odds: OK")
    else:
        data_quality_parts.append("Odds: unavailable")

    data_quality = " | ".join(data_quality_parts)

    # Format and print
    output = format_prediction_output(predictions, race_name, year, data_quality)
    print()
    print(output)

    # Log prediction
    log_prediction(race_name, year, predictions)

    # If we have actual results, show accuracy
    if "FinishPosition" in predictions.columns and predictions["FinishPosition"].notna().any():
        spearman = evaluate_spearman(predictions)
        print(f"\nAccuracy (Spearman rho): {spearman:.3f}")


def run_test():
    """Run prediction on 2025 holdout data to validate the model."""
    enable_cache()

    all_data = load_from_parquet("all_races.parquet")
    if all_data is None:
        print("No cached data. Run 'python predict.py --fetch' first to download historical data.")
        return

    featured_data = build_feature_matrix(all_data)
    featured_data = compute_elo_ratings(featured_data)

    # Use 2024 as holdout
    train = featured_data[featured_data["Year"] <= 2023]
    holdout = featured_data[featured_data["Year"] == 2024]

    if holdout.empty:
        print("No 2024 data available for testing.")
        return

    X_train, y_train, groups_train = prepare_ranking_data(train)
    model = train_ranker(X_train, y_train, groups_train)

    # Predict each race in holdout
    all_predictions = []
    for (year, rnd), race in holdout.groupby(["Year", "RoundNumber"]):
        preds = predict_race_order(model, race)
        preds["FinishPosition"] = race["FinishPosition"].values
        all_predictions.append(preds)

    all_preds = pd.concat(all_predictions, ignore_index=True)
    overall_spearman = evaluate_spearman(all_preds)

    print(f"2024 Holdout Results:")
    print(f"  Spearman rho: {overall_spearman:.3f}")
    print(f"  Races evaluated: {holdout.groupby(['Year', 'RoundNumber']).ngroups}")
    print(f"  Total predictions: {len(all_preds)}")


def main():
    parser = argparse.ArgumentParser(description="F1 Race Prediction")
    parser.add_argument("--race", type=str, help="Race name (e.g., 'Bahrain Grand Prix')")
    parser.add_argument("--year", type=int, default=2026, help="Race year (default: 2026)")
    parser.add_argument("--test", action="store_true", help="Run on 2024 holdout data")
    parser.add_argument("--fetch", action="store_true", help="Fetch and cache historical data")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap confidence")

    args = parser.parse_args()

    if args.fetch:
        enable_cache()
        print("Fetching historical data (2019-2025)...")
        data = fetch_historical_data(2019, 2025)
        save_to_parquet(data, "all_races.parquet")
        print("Done.")
    elif args.test:
        run_test()
    elif args.race:
        run_prediction(args.race, args.year, use_bootstrap=not args.no_bootstrap)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
