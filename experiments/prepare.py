"""
IMMUTABLE evaluation script for autoresearch.

DO NOT MODIFY THIS FILE during autoresearch runs.
This is the scoring function that autoresearch uses to evaluate experiments.

It loads pre-computed features, splits into temporal train/holdout,
and evaluates the model trained by train.py using Spearman correlation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"

# Temporal split: train on 2019-2023, holdout on 2024
HOLDOUT_YEAR = 2024


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-computed features and split into train/holdout.

    Returns:
        (train_df, holdout_df) split by HOLDOUT_YEAR
    """
    if not FEATURES_PARQUET.exists():
        print(f"ERROR: {FEATURES_PARQUET} not found.")
        print("Run: python -c \"from src.data_pipeline import *; from src.features import *; "
              "from src.elo import *; d=fetch_historical_data(); d=build_feature_matrix(d); "
              "save_to_parquet(d, 'features.parquet')\"")
        sys.exit(1)

    df = pd.read_parquet(FEATURES_PARQUET)

    train = df[df["Year"] < HOLDOUT_YEAR].copy()
    holdout = df[df["Year"] == HOLDOUT_YEAR].copy()

    if holdout.empty:
        print(f"ERROR: No data for holdout year {HOLDOUT_YEAR}")
        sys.exit(1)

    return train, holdout


def evaluate(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """
    Compute Spearman rank correlation between predictions and actuals.

    This is THE metric that autoresearch optimizes.

    Args:
        predictions: DataFrame with Abbreviation, PredictedPosition per race
        actuals: DataFrame with Abbreviation, FinishPosition per race

    Returns:
        Average Spearman correlation across all holdout races
    """
    merged = predictions.merge(
        actuals[["Abbreviation", "Year", "RoundNumber", "FinishPosition"]],
        on=["Abbreviation", "Year", "RoundNumber"],
        how="inner",
    )

    if merged.empty:
        return 0.0

    race_spearmans = []
    for (year, rnd), race in merged.groupby(["Year", "RoundNumber"]):
        if len(race) < 3:
            continue
        corr, _ = spearmanr(race["PredictedPosition"], race["FinishPosition"])
        if not np.isnan(corr):
            race_spearmans.append(corr)

    if not race_spearmans:
        return 0.0

    return float(np.mean(race_spearmans))


def main():
    """
    Evaluate the model from train.py on the holdout set.

    Prints the Spearman correlation as the LAST LINE of stdout.
    Autoresearch reads this float to score the experiment.
    """
    train_df, holdout_df = load_data()

    # Import train.py's model training and prediction functions
    # train.py is agent-modifiable, so its interface may change
    try:
        from train import train_and_predict
    except ImportError:
        print("ERROR: Could not import train_and_predict from train.py")
        print("0.0")
        return

    # train.py must implement: train_and_predict(train_df, holdout_df) -> predictions_df
    # predictions_df must have: Abbreviation, Year, RoundNumber, PredictedPosition
    predictions = train_and_predict(train_df, holdout_df)

    score = evaluate(predictions, holdout_df)

    # Print score as last line (autoresearch reads this)
    print(f"Spearman correlation on {HOLDOUT_YEAR} holdout: {score:.4f}")
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
