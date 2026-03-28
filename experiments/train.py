"""
AGENT-MODIFIABLE training script for autoresearch.

Autoresearch's AI agent can modify this file to explore different:
- K-factor schedules for Elo
- Feature subsets
- Model hyperparameters
- Cold-start decay rates

The agent modifies this file, runs it, prepare.py scores the result,
and the agent decides whether to keep or discard the changes.

Interface contract: must export train_and_predict(train_df, holdout_df) -> predictions_df
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import compute_elo_ratings


# === CONFIGURATION (autoresearch agent modifies these) ===

# Elo K-factor schedule
K_STANDARD = 16
K_HIGH = 48      # First 3 races of regulation change
K_MEDIUM = 32    # Races 4-8 of regulation change

# Features to use (toggle on/off)
USE_GRID_POSITION = True
USE_QUALIFYING_POSITION = True
USE_DRIVER_ELO = True
USE_CONSTRUCTOR_ELO = True
USE_ROLLING_AVG_3 = True
USE_ROLLING_AVG_5 = True
USE_ROLLING_AVG_10 = True
USE_DNF_RATE = True
USE_QUALI_TEAMMATE_DELTA = True
USE_TRACK_AVG = True
USE_CONSTRUCTOR_POINTS = True
USE_CONSTRUCTOR_RELIABILITY = True
USE_REGULATION_CONFIDENCE = True

# LGBMRanker hyperparameters
LEARNING_RATE = 0.05
NUM_LEAVES = 31
N_ESTIMATORS = 200
MIN_CHILD_SAMPLES = 5

# Cold-start decay
COLD_START_MIN_WEIGHT = 0.3
COLD_START_FULL_RACE = 10


# === IMPLEMENTATION ===

def get_feature_columns() -> list[str]:
    """Return list of feature columns based on configuration flags."""
    features = []
    if USE_GRID_POSITION: features.append("GridPosition")
    if USE_QUALIFYING_POSITION: features.append("QualifyingPosition")
    if USE_DRIVER_ELO: features.append("driver_elo")
    if USE_CONSTRUCTOR_ELO: features.append("constructor_elo")
    if USE_ROLLING_AVG_3: features.append("driver_rolling_avg_3")
    if USE_ROLLING_AVG_5: features.append("driver_rolling_avg_5")
    if USE_ROLLING_AVG_10: features.append("driver_rolling_avg_10")
    if USE_DNF_RATE: features.append("driver_dnf_rate")
    if USE_QUALI_TEAMMATE_DELTA: features.append("quali_teammate_delta")
    if USE_TRACK_AVG: features.append("driver_track_avg")
    if USE_CONSTRUCTOR_POINTS: features.append("constructor_rolling_points")
    if USE_CONSTRUCTOR_RELIABILITY: features.append("constructor_reliability")
    if USE_REGULATION_CONFIDENCE: features.append("regulation_confidence")
    return features


def train_and_predict(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train on training data and predict on holdout races.

    This is the function that prepare.py calls. Its signature is fixed.
    The agent modifies the CONFIGURATION section above and the
    implementation details below.

    Args:
        train_df: Historical race data for training
        holdout_df: Holdout race data to predict

    Returns:
        DataFrame with Abbreviation, Year, RoundNumber, PredictedPosition
    """
    # Compute Elo ratings with current K-factor schedule
    all_data = pd.concat([train_df, holdout_df], ignore_index=True)
    all_data = compute_elo_ratings(
        all_data,
        k_standard=K_STANDARD,
        k_high=K_HIGH,
        k_medium=K_MEDIUM,
    )

    # Split back
    train = all_data[all_data["Year"] < holdout_df["Year"].min()].copy()
    holdout = all_data[all_data["Year"] >= holdout_df["Year"].min()].copy()

    # Prepare features
    feature_cols = get_feature_columns()
    available = [f for f in feature_cols if f in train.columns]

    X_train = train[available].copy()
    for col in available:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med if not np.isnan(med) else 0)

    y_train = 21 - train["FinishPosition"].values
    groups_train = train.groupby(["Year", "RoundNumber"]).size().values

    # Train LGBMRanker
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 5, 10],
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        n_estimators=N_ESTIMATORS,
        min_child_samples=MIN_CHILD_SAMPLES,
        verbose=-1,
    )
    model.fit(X_train.values, y_train, group=groups_train)

    # Predict each holdout race
    all_predictions = []
    for (year, rnd), race in holdout.groupby(["Year", "RoundNumber"]):
        X_race = race[available].copy()
        for col in available:
            med = X_race[col].median()
            X_race[col] = X_race[col].fillna(med if not np.isnan(med) else 0)

        scores = model.predict(X_race.values)
        race = race.copy()
        race["PredictionScore"] = scores
        race = race.sort_values("PredictionScore", ascending=False).reset_index(drop=True)
        race["PredictedPosition"] = range(1, len(race) + 1)
        all_predictions.append(race[["Abbreviation", "Year", "RoundNumber", "PredictedPosition"]])

    return pd.concat(all_predictions, ignore_index=True)
