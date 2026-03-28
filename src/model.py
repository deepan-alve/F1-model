"""
F1 prediction model using LGBMRanker + DNF classifier.

Two-stage prediction:
1. DNF probability (LGBMClassifier)
2. Finishing order (LGBMRanker with LambdaRank)

Bootstrap confidence computed only during prediction (50 samples),
not during autoresearch experiments.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMClassifier
from scipy.stats import spearmanr


# Features used by the model. Add/remove here to change the feature set.
RANKING_FEATURES = [
    "GridPosition",
    "QualifyingPosition",
    "driver_elo",
    "constructor_elo",
    "driver_rolling_avg_3",
    "driver_rolling_avg_5",
    "driver_rolling_avg_10",
    "driver_dnf_rate",
    "quali_teammate_delta",
    "driver_track_avg",
    "constructor_rolling_points",
    "constructor_reliability",
    "regulation_confidence",
]

DNF_FEATURES = [
    "GridPosition",
    "driver_elo",
    "driver_dnf_rate",
    "constructor_reliability",
]


def prepare_ranking_data(
    df: pd.DataFrame,
    feature_cols: list[str] = RANKING_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for LGBMRanker.

    Args:
        df: Feature-enriched race data

    Returns:
        (X, y, groups) where:
        - X: feature matrix
        - y: relevance labels (inverse of finish position, higher = better)
        - groups: number of drivers per race (for query grouping)
    """
    df = df.copy()

    # Only use rows that have core features
    available_features = [f for f in feature_cols if f in df.columns]

    # Fill NaN with median for features (LGBMRanker doesn't handle NaN natively in ranking mode)
    X = df[available_features].copy()
    for col in available_features:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    # Relevance: higher is better. Max position (21 for DNF) minus actual position.
    y = 21 - df["FinishPosition"].values

    # Groups: number of drivers per race
    groups = df.groupby(["Year", "RoundNumber"]).size().values

    return X.values, y, groups


def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: dict | None = None,
) -> LGBMRanker:
    """
    Train the LGBMRanker model.

    Args:
        X: Feature matrix
        y: Relevance labels
        groups: Query group sizes
        params: LGBMRanker parameters (optional)

    Returns:
        Trained LGBMRanker model
    """
    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 200,
        "min_child_samples": 5,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    model = LGBMRanker(**default_params)
    model.fit(X, y, group=groups)
    return model


def train_dnf_classifier(
    df: pd.DataFrame,
    feature_cols: list[str] = DNF_FEATURES,
) -> LGBMClassifier:
    """
    Train the DNF probability classifier.

    Args:
        df: Feature-enriched race data with DNF column
        feature_cols: Features for DNF prediction

    Returns:
        Trained LGBMClassifier model
    """
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].copy()
    for col in available_features:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    y = df["DNF"].astype(int).values

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        verbose=-1,
    )
    model.fit(X.values, y)
    return model


def predict_race_order(
    model: LGBMRanker,
    race_features: pd.DataFrame,
    feature_cols: list[str] = RANKING_FEATURES,
) -> pd.DataFrame:
    """
    Predict the finishing order for a single race.

    Args:
        model: Trained LGBMRanker
        race_features: Feature data for all drivers in one race
        feature_cols: Feature columns to use

    Returns:
        DataFrame sorted by predicted finish position with PredictedPosition column
    """
    available_features = [f for f in feature_cols if f in race_features.columns]
    X = race_features[available_features].copy()
    for col in available_features:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    scores = model.predict(X.values)
    result = race_features.copy()
    result["PredictionScore"] = scores
    result = result.sort_values("PredictionScore", ascending=False).reset_index(drop=True)
    result["PredictedPosition"] = range(1, len(result) + 1)
    return result


def predict_with_confidence(
    df_train: pd.DataFrame,
    race_features: pd.DataFrame,
    n_bootstrap: int = 50,
    feature_cols: list[str] = RANKING_FEATURES,
) -> pd.DataFrame:
    """
    Predict finishing order with bootstrap confidence estimates.

    Trains n_bootstrap models on different bootstrap samples of the
    training data. Confidence = inverse of prediction variance.

    Only used in predict.py, NOT during autoresearch experiments.

    Args:
        df_train: Training data
        race_features: Features for the race to predict
        n_bootstrap: Number of bootstrap samples
        feature_cols: Feature columns to use

    Returns:
        DataFrame with PredictedPosition and Confidence columns
    """
    n_drivers = len(race_features)
    position_matrix = np.zeros((n_bootstrap, n_drivers))

    for i in range(n_bootstrap):
        # Bootstrap sample of training races
        races = df_train.groupby(["Year", "RoundNumber"])
        race_keys = list(races.groups.keys())
        sampled_keys = [race_keys[j] for j in np.random.choice(len(race_keys), len(race_keys), replace=True)]
        sampled_dfs = [races.get_group(k) for k in sampled_keys]
        bootstrap_df = pd.concat(sampled_dfs, ignore_index=True)

        X_train, y_train, groups_train = prepare_ranking_data(bootstrap_df, feature_cols)
        model = train_ranker(X_train, y_train, groups_train)

        result = predict_race_order(model, race_features, feature_cols)

        # Store predicted positions indexed by driver abbreviation
        for _, row in result.iterrows():
            driver_idx = race_features.index.get_loc(
                race_features[race_features["Abbreviation"] == row["Abbreviation"]].index[0]
            )
            position_matrix[i, driver_idx] = row["PredictedPosition"]

    # Compute mean position and confidence
    mean_positions = position_matrix.mean(axis=0)
    std_positions = position_matrix.std(axis=0)
    max_std = n_drivers / 2  # Theoretical max std for uniform distribution
    confidence = np.clip(1.0 - (std_positions / max_std), 0.0, 1.0) * 100

    result = race_features.copy()
    result["MeanPredictedPosition"] = mean_positions
    result["Confidence"] = confidence
    result = result.sort_values("MeanPredictedPosition").reset_index(drop=True)
    result["PredictedPosition"] = range(1, len(result) + 1)

    return result


def evaluate_spearman(predictions: pd.DataFrame) -> float:
    """
    Compute Spearman rank correlation between predicted and actual positions.

    Args:
        predictions: DataFrame with PredictedPosition and FinishPosition columns

    Returns:
        Spearman correlation coefficient (-1 to 1, higher is better)
    """
    if "PredictedPosition" not in predictions.columns or "FinishPosition" not in predictions.columns:
        raise ValueError("DataFrame must have PredictedPosition and FinishPosition columns")

    valid = predictions.dropna(subset=["PredictedPosition", "FinishPosition"])
    if len(valid) < 3:
        return 0.0

    corr, _ = spearmanr(valid["PredictedPosition"], valid["FinishPosition"])
    return float(corr) if not np.isnan(corr) else 0.0
