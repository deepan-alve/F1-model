"""
IMMUTABLE evaluation script for experiments.

DO NOT MODIFY THIS FILE during experiment runs.
This is the scoring function that the experiment uses to evaluate experiments.

It loads pre-computed features, splits into temporal train/holdout,
and evaluates the model trained by train.py using Spearman correlation.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "features.parquet"

# Temporal split: train on 2019-2023, holdout on 2024
HOLDOUT_YEAR = 2024

from src.tracking import (
    build_run_name,
    configure_mlflow,
    log_dataframe_artifact,
    log_json_artifact,
    log_params,
    set_run_context,
)


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

    This is THE metric that the experiment optimizes.

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
    configure_mlflow("f1-experiments")
    train_df, holdout_df = load_data()

    # Import train.py's model training and prediction functions
    # train.py is agent-modifiable, so its interface may change
    try:
        import train as train_module
        from train import train_and_predict
    except ImportError:
        print("ERROR: Could not import train_and_predict from train.py")
        print("0.0")
        return

    with mlflow.start_run(run_name=build_run_name("experiments", "holdout", HOLDOUT_YEAR)) as run:
        set_run_context(
            component="experiments",
            script="experiments/prepare.py",
            stage="evaluation",
            dataset="data/processed/features.parquet",
            split=f"train<{HOLDOUT_YEAR},holdout={HOLDOUT_YEAR}",
            notes="Base holdout evaluation pipeline",
        )
        mlflow.set_tag("holdout_year", str(HOLDOUT_YEAR))
        log_params({
            "features_parquet": str(FEATURES_PARQUET),
            "holdout_year": HOLDOUT_YEAR,
            "train_rows": len(train_df),
            "holdout_rows": len(holdout_df),
            "train_races": int(train_df.groupby(["Year", "RoundNumber"]).ngroups),
            "holdout_races": int(holdout_df.groupby(["Year", "RoundNumber"]).ngroups),
            "feature_columns": train_module.get_feature_columns(),
            "k_standard": train_module.K_STANDARD,
            "k_high": train_module.K_HIGH,
            "k_medium": train_module.K_MEDIUM,
            "cold_start_min_weight": train_module.COLD_START_MIN_WEIGHT,
            "cold_start_full_race": train_module.COLD_START_FULL_RACE,
            "learning_rate": train_module.LEARNING_RATE,
            "num_leaves": train_module.NUM_LEAVES,
            "n_estimators": train_module.N_ESTIMATORS,
            "min_child_samples": train_module.MIN_CHILD_SAMPLES,
        })

        # train.py must implement: train_and_predict(train_df, holdout_df) -> predictions_df
        # predictions_df must have: Abbreviation, Year, RoundNumber, PredictedPosition
        predictions = train_and_predict(train_df, holdout_df)
        score = evaluate(predictions, holdout_df)

        merged = predictions.merge(
            holdout_df[["Abbreviation", "Year", "RoundNumber", "EventName", "FinishPosition"]],
            on=["Abbreviation", "Year", "RoundNumber"],
            how="inner",
        )
        per_race = []
        for (year, rnd), race in merged.groupby(["Year", "RoundNumber"]):
            if len(race) < 3:
                continue
            corr, _ = spearmanr(race["PredictedPosition"], race["FinishPosition"])
            if np.isnan(corr):
                continue
            per_race.append({
                "Year": year,
                "RoundNumber": rnd,
                "EventName": race["EventName"].iloc[0],
                "Spearman": float(corr),
                "Drivers": len(race),
            })

        mlflow.log_metric("mean_holdout_spearman", score)
        mlflow.log_metric("prediction_rows", len(predictions))

        log_dataframe_artifact(predictions, "tables", "holdout_predictions.csv")
        log_dataframe_artifact(merged, "tables", "holdout_predictions_with_actuals.csv")
        if per_race:
            log_dataframe_artifact(pd.DataFrame(per_race), "tables", "per_race_holdout_metrics.csv")

        log_json_artifact(
            {
                "run_id": run.info.run_id,
                "holdout_year": HOLDOUT_YEAR,
                "score": score,
                "feature_columns": train_module.get_feature_columns(),
            },
            "reports",
            "holdout_summary.json",
        )

        train = pd.concat([train_df, holdout_df], ignore_index=True)
        available = [f for f in train_module.get_feature_columns() if f in train.columns]
        model_training_df = train[train["Year"] < HOLDOUT_YEAR].copy()
        X_train = model_training_df[available].copy()
        for col in available:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med if not np.isnan(med) else 0)
        y_train = 21 - model_training_df["FinishPosition"].values
        groups_train = model_training_df.groupby(["Year", "RoundNumber"]).size().values
        ranker = train_module.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[3, 5, 10],
            learning_rate=train_module.LEARNING_RATE,
            num_leaves=train_module.NUM_LEAVES,
            n_estimators=train_module.N_ESTIMATORS,
            min_child_samples=train_module.MIN_CHILD_SAMPLES,
            verbose=-1,
        )
        ranker.fit(X_train.values, y_train, group=groups_train)
        mlflow.sklearn.log_model(ranker, name="ranker", registered_model_name="f1-ranker")

    # Print score as last line (the experiment runner reads this)
    print(f"Spearman correlation on {HOLDOUT_YEAR} holdout: {score:.4f}")
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
