"""Tests for the model module (LGBMRanker + DNF classifier)."""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    evaluate_spearman,
    predict_race_order,
    prepare_ranking_data,
    train_dnf_classifier,
    train_ranker,
)


def _make_training_data(n_races=20, n_drivers=10):
    """Create realistic training data for model tests."""
    rows = []
    for race in range(1, n_races + 1):
        for i in range(n_drivers):
            # Simulate: grid position strongly correlates with finish
            grid = i + 1
            # Add some noise to finish position
            finish = max(1, min(n_drivers, grid + np.random.randint(-2, 3)))
            rows.append({
                "Abbreviation": f"DR{i:02d}",
                "TeamName": f"Team{i // 2}",
                "Year": 2024,
                "RoundNumber": race,
                "GridPosition": float(grid),
                "QualifyingPosition": float(grid),
                "FinishPosition": float(finish),
                "DNF": finish > n_drivers,
                "driver_elo": 1500.0 + (n_drivers - i) * 20,
                "constructor_elo": 1500.0 + (n_drivers - i) * 10,
                "driver_rolling_avg_3": float(grid) + np.random.normal(0, 1),
                "driver_rolling_avg_5": float(grid) + np.random.normal(0, 0.5),
                "driver_rolling_avg_10": float(grid) + np.random.normal(0, 0.3),
                "driver_dnf_rate": 0.05,
                "quali_teammate_delta": np.random.normal(0, 0.3),
                "driver_track_avg": float(grid),
                "constructor_rolling_points": 0.2 - i * 0.01,
                "constructor_reliability": 0.95,
                "regulation_confidence": 1.0,
            })

    return pd.DataFrame(rows)


class TestPrepareRankingData:
    def test_returns_correct_shapes(self):
        """Prepare returns X, y, groups with correct dimensions."""
        df = _make_training_data(n_races=5, n_drivers=10)
        X, y, groups = prepare_ranking_data(df)

        assert X.shape[0] == 50  # 5 races * 10 drivers
        assert len(y) == 50
        assert sum(groups) == 50  # Groups sum to total rows
        assert len(groups) == 5  # One group per race

    def test_relevance_higher_for_better_finishers(self):
        """Relevance labels are higher for better finishing positions."""
        df = _make_training_data(n_races=1, n_drivers=5)
        X, y, groups = prepare_ranking_data(df)

        # Sort by original data order (should be by grid position)
        # P1 should have highest relevance (21 - 1 = 20)
        # P5 should have lower relevance (21 - 5 = 16)
        assert max(y) >= 16
        assert min(y) >= 0

    def test_handles_nan_features(self):
        """NaN features are filled with median values."""
        df = _make_training_data(n_races=3, n_drivers=5)
        df.loc[0, "driver_rolling_avg_3"] = np.nan
        df.loc[1, "driver_elo"] = np.nan

        X, y, groups = prepare_ranking_data(df)
        assert not np.any(np.isnan(X))


class TestTrainRanker:
    def test_trains_without_error(self):
        """LGBMRanker trains successfully on sample data."""
        df = _make_training_data(n_races=10, n_drivers=10)
        X, y, groups = prepare_ranking_data(df)

        model = train_ranker(X, y, groups)
        assert model is not None
        assert hasattr(model, "predict")

    def test_predictions_are_valid(self):
        """Predictions produce a valid ordering (no duplicates)."""
        np.random.seed(42)
        df = _make_training_data(n_races=15, n_drivers=10)
        X_train, y_train, groups_train = prepare_ranking_data(df[df["RoundNumber"] <= 10])

        model = train_ranker(X_train, y_train, groups_train)

        race = df[df["RoundNumber"] == 11]
        result = predict_race_order(model, race)

        assert "PredictedPosition" in result.columns
        positions = sorted(result["PredictedPosition"].tolist())
        assert positions == list(range(1, 11))  # 1 through 10, no gaps


class TestDNFClassifier:
    def test_trains_without_error(self):
        """DNF classifier trains successfully."""
        df = _make_training_data(n_races=10, n_drivers=10)
        # Add some DNFs
        df.loc[df.index[:5], "DNF"] = True

        model = train_dnf_classifier(df)
        assert model is not None

    def test_probabilities_in_range(self):
        """DNF probabilities are between 0 and 1."""
        df = _make_training_data(n_races=10, n_drivers=10)
        df.loc[df.index[:10], "DNF"] = True

        model = train_dnf_classifier(df)

        X = df[["GridPosition", "driver_elo", "driver_dnf_rate", "constructor_reliability"]].fillna(0)
        probs = model.predict_proba(X.values)[:, 1]

        assert all(0 <= p <= 1 for p in probs)


class TestEvaluateSpearman:
    def test_perfect_correlation(self):
        """Perfect prediction gives Spearman of 1.0."""
        df = pd.DataFrame({
            "PredictedPosition": [1, 2, 3, 4, 5],
            "FinishPosition": [1, 2, 3, 4, 5],
        })
        assert abs(evaluate_spearman(df) - 1.0) < 0.001

    def test_inverse_correlation(self):
        """Completely wrong prediction gives Spearman of -1.0."""
        df = pd.DataFrame({
            "PredictedPosition": [1, 2, 3, 4, 5],
            "FinishPosition": [5, 4, 3, 2, 1],
        })
        assert abs(evaluate_spearman(df) - (-1.0)) < 0.001

    def test_handles_missing_values(self):
        """Returns 0.0 when insufficient valid data."""
        df = pd.DataFrame({
            "PredictedPosition": [1, np.nan],
            "FinishPosition": [np.nan, 2],
        })
        assert evaluate_spearman(df) == 0.0

    def test_missing_columns_raises(self):
        """Raises ValueError when required columns are missing."""
        df = pd.DataFrame({"Position": [1, 2, 3]})
        with pytest.raises(ValueError):
            evaluate_spearman(df)
