"""Tests for the accuracy tracker module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accuracy_tracker import (
    _compute_race_spearman,
    _compute_top3_accuracy,
    generate_season_report,
    load_season_results,
    log_prediction,
)


class TestSpearmanComputation:
    def test_perfect_prediction(self):
        preds = pd.DataFrame({"Abbreviation": ["A", "B", "C"], "PredictedPosition": [1, 2, 3]})
        actuals = pd.DataFrame({"Abbreviation": ["A", "B", "C"], "FinishPosition": [1, 2, 3]})
        assert abs(_compute_race_spearman(preds, actuals) - 1.0) < 0.01

    def test_bad_prediction(self):
        preds = pd.DataFrame({"Abbreviation": ["A", "B", "C"], "PredictedPosition": [1, 2, 3]})
        actuals = pd.DataFrame({"Abbreviation": ["A", "B", "C"], "FinishPosition": [3, 2, 1]})
        assert _compute_race_spearman(preds, actuals) < 0


class TestTop3Accuracy:
    def test_perfect_top3(self):
        preds = pd.DataFrame({"Abbreviation": ["A", "B", "C", "D"], "PredictedPosition": [1, 2, 3, 4]})
        actuals = pd.DataFrame({"Abbreviation": ["A", "B", "C", "D"], "FinishPosition": [1, 2, 3, 4]})
        assert _compute_top3_accuracy(preds, actuals) == 3

    def test_zero_top3(self):
        preds = pd.DataFrame({"Abbreviation": ["A", "B", "C", "D", "E", "F"], "PredictedPosition": [1, 2, 3, 4, 5, 6]})
        actuals = pd.DataFrame({"Abbreviation": ["A", "B", "C", "D", "E", "F"], "FinishPosition": [4, 5, 6, 1, 2, 3]})
        # Predicted top 3: A, B, C. Actual top 3: D, E, F. Zero overlap.
        assert _compute_top3_accuracy(preds, actuals) == 0


class TestLogPrediction:
    def test_logs_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("accuracy_tracker.RESULTS_DIR", tmp_path)

        preds = pd.DataFrame({
            "Abbreviation": ["VER", "NOR"],
            "PredictedPosition": [1, 2],
            "Confidence": [85.0, 65.0],
        })

        log_prediction("Bahrain GP", 2026, preds)

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

        with open(files[0]) as f:
            data = json.load(f)

        assert data["race_name"] == "Bahrain GP"
        assert len(data["predictions"]) == 2

    def test_logs_with_actuals(self, tmp_path, monkeypatch):
        monkeypatch.setattr("accuracy_tracker.RESULTS_DIR", tmp_path)

        preds = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "PIA", "HAM"],
            "PredictedPosition": [1, 2, 3, 4],
        })
        actuals = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "PIA", "HAM"],
            "FinishPosition": [1, 2, 3, 4],
        })

        log_prediction("Bahrain GP", 2026, preds, actuals)

        files = list(tmp_path.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)

        assert "spearman" in data
        assert data["spearman"] == 1.0


class TestSeasonReport:
    def test_empty_season(self, tmp_path, monkeypatch):
        monkeypatch.setattr("accuracy_tracker.RESULTS_DIR", tmp_path)
        report = generate_season_report(2026)
        assert "No predictions logged" in report

    def test_report_with_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("accuracy_tracker.RESULTS_DIR", tmp_path)

        # Write a result file
        entry = {
            "race_name": "Bahrain GP",
            "year": 2026,
            "predictions": [{"Abbreviation": "VER", "PredictedPosition": 1}],
            "actuals": [{"Abbreviation": "VER", "FinishPosition": 1}],
            "spearman": 0.85,
            "top3_correct": 2,
        }
        with open(tmp_path / "2026_bahrain_gp.json", "w") as f:
            json.dump(entry, f)

        report = generate_season_report(2026)
        assert "0.850" in report
        assert "Bahrain GP" in report
