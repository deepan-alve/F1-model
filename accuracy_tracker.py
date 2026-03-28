"""
Season-long accuracy tracker.

Logs model predictions and actual results after each race.
Computes Spearman correlation and top-3 accuracy over the season.
Outputs a formatted season report for the group chat.

Usage:
    python accuracy_tracker.py --log --race "Bahrain Grand Prix" --year 2026
    python accuracy_tracker.py --report
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RESULTS_DIR = Path(__file__).parent / "data" / "results"


def log_prediction(
    race_name: str,
    year: int,
    predictions: pd.DataFrame,
    actuals: pd.DataFrame | None = None,
):
    """
    Log a prediction (and optionally actual results) for a race.

    Args:
        race_name: Name of the race
        year: Race year
        predictions: DataFrame with Abbreviation, PredictedPosition, Confidence
        actuals: DataFrame with Abbreviation, FinishPosition (None if race hasn't happened)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "race_name": race_name,
        "year": year,
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions[["Abbreviation", "PredictedPosition"]].to_dict(orient="records"),
    }

    if "Confidence" in predictions.columns:
        conf_map = dict(zip(predictions["Abbreviation"], predictions["Confidence"]))
        for p in entry["predictions"]:
            p["Confidence"] = conf_map.get(p["Abbreviation"], 0)

    if actuals is not None:
        entry["actuals"] = actuals[["Abbreviation", "FinishPosition"]].to_dict(orient="records")
        entry["spearman"] = _compute_race_spearman(predictions, actuals)
        entry["top3_correct"] = _compute_top3_accuracy(predictions, actuals)

    filename = f"{year}_{race_name.lower().replace(' ', '_')}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2)

    print(f"Logged prediction for {race_name} {year} -> {filepath}")


def update_with_actuals(race_name: str, year: int, actuals: pd.DataFrame):
    """
    Update a previously logged prediction with actual race results.

    Args:
        race_name: Name of the race
        year: Race year
        actuals: DataFrame with Abbreviation, FinishPosition
    """
    filename = f"{year}_{race_name.lower().replace(' ', '_')}.json"
    filepath = RESULTS_DIR / filename

    if not filepath.exists():
        print(f"No prediction found for {race_name} {year}")
        return

    with open(filepath) as f:
        entry = json.load(f)

    entry["actuals"] = actuals[["Abbreviation", "FinishPosition"]].to_dict(orient="records")

    preds = pd.DataFrame(entry["predictions"])
    entry["spearman"] = _compute_race_spearman(preds, actuals)
    entry["top3_correct"] = _compute_top3_accuracy(preds, actuals)

    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2)

    print(f"Updated {race_name} {year} with actual results. Spearman: {entry['spearman']:.3f}")


def _compute_race_spearman(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Compute Spearman correlation for a single race."""
    merged = predictions.merge(actuals, on="Abbreviation", how="inner")
    if len(merged) < 3:
        return 0.0

    corr, _ = spearmanr(merged["PredictedPosition"], merged["FinishPosition"])
    return float(corr) if not np.isnan(corr) else 0.0


def _compute_top3_accuracy(predictions: pd.DataFrame, actuals: pd.DataFrame) -> int:
    """Count how many of the predicted top-3 actually finished in the top-3."""
    pred_top3 = set(predictions.nsmallest(3, "PredictedPosition")["Abbreviation"])
    actual_top3 = set(actuals.nsmallest(3, "FinishPosition")["Abbreviation"])
    return len(pred_top3 & actual_top3)


def load_season_results(year: int) -> list[dict]:
    """Load all logged results for a season."""
    if not RESULTS_DIR.exists():
        return []

    results = []
    for filepath in sorted(RESULTS_DIR.glob(f"{year}_*.json")):
        with open(filepath) as f:
            results.append(json.load(f))

    return results


def generate_season_report(year: int) -> str:
    """
    Generate a season accuracy report formatted for group chat.

    Returns a formatted string showing per-race and aggregate accuracy.
    """
    results = load_season_results(year)

    if not results:
        return f"No predictions logged for {year} yet."

    lines = []
    lines.append(f"F1 {year} Season - Model Accuracy Report")
    lines.append("\u2501" * 55)

    races_with_results = [r for r in results if "spearman" in r]
    races_pending = [r for r in results if "spearman" not in r]

    if races_with_results:
        lines.append(f"{'Race':<30} {'Spearman':>8} {'Top-3':>6}")
        lines.append("-" * 50)

        spearmans = []
        top3s = []

        for r in races_with_results:
            sp = r["spearman"]
            t3 = r.get("top3_correct", 0)
            spearmans.append(sp)
            top3s.append(t3)
            lines.append(f"{r['race_name']:<30} {sp:>8.3f} {t3:>4}/3")

        lines.append("-" * 50)
        avg_sp = np.mean(spearmans)
        avg_t3 = np.mean(top3s)
        lines.append(f"{'SEASON AVERAGE':<30} {avg_sp:>8.3f} {avg_t3:>4.1f}/3")

        lines.append("")
        lines.append(f"Races evaluated: {len(races_with_results)}")

        # Rating
        if avg_sp >= 0.7:
            lines.append("Rating: STRONG - model is beating the grid-position baseline")
        elif avg_sp >= 0.5:
            lines.append("Rating: DECENT - model adds value over random")
        else:
            lines.append("Rating: NEEDS WORK - autoresearch time")

    if races_pending:
        lines.append("")
        lines.append(f"Pending results: {', '.join(r['race_name'] for r in races_pending)}")

    lines.append("\u2501" * 55)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="F1 Accuracy Tracker")
    parser.add_argument("--report", action="store_true", help="Show season accuracy report")
    parser.add_argument("--year", type=int, default=2026, help="Season year")

    args = parser.parse_args()

    if args.report:
        print(generate_season_report(args.year))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
