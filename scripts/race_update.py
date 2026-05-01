"""
Race update orchestrator.

Two crons in .github/workflows/race-update.yml hit this script:

  Saturday 23:00 UTC — refresh the upcoming-race prediction with the
                       latest qualifying data (and any other newer rows).
  Monday   12:00 UTC — score the just-completed race against the
                       prediction we made before it, then refresh the
                       upcoming prediction again.

Each invocation:
  1. Loads / refreshes the cached parquet of historical race data.
  2. Builds features + Elo ratings (same pipeline as predict.py).
  3. If a new race finished since the last run, scores it via a
     model trained on data STRICTLY BEFORE the race (chronological
     split — no leakage).
  4. Re-predicts the next upcoming race using a model trained on
     every completed race so far.
  5. Writes data/results/{year}_{race}.json (per-race log) and
     data/results/upcoming_prediction.json.

Idempotent. Idempotency key: data/results/last_processed.json.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

# Make src/ + accuracy_tracker.py importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    DNF_FEATURES,
    RANKING_FEATURES,
    predict_race_order,
    predict_with_confidence,
    prepare_ranking_data,
    train_dnf_classifier,
    train_ranker,
)

import accuracy_tracker

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAST_PROCESSED_FILE = RESULTS_DIR / "last_processed.json"
UPCOMING_FILE = RESULTS_DIR / "upcoming_prediction.json"

# How far back to backfill on a cold cache. Ergast (FastF1's data backend)
# rate-limits at 500 calls/hour. 2024+ stays well within budget; subsequent
# runs use FastF1's HTTP cache (persisted via actions/cache) so this only
# matters on the very first run.
HISTORY_START_YEAR = int(os.environ.get("RACE_UPDATE_START_YEAR", "2024"))

# Bootstrap sample count for confidence estimates. predict.py uses 50;
# we keep it conservative for CI runtime budget.
N_BOOTSTRAP = int(os.environ.get("RACE_UPDATE_N_BOOTSTRAP", "30"))

PARQUET_NAME = "all_races.parquet"


# ---------------------------------------------------------------------------
# Schedule helpers (FastF1)
# ---------------------------------------------------------------------------
def current_year() -> int:
    return dt.datetime.now(dt.timezone.utc).year


def get_schedule(year: int) -> pd.DataFrame:
    enable_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if "Session5DateUtc" in schedule.columns:
        schedule["RaceEndUtc"] = pd.to_datetime(schedule["Session5DateUtc"], utc=True)
    elif "EventDate" in schedule.columns:
        schedule["RaceEndUtc"] = pd.to_datetime(schedule["EventDate"], utc=True)
    else:
        raise RuntimeError("FastF1 schedule has no recognizable race-end column.")
    return schedule


def find_latest_completed(schedule: pd.DataFrame) -> pd.Series | None:
    now = pd.Timestamp.now(tz="UTC")
    completed = schedule[schedule["RaceEndUtc"] < now]
    return completed.iloc[-1] if len(completed) else None


def find_next_upcoming(schedule: pd.DataFrame) -> pd.Series | None:
    now = pd.Timestamp.now(tz="UTC")
    upcoming = schedule[schedule["RaceEndUtc"] >= now]
    return upcoming.iloc[0] if len(upcoming) else None


def race_json_path(year: int, race_name: str) -> Path:
    """Match accuracy_tracker.log_prediction's filename convention."""
    slug = race_name.lower().replace(" ", "_")
    return RESULTS_DIR / f"{year}_{slug}.json"


def find_unscored_completed(schedule: pd.DataFrame, year: int) -> list[pd.Series]:
    """All completed races (chronological order) that don't yet have a JSON."""
    now = pd.Timestamp.now(tz="UTC")
    completed = schedule[schedule["RaceEndUtc"] < now]
    unscored: list[pd.Series] = []
    for _, race in completed.iterrows():
        if not race_json_path(year, str(race["EventName"])).exists():
            unscored.append(race)
    return unscored


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_last_processed() -> dict:
    if LAST_PROCESSED_FILE.exists():
        return json.loads(LAST_PROCESSED_FILE.read_text())
    return {"year": None, "round": None}


def save_last_processed(year: int, round_num: int, race_name: str) -> None:
    LAST_PROCESSED_FILE.write_text(
        json.dumps(
            {
                "year": int(year),
                "round": int(round_num),
                "race_name": race_name,
                "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Data loading + feature pipeline (mirrors predict.py)
# ---------------------------------------------------------------------------
def load_and_refresh(year: int) -> pd.DataFrame:
    """
    Load the parquet cache (or build it from scratch), incrementally refresh
    the current season, and return the feature-engineered + Elo-rated DataFrame.
    """
    enable_cache()
    all_data = load_from_parquet(PARQUET_NAME)

    if all_data is None or all_data.empty:
        print(
            f"[data] No cached parquet — fetching {HISTORY_START_YEAR}–{year - 1} "
            f"from FastF1 (slow on first run)..."
        )
        all_data = fetch_historical_data(HISTORY_START_YEAR, year - 1)
        save_to_parquet(all_data, PARQUET_NAME)

    print(f"[data] Refreshing {year} season (incremental)...")
    all_data = refresh_current_season(year, all_data)
    save_to_parquet(all_data, PARQUET_NAME)

    print("[data] Building feature matrix...")
    featured = build_feature_matrix(all_data)

    print("[data] Computing Elo ratings...")
    featured = compute_elo_ratings(featured)

    return featured


def attach_dnf_probabilities(predictions: pd.DataFrame, history: pd.DataFrame, race_data: pd.DataFrame) -> pd.DataFrame:
    """Train DNF classifier on history and append a DNFProbability column."""
    dnf_model = train_dnf_classifier(history)
    if dnf_model is None:
        predictions["DNFProbability"] = 0.05
        return predictions

    dnf_features = [f for f in DNF_FEATURES if f in race_data.columns]
    X_dnf = race_data[dnf_features].fillna(0)
    abbreviation_to_dnf = dict(zip(race_data["Abbreviation"], dnf_model.predict_proba(X_dnf.values)[:, 1]))
    predictions["DNFProbability"] = predictions["Abbreviation"].map(abbreviation_to_dnf).fillna(0.05)
    return predictions


# ---------------------------------------------------------------------------
# Scoring + upcoming prediction
# ---------------------------------------------------------------------------
def score_completed_race(featured: pd.DataFrame, year: int, latest: pd.Series) -> bool:
    """
    Train on data STRICTLY BEFORE the latest race, predict it, fetch the
    actuals, and log via accuracy_tracker. Returns True if a prediction was
    written, False if skipped.
    """
    race_name = str(latest["EventName"])
    round_number = int(latest["RoundNumber"])

    # Locate the target race rows (fuzzy on event name like predict.py does).
    race_mask = (
        featured["EventName"].str.contains(race_name, case=False, na=False)
        & (featured["Year"] == year)
    )
    if race_mask.sum() == 0:
        print(f"[score] {race_name} not in dataset yet — FastF1 may not have published results.")
        return False

    race_data = featured[race_mask]

    # Strict chronological history: everything before this race.
    history = featured[
        (featured["Year"] < year)
        | ((featured["Year"] == year) & (featured["RoundNumber"] < round_number))
    ]
    if len(history) < 100:
        print(f"[score] Only {len(history)} historical rows — too thin to train; skipping.")
        return False

    # Need finished positions to score; if FinishPosition is all NaN, the race
    # results haven't propagated yet.
    if race_data["FinishPosition"].isna().all():
        print(f"[score] {race_name} has no FinishPosition yet — actuals not available.")
        return False

    print(f"[score] Bootstrap-predicting {race_name} (n={N_BOOTSTRAP})...")
    predictions = predict_with_confidence(history, race_data, n_bootstrap=N_BOOTSTRAP)
    predictions = attach_dnf_probabilities(predictions, history, race_data)

    actuals = race_data[["Abbreviation", "FinishPosition"]].copy()

    # Two-call flow: log_prediction stores the prediction JSON (using only
    # Abbreviation / PredictedPosition / Confidence columns), then
    # update_with_actuals reads that slim JSON back, computes Spearman, and
    # writes actuals. Doing this in one combined call would feed _compute_-
    # race_spearman a DataFrame that still carries FinishPosition from the
    # source feature matrix, which collides with actuals during merge.
    print(f"[score] Logging {race_name} prediction via accuracy_tracker...")
    accuracy_tracker.log_prediction(race_name, year, predictions)

    print(f"[score] Updating {race_name} with actual results...")
    accuracy_tracker.update_with_actuals(race_name, year, actuals)

    save_last_processed(year, round_number, race_name)
    return True


def refresh_upcoming_prediction(featured: pd.DataFrame, year: int, upcoming: pd.Series) -> bool:
    """
    Predict the next upcoming race using everything completed so far (training
    set = all races where FinishPosition is known).

    Returns True if a prediction was written.
    """
    up_name = str(upcoming["EventName"])
    up_round = int(upcoming["RoundNumber"])

    race_mask = (
        featured["EventName"].str.contains(up_name, case=False, na=False)
        & (featured["Year"] == year)
    )
    if race_mask.sum() == 0:
        print(f"[upcoming] {up_name} not in dataset yet (no qualifying / entry list).")
        return False

    race_data = featured[race_mask]

    # Train on everything that has a known finish (i.e. completed races only).
    history = featured[featured["FinishPosition"].notna()]
    if len(history) < 100:
        print(f"[upcoming] Only {len(history)} completed rows — too thin to train; skipping.")
        return False

    print(f"[upcoming] Bootstrap-predicting {up_name} (n={N_BOOTSTRAP})...")
    predictions = predict_with_confidence(history, race_data, n_bootstrap=N_BOOTSTRAP)
    predictions = attach_dnf_probabilities(predictions, history, race_data)

    cols = [
        c
        for c in ["Abbreviation", "TeamName", "PredictedPosition", "Confidence", "DNFProbability"]
        if c in predictions.columns
    ]

    UPCOMING_FILE.write_text(
        json.dumps(
            {
                "year": year,
                "round": up_round,
                "race_name": up_name,
                "race_date_utc": str(upcoming["RaceEndUtc"]),
                "predictions": predictions[cols].to_dict(orient="records"),
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "training_rows": int(len(history)),
            },
            indent=2,
            default=str,
        )
    )
    print(f"[upcoming] Wrote {UPCOMING_FILE}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    year = current_year()
    print(f"[race_update] Season: {year}")

    schedule = get_schedule(year)
    latest = find_latest_completed(schedule)
    upcoming = find_next_upcoming(schedule)

    print(
        f"[race_update] Latest completed: "
        f"{(latest['EventName'] + ' (R' + str(int(latest['RoundNumber'])) + ')') if latest is not None else 'none'}"
    )
    print(
        f"[race_update] Next upcoming:    "
        f"{(upcoming['EventName'] + ' (R' + str(int(upcoming['RoundNumber'])) + ')') if upcoming is not None else 'none'}"
    )

    # Bail before expensive data loading if there's literally nothing to do.
    if latest is None and upcoming is None:
        print("[race_update] Nothing to score, nothing to predict.")
        return 0

    featured = load_and_refresh(year)

    # 1. Score every completed race that doesn't yet have a JSON in
    #    data/results/. On the first real run this backfills the whole
    #    season; subsequent runs only score the newly-finished race.
    #    FORCE_REPROCESS=true re-scores every completed race regardless.
    force = os.environ.get("FORCE_REPROCESS", "false").lower() == "true"

    if force:
        now = pd.Timestamp.now(tz="UTC")
        races_to_score = [r for _, r in schedule[schedule["RaceEndUtc"] < now].iterrows()]
    else:
        races_to_score = find_unscored_completed(schedule, year)

    if not races_to_score:
        print("[race_update] No new completed races to score.")
    else:
        names = ", ".join(str(r["EventName"]) for r in races_to_score)
        print(f"[race_update] Scoring {len(races_to_score)} race(s): {names}")
        for race in races_to_score:
            score_completed_race(featured, year, race)

    # 2. Always refresh the upcoming-race prediction with the latest data.
    if upcoming is not None:
        refresh_upcoming_prediction(featured, year, upcoming)

    print("[race_update] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
