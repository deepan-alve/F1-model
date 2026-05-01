"""
Race update orchestrator.

Run weekly (or manually). Detects whether a new F1 race has finished since
the last run, and if so:

  1. Refreshes the historical FastF1 data so the latest race is in the
     dataset (re-trains "the model" implicitly because src.model.train_ranker
     is called with up-to-date data inside generate_prediction()).
  2. Computes the model's predicted finishing order for the just-completed
     race using a model trained on data BEFORE that race (honest scoring).
  3. Logs prediction + actuals via accuracy_tracker.log_prediction so the
     per-race JSON files in data/results/ get created.
  4. Predicts the next upcoming race using a model trained on data INCLUDING
     the just-completed race, and writes data/results/upcoming_prediction.json.
  5. Persists data/results/last_processed.json so the next run skips work
     when nothing new has happened.

Idempotent. Safe to run on a cron.
"""
from __future__ import annotations

import json
import os
import sys
import datetime as dt
from pathlib import Path

import pandas as pd

# Make src/ + accuracy_tracker.py importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fastf1

from src import data_pipeline, features, model
import accuracy_tracker

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAST_PROCESSED_FILE = RESULTS_DIR / "last_processed.json"
UPCOMING_FILE = RESULTS_DIR / "upcoming_prediction.json"

# Enable FastF1 cache (path matches the workflow's actions/cache target).
CACHE_DIR = ROOT / "data" / "raw" / "fastf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def current_year() -> int:
    return dt.datetime.now(dt.timezone.utc).year


def get_schedule(year: int) -> pd.DataFrame:
    """Race-only schedule with a normalized UTC race-end column."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if "Session5DateUtc" in schedule.columns:
        schedule["RaceEndUtc"] = pd.to_datetime(schedule["Session5DateUtc"], utc=True)
    elif "EventDate" in schedule.columns:
        schedule["RaceEndUtc"] = pd.to_datetime(schedule["EventDate"], utc=True)
    else:
        raise RuntimeError("Couldn't find a race date column in FastF1 schedule.")
    return schedule


def find_latest_completed(schedule: pd.DataFrame) -> pd.Series | None:
    now = pd.Timestamp.now(tz="UTC")
    completed = schedule[schedule["RaceEndUtc"] < now]
    return completed.iloc[-1] if len(completed) else None


def find_next_upcoming(schedule: pd.DataFrame) -> pd.Series | None:
    now = pd.Timestamp.now(tz="UTC")
    upcoming = schedule[schedule["RaceEndUtc"] >= now]
    return upcoming.iloc[0] if len(upcoming) else None


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
# Modeling
# ---------------------------------------------------------------------------

def build_training_dataset(through_year: int, through_round: int | None) -> pd.DataFrame:
    """
    Return a feature-engineered dataset including all races up to (and
    including) through_year/through_round.

    If through_round is None, include the entire `through_year` season.
    """
    df = data_pipeline.fetch_historical_data(start_year=2018, end_year=through_year)
    if through_round is not None:
        # Drop everything strictly after the cutoff in the cutoff year.
        cutoff_mask = (df["Year"] == through_year) & (df["RoundNumber"] > through_round)
        df = df[~cutoff_mask].reset_index(drop=True)
    return features.build_feature_matrix(df)


def predict_one_race(train_df: pd.DataFrame, year: int, round_number: int) -> pd.DataFrame:
    """
    Train a ranker on train_df, then score the entrants of the target race
    and return a DataFrame with Abbreviation + PredictedPosition (+ Confidence
    if available).
    """
    target = train_df[(train_df["Year"] == year) & (train_df["RoundNumber"] == round_number)].copy()
    if target.empty:
        raise RuntimeError(f"Target race not in dataset: {year} R{round_number}")

    # Train on everything strictly before the target race.
    history = train_df[
        (train_df["Year"] < year)
        | ((train_df["Year"] == year) & (train_df["RoundNumber"] < round_number))
    ].copy()

    ranker = model.train_ranker(history)
    return model.predict_race_order(ranker, target)


def fetch_actuals(year: int, round_number: int) -> pd.DataFrame:
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=False, telemetry=False, weather=False)
    res = session.results.copy()
    res = res.rename(columns={"Position": "FinishPosition"})
    return res[["Abbreviation", "FinishPosition"]]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def score_completed_race(year: int, latest: pd.Series) -> None:
    """Train a ranker excluding the latest race, score predictions vs actuals."""
    race_name = str(latest["EventName"])
    round_number = int(latest["RoundNumber"])

    print(f"[score] Building training set up to (excluding) {race_name} R{round_number}...")
    if round_number == 1:
        # First race of the season: train on full prior season.
        train_excl = build_training_dataset(through_year=year - 1, through_round=None)
    else:
        train_excl = build_training_dataset(through_year=year, through_round=round_number - 1)

    # NOTE: predict_one_race re-trains internally on data strictly before the target.
    # We pass the full dataset so the target race rows are present to score, but
    # the ranker is fit only on history.
    print(f"[score] Predicting {race_name}...")
    full_dataset = build_training_dataset(through_year=year, through_round=round_number)
    predictions = predict_one_race(full_dataset, year, round_number)

    print(f"[score] Fetching actuals for {race_name}...")
    actuals = fetch_actuals(year, round_number)

    print(f"[score] Logging {race_name} predictions + actuals...")
    accuracy_tracker.log_prediction(race_name, year, predictions, actuals=actuals)

    save_last_processed(year, round_number, race_name)


def refresh_upcoming_prediction(year: int, upcoming: pd.Series, latest: pd.Series | None) -> None:
    """
    (Re-)generate the prediction for the next upcoming race using the freshest
    available training data — including any qualifying / sprint sessions that
    may have happened since the previous run.
    """
    up_name = str(upcoming["EventName"])
    up_round = int(upcoming["RoundNumber"])

    # Train on every race up to (and including) the most recently completed one.
    if latest is not None:
        train_through_round = int(latest["RoundNumber"])
        train_through_year = year
    else:
        train_through_round = None
        train_through_year = year - 1

    print(f"[upcoming] Building training set through {train_through_year} R{train_through_round}...")
    full_dataset = build_training_dataset(through_year=train_through_year, through_round=train_through_round)

    print(f"[upcoming] Predicting {up_name} (R{up_round})...")
    try:
        upcoming_predictions = predict_one_race(full_dataset, year, up_round)
    except RuntimeError:
        print(f"[upcoming] {up_name} not yet in dataset — skipping (this is normal if FastF1 hasn't published the entry list yet).")
        return

    cols = [c for c in ["Abbreviation", "PredictedPosition", "Confidence"] if c in upcoming_predictions.columns]
    UPCOMING_FILE.write_text(
        json.dumps(
            {
                "year": year,
                "round": up_round,
                "race_name": up_name,
                "race_date_utc": str(upcoming["RaceEndUtc"]),
                "predictions": upcoming_predictions[cols].to_dict(orient="records"),
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    print(f"[upcoming] Wrote {UPCOMING_FILE}")


def main() -> int:
    year = current_year()
    print(f"[race_update] Season: {year}")

    schedule = get_schedule(year)
    latest = find_latest_completed(schedule)
    upcoming = find_next_upcoming(schedule)

    state = load_last_processed()
    force = os.environ.get("FORCE_REPROCESS", "false").lower() == "true"

    # 1. Score the most recent completed race iff it's new since the last run.
    if latest is not None:
        already_scored = (
            state.get("year") == year and state.get("round") == int(latest["RoundNumber"])
        )
        if force or not already_scored:
            score_completed_race(year, latest)
        else:
            print(f"[race_update] {latest['EventName']} already scored — skipping post-race step.")
    else:
        print("[race_update] No completed races yet this season.")

    # 2. ALWAYS refresh the upcoming-race prediction with the latest available data.
    #    This is the bit that picks up Saturday's qualifying results when the
    #    workflow runs Sat 23:00 UTC — the prediction in the README updates with
    #    real grid info before the race.
    if upcoming is not None:
        refresh_upcoming_prediction(year, upcoming, latest)
    else:
        print("[race_update] No upcoming race on the schedule.")

    print("[race_update] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
