# F1 Race Prediction Model

**Predict Formula 1 race finishing order with engineered features, adaptive Elo ratings, and a LightGBM ranker.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Ranker-3CB371)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![FastF1](https://img.shields.io/badge/FastF1-Data-E10600)](https://docs.fastf1.dev)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

The model treats each race as a **learning-to-rank** problem rather than position regression: a LightGBM ranker scores every entrant, and the predicted finishing order is the score-sorted list. Driver and team strength are tracked over time with an adaptive **Elo rating system** that adjusts after every race. A second classifier estimates **DNF probability** so the predicted ranking can downweight likely retirements.

Every training run, backtest, and prediction is logged to **MLflow** — parameters, metrics, feature importance, predictions, and the serialized model — so each run is reproducible and comparable.

<!-- accuracy-start -->
## Live model accuracy

_Auto-updated by [.github/workflows/race-update.yml](.github/workflows/race-update.yml). Pre-race prediction generated Saturday 23:00 UTC after qualifying; race scored Monday 12:00 UTC._

### 2026 season — 4 race(s) scored

**Latest:** Miami Grand Prix (Round 4) — Spearman **0.720**, top-3 **1/3**.

| Mean Spearman | Mean Top-3 (out of 3) | Rating |
|---|---|---|
| 0.728 | 1.75 | STRONG |

### Latest race — Miami Grand Prix: predicted vs actual

| Pos | Predicted | Actual | Hit |
|---|---|---|---|
| 1 | ANT | ANT | ✓ |
| 2 | VER | NOR |   |
| 3 | LEC | PIA |   |
| 4 | RUS | RUS | ✓ |
| 5 | NOR | VER |   |
| 6 | PIA | HAM |   |
| 7 | HAM | COL |   |
| 8 | GAS | LEC |   |
| 9 | HUL | SAI |   |
| 10 | BEA | ALB |   |

**Per-driver delta** (sorted by actual finish; positive Δ = model placed them lower than they finished):

| Driver | Predicted | Actual | Δ |
|---|---|---|---|
| ANT | 1 | 1 | 0 |
| NOR | 5 | 2 | +3 |
| PIA | 6 | 3 | +3 |
| RUS | 4 | 4 | 0 |
| VER | 2 | 5 | -3 |
| HAM | 7 | 6 | +1 |
| COL | 13 | 7 | +6 |
| LEC | 3 | 8 | -5 |
| SAI | 15 | 9 | +6 |
| ALB | 16 | 10 | +6 |

### Per-race results

| Round | Race | Spearman | Top-3 | Predicted P1 → P3 | Actual P1 → P3 |
|---|---|---|---|---|---|
| 1 | Australian Grand Prix | 0.676 | 2/3 | LEC → RUS → PIA | RUS → ANT → LEC |
| 2 | Chinese Grand Prix | 0.622 | 2/3 | RUS → LEC → ANT | ANT → RUS → HAM |
| 3 | Japanese Grand Prix | 0.894 | 2/3 | ANT → RUS → LEC | ANT → PIA → LEC |
| 4 | Miami Grand Prix | 0.720 | 1/3 | ANT → VER → LEC | ANT → NOR → PIA |
<!-- accuracy-end -->

<!-- next-race-start -->
## Next race prediction

**2026 Canadian Grand Prix — Round 5**

_Prediction will appear here after qualifying._
<!-- next-race-end -->

## Pipelines

| Script                  | Purpose                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `experiments/prepare.py`| Holdout experiment — train on pre-2024 data, evaluate on 2024.       |
| `experiments/train.py`  | Tunable training script (hyperparameter search target).              |
| `backtest.py`           | Season-by-season historical backtest with per-race / per-season metrics. |
| `run_final.py`          | "Everything model" — adds car-performance + power-unit features and the DNF classifier. |
| `predict.py`            | Race-day prediction flow + 2024 holdout validation path.             |
| `optimize.py` / `optimize_v2.py` | Hyperparameter optimization wrappers.                       |
| `accuracy_tracker.py`   | Trend tracking across runs to surface regression vs. baseline.       |
| `build_features.py`     | One-shot build of `data/processed/features.parquet`.                 |

## How it works

```
                     ┌──────────────────────┐
   FastF1 API ──────►│  src/data_pipeline   │── pre-cleaned race results
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │  src/features        │── per-race feature matrix
                     │  src/elo             │── adaptive driver / team Elo
                     │  src/odds            │── implied-probability features
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
   LightGBM ranker ◄─┤  src/model           │── learning-to-rank
   DNF classifier  ◄─┤                      │── auxiliary head
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │  MLflow tracking     │── params, metrics, model, predictions
                     └──────────────────────┘
```

The Elo update is online: after each historical race, every driver's and constructor's rating is bumped based on actual vs. expected finishing position. By race day, the model sees an Elo state that reflects all preceding form, not just season averages.

## MLflow experiments

Runs land in four named experiments:

| Experiment           | Logged from        | What's tracked                                |
| -------------------- | ------------------ | --------------------------------------------- |
| `f1-experiments`     | `experiments/prepare.py` | Holdout split, feature list, Elo settings, ranker hyperparams, mean Spearman |
| `f1-backtest`        | `backtest.py`      | Per-race + per-season Spearman, top-N hit rate |
| `f1-final-model`     | `run_final.py`     | Full model w/ car perf + DNF, feature importance, prediction artifacts |
| `f1-prediction`      | `predict.py`       | Held-out test metrics + race-day prediction snapshots |

Two registered models: **`f1-ranker`** and **`f1-dnf-classifier`**.

## Project layout

```
F1-model/
├── src/
│   ├── data_pipeline.py    # FastF1 data loading + cleaning
│   ├── features.py         # Feature engineering (track, weather, recent form)
│   ├── elo.py              # Adaptive driver + team Elo ratings
│   ├── odds.py             # Implied-probability features from odds data
│   ├── model.py            # LightGBM ranker + DNF classifier wrappers
│   └── tracking.py         # MLflow helpers (run naming, metric logging)
├── experiments/
│   ├── prepare.py          # IMMUTABLE scoring script
│   ├── train.py            # Tunable training script
│   └── program.md          # Notes on the experiment loop
├── tests/                  # Pytest suite — one file per src module
├── data/
│   ├── tracks.csv          # Static reference: circuits
│   ├── power_units.csv     # Static reference: PU manufacturers per season
│   ├── raw/                # FastF1 cache (gitignored)
│   └── processed/          # Built feature parquet (gitignored)
├── notebooks/              # Exploratory analysis
├── backtest.py             # Historical backtesting workflow
├── predict.py              # Race-day prediction
├── run_final.py            # End-to-end "everything" model
├── optimize.py             # Hyperparameter sweeps
├── optimize_v2.py          # Iteration on optimize.py
├── build_features.py       # Pre-compute features parquet
├── accuracy_tracker.py     # Cross-run trend tracking
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/deepan-alve/F1-model.git
cd F1-model
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Build the pre-computed features parquet (one-time)
python build_features.py

# 2. Run the holdout experiment (logs to MLflow)
python experiments/prepare.py

# 3. Inspect runs
mlflow ui
```

To use a custom MLflow backend instead of the local `mlruns/` folder:

```bash
export MLFLOW_TRACKING_URI=postgresql://...     # or http://your-mlflow-server
```

## Running tests

```bash
pytest                  # full suite
pytest -k elo           # just Elo tests
pytest --cov=src        # with coverage
```

## Continuous race updates

`.github/workflows/race-update.yml` runs on two crons (or on demand via *Run workflow*):

| Cron               | What it does                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sat 23:00 UTC**  | Re-trains and refreshes the **next-race prediction** using the latest available data — including that day's qualifying. The README's *Next race prediction* section updates with the real grid. |
| **Mon 12:00 UTC**  | Scores the race that just finished against the prediction made before it, logs the result via `accuracy_tracker.log_prediction`, then refreshes the next-race prediction again.                |

Each run:

1. Refreshes FastF1 historical data (cached between runs).
2. **If a new race finished** since the last run: trains the ranker on data *excluding* that race, predicts it honestly, fetches actuals, and writes `data/results/{year}_{race}.json`.
3. **Always**: re-trains on data through the most recent completed race and predicts the next upcoming round into `data/results/upcoming_prediction.json`.
4. Regenerates the *Live model accuracy* and *Next race prediction* sections of this README via `scripts/update_readme.py`.
5. Commits any changes back to `main`.

Run it manually in the [Actions tab](../../actions/workflows/race-update.yml).

## License

[GPL-3.0-or-later](LICENSE) — see [NOTICE](NOTICE) for attribution. Copyright (C) 2026 Deepan Alve
