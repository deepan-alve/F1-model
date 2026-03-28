# F1 Prediction Model - Autoresearch Program

## Objective

Maximize Spearman rank correlation between predicted and actual finishing
positions on the 2024 holdout races.

## What you can modify

Only `train.py`. Never modify `prepare.py`.

## Search space

Explore these dimensions, one at a time:

### 1. Elo K-factor schedule
- K_STANDARD: try 8, 16, 24, 32
- K_HIGH: try 32, 48, 64, 80 (for regulation change years)
- K_MEDIUM: try 16, 24, 32, 48

### 2. Feature selection
- Toggle USE_* flags. Start by removing one feature at a time
  and checking if Spearman improves (ablation study).
- Try removing correlated features (rolling_avg_3 vs rolling_avg_5 vs rolling_avg_10).

### 3. LGBMRanker hyperparameters
- LEARNING_RATE: try 0.01, 0.03, 0.05, 0.1
- NUM_LEAVES: try 15, 31, 63
- N_ESTIMATORS: try 100, 200, 500
- MIN_CHILD_SAMPLES: try 3, 5, 10, 20

### 4. Cold-start decay
- COLD_START_MIN_WEIGHT: try 0.1, 0.3, 0.5
- COLD_START_FULL_RACE: try 5, 8, 10, 15

## Constraints

- Each experiment must complete in under 5 minutes.
- Do not install new packages.
- Do not modify prepare.py.
- Apply simplicity criterion: a tiny improvement that adds 50 lines
  of tangled code isn't worth keeping.
- Keep the `train_and_predict` function signature unchanged.

## Current baseline

Run `python prepare.py` to see the current score with default config.
