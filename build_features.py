"""
Build the pre-computed features parquet file for experiments.

This script:
1. Loads historical data (from cache or fetches it)
2. Computes all features
3. Computes Elo ratings
4. Saves to data/processed/features.parquet

Run this before using experiments:
    python build_features.py
"""

from src.data_pipeline import (
    enable_cache,
    fetch_historical_data,
    load_from_parquet,
    save_to_parquet,
)
from src.elo import compute_elo_ratings
from src.features import build_feature_matrix


def main():
    enable_cache()

    # Load or fetch raw data
    print("Loading raw data...")
    data = load_from_parquet("all_races.parquet")
    if data is None:
        print("No cached data. Fetching 2019-2025 (this takes ~30 min first time)...")
        data = fetch_historical_data(2019, 2025)
        save_to_parquet(data, "all_races.parquet")

    # Build features
    print("Computing features...")
    featured = build_feature_matrix(data)

    # Compute Elo with default K-factors
    print("Computing Elo ratings...")
    featured = compute_elo_ratings(featured)

    # Save for experiments
    save_to_parquet(featured, "features.parquet")
    print("Done. experiments/prepare.py can now load data/processed/features.parquet")


if __name__ == "__main__":
    main()
