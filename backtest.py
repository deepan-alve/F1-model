"""
Full historical backtest across all seasons.
For each year, train on all previous years and predict every race.
Report per-race and per-season accuracy.
"""
import pandas as pd
import numpy as np
from src.features import build_feature_matrix
from src.elo import compute_elo_ratings
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/processed/all_races.parquet")
df = build_feature_matrix(df)
df = compute_elo_ratings(df)

FEATURES = [
    "GridPosition","QualifyingPosition","driver_elo","constructor_elo",
    "driver_rolling_avg_3","driver_rolling_avg_5","driver_rolling_avg_10",
    "driver_dnf_rate","quali_teammate_delta","driver_track_avg",
    "constructor_rolling_points","constructor_reliability","regulation_confidence",
    "weather_temp_max","weather_rain_mm","weather_wind_max",
    "num_pit_stops","used_soft","used_medium","used_hard",
    "used_intermediate","used_wet","driver_avg_pit_stops",
    "speed_trap","car_speed_delta","car_aero_balance",
    "car_rolling_speed","car_rolling_aero","team_avg_speed","pu_encoded",
]
available = [f for f in FEATURES if f in df.columns]

# Best hyperparameters from optimization
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "learning_rate": 0.1,
    "num_leaves": 63,
    "n_estimators": 300,
    "min_child_samples": 5,
    "verbose": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

def prepare(data, features):
    data = data.copy().sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    data = data.dropna(subset=["FinishPosition"])
    X = data[features].copy()
    for col in features:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    y = (21 - data["FinishPosition"].values).astype(int)
    groups = data.groupby(["Year", "RoundNumber"]).size().values
    return X.values, y, groups

# ==========================================
# BACKTEST: For each year, train on all prior years
# ==========================================
test_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
all_results = []
season_summaries = []

print("=" * 75)
print("  FULL HISTORICAL BACKTEST")
print("  Train on past years, predict each race of the next year")
print("=" * 75)

for test_year in test_years:
    train = df[df["Year"] < test_year]
    test = df[df["Year"] == test_year]

    if train.empty or test.empty:
        continue

    # Need minimum training data
    train_races = train.groupby(["Year", "RoundNumber"]).ngroups
    if train_races < 10:
        continue

    X_tr, y_tr, g_tr = prepare(train, available)
    model = LGBMRanker(**PARAMS)
    model.fit(X_tr, y_tr, group=g_tr)

    race_results = []
    for (year, rnd), race in test.groupby(["Year", "RoundNumber"]):
        X_race = race[available].copy()
        for col in available:
            med = X_race[col].median()
            X_race[col] = X_race[col].fillna(med if not np.isnan(med) else 0)

        scores = model.predict(X_race.values)
        result = race.copy()
        result["PredictionScore"] = scores
        result = result.sort_values("PredictionScore", ascending=False).reset_index(drop=True)
        result["PredictedPosition"] = range(1, len(result) + 1)

        # Merge actuals
        actuals = race[["Abbreviation", "FinishPosition"]].copy()
        result = result.drop(columns=["FinishPosition"], errors="ignore")
        result = result.merge(actuals, on="Abbreviation", how="left")

        valid = result.dropna(subset=["PredictedPosition", "FinishPosition"])
        if len(valid) < 3:
            continue

        corr, _ = spearmanr(valid["PredictedPosition"], valid["FinishPosition"])
        if np.isnan(corr):
            continue

        winner_pred = result.iloc[0]["Abbreviation"]
        winner_actual = valid.nsmallest(1, "FinishPosition")["Abbreviation"].iloc[0]

        top3_pred = set(result.nsmallest(3, "PredictedPosition")["Abbreviation"])
        top3_actual = set(valid.nsmallest(3, "FinishPosition")["Abbreviation"])

        top5_pred = set(result.nsmallest(5, "PredictedPosition")["Abbreviation"])
        top5_actual = set(valid.nsmallest(5, "FinishPosition")["Abbreviation"])

        result["pos_error"] = abs(result["PredictedPosition"] - result["FinishPosition"])
        avg_err = result["pos_error"].mean()
        within2 = (result["pos_error"] <= 2).sum()
        exact = (result["pos_error"] == 0).sum()

        event = race["EventName"].iloc[0]
        race_results.append({
            "year": year, "round": rnd, "event": event,
            "spearman": corr,
            "winner_correct": winner_pred == winner_actual,
            "top3_overlap": len(top3_pred & top3_actual),
            "top5_overlap": len(top5_pred & top5_actual),
            "avg_error": avg_err,
            "within2": within2,
            "exact": exact,
            "n_drivers": len(valid),
        })

    if not race_results:
        continue

    rr = pd.DataFrame(race_results)
    all_results.append(rr)

    # Season summary
    avg_spearman = rr["spearman"].mean()
    winner_pct = rr["winner_correct"].mean() * 100
    avg_top3 = rr["top3_overlap"].mean()
    avg_top5 = rr["top5_overlap"].mean()
    avg_within2 = rr["within2"].mean()
    avg_exact = rr["exact"].mean()
    avg_error = rr["avg_error"].mean()
    n_races = len(rr)
    is_reg_change = test_year in {2022, 2026}

    season_summaries.append({
        "year": test_year,
        "races": n_races,
        "spearman": avg_spearman,
        "winner_pct": winner_pct,
        "top3": avg_top3,
        "top5": avg_top5,
        "within2": avg_within2,
        "exact": avg_exact,
        "avg_error": avg_error,
        "reg_change": is_reg_change,
        "train_races": train_races,
    })

    print(f"\n{'='*75}")
    reg_tag = " ** REGULATION CHANGE **" if is_reg_change else ""
    print(f"  {test_year} SEASON{reg_tag}")
    print(f"  Trained on: {train_races} races ({train['Year'].min()}-{train['Year'].max()})")
    print(f"  Tested on:  {n_races} races")
    print(f"{'='*75}")
    print(f"  {'Race':<35} {'Spearman':>8} {'Winner':>7} {'Top3':>5} {'Top5':>5} {'AvgErr':>7}")
    print(f"  {'-'*70}")

    for _, r in rr.iterrows():
        name = r["event"][:33]
        sp = r["spearman"]
        win = "YES" if r["winner_correct"] else "no"
        t3 = f"{r['top3_overlap']}/3"
        t5 = f"{r['top5_overlap']}/5"
        err = f"{r['avg_error']:.1f}"
        print(f"  {name:<35} {sp:>8.3f} {win:>7} {t3:>5} {t5:>5} {err:>7}")

    print(f"  {'-'*70}")
    print(f"  {'SEASON AVERAGE':<35} {avg_spearman:>8.3f} {winner_pct:>6.0f}% {avg_top3:>4.1f}/3 {avg_top5:>4.1f}/5 {avg_error:>7.1f}")

# ==========================================
# OVERALL SUMMARY
# ==========================================
all_df = pd.concat(all_results, ignore_index=True)
ss = pd.DataFrame(season_summaries)

print(f"\n\n{'='*75}")
print(f"  OVERALL MODEL PERFORMANCE SUMMARY")
print(f"  Total: {len(all_df)} races backtested across {len(ss)} seasons")
print(f"{'='*75}")
print(f"\n  {'Season':<8} {'Races':>6} {'Spearman':>9} {'Winner%':>8} {'Top3':>6} {'Top5':>6} {'AvgErr':>7} {'Note'}")
print(f"  {'-'*65}")
for _, s in ss.iterrows():
    note = "REG CHANGE" if s["reg_change"] else ""
    print(f"  {int(s['year']):<8} {int(s['races']):>6} {s['spearman']:>9.3f} {s['winner_pct']:>7.0f}% {s['top3']:>5.1f}/3 {s['top5']:>5.1f}/5 {s['avg_error']:>7.1f}  {note}")

print(f"  {'-'*65}")
overall_sp = all_df["spearman"].mean()
overall_win = all_df["winner_correct"].mean() * 100
overall_t3 = all_df["top3_overlap"].mean()
overall_t5 = all_df["top5_overlap"].mean()
overall_err = all_df["avg_error"].mean()
print(f"  {'OVERALL':<8} {len(all_df):>6} {overall_sp:>9.3f} {overall_win:>7.0f}% {overall_t3:>5.1f}/3 {overall_t5:>5.1f}/5 {overall_err:>7.1f}")

print(f"\n  KEY METRICS FOR PRESENTATION:")
print(f"  ================================")
print(f"  Average Spearman Correlation:  {overall_sp:.3f}")
print(f"  Winner Prediction Accuracy:    {overall_win:.1f}%")
print(f"  Top-3 Accuracy:                {overall_t3:.1f}/3 ({overall_t3/3*100:.0f}%)")
print(f"  Top-5 Accuracy:                {overall_t5:.1f}/5 ({overall_t5/5*100:.0f}%)")
print(f"  Average Position Error:        {overall_err:.1f} places")
print(f"  Races Backtested:              {len(all_df)}")
print(f"  Seasons Covered:               {len(ss)} (2020-2026)")

# Best and worst races
best = all_df.nlargest(5, "spearman")
worst = all_df.nsmallest(5, "spearman")
print(f"\n  TOP 5 BEST PREDICTED RACES:")
for _, r in best.iterrows():
    print(f"    {r['event']:<35} {int(r['year'])} Spearman={r['spearman']:.3f} Winner:{'YES' if r['winner_correct'] else 'no'}")

print(f"\n  TOP 5 WORST PREDICTED RACES:")
for _, r in worst.iterrows():
    print(f"    {r['event']:<35} {int(r['year'])} Spearman={r['spearman']:.3f} Winner:{'YES' if r['winner_correct'] else 'no'}")

# Regulation change vs normal
reg_races = all_df[all_df["year"].isin([2022, 2026])]
normal_races = all_df[~all_df["year"].isin([2022, 2026])]
if not reg_races.empty and not normal_races.empty:
    print(f"\n  REGULATION CHANGE IMPACT:")
    print(f"    Normal years Spearman:     {normal_races['spearman'].mean():.3f} ({len(normal_races)} races)")
    print(f"    Reg change years Spearman: {reg_races['spearman'].mean():.3f} ({len(reg_races)} races)")
    print(f"    Difference:                {normal_races['spearman'].mean() - reg_races['spearman'].mean():.3f}")

# Safety car impact proxy: races with DNFs tend to be more chaotic
print(f"\n{'='*75}")
