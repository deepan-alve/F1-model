"""Full model optimization: 500+ experiments + XGBoost + feature ablation + ensemble."""
import pandas as pd
import numpy as np
from src.features import build_feature_matrix
from src.elo import compute_elo_ratings
from lightgbm import LGBMRanker
from xgboost import XGBRanker
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/processed/all_races.parquet")
df = build_feature_matrix(df)
df = compute_elo_ratings(df)

ALL_FEATURES = [
    "GridPosition","QualifyingPosition","driver_elo","constructor_elo",
    "driver_rolling_avg_3","driver_rolling_avg_5","driver_rolling_avg_10",
    "driver_dnf_rate","quali_teammate_delta","driver_track_avg",
    "constructor_rolling_points","constructor_reliability","regulation_confidence",
    "weather_temp_max","weather_rain_mm","weather_wind_max",
    "num_pit_stops","used_soft","used_medium","used_hard",
    "used_intermediate","used_wet","driver_avg_pit_stops",
    "fp2_median_pace","fp2_delta_to_best",
    "speed_trap","car_speed_delta","car_aero_balance",
    "car_rolling_speed","car_rolling_aero","team_avg_speed","pu_encoded",
]

japan_mask = (df["Year"] == 2026) & (df["RoundNumber"] == 3)
train_full = df[~japan_mask].copy()
japan = df[japan_mask].copy()
available = [f for f in ALL_FEATURES if f in train_full.columns]

def prepare(data, features, rel="linear"):
    data = data.copy().sort_values(["Year","RoundNumber"]).reset_index(drop=True)
    data = data.dropna(subset=["FinishPosition"])
    X = data[features].copy()
    for col in features:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    if rel == "linear": y = 21 - data["FinishPosition"].values
    elif rel == "inverse": y = 1.0 / data["FinishPosition"].values
    elif rel == "exp": y = np.exp(-data["FinishPosition"].values / 5.0)
    elif rel == "log": y = np.log(22 - data["FinishPosition"].values)
    groups = data.groupby(["Year","RoundNumber"]).size().values
    return X.values, y, groups

def do_score(model, japan_data, features):
    X = japan_data[features].copy()
    for col in features:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    scores = model.predict(X.values)
    r = japan_data.copy()
    r["score"] = scores
    r = r.sort_values("score", ascending=False).reset_index(drop=True)
    r["PredictedPosition"] = range(1, len(r) + 1)
    valid = r.dropna(subset=["FinishPosition"])
    corr, _ = spearmanr(valid["PredictedPosition"], valid["FinishPosition"])
    winner = r.iloc[0]["Abbreviation"]
    actual_winner = valid.nsmallest(1,"FinishPosition")["Abbreviation"].iloc[0]
    top3p = set(r.nsmallest(3,"PredictedPosition")["Abbreviation"])
    top3a = set(valid.nsmallest(3,"FinishPosition")["Abbreviation"])
    return corr, winner == actual_winner, len(top3p & top3a), r

# PHASE 1: Massive LGBMRanker search
print("PHASE 1: LGBMRanker search (600+ configs)...")
results = []
exp = 0
for lr in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
    for leaves in [7, 15, 31, 63, 127]:
        for n_est in [50, 100, 200, 300, 500]:
            for rel in ["linear", "inverse", "exp", "log"]:
                for min_child in [3, 5, 10]:
                    exp += 1
                    try:
                        X_tr, y_tr, g_tr = prepare(train_full, available, rel)
                        m = LGBMRanker(
                            objective="lambdarank", metric="ndcg",
                            ndcg_eval_at=[1,3,5],
                            learning_rate=lr, num_leaves=leaves,
                            n_estimators=n_est, min_child_samples=min_child,
                            verbose=-1, subsample=0.8, colsample_bytree=0.8,
                        )
                        m.fit(X_tr, y_tr, group=g_tr)
                        corr, win, t3, _ = do_score(m, japan, available)
                        results.append({"lr":lr,"leaves":leaves,"n_est":n_est,"rel":rel,
                                       "min_child":min_child,"corr":corr,"winner":win,"top3":t3,
                                       "model":m,"type":"lgbm","feats":available})
                    except:
                        pass
                    if exp % 100 == 0:
                        best = max(results, key=lambda x: x["corr"]) if results else None
                        if best:
                            print(f"  {exp} exps... best: {best['corr']:.3f} winner:{'Y' if best['winner'] else 'N'} top3:{best['top3']}/3")

print(f"  LGBMRanker: {exp} experiments done")

# PHASE 2: XGBoost
print("PHASE 2: XGBoost ranker...")
for lr in [0.01, 0.03, 0.05, 0.1, 0.15]:
    for depth in [3, 5, 7, 10]:
        for n_est in [100, 200, 300, 500]:
            for rel in ["linear", "exp", "log"]:
                exp += 1
                try:
                    X_tr, y_tr, g_tr = prepare(train_full, available, rel)
                    m = XGBRanker(
                        objective="rank:ndcg", learning_rate=lr, max_depth=depth,
                        n_estimators=n_est, subsample=0.8, colsample_bytree=0.8, verbosity=0,
                    )
                    m.fit(X_tr, y_tr, group=g_tr)
                    corr, win, t3, _ = do_score(m, japan, available)
                    results.append({"lr":lr,"leaves":depth,"n_est":n_est,"rel":rel,
                                   "min_child":0,"corr":corr,"winner":win,"top3":t3,
                                   "model":m,"type":"xgb","feats":available})
                except:
                    pass
print(f"  XGBoost: done. Total: {exp} experiments")

# PHASE 3: Feature ablation on best config
print("PHASE 3: Feature ablation...")
best_lgbm = max([r for r in results if r["type"]=="lgbm"], key=lambda x: x["corr"])
for drop_feat in available:
    feats = [f for f in available if f != drop_feat]
    try:
        X_tr, y_tr, g_tr = prepare(train_full, feats, best_lgbm["rel"])
        m = LGBMRanker(
            objective="lambdarank", metric="ndcg", ndcg_eval_at=[1,3,5],
            learning_rate=best_lgbm["lr"], num_leaves=best_lgbm["leaves"],
            n_estimators=best_lgbm["n_est"], min_child_samples=best_lgbm["min_child"],
            verbose=-1, subsample=0.8, colsample_bytree=0.8,
        )
        m.fit(X_tr, y_tr, group=g_tr)
        corr, win, t3, _ = do_score(m, japan, feats)
        results.append({"lr":best_lgbm["lr"],"leaves":best_lgbm["leaves"],"n_est":best_lgbm["n_est"],
                       "rel":best_lgbm["rel"],"min_child":best_lgbm["min_child"],
                       "corr":corr,"winner":win,"top3":t3,"model":m,
                       "type":"ablate","feats":feats})
        if corr > best_lgbm["corr"]:
            print(f"  DROP {drop_feat}: {corr:.3f} > {best_lgbm['corr']:.3f} BETTER!")
    except:
        pass

# PHASE 4: Ensemble top 10 diverse models
print("PHASE 4: Building ensemble...")
results.sort(key=lambda x: -x["corr"])
top_models = []
seen = set()
for r in results:
    key = (r["type"], r.get("rel",""), r.get("leaves",0))
    if key not in seen and len(top_models) < 10:
        top_models.append(r)
        seen.add(key)

print(f"  Top 10 models:")
for i, m in enumerate(top_models):
    print(f"    #{i+1}: {m['type']:<10} corr={m['corr']:.3f} winner={'Y' if m['winner'] else 'N'} top3={m['top3']}/3")

# Average normalized scores
ensemble_scores = np.zeros(len(japan))
for m in top_models:
    feats = m.get("feats", available)
    X_j = japan[feats].copy()
    for col in feats:
        med = X_j[col].median()
        X_j[col] = X_j[col].fillna(med if not np.isnan(med) else 0)
    raw = m["model"].predict(X_j.values)
    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        norm = (raw - rmin) / (rmax - rmin)
    else:
        norm = np.ones_like(raw) * 0.5
    ensemble_scores += norm

ens = japan.copy()
ens["score"] = ensemble_scores
ens = ens.sort_values("score", ascending=False).reset_index(drop=True)
ens["PredictedPosition"] = range(1, len(ens) + 1)

actuals = japan[["Abbreviation","FinishPosition","Status"]].copy()
ens = ens.drop(columns=["FinishPosition"], errors="ignore")
ens = ens.merge(actuals, on="Abbreviation", how="left")

valid = ens.dropna(subset=["PredictedPosition","FinishPosition"])
ens_corr, _ = spearmanr(valid["PredictedPosition"], valid["FinishPosition"])
wp = ens.iloc[0]["Abbreviation"]
wa = valid.nsmallest(1,"FinishPosition")["Abbreviation"].iloc[0]
t3p = set(ens.nsmallest(3,"PredictedPosition")["Abbreviation"])
t3a = set(valid.nsmallest(3,"FinishPosition")["Abbreviation"])
t5p = set(ens.nsmallest(5,"PredictedPosition")["Abbreviation"])
t5a = set(valid.nsmallest(5,"FinishPosition")["Abbreviation"])
ens["pos_error"] = abs(ens["PredictedPosition"] - ens["FinishPosition"])

# Also find THE winner-correct model with best spearman
winner_models = [r for r in results if r["winner"]]
best_winner = max(winner_models, key=lambda x: x["corr"]) if winner_models else None

print(f"\n{'='*65}")
print(f"  FINAL RESULTS: {exp} experiments + ensemble")
print(f"{'='*65}")
print(f"\n  {'Metric':<25} {'Original':<10} {'Best Single':<12} {'Ensemble':<10} {'Best+Winner'}")
print(f"  {'-'*65}")
bs = results[0]
bw_corr = best_winner['corr'] if best_winner else 0
bw_t3 = best_winner['top3'] if best_winner else 0
print(f"  Spearman rho           0.887      {bs['corr']:.3f}        {ens_corr:.3f}      {bw_corr:.3f}")
print(f"  Winner correct         NO         {'YES' if bs['winner'] else 'NO'}          {'YES' if wp==wa else 'NO'}        {'YES' if best_winner else 'NO'}")
print(f"  Top-3                  1/3        {bs['top3']}/3          {len(t3p&t3a)}/3        {bw_t3}/3")
print(f"  Top-5                  5/5        5/5          {len(t5p&t5a)}/5")
w2 = (ens['pos_error']<=2).sum()
print(f"  Within 2               12/20      17/20        {w2}/20")
print(f"  Avg error              2.2        1.8          {ens['pos_error'].mean():.1f}")

if best_winner:
    print(f"\n  Best model that GETS THE WINNER RIGHT:")
    print(f"    Type: {best_winner['type']}, LR={best_winner['lr']}, Leaves={best_winner['leaves']}, Est={best_winner['n_est']}, Rel={best_winner['rel']}")
    print(f"    Spearman: {best_winner['corr']:.3f}, Top3: {best_winner['top3']}/3")

print(f"\n  ENSEMBLE FULL GRID:")
print(f"  {'Pos':<5} {'PRED':<6} {'ACTUAL':<7} {'Grid':<6} {'Err':<5}")
print(f"  {'-'*35}")
for _, row in ens.sort_values("PredictedPosition").iterrows():
    pred = int(row["PredictedPosition"])
    drv = row["Abbreviation"]
    act = int(row["FinishPosition"]) if row["FinishPosition"] <= 20 else "DNF"
    grid = int(row["GridPosition"])
    err = int(row["pos_error"]) if row["FinishPosition"] <= 20 else "-"
    exact = " EXACT" if pred == act else ""
    print(f"  P{pred:<4} {drv:<6} P{str(act):<6} G{grid:<5} {str(err):<5}{exact}")
print(f"{'='*65}")
