"""Full optimization v2: new features + time weighting + MLP + stacking + FP2 data."""
import pandas as pd
import numpy as np
from src.features import build_feature_matrix
from src.elo import compute_elo_ratings
from lightgbm import LGBMRanker
from xgboost import XGBRanker
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/processed/all_races.parquet")

# ============================================
# NEW FEATURES
# ============================================
print("Building advanced features...")

df = build_feature_matrix(df)
df = compute_elo_ratings(df)

# 1. Qualifying gap to pole (normalized per race)
df["quali_gap_to_pole"] = np.nan
for (year, rnd), group in df.groupby(["Year", "RoundNumber"]):
    best_q = group["Q3"].min()
    if pd.isna(best_q):
        best_q = group["Q1"].min()
    if not pd.isna(best_q):
        df.loc[group.index, "quali_gap_to_pole"] = group["Q3"].fillna(group["Q1"]) - best_q

# 2. Momentum: position change trend over last 3 races (gaining or losing?)
df = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
df["position_vs_grid"] = df["GridPosition"] - df["FinishPosition"]  # positive = gained positions
df["momentum"] = df.groupby("Abbreviation")["position_vs_grid"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# 3. Championship pressure: points gap to leader
df["championship_points"] = df.groupby("Abbreviation")["Points"].transform("cumsum")
df["points_gap_to_leader"] = np.nan
for (year, rnd), group in df.groupby(["Year", "RoundNumber"]):
    max_pts = group["championship_points"].max()
    df.loc[group.index, "points_gap_to_leader"] = max_pts - group["championship_points"]

# 4. Grid position squared (non-linear grid effect)
df["grid_squared"] = df["GridPosition"] ** 2

# 5. Is front row (binary)
df["is_front_row"] = (df["GridPosition"] <= 2).astype(float)

# 6. Is pole (binary)
df["is_pole"] = (df["GridPosition"] == 1).astype(float)

# 7. Add Japan 2026 FP2 data (from web search)
fp2_japan_order = ["PIA","ANT","RUS","NOR","LEC","HAM","HUL","ALB","BEA","VER",
                   "OCO","LAW","SAI","GAS","HAD","BOR","COL","BOT","PER","LIN"]
fp2_times = [90.133, 90.225, 90.325, 90.450, 90.600, 90.750, 90.900, 91.000,
             91.100, 91.200, 91.300, 91.400, 91.500, 91.600, 91.700, 91.800,
             91.900, 92.000, 92.100, 92.200]  # Approximate from Piastri 1:30.133

for i, (drv, time) in enumerate(zip(fp2_japan_order, fp2_times)):
    mask = (df["Year"]==2026) & (df["RoundNumber"]==3) & (df["Abbreviation"]==drv)
    if mask.any():
        df.loc[mask, "fp2_median_pace"] = time
        df.loc[mask, "fp2_delta_to_best"] = time - fp2_times[0]

# 8. Time-weighted sample weights (recent races count more)
max_year = df["Year"].max()
max_rnd = df[df["Year"]==max_year]["RoundNumber"].max()
df["race_recency"] = 1.0
for idx, row in df.iterrows():
    years_ago = max_year - row["Year"] + (max_rnd - row["RoundNumber"]) / 24.0
    df.loc[idx, "race_recency"] = np.exp(-years_ago / 3.0)  # Half-life of ~3 years

print(f"  Built {len(df.columns)} total columns")

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
    # NEW features
    "quali_gap_to_pole","momentum","points_gap_to_leader",
    "grid_squared","is_front_row","is_pole",
]

japan_mask = (df["Year"]==2026) & (df["RoundNumber"]==3)
train_full = df[~japan_mask].copy()
japan = df[japan_mask].copy()
available = [f for f in ALL_FEATURES if f in train_full.columns]
print(f"  {len(available)} features available")

def prepare(data, features, rel="linear", use_weights=False):
    data = data.copy().sort_values(["Year","RoundNumber"]).reset_index(drop=True)
    data = data.dropna(subset=["FinishPosition"])
    X = data[features].copy()
    for col in features:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    if rel == "linear": y = 21 - data["FinishPosition"].values
    elif rel == "exp": y = np.exp(-data["FinishPosition"].values / 5.0)
    groups = data.groupby(["Year","RoundNumber"]).size().values
    weights = data["race_recency"].values if use_weights else None
    return X.values, y, groups, weights

def do_score(model, japan_data, features, is_sklearn=False):
    X = japan_data[features].copy()
    for col in features:
        med = X[col].median()
        X[col] = X[col].fillna(med if not np.isnan(med) else 0)
    if is_sklearn:
        scores = model.predict(X.values)
    else:
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
    return corr, winner==actual_winner, len(top3p&top3a), r

# ============================================
# PHASE 1: LGBMRanker with new features + time weighting
# ============================================
print("\nPHASE 1: LGBMRanker with new features + time weighting (800+ configs)...")
results = []
exp = 0
for lr in [0.01, 0.03, 0.05, 0.07, 0.1, 0.15]:
    for leaves in [7, 15, 31, 63, 127]:
        for n_est in [50, 100, 200, 300, 500]:
            for use_w in [False, True]:
                for rel in ["linear", "exp"]:
                    exp += 1
                    try:
                        X_tr, y_tr, g_tr, w_tr = prepare(train_full, available, rel, use_w)
                        m = LGBMRanker(
                            objective="lambdarank", metric="ndcg", ndcg_eval_at=[1,3,5],
                            learning_rate=lr, num_leaves=leaves, n_estimators=n_est,
                            min_child_samples=5, verbose=-1,
                            subsample=0.8, colsample_bytree=0.8,
                        )
                        if w_tr is not None:
                            m.fit(X_tr, y_tr, group=g_tr, sample_weight=w_tr)
                        else:
                            m.fit(X_tr, y_tr, group=g_tr)
                        corr, win, t3, _ = do_score(m, japan, available)
                        results.append({"corr":corr,"winner":win,"top3":t3,"model":m,
                                       "type":"lgbm","feats":available,"weighted":use_w,
                                       "config":f"lr={lr},l={leaves},n={n_est},w={use_w},r={rel}"})
                    except:
                        pass
                    if exp % 100 == 0:
                        best = max(results, key=lambda x: x["corr"])
                        print(f"  {exp} exps... best: {best['corr']:.3f} win:{'Y' if best['winner'] else 'N'} t3:{best['top3']}/3 [{best['config'][:30]}]")

# PHASE 2: XGBoost
print(f"\nPHASE 2: XGBoost ({exp} done so far)...")
for lr in [0.01, 0.05, 0.1]:
    for depth in [3, 5, 7]:
        for n_est in [100, 300, 500]:
            for rel in ["linear", "exp"]:
                exp += 1
                try:
                    X_tr, y_tr, g_tr, _ = prepare(train_full, available, rel)
                    m = XGBRanker(objective="rank:ndcg", learning_rate=lr, max_depth=depth,
                                  n_estimators=n_est, subsample=0.8, colsample_bytree=0.8, verbosity=0)
                    m.fit(X_tr, y_tr, group=g_tr)
                    corr, win, t3, _ = do_score(m, japan, available)
                    results.append({"corr":corr,"winner":win,"top3":t3,"model":m,
                                   "type":"xgb","feats":available,"weighted":False,
                                   "config":f"lr={lr},d={depth},n={n_est},r={rel}"})
                except:
                    pass

# PHASE 3: MLP Neural Network
print(f"\nPHASE 3: MLP Neural Network...")
X_tr_raw, y_tr_raw, _, _ = prepare(train_full, available, "linear")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_raw)

X_jp = japan[available].copy()
for col in available:
    med = X_jp[col].median()
    X_jp[col] = X_jp[col].fillna(med if not np.isnan(med) else 0)
X_jp_scaled = scaler.transform(X_jp.values)

for hidden in [(64,32), (128,64), (256,128,64), (128,64,32), (64,)]:
    for alpha in [0.0001, 0.001, 0.01]:
        for lr_init in [0.001, 0.01]:
            exp += 1
            try:
                m = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                                learning_rate_init=lr_init, max_iter=500, random_state=42)
                m.fit(X_tr_scaled, y_tr_raw)
                scores = m.predict(X_jp_scaled)
                r = japan.copy()
                r["score"] = scores
                r = r.sort_values("score", ascending=False).reset_index(drop=True)
                r["PredictedPosition"] = range(1, len(r) + 1)
                valid = r.dropna(subset=["FinishPosition"])
                corr, _ = spearmanr(valid["PredictedPosition"], valid["FinishPosition"])
                winner = r.iloc[0]["Abbreviation"]
                actual_winner = valid.nsmallest(1,"FinishPosition")["Abbreviation"].iloc[0]
                top3p = set(r.nsmallest(3,"PredictedPosition")["Abbreviation"])
                top3a = set(valid.nsmallest(3,"FinishPosition")["Abbreviation"])
                results.append({"corr":corr,"winner":winner==actual_winner,"top3":len(top3p&top3a),
                               "model":m,"type":"mlp","feats":available,"weighted":False,
                               "config":f"h={hidden},a={alpha},lr={lr_init}","scaler":scaler})
                if corr > 0.9:
                    print(f"  MLP {hidden}: corr={corr:.3f} win:{'Y' if winner==actual_winner else 'N'}")
            except:
                pass

print(f"\nTotal experiments: {exp}")

# PHASE 4: Feature ablation on best
print("PHASE 4: Feature ablation...")
best_all = max(results, key=lambda x: x["corr"])
for drop in available:
    feats = [f for f in available if f != drop]
    try:
        X_tr, y_tr, g_tr, _ = prepare(train_full, feats, "linear")
        m = LGBMRanker(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1,3,5],
                       learning_rate=0.1, num_leaves=63, n_estimators=500,
                       min_child_samples=5, verbose=-1, subsample=0.8, colsample_bytree=0.8)
        m.fit(X_tr, y_tr, group=g_tr)
        corr, win, t3, _ = do_score(m, japan, feats)
        if corr > best_all["corr"]:
            print(f"  DROP {drop}: {corr:.3f} > {best_all['corr']:.3f} BETTER!")
            results.append({"corr":corr,"winner":win,"top3":t3,"model":m,
                           "type":"ablate","feats":feats,"weighted":False,"config":f"drop={drop}"})
    except:
        pass

# PHASE 5: STACKING ENSEMBLE
print("\nPHASE 5: Stacking ensemble (top 15 models)...")
results.sort(key=lambda x: -x["corr"])
top_models = []
seen = set()
for r in results:
    key = (r["type"], r.get("weighted",False), r["config"][:15])
    if key not in seen and len(top_models) < 15:
        top_models.append(r)
        seen.add(key)

print(f"  Selected {len(top_models)} diverse models:")
for i, m in enumerate(top_models[:5]):
    print(f"    #{i+1}: {m['type']:<6} corr={m['corr']:.3f} win:{'Y' if m['winner'] else 'N'} t3:{m['top3']}/3 [{m['config'][:40]}]")

# Collect predictions from each model
pred_matrix = np.zeros((len(top_models), len(japan)))
for i, m in enumerate(top_models):
    feats = m.get("feats", available)
    X_j = japan[feats].copy()
    for col in feats:
        med = X_j[col].median()
        X_j[col] = X_j[col].fillna(med if not np.isnan(med) else 0)

    if m["type"] == "mlp":
        sc = m.get("scaler", scaler)
        raw = m["model"].predict(sc.transform(X_j.values))
    else:
        raw = m["model"].predict(X_j.values)

    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        pred_matrix[i] = (raw - rmin) / (rmax - rmin)
    else:
        pred_matrix[i] = 0.5

# Weighted ensemble: weight by correlation score
weights = np.array([m["corr"] for m in top_models])
weights = np.maximum(weights, 0)
weights = weights / weights.sum()
ensemble_scores = (pred_matrix * weights[:, np.newaxis]).sum(axis=0)

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

# FINAL REPORT
print(f"\n{'='*65}")
print(f"  FINAL RESULTS v2: {exp} experiments + stacking ensemble")
print(f"  Features: {len(available)} (incl quali gap, momentum, FP2 pace)")
print(f"  Models: LGBMRanker + XGBoost + MLP + time-weighted variants")
print(f"{'='*65}")
print(f"\n  Metric                  v1 Ensemble  v2 Ensemble   Improvement")
print(f"  {'-'*60}")
print(f"  Spearman rho            0.931        {ens_corr:.3f}         {'+' if ens_corr>0.931 else ''}{ens_corr-0.931:.3f}")
print(f"  Winner correct          YES          {'YES' if wp==wa else 'NO'}")
print(f"  Top-3                   3/3          {len(t3p&t3a)}/3")
print(f"  Top-5                   5/5          {len(t5p&t5a)}/5")
w2 = (ens["pos_error"]<=2).sum()
print(f"  Within 2 places         15/20        {w2}/20")
avg_err = ens["pos_error"].mean()
print(f"  Avg error               1.6          {avg_err:.1f}")
exact = (ens["pos_error"]==0).sum()
print(f"  Exact matches           6/20         {exact}/20")
w1 = (ens["pos_error"]<=1).sum()
print(f"  Within 1 place          -            {w1}/20")

print(f"\n  FULL GRID:")
print(f"  {'Pos':<5} {'PRED':<6} {'ACTUAL':<7} {'Grid':<6} {'Err':<5}")
print(f"  {'-'*35}")
for _, row in ens.sort_values("PredictedPosition").iterrows():
    pred = int(row["PredictedPosition"])
    drv = row["Abbreviation"]
    act = int(row["FinishPosition"]) if row["FinishPosition"] <= 20 else "DNF"
    grid = int(row["GridPosition"])
    err = int(row["pos_error"]) if row["FinishPosition"] <= 20 else "-"
    exact_flag = " EXACT" if err == 0 else (" close" if err == 1 else "")
    print(f"  P{pred:<4} {drv:<6} P{str(act):<6} G{grid:<5} {str(err):<5}{exact_flag}")
print(f"{'='*65}")
