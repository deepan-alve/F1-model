"""Final prediction with ALL features including car performance + power unit."""
import pandas as pd
import numpy as np
from src.features import build_feature_matrix
from src.elo import compute_elo_ratings
from src.model import prepare_ranking_data, train_ranker, predict_race_order, train_dnf_classifier

df = pd.read_parquet("data/processed/all_races.parquet")
car = pd.read_parquet("data/processed/car_performance.parquet")
pu = pd.read_csv("data/power_units.csv")

# Drop old car columns
for col in ["speed_trap","car_speed_delta","car_aero_balance","sector1_time","sector2_time",
            "sector3_time","car_rolling_speed","car_rolling_aero","team_avg_speed","pu_encoded","pu_manufacturer"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Merge car performance
car_cols = ["Abbreviation","Year","RoundNumber","speed_trap","car_speed_delta","car_aero_balance"]
df = df.merge(car[car_cols], on=["Abbreviation","Year","RoundNumber"], how="left")

# Power unit mapping
pu_map = {}
for _, row in pu.iterrows():
    pu_map[(row["year"], row["team_name"])] = row["pu_manufacturer"]

def get_pu(row):
    key = (row["Year"], row["TeamName"])
    if key in pu_map:
        return pu_map[key]
    for (y, t), m in pu_map.items():
        if y == row["Year"] and (t.lower() in row["TeamName"].lower() or row["TeamName"].lower() in t.lower()):
            return m
    return "Unknown"

df["pu_manufacturer"] = df.apply(get_pu, axis=1)
pu_encoding = {"Mercedes": 1, "Ferrari": 2, "Honda": 3, "Renault": 4, "Ford": 5, "Unknown": 0}
df["pu_encoded"] = df["pu_manufacturer"].map(pu_encoding).fillna(0)

# Rolling car features
df = df.sort_values(["Year","RoundNumber"]).reset_index(drop=True)
df["car_rolling_speed"] = df.groupby("Abbreviation")["speed_trap"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["car_rolling_aero"] = df.groupby("Abbreviation")["car_aero_balance"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
team_speed = df.groupby(["Year","RoundNumber","TeamName"])["speed_trap"].mean().reset_index()
team_speed.rename(columns={"speed_trap": "team_avg_speed"}, inplace=True)
df = df.merge(team_speed, on=["Year","RoundNumber","TeamName"], how="left")

df.to_parquet("data/processed/all_races.parquet", index=False)

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

X_train, y_train, groups_train = prepare_ranking_data(df, ALL_FEATURES)
model = train_ranker(X_train, y_train, groups_train)
dnf_model = train_dnf_classifier(df)

available = [f for f in ALL_FEATURES if f in df.columns]
print(f"Model trained: {len(X_train)} samples, {len(available)} features")

importances = model.feature_importances_
print("\nFeature importance (top 15):")
for name, imp in sorted(zip(available, importances), key=lambda x: -x[1])[:15]:
    print(f"  {name:<30} {imp}")

latest = df.sort_values(["Year","RoundNumber"]).groupby("Abbreviation").last()

# Japanese GP
quali_data = pd.DataFrame({
    "Abbreviation": ["ANT","RUS","PIA","LEC","NOR","HAM","GAS","HAD","BOR","LIN",
                     "VER","OCO","HUL","LAW","COL","SAI","ALB","BEA","PER","BOT"],
    "TeamName": ["Mercedes","Mercedes","McLaren","Ferrari","McLaren","Ferrari",
                 "Alpine","Red Bull Racing","Sauber","RB",
                 "Red Bull Racing","Haas","Sauber","RB","Alpine","Williams",
                 "Williams","Haas","Cadillac","Cadillac"],
    "GridPosition": list(range(1,21)), "QualifyingPosition": list(range(1,21)),
    "Q1": [88.778+0.298*i for i in range(20)],
    "Q2": [88.5+0.3*i for i in range(20)],
    "Q3": [88.778,89.076,89.132,89.405,89.409,89.567,89.691,89.978,90.274,90.319]+[np.nan]*10,
    "Year": [2026]*20, "RoundNumber": [3]*20,
    "EventName": ["Japanese Grand Prix"]*20,
    "Position": [np.nan]*20, "FinishPosition": [np.nan]*20,
    "Status": ["Upcoming"]*20, "DNF": [False]*20,
    "Points": [0.0]*20, "DriverNumber": [""]*20, "BroadcastName": [""]*20,
    "weather_temp_max": [18.0]*20, "weather_rain_mm": [0.0]*20, "weather_wind_max": [10.0]*20,
    "weather_temp_min": [9.0]*20,
    "num_pit_stops": [np.nan]*20, "used_soft": [np.nan]*20, "used_medium": [np.nan]*20,
    "used_hard": [np.nan]*20, "used_intermediate": [False]*20, "used_wet": [False]*20,
    "driver_avg_pit_stops": [np.nan]*20, "avg_pit_duration": [np.nan]*20,
    "fp2_median_pace": [np.nan]*20, "fp2_delta_to_best": [np.nan]*20,
    "speed_trap": [np.nan]*20, "car_speed_delta": [np.nan]*20, "car_aero_balance": [np.nan]*20,
    "car_rolling_speed": [np.nan]*20, "car_rolling_aero": [np.nan]*20,
    "team_avg_speed": [np.nan]*20,
})

pu_2026 = {"Mercedes": 1, "McLaren": 1, "Alpine": 1, "Williams": 1,
           "Ferrari": 2, "Sauber": 2, "Haas": 2, "Cadillac": 2,
           "Red Bull Racing": 5, "RB": 5}
quali_data["pu_encoded"] = quali_data["TeamName"].map(pu_2026).fillna(0)

quali_data = build_feature_matrix(quali_data)
for col in ["driver_elo","constructor_elo","driver_rolling_avg_3","driver_rolling_avg_5",
            "driver_rolling_avg_10","driver_dnf_rate","constructor_rolling_points",
            "constructor_reliability","driver_avg_pit_stops",
            "car_rolling_speed","car_rolling_aero","team_avg_speed"]:
    if col in latest.columns:
        quali_data[col] = quali_data["Abbreviation"].map(latest[col].to_dict())

predictions = predict_race_order(model, quali_data, ALL_FEATURES)

if dnf_model is not None:
    dnf_feats = [f for f in ["GridPosition","driver_elo","driver_dnf_rate","constructor_reliability"] if f in quali_data.columns]
    X_dnf = quali_data[dnf_feats].fillna(0)
    probs = dnf_model.predict_proba(X_dnf.values)[:, 1]
    predictions["DNFProb"] = predictions["Abbreviation"].map(dict(zip(quali_data["Abbreviation"], probs)))

pu_names = {"Mercedes": "Merc PU", "McLaren": "Merc PU", "Alpine": "Merc PU", "Williams": "Merc PU",
            "Ferrari": "Ferrari PU", "Sauber": "Ferrari PU", "Haas": "Ferrari PU", "Cadillac": "Ferrari PU",
            "Red Bull Racing": "Ford PU", "RB": "Ford PU"}

print()
print("================================================================")
print("  JAPANESE GRAND PRIX 2026 - SUZUKA")
print("  EVERYTHING MODEL v3 (+ speed trap + aero + power unit)")
print("================================================================")
print()
for _, row in predictions.iterrows():
    pos = int(row["PredictedPosition"])
    driver = row["Abbreviation"]
    team = row["TeamName"][:12]
    grid = int(row["GridPosition"])
    dnf = row.get("DNFProb", 0) * 100
    pu = pu_names.get(row["TeamName"], "?")
    delta = grid - pos
    move = ""
    if delta > 0: move = f" (+{delta})"
    elif delta < 0: move = f" ({delta})"
    dnf_warn = f" DNF:{dnf:.0f}%" if dnf > 10 else ""
    print(f"  P{pos:<3} {driver:<5} {team:<12} [{pu:<10}] grid P{grid}{move}{dnf_warn}")

print()
print("----------------------------------------------------------------")
print(f"DATA: {len(df)} rows | 154 races | 8 seasons | {len(available)} features")
print(f"CAR DATA: speed trap (engine+drag), aero balance (downforce vs")
print(f"  straight speed), sector times, team avg speed, power unit mfr")
print(f"+ Elo + quali + weather + tire + pit stops + FP2 pace + track")
print(f"MISSING: news sentiment, betting odds history only")
print("================================================================")
