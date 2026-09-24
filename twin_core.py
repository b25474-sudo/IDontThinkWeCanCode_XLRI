"""GlucoTwin core: data cleaning, feature engineering and model helpers.

The feature code here is identical to notebooks/02_train_the_twin.ipynb, so the dashboard
computes exactly the same inputs the model was trained on.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SPIKE = 180
HORIZON = 24  # 24 x 5 min = 2 hours

EHR_FEATURES = ["age", "sex_male", "bmi", "years_with_diabetes", "hba1c_pct", "fasting_glucose_mg_dl",
                "tcf7l2_risk_alleles", "family_history", "on_metformin", "on_second_drug", "on_insulin",
                "systolic_bp", "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl", "creatinine_mg_dl",
                "has_hypertension", "has_dyslipidemia", "diet_rice", "activity_score"]
WEARABLE_FEATURES = ["cgm", "cgm_lag_15m", "cgm_lag_30m", "cgm_lag_60m", "cgm_slope_15m", "cgm_slope_30m",
                     "cgm_mean_2h", "cgm_std_2h", "cgm_max_2h",
                     "carbs_last_30m", "carbs_last_1h", "carbs_last_2h", "carbs_last_4h", "minutes_since_meal",
                     "steps_last_30m", "steps_last_1h", "hr_now", "hr_mean_30m", "asleep_now",
                     "sleep_hours", "sleep_efficiency_pct", "deep_sleep_pct", "overnight_hrv_rmssd_ms",
                     "hrv_vs_personal_avg", "prev_day_steps", "hour_sin", "hour_cos", "is_weekend"]
TWIN_FEATURES = ["personal_mean_glucose", "personal_pct_above_180", "personal_meal_rise",
                 "personal_rise_per_gram", "expected_rise_now"]
ALL_FEATURES = WEARABLE_FEATURES + TWIN_FEATURES + EHR_FEATURES

# Plain-language names a doctor understands, grouped by data source
FEATURE_LABELS = {
    "cgm": ("Glucose right now", "mg/dL", "Wearable"),
    "cgm_lag_15m": ("Glucose 15 min ago", "mg/dL", "Wearable"),
    "cgm_lag_30m": ("Glucose 30 min ago", "mg/dL", "Wearable"),
    "cgm_lag_60m": ("Glucose 1 h ago", "mg/dL", "Wearable"),
    "cgm_slope_15m": ("Glucose trend, last 15 min", "mg/dL per min", "Wearable"),
    "cgm_slope_30m": ("Glucose trend, last 30 min", "mg/dL per min", "Wearable"),
    "cgm_mean_2h": ("Average glucose, last 2 h", "mg/dL", "Wearable"),
    "cgm_std_2h": ("Glucose swings, last 2 h", "mg/dL", "Wearable"),
    "cgm_max_2h": ("Highest glucose, last 2 h", "mg/dL", "Wearable"),
    "carbs_last_30m": ("Carbs eaten, last 30 min", "g", "Wearable"),
    "carbs_last_1h": ("Carbs eaten, last 1 h", "g", "Wearable"),
    "carbs_last_2h": ("Carbs eaten, last 2 h", "g", "Wearable"),
    "carbs_last_4h": ("Carbs eaten, last 4 h", "g", "Wearable"),
    "minutes_since_meal": ("Time since last meal", "min", "Wearable"),
    "steps_last_30m": ("Steps, last 30 min", "steps", "Wearable"),
    "steps_last_1h": ("Steps, last 1 h", "steps", "Wearable"),
    "hr_now": ("Heart rate now", "bpm", "Wearable"),
    "hr_mean_30m": ("Heart rate, last 30 min", "bpm", "Wearable"),
    "asleep_now": ("Asleep now", "", "Wearable"),
    "sleep_hours": ("Sleep last night", "h", "Wearable"),
    "sleep_efficiency_pct": ("Sleep efficiency last night", "%", "Wearable"),
    "deep_sleep_pct": ("Deep sleep last night", "%", "Wearable"),
    "overnight_hrv_rmssd_ms": ("Overnight HRV", "ms", "Wearable"),
    "hrv_vs_personal_avg": ("HRV vs personal average", "x", "Wearable"),
    "prev_day_steps": ("Steps yesterday", "steps", "Wearable"),
    "hour_sin": ("Time of day", "", "Routine"),
    "hour_cos": ("Time of day", "", "Routine"),
    "is_weekend": ("Weekend", "", "Routine"),
    "personal_mean_glucose": ("Personal average glucose", "mg/dL", "Twin memory"),
    "personal_pct_above_180": ("Personal share of time above 180", "", "Twin memory"),
    "personal_meal_rise": ("Usual rise after a meal", "mg/dL", "Twin memory"),
    "personal_rise_per_gram": ("Usual rise per gram of carbs", "mg/dL per g", "Twin memory"),
    "expected_rise_now": ("Twin's expected rise from recent carbs", "mg/dL", "Twin memory"),
    "age": ("Age", "years", "EHR"),
    "sex_male": ("Male", "", "EHR"),
    "bmi": ("BMI", "", "EHR"),
    "years_with_diabetes": ("Years with diabetes", "years", "EHR"),
    "hba1c_pct": ("HbA1c", "%", "EHR"),
    "fasting_glucose_mg_dl": ("Fasting glucose", "mg/dL", "EHR"),
    "tcf7l2_risk_alleles": ("TCF7L2 risk alleles", "", "EHR"),
    "family_history": ("Family history of diabetes", "", "EHR"),
    "on_metformin": ("On metformin", "", "EHR"),
    "on_second_drug": ("On a second diabetes drug", "", "EHR"),
    "on_insulin": ("On insulin", "", "EHR"),
    "systolic_bp": ("Systolic BP", "mmHg", "EHR"),
    "ldl_mg_dl": ("LDL cholesterol", "mg/dL", "EHR"),
    "hdl_mg_dl": ("HDL cholesterol", "mg/dL", "EHR"),
    "triglycerides_mg_dl": ("Triglycerides", "mg/dL", "EHR"),
    "creatinine_mg_dl": ("Creatinine", "mg/dL", "EHR"),
    "has_hypertension": ("Hypertension", "", "EHR"),
    "has_dyslipidemia": ("Dyslipidemia", "", "EHR"),
    "diet_rice": ("Rice-dominant diet", "", "EHR"),
    "activity_score": ("Activity level", "", "EHR"),
}


def find_file(name):
    """Look for a file in data/, model/ or the repository root (in case folders were flattened on upload)."""
    for folder in (ROOT / "data", ROOT / "model", ROOT):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(
        f"Could not find '{name}'. Upload it to the repository's data/ or model/ folder (or the main page).")


def load_raw():
    ehr = pd.read_csv(find_file("ehr_patients.csv"))
    wear = pd.read_csv(find_file("wearable_timeseries.csv"), parse_dates=["timestamp"])
    daily = pd.read_csv(find_file("daily_summary.csv"), parse_dates=["date"])
    return ehr, wear, daily


def train_model(ehr, wear, daily):
    """Re-train the early-warning model exactly as in notebooks/02_train_the_twin.ipynb
    (same 60 training patients, same settings, same random seed)."""
    pids = np.array(sorted(ehr.patient_id))
    np.random.default_rng(7).shuffle(pids)
    train_ids = pids[:60]
    f = build_features(ehr, wear, daily, train_ids)
    rows = f[(f.timestamp.dt.minute % 15 == 0) & f.cgm.notna() & (f.cgm < SPIKE)
             & f.future_max_2h.notna() & f.cgm_lag_60m.notna()]
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                             random_state=42)
    clf.fit(rows[ALL_FEATURES], (rows.future_max_2h > SPIKE).astype(int))
    return clf.booster_


def load_model():
    """Load the saved model; if the file is missing or unreadable, rebuild it from the data."""
    try:
        return lgb.Booster(model_file=str(find_file("twin_model.txt")))
    except Exception:
        return train_model(*load_raw())


def _future_max(s, n):
    return s[::-1].rolling(n, min_periods=int(n * 0.5)).max()[::-1].shift(-1)


def _patient_features(d):
    d = d.copy()
    g = d.cgm
    for lag in [3, 6, 12]:
        d[f"cgm_lag_{lag * 5}m"] = g.shift(lag)
    d["cgm_slope_15m"] = (g - g.shift(3)) / 15
    d["cgm_slope_30m"] = (g - g.shift(6)) / 30
    d["cgm_mean_2h"] = g.rolling(24, min_periods=12).mean()
    d["cgm_std_2h"] = g.rolling(24, min_periods=12).std()
    d["cgm_max_2h"] = g.rolling(24, min_periods=12).max()

    c = d.carbs_logged_g
    d["carbs_last_30m"] = c.rolling(6, min_periods=1).sum()
    d["carbs_last_1h"] = c.rolling(12, min_periods=1).sum()
    d["carbs_last_2h"] = c.rolling(24, min_periods=1).sum()
    d["carbs_last_4h"] = c.rolling(48, min_periods=1).sum()
    meal_idx = pd.Series(np.where(c > 0, np.arange(len(d)), np.nan), index=d.index).ffill()
    d["minutes_since_meal"] = ((np.arange(len(d)) - meal_idx) * 5).clip(upper=720).fillna(720)

    d["steps_last_30m"] = d.steps.rolling(6, min_periods=1).sum()
    d["steps_last_1h"] = d.steps.rolling(12, min_periods=1).sum()
    d["hr_now"] = d.heart_rate_bpm
    d["hr_mean_30m"] = d.heart_rate_bpm.rolling(6, min_periods=1).mean()
    d["asleep_now"] = (~d.sleep_stage.isin(["Not asleep"])).astype(int)

    hour = d.timestamp.dt.hour + d.timestamp.dt.minute / 60
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["is_weekend"] = (d.timestamp.dt.dayofweek >= 5).astype(int)

    d["personal_mean_glucose"] = g.expanding(min_periods=12).mean().shift(1)
    d["personal_pct_above_180"] = (g > SPIKE).astype(float).expanding(min_periods=12).mean().shift(1)

    rises = np.full(len(d), np.nan)
    per_gram = np.full(len(d), np.nan)
    history, history_pg, completed = [], [], []
    gv = g.values
    for m in np.where(c.values > 0)[0]:
        window = gv[m:m + 25]
        if np.isnan(gv[m]) or np.all(np.isnan(window)):
            continue
        rise = np.nanmax(window) - gv[m]
        completed.append((m + 24, rise, rise / c.values[m]))
    completed.sort()
    j = 0
    for t in range(len(d)):
        while j < len(completed) and completed[j][0] < t:
            history.append(completed[j][1]); history_pg.append(completed[j][2]); j += 1
        if history:
            rises[t] = np.mean(history[-10:])
            per_gram[t] = np.mean(history_pg[-10:])
    d["personal_meal_rise"] = rises
    d["personal_rise_per_gram"] = per_gram
    d["expected_rise_now"] = d.personal_rise_per_gram * d.carbs_last_2h
    return d


def build_features(ehr, wear, daily, patient_ids=None):
    if patient_ids is not None:
        wear = wear[wear.patient_id.isin(patient_ids)]
    wear = wear.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    wear["cgm"] = wear.cgm_glucose_mg_dl.clip(40, 400)
    wear["cgm"] = wear.groupby("patient_id").cgm.transform(
        lambda s: s.interpolate(limit=6, limit_area="inside"))
    wear["future_max_2h"] = wear.groupby("patient_id").cgm.transform(lambda s: _future_max(s, HORIZON))
    wear = pd.concat([_patient_features(d) for _, d in wear.groupby("patient_id")], ignore_index=True)

    daily = daily.copy()
    daily["prev_day_steps"] = daily.groupby("patient_id").total_steps.shift(1)
    daily["hrv_vs_personal_avg"] = daily.overnight_hrv_rmssd_ms / daily.groupby("patient_id") \
        .overnight_hrv_rmssd_ms.transform(lambda s: s.expanding().mean().shift(1))
    wear["date"] = wear.timestamp.dt.normalize()
    wear = wear.merge(daily[["patient_id", "date", "sleep_hours", "sleep_efficiency_pct", "deep_sleep_pct",
                             "overnight_hrv_rmssd_ms", "hrv_vs_personal_avg", "prev_day_steps"]],
                      on=["patient_id", "date"], how="left")

    e = ehr.copy()
    e["sex_male"] = (e.sex == "Male").astype(int)
    e["on_second_drug"] = e.medication.str.contains("DPP-4|SGLT2").astype(int)
    e["on_insulin"] = e.medication.str.contains("insulin").astype(int)
    e["on_metformin"] = e.medication.str.contains("Metformin").astype(int)
    e["has_hypertension"] = e.past_diagnoses.str.contains("Hypertension").astype(int)
    e["has_dyslipidemia"] = e.past_diagnoses.str.contains("Dyslipidemia").astype(int)
    e["family_history"] = e.family_history_diabetes.astype(int)
    e["diet_rice"] = (e.diet_pattern == "Rice-dominant").astype(int)
    e["activity_score"] = e.activity_level.map({"Low": 0, "Moderate": 1, "High": 2})
    return wear.merge(e[["patient_id"] + EHR_FEATURES], on="patient_id", how="left")


def predict(model, X):
    return model.predict(X[ALL_FEATURES])


def explain(model, x_row):
    """Per-feature contributions (SHAP values, in log-odds) for a single prediction.
    LightGBM computes these exactly for tree models via pred_contrib."""
    contrib = model.predict(x_row[ALL_FEATURES], pred_contrib=True)[0]
    out = pd.DataFrame({"feature": ALL_FEATURES, "value": x_row[ALL_FEATURES].values[0],
                        "contribution": contrib[:-1]})
    # merge the two time-of-day components into one readable row
    tod = out[out.feature.isin(["hour_sin", "hour_cos"])]
    out = out[~out.feature.isin(["hour_sin", "hour_cos"])]
    out = pd.concat([out, pd.DataFrame([{"feature": "time_of_day", "value": np.nan,
                                          "contribution": tod.contribution.sum()}])], ignore_index=True)
    return out, contrib[-1]


# ============================================================================
# Physiology twin: a small personalised glucose model fitted to each patient.
# The ML model above answers "will a spike happen?"; this model answers
# "what would happen to this patient's glucose if they did X?"
#
#   excess[t+1] = excess[t] + carb_sensitivity * carbs_absorbed[t]
#                 - 5 min * clearance * (1 + walk_boost * activity[t]) * excess[t]
#   glucose     = baseline + excess
#
# Four personal parameters (baseline, carb_sensitivity, clearance, walk_boost)
# are fitted to the patient's last 3 days of CGM, meal logs and steps.
# Tested on 30 unseen patients: 1-hour forecast error ~12 mg/dL, 2-hour ~14 mg/dL,
# versus 23 and 37 mg/dL for "glucose stays where it is".
# ============================================================================
from scipy.optimize import least_squares

_TAU = 35.0                                            # typical carb absorption time (min)
_K = np.arange(48) * 5 + 2.5
_KERNEL = _K / _TAU ** 2 * np.exp(-_K / _TAU) * 5      # share of carbs absorbed in each 5-min slot
PARAM_NAMES = ["baseline_mg_dl", "carb_sensitivity", "clearance_per_min", "walk_boost"]


def absorption(carbs):
    return np.convolve(np.asarray(carbs, float), _KERNEL)[:len(carbs)]


def activity(steps):
    return np.clip(pd.Series(np.asarray(steps, float)).rolling(6, min_periods=1).sum().values / 2500, 0, 1)


def _simulate(p, ra, act, x0):
    base, k, s, a = p
    x = np.empty(len(ra))
    x[0] = x0
    for t in range(1, len(ra)):
        x[t] = x[t - 1] + k * ra[t] - 5 * s * (1 + a * act[t]) * x[t - 1]
    return base + x


def fit_physiology(history):
    """history: this patient's rows (5-min) up to now, with cgm, carbs_logged_g, steps."""
    y = history.cgm.values
    ra = absorption(history.carbs_logged_g.values)
    act = activity(history.steps.values)
    m = ~np.isnan(y)
    y0 = y[m][0]
    w = np.sqrt(m.sum())

    def residuals(p):
        # gentle priors keep parameters physiologically sensible when data is thin
        prior = np.array([(p[3] - 1.2) / 0.4, (np.log(p[1]) - np.log(2.8)) / 0.5]) * w
        return np.concatenate([(_simulate(p, ra, act, y0 - p[0]) - y)[m], prior])

    r = least_squares(residuals, [np.nanpercentile(y, 20), 2.5, 0.012, 1.2],
                      bounds=([60, 0.3, 0.002, 0], [260, 10, 0.06, 4]), x_scale=[50, 1, 0.01, 1])
    fit_error = float(np.sqrt(np.mean(r.fun[:-2] ** 2)))
    return dict(zip(PARAM_NAMES, r.x)), fit_error


def forecast(params, history, meal_now_g=0.0, walk_now_min=0, hours=3):
    """Simulate glucose for the next `hours` from the latest reading, under a scenario."""
    n = hours * 12
    carbs = np.concatenate([history.carbs_logged_g.values.astype(float), np.zeros(n)])
    steps = np.concatenate([history.steps.values.astype(float), np.zeros(n)])
    now_i = len(history) - 1
    carbs[now_i + 1] += meal_now_g
    steps[now_i + 1: now_i + 1 + int(walk_now_min // 5)] += 500    # brisk walk ~100 steps/min
    ra, act = absorption(carbs), activity(steps)
    p = [params[k] for k in PARAM_NAMES]
    g_now = history.cgm.dropna().iloc[-1]
    return _simulate(p, ra[now_i:], act[now_i:], g_now - p[0])
