"""Reproduces the physiology-simulator accuracy reported in the dashboard.

For each of the 30 held-out patients, on days 9, 11 and 13 we fit the personal physiology
model to the previous 3 days, then forecast glucose every hour for that day (given the meals
and steps that actually followed) and compare with the measured CGM 1 and 2 hours later.

Run from the repository root:  python scripts/evaluate_physiology.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin_core import ROOT, _simulate, absorption, activity, build_features, fit_physiology, load_raw, PARAM_NAMES  # noqa: E402

ehr, wear, daily = load_raw()
test_ids = pd.read_csv(ROOT / "model" / "test_patients.csv").patient_id.tolist()
feats = build_features(ehr, wear, daily, test_ids)

errors = {60: [], 120: []}
persistence = {60: [], 120: []}
for pid, d in feats.groupby("patient_id"):
    d = d.reset_index(drop=True)
    ra = absorption(d.carbs_logged_g.values)
    act = activity(d.steps.values)
    for day in (8, 10, 12):
        start = day * 288
        params, _ = fit_physiology(d.iloc[start - 864:start])
        p = [params[k] for k in PARAM_NAMES]
        for i in range(start, start + 288, 12):
            if np.isnan(d.cgm[i]):
                continue
            sim = _simulate(p, ra[i:i + 25], act[i:i + 25], d.cgm[i] - p[0])
            for h in (60, 120):
                actual = d.cgm[i + h // 5]
                if not np.isnan(actual):
                    errors[h].append(abs(sim[h // 5] - actual))
                    persistence[h].append(abs(d.cgm[i] - actual))

for h in (60, 120):
    print(f"{h:>3} min ahead | twin MAE {np.mean(errors[h]):5.1f} mg/dL | "
          f"'no change' MAE {np.mean(persistence[h]):5.1f} mg/dL | n={len(errors[h])}")
