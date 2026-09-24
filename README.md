# GlucoTwin — a digital twin that warns of blood-sugar spikes two hours ahead

**Happiest Health Reimagining and Reforming Healthcare in India Summit 2026 — Digital Twin Challenge, Phase 1**

| | |
|---|---|
| **Team** | I Dont Think We Can Code |
| **College** | XLRI – Xavier School of Management, Jamshedpur |
| **Team leader** | Sarah Dhamija · PGD-BM 2025–27 |
| **Team member** | Yashas Tarakaram · PGD-BM 2025–27 |
| **Condition** | Type 2 Diabetes |
| **Live dashboard** | **https://idontthinkwecancodexlri.streamlit.app** — try it, no installation needed |
| **Demo video (2–5 min)** | _[Unlisted YouTube link]_ |
| **Architecture diagram** | [`GlucoTwin_Architecture.pdf`](GlucoTwin_Architecture.pdf) |
| **Presentation** | [`GlucoTwin_Presentation.pdf`](GlucoTwin_Presentation.pdf) (PowerPoint version: [`GlucoTwin_Presentation.pptx`](GlucoTwin_Presentation.pptx)) |
| **License** | MIT (see [`LICENSE`](LICENSE)) |

---

## 1. Problem statement and healthcare use case

India has over 100 million adults living with diabetes. Repeated blood-sugar spikes after meals (above
180 mg/dL) damage blood vessels, nerves, eyes and kidneys over time, and in India's carbohydrate-heavy diet
they are common. Today, patients and doctors usually learn about a spike only **after** it has happened, from a
glucometer or a lab report weeks later.

**GlucoTwin** keeps a virtual copy of each patient that fuses their hospital record with live wearable data.
Every 15 minutes it:

1. **Predicts** the chance that glucose will cross 180 mg/dL in the next two hours,
2. **Explains** why, in plain language (e.g. "95 g carbs 40 minutes ago, no walk, 5 hours of sleep last night"),
3. **Simulates** what would happen if the patient changed something now — a 15-minute walk, a smaller meal.

**Users.** A diabetologist or primary-care doctor reviewing many patients remotely (ward view sorted by risk),
and, in future, the patient through a phone app.

## 2. Data (synthetic only — DPDP Act / HIPAA compliant)

No real patient data is used. We generated 100 synthetic patients with physiology-based rules calibrated to
published clinical relationships (see `01_create_virtual_patients.ipynb`):

| Stream | Contents | Resolution |
|---|---|---|
| **Static EHR** | Age, sex, BMI, years with diabetes, HbA1c, fasting glucose, medication, BP, LDL/HDL, triglycerides, creatinine, past diagnoses, family history, **TCF7L2 genetic risk alleles**, diet pattern, activity level | 1 row per patient |
| **Dynamic wearables** | Continuous glucose monitor, heart rate, steps, sleep stages (Light/Deep/REM/Awake), logged meal carbs | Every 5 minutes, 14 days |
| **Daily summary** | Sleep hours, sleep efficiency, deep-sleep %, overnight HRV (RMSSD), total steps | 1 row per patient-day |

Indian context is built in: rice- or wheat-dominant diets, heavy lunches, evening chai and snacks, late
dinners, weekend and festival meals. Realistic imperfections are included: ~15% of meals are not logged,
portion estimates are off by up to ±20%, and the CGM has noise and signal gaps.

**Validation against clinical benchmarks.** Mean CGM glucose matches the ADAG formula
(eAG = 28.7 × HbA1c − 46.7); time-in-range falls with HbA1c (≈70% at HbA1c 7–8%, correlation −0.96),
in line with international consensus; median 5,200 steps/day and 6.8 h sleep.

## 3. How the twin works — two engines

```
 EHR (static) ──┐
                ├─► Feature builder ─► ① Early-warning model (LightGBM) ─► risk % + SHAP "why"
 Wearables ─────┤     (53 features,                                           │
 (5-min stream) │      past data only)                                        ▼
                └─► ② Personal physiology model ─► "What if…" glucose forecast ─► Doctor dashboard
                     (4 parameters refitted to each patient's last 3 days)
```

**① Early warning — machine learning.** A LightGBM gradient-boosted tree classifier predicts
*"will glucose exceed 180 mg/dL within 2 hours?"* using 53 features in four groups:

- *Wearable:* current glucose, 15/30/60-min lags, trend, 2-h mean/variability, carbs in last 30 min–4 h,
  time since meal, steps, heart rate, last night's sleep, deep sleep, HRV vs personal average.
- *EHR:* age, BMI, HbA1c, fasting glucose, medication, BP, lipids, creatinine, TCF7L2, diagnoses.
- *Twin memory:* the patient's own average glucose, usual post-meal rise, rise per gram of carbs — learned
  continuously from their past data only.
- *Routine:* time of day, weekend.

Explanations use exact TreeSHAP contributions computed by LightGBM.

**② Scenario simulator — physiology.** A compact glucose model with four personal parameters (baseline,
rise per gram of absorbed carbs, clearance speed, and how much walking speeds clearance) is fitted to each
patient's last three days by least squares. It forecasts the next three hours under a scenario the doctor
chooses. We use a mechanistic model here because a purely statistical model can't be trusted to answer
counterfactual questions.

## 4. Results (30 patients never seen in training)

Patients were split **by person** — 60 train, 10 validation (to choose the alert level), 30 test — so results
reflect a new patient being onboarded. Predictions are made only while glucose is below 180 (warning about a
spike that has already started is useless) and only from past data (no leakage).

| Model | AUROC | AUPRC |
|---|---|---|
| Simple rule (glucose > 150 and rising) | 0.573 | 0.336 |
| EHR only | 0.632 | 0.381 |
| Wearables only | 0.928 | 0.836 |
| **GlucoTwin (EHR + wearables + twin memory)** | **0.943** | **0.873** |

At the 60% alert level: **97% of spikes flagged in advance**, median warning **≈95 minutes** before glucose
crosses 180, **0.7 false alarms per patient per day**.

Physiology simulator (`python evaluate_physiology.py`): mean absolute error **11.9 mg/dL at 1 h** and
**14.0 mg/dL at 2 h**, versus 23.1 and 36.7 mg/dL for assuming glucose stays unchanged.

**Limitations.** The patients are synthetic and follow steadier routines than real people, so the model can
partly anticipate meals from time of day; real-world performance will be lower. What-if forecasts are decision
aids, not treatment advice. Next steps: validation on public CGM datasets (e.g. OhioT1DM, Shanghai T2DM),
robustness to irregular routines, and a clinician-in-the-loop pilot.

## 5. Doctor dashboard

- **Ward overview:** all patients sorted by current spike risk, like a triage board.
- **Patient twin:** EHR summary, risk gauge, 24-hour timeline (glucose, meals, alerts, risk, steps),
  "Why the twin thinks this" (SHAP), and "What if…" (meal / walk sliders with a 3-hour forecast, plus an
  option to reveal what actually happened for validation).
- **How well it works:** model comparison and which data sources drive decisions.

A "clinic clock" replays the 14 days as if live; the twin only ever sees data up to that moment.

## 6. Technical stack

Python 3 · pandas · NumPy · LightGBM (model + TreeSHAP) · SciPy (physiology fitting) · scikit-learn (metrics) ·
Streamlit (dashboard) · Plotly (charts) · Google Colab (notebooks).

## 7. Repository structure

| File | What it is |
|---|---|
| `app.py` | Streamlit doctor dashboard |
| `twin_core.py` | Feature engineering, model helpers and the physiology twin |
| `01_create_virtual_patients.ipynb` | Generates the 100 synthetic patients and validates them against clinical benchmarks |
| `02_train_the_twin.ipynb` | Builds features and labels, trains and evaluates the early-warning model |
| `evaluate_physiology.py` | Reproduces the physiology simulator's accuracy |
| `ehr_patients.csv` | Synthetic EHR, one row per patient |
| `wearable_timeseries.csv` | Synthetic wearable stream, every 5 minutes for 14 days |
| `daily_summary.csv` | Daily sleep, HRV and step summaries |
| `twin_model.txt` | Trained LightGBM model |
| `test_patients.csv` | The 30 patients held out for testing |
| `model_comparison.csv` | EHR-only vs wearables-only vs fused results |
| `requirements.txt` | Python packages |
| `GlucoTwin_Architecture.pdf` | Architecture diagram |
| `GlucoTwin_Presentation.pdf`, `.pptx` | Presentation with project details and outcomes |

## 8. Run it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or simply open the live dashboard linked above. To regenerate everything from scratch, run the two notebooks
in order (Google Colab works with no setup).
All randomness is seeded, so results reproduce exactly.

## 9. License

Released under the MIT License. All data in this repository is synthetic.

*GlucoTwin is a research proof-of-concept, not a medical device.*
