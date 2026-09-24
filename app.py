"""GlucoTwin — doctor dashboard for a Type 2 Diabetes digital twin.
Team: I Dont Think We Can Code (XLRI). Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from twin_core import (ALL_FEATURES, FEATURE_LABELS, ROOT, SPIKE, build_features, explain, find_file, fit_physiology,
                       forecast, load_model, load_raw, predict)

st.set_page_config(page_title="GlucoTwin", page_icon="🩺", layout="wide")

INK, TEAL, GREEN, AMBER, RED, MUTED = "#1F2A30", "#0E6E6B", "#2E7D5B", "#C98A12", "#B83A2B", "#6B7A80"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.block-container { padding-top: 1.6rem; }
.brand { font-size: 1.9rem; font-weight: 600; color: #0E6E6B; margin-bottom: 0; }
.brand-sub { color: #6B7A80; margin-top: 0; }
.status { padding: 0.9rem 1.1rem; border-left: 5px solid; background: white; border-radius: 4px; }
.status b { font-size: 1.05rem; }
.ehr-grid { display: grid; grid-template-columns: auto 1fr; gap: 0.25rem 1rem; font-size: 0.93rem; }
.ehr-grid span:nth-child(odd) { color: #6B7A80; }
.small-note { color: #6B7A80; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(show_spinner="Building the virtual patients…")
def get_data():
    ehr, wear, daily = load_raw()
    test_ids = pd.read_csv(find_file("test_patients.csv")).patient_id.tolist()
    feats = build_features(ehr, wear, daily, test_ids)
    ok = feats.cgm.notna() & (feats.cgm < SPIKE) & feats.cgm_lag_60m.notna()
    feats["risk"] = np.nan
    feats.loc[ok, "risk"] = predict(get_model(), feats[ok])
    return ehr.set_index("patient_id"), daily, feats, test_ids


@st.cache_data(show_spinner=False)
def get_physiology(pid, now):
    pf_ = feats[feats.patient_id == pid].set_index("timestamp")
    return fit_physiology(pf_.loc[now - pd.Timedelta(days=3) + pd.Timedelta(minutes=5): now])


model = get_model()
ehr, daily, feats, test_ids = get_data()


def risk_band(r, threshold):
    if pd.isna(r):
        return "—", MUTED
    if r >= threshold:
        return "Spike likely", RED
    if r >= threshold / 2:
        return "Watch", AMBER
    return "Stable", GREEN


def trend_arrow(slope):
    if pd.isna(slope):
        return ""
    if slope > 2: return "⇈"
    if slope > 1: return "↑"
    if slope > 0.3: return "↗"
    if slope < -2: return "⇊"
    if slope < -1: return "↓"
    if slope < -0.3: return "↘"
    return "→"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<p class="brand">GlucoTwin</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Digital twin for Type 2 Diabetes</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Clinic clock**")
    st.caption("Replay the 14 days of wearable data as if it were live. The twin only sees data up to this moment.")
    days = pd.date_range(feats.timestamp.min().normalize() + pd.Timedelta(days=1), feats.timestamp.max().normalize(), freq="D")
    day = st.select_slider("Day", options=list(days), value=days[4], format_func=lambda d: d.strftime("%a %d %b"))
    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
    tod = st.select_slider("Time", options=times, value="12:00")
    now = day + pd.Timedelta(hours=int(tod[:2]), minutes=int(tod[3:]))
    st.divider()
    threshold = st.slider("Alert when spike risk reaches", 30, 90, 60, 5, format="%d%%") / 100
    st.caption("Chosen on validation patients: catches most spikes with under 1 false alarm per patient per day.")
    st.divider()
    st.caption("Team *I Dont Think We Can Code* · XLRI Jamshedpur\n\nSynthetic data only. Not a medical device.")

snap = feats[feats.timestamp == now].set_index("patient_id")

tab_ward, tab_twin, tab_perf, tab_about = st.tabs(["Ward overview", "Patient twin", "How well it works", "About"])

# ---------------------------------------------------------------- ward overview
with tab_ward:
    st.subheader(f"Who needs attention at {now:%H:%M on %a %d %b}")
    rows = []
    for pid in test_ids:
        if pid not in snap.index:
            continue
        s, e = snap.loc[pid], ehr.loc[pid]
        if pd.isna(s.cgm):
            status = "Sensor gap"
        elif s.cgm >= SPIKE:
            status = "Above 180 now"
        else:
            status = risk_band(s.risk, threshold)[0]
        rows.append(dict(Patient=pid, Status=status, Risk=None if pd.isna(s.risk) else round(100 * s.risk),
                         Glucose=None if pd.isna(s.cgm) else f"{s.cgm:.0f} {trend_arrow(s.cgm_slope_15m)}",
                         **{"Carbs last 2 h (g)": int(s.carbs_last_2h), "Steps last 30 min": int(s.steps_last_30m),
                            "Sleep last night (h)": None if pd.isna(s.sleep_hours) else round(s.sleep_hours, 1),
                            "HbA1c %": e.hba1c_pct, "Age": e.age, "Medication": e.medication}))
    ward = pd.DataFrame(rows)
    order = {"Spike likely": 0, "Above 180 now": 1, "Watch": 2, "Stable": 3, "Sensor gap": 4}
    ward = ward.sort_values(["Status", "Risk"], key=lambda c: c.map(order) if c.name == "Status" else -c.fillna(-1))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients monitored", len(ward))
    c2.metric("Spike likely in next 2 h", int((ward.Status == "Spike likely").sum()))
    c3.metric("Above 180 right now", int((ward.Status == "Above 180 now").sum()))
    c4.metric("Stable", int((ward.Status == "Stable").sum()))

    st.dataframe(ward, hide_index=True, width="stretch", height=560, column_config={
        "Risk": st.column_config.ProgressColumn("Spike risk, next 2 h", format="%d%%", min_value=0, max_value=100),
    })
    st.caption("All 30 patients here were held out during training — the twin has never seen them before. "
               "Open **Patient twin** to see why a patient is flagged and test what could lower their risk.")

# ---------------------------------------------------------------- patient twin
with tab_twin:
    ranked = ward.Patient.tolist()
    pid = st.selectbox("Patient", ranked, index=0,
                       format_func=lambda p: f"{p} · {ward.set_index('Patient').loc[p, 'Status']}")
    e = ehr.loc[pid]
    pf = feats[feats.patient_id == pid].set_index("timestamp")
    s = pf.loc[now]

    left, mid, right = st.columns([1.1, 1, 1.3])
    with left:
        st.markdown(f"#### {pid}")
        st.markdown(f"""<div class="ehr-grid">
            <span>Age / sex</span><span>{e.age} / {e.sex}</span>
            <span>BMI</span><span>{e.bmi}</span>
            <span>HbA1c</span><span>{e.hba1c_pct}%</span>
            <span>Diabetes for</span><span>{e.years_with_diabetes} years</span>
            <span>Medication</span><span>{e.medication}</span>
            <span>Diagnoses</span><span>{e.past_diagnoses}</span>
            <span>BP</span><span>{e.systolic_bp}/{e.diastolic_bp} mmHg</span>
            <span>LDL / HDL</span><span>{e.ldl_mg_dl} / {e.hdl_mg_dl} mg/dL</span>
            <span>TCF7L2 risk alleles</span><span>{e.tcf7l2_risk_alleles} of 2</span>
            <span>Diet / activity</span><span>{e.diet_pattern} / {e.activity_level}</span>
        </div>""", unsafe_allow_html=True)

    with mid:
        if pd.isna(s.risk):
            label = "Sensor gap — no reading" if pd.isna(s.cgm) else "Already above 180 mg/dL"
            st.markdown(f'<div class="status" style="border-color:{MUTED}"><b>{label}</b><br>'
                        f'The twin predicts only while glucose is below 180 and the sensor is reporting.</div>',
                        unsafe_allow_html=True)
        else:
            band, color = risk_band(s.risk, threshold)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=100 * s.risk, number={"suffix": "%", "font": {"size": 44, "color": color}},
                gauge={"axis": {"range": [0, 100], "tickwidth": 0}, "bar": {"color": color, "thickness": 0.3},
                       "steps": [{"range": [0, 50 * threshold], "color": "#E3EFE8"},
                                 {"range": [50 * threshold, 100 * threshold], "color": "#F6ECD6"},
                                 {"range": [100 * threshold, 100], "color": "#F4DEDA"}],
                       "threshold": {"line": {"color": INK, "width": 2}, "value": 100 * threshold}}))
            gauge.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
            st.markdown(f'<div class="status" style="border-color:{color}"><b>{band}</b><br>'
                        f'{100 * s.risk:.0f}% chance glucose crosses 180 mg/dL before '
                        f'{now + pd.Timedelta(hours=2):%H:%M}.</div>', unsafe_allow_html=True)

    with right:
        g1, g2 = st.columns(2)
        g1.metric("Glucose now", "—" if pd.isna(s.cgm) else f"{s.cgm:.0f} mg/dL",
                  None if pd.isna(s.cgm_slope_15m) else f"{15 * s.cgm_slope_15m:+.0f} in 15 min",
                  delta_color="inverse")
        g2.metric("Heart rate", f"{s.hr_now:.0f} bpm")
        g3, g4 = st.columns(2)
        g3.metric("Sleep last night", "—" if pd.isna(s.sleep_hours) else f"{s.sleep_hours:.1f} h",
                  None if pd.isna(s.deep_sleep_pct) else f"{s.deep_sleep_pct:.0f}% deep", delta_color="off")
        g4.metric("Overnight HRV", "—" if pd.isna(s.overnight_hrv_rmssd_ms) else f"{s.overnight_hrv_rmssd_ms:.0f} ms",
                  None if pd.isna(s.hrv_vs_personal_avg) else f"{100 * (s.hrv_vs_personal_avg - 1):+.0f}% vs usual")
        st.caption(f"Twin memory: this patient's glucose usually rises "
                   f"**{s.personal_meal_rise:.0f} mg/dL** after a meal "
                   f"(~{s.personal_rise_per_gram:.1f} mg/dL per gram of carbs)." if pd.notna(s.personal_meal_rise)
                   else "Twin memory: still learning this patient's meal response.")

    # --- timeline chart
    window = pf.loc[now - pd.Timedelta(hours=24): now]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.04)
    fig.add_hrect(y0=70, y1=180, fillcolor=GREEN, opacity=0.06, line_width=0, row=1, col=1)
    fig.add_hline(y=SPIKE, line=dict(color=RED, dash="dash", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=window.index, y=window.cgm, name="Glucose", line=dict(color=TEAL, width=2)), row=1, col=1)
    meals = window[window.carbs_logged_g > 0]
    fig.add_trace(go.Scatter(x=meals.index, y=[window.cgm.min() - 12] * len(meals), mode="markers+text",
                             text=[f"{c:.0f} g" for c in meals.carbs_logged_g], textposition="top center",
                             marker=dict(symbol="triangle-up", size=11, color=AMBER), name="Meal logged"), row=1, col=1)
    alarms = window[window.risk >= threshold]
    fig.add_trace(go.Scatter(x=alarms.index, y=alarms.cgm, mode="markers", name="Twin alert",
                             marker=dict(color=RED, size=7)), row=1, col=1)
    fig.add_vrect(x0=now, x1=now + pd.Timedelta(hours=2), fillcolor=INK, opacity=0.05, line_width=0,
                  annotation_text="prediction window", annotation_position="top left", row=1, col=1)
    rgrid = window.risk.resample("15min").first()
    fig.add_trace(go.Scatter(x=rgrid.index, y=100 * rgrid, name="Spike risk", fill="tozeroy",
                             line=dict(color="#6B4C9A", width=1.5), connectgaps=False), row=2, col=1)
    fig.add_hline(y=100 * threshold, line=dict(color="#6B4C9A", dash="dot", width=1), row=2, col=1)
    fig.add_trace(go.Bar(x=window.index, y=window.steps, name="Steps", marker_color="#5B8C85"), row=3, col=1)
    fig.update_yaxes(title_text="mg/dL", row=1, col=1)
    fig.update_yaxes(title_text="Risk %", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Steps", row=3, col=1)
    fig.update_xaxes(range=[now - pd.Timedelta(hours=24), now + pd.Timedelta(hours=2)])
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.06, x=0),
                      font=dict(family="IBM Plex Sans, sans-serif", color=INK))
    st.markdown("##### Last 24 hours")
    st.plotly_chart(fig, width="stretch")

    # --- why + what-if
    why_col, whatif_col = st.columns(2)
    x_now = pf.loc[[now]].reset_index()

    with why_col:
        st.markdown("##### Why the twin thinks this")
        if pd.isna(s.risk):
            st.info("Explanations appear when the twin is making a prediction (glucose below 180, sensor reporting).")
        else:
            contrib, base = explain(model, x_now)
            top = contrib.reindex(contrib.contribution.abs().sort_values(ascending=False).index).head(7)

            def describe(r):
                if r.feature == "time_of_day":
                    return f"Time of day ({now:%H:%M}) — routine", "Routine"
                name, unit, source = FEATURE_LABELS[r.feature]
                v = r.value
                if unit == "" and v in (0, 1):
                    val = "yes" if v == 1 else "no"
                elif r.feature == "personal_pct_above_180":
                    val = f"{100 * v:.0f}%"
                elif r.feature in ("cgm_slope_15m", "cgm_slope_30m"):
                    val = f"{v:+.1f} {unit}"
                else:
                    val = f"{v:.1f} {unit}" if abs(v) < 10 else f"{v:.0f} {unit}"
                return f"{name}: {val}", source

            labels, sources = zip(*[describe(r) for r in top.itertuples()])
            bar = go.Figure(go.Bar(
                y=[f"{l}  [{src}]" for l, src in zip(labels, sources)][::-1], x=top.contribution[::-1], orientation="h",
                marker_color=[RED if c > 0 else GREEN for c in top.contribution[::-1]]))
            bar.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor="white",
                              paper_bgcolor="rgba(0,0,0,0)", xaxis_title="← lowers risk      raises risk →",
                              font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=12))
            st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
            st.caption("Bars are SHAP values: each clue's push on this prediction. "
                       "Sources: Wearable, EHR, Twin memory (learned from this patient's past), Routine.")

    with whatif_col:
        st.markdown("##### What if…")
        history = pf.loc[now - pd.Timedelta(days=3) + pd.Timedelta(minutes=5): now]
        if history.cgm.notna().sum() < 100:
            st.info("Not enough recent glucose data to calibrate this patient's physiology yet.")
        else:
            params, fit_err = get_physiology(pid, now)
            meal = st.slider("Meal eaten now", 0, 150, 0, 10, format="%d g carbs")
            walk = st.slider("Walk starting now", 0, 45, 0, 5, format="%d min")
            reveal = st.toggle("Show what actually happened", value=False,
                               help="For validation only: the twin never sees future data.")

            base_fc = forecast(params, history)
            scen_fc = forecast(params, history, meal, walk)
            t_fc = pd.date_range(now, periods=len(base_fc), freq="5min")
            past = history.loc[now - pd.Timedelta(hours=2):]

            w = go.Figure()
            w.add_hrect(y0=70, y1=180, fillcolor=GREEN, opacity=0.06, line_width=0)
            w.add_hline(y=SPIKE, line=dict(color=RED, dash="dash", width=1))
            w.add_trace(go.Scatter(x=past.index, y=past.cgm, name="Measured", line=dict(color=TEAL, width=2)))
            w.add_trace(go.Scatter(x=t_fc, y=base_fc, name="As things stand",
                                   line=dict(color=MUTED, dash="dash", width=2)))
            if meal or walk:
                w.add_trace(go.Scatter(x=t_fc, y=scen_fc, name="With your changes",
                                       line=dict(color="#6B4C9A", width=3)))
            if reveal:
                actual = pf.loc[now: now + pd.Timedelta(hours=3)]
                w.add_trace(go.Scatter(x=actual.index, y=actual.cgm, name="What actually happened",
                                       line=dict(color=INK, dash="dot", width=1.5)))
            w.add_vline(x=now, line=dict(color=INK, width=1))
            w.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                            paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.15),
                            yaxis_title="mg/dL", font=dict(family="IBM Plex Sans, sans-serif", color=INK))
            st.plotly_chart(w, width="stretch", config={"displayModeBar": False})

            def minutes_above(fc):
                return int((fc > SPIKE).sum() * 5)
            m1, m2 = st.columns(2)
            m1.metric("Peak, next 3 h", f"{scen_fc.max():.0f} mg/dL",
                      f"{scen_fc.max() - base_fc.max():+.0f} vs as things stand" if (meal or walk) else None,
                      delta_color="inverse")
            m2.metric("Time above 180", f"{minutes_above(scen_fc)} min",
                      f"{minutes_above(scen_fc) - minutes_above(base_fc):+d} min" if (meal or walk) else None,
                      delta_color="inverse")
            st.markdown(f'<p class="small-note">A personal physiology model fitted to {pid}\'s last 3 days of '
                        f'glucose, meals and steps (typical error ±{fit_err:.0f} mg/dL). It learned that this '
                        f'patient\'s glucose rises about {params["carb_sensitivity"]:.1f} mg/dL per gram of '
                        f'absorbed carbs and that walking speeds up sugar clearance by about '
                        f'{100 * params["walk_boost"]:.0f}%. A decision aid for conversations with the patient, '
                        f'not a treatment instruction.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- performance
with tab_perf:
    st.subheader("Does fusing EHR and wearable data help?")
    comp = pd.read_csv(find_file("model_comparison.csv"))
    fig = go.Figure(go.Bar(x=comp.AUROC, y=comp.model, orientation="h",
                           marker_color=["#B7C4C1", "#8FB0AB", "#4E8E87", TEAL],
                           text=comp.AUROC.map("{:.3f}".format), textposition="outside"))
    fig.update_layout(height=260, xaxis=dict(range=[0.5, 1.0], title="AUROC on 30 unseen patients"),
                      margin=dict(l=10, r=40, t=10, b=30), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans, sans-serif", color=INK))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("AUROC runs from 0.5 (coin toss) to 1.0 (perfect). The fused twin beats both single sources "
                "and a simple clinical rule.")

    @st.cache_data
    def source_importance():
        sample = feats[feats.risk.notna()].sample(3000, random_state=0)
        contrib = model.predict(sample[ALL_FEATURES], pred_contrib=True)[:, :-1]
        imp = pd.DataFrame({"feature": ALL_FEATURES, "importance": np.abs(contrib).mean(axis=0)})
        imp["source"] = imp.feature.map(lambda f: FEATURE_LABELS[f][2])
        return imp.groupby("source").importance.sum().sort_values()

    st.markdown("##### Where the twin's decisions come from")
    imp = source_importance()
    fig2 = go.Figure(go.Bar(x=100 * imp / imp.sum(), y=imp.index, orientation="h", marker_color=TEAL,
                            text=(100 * imp / imp.sum()).map("{:.0f}%".format), textposition="outside"))
    fig2.update_layout(height=220, margin=dict(l=10, r=40, t=10, b=30), plot_bgcolor="white",
                       paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Share of total influence (mean |SHAP|)",
                       font=dict(family="IBM Plex Sans, sans-serif", color=INK))
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
    st.markdown("""
**In clinical terms, on 30 patients the twin never trained on (alert level 60%):**
spikes caught in advance **97%**, median warning **~95 minutes** before glucose crosses 180,
false alarms **0.7 per patient per day**.

**The physiology simulator** behind *What if…* forecasts glucose with a typical error of **12 mg/dL one hour
ahead and 14 mg/dL two hours ahead**, compared with 23 and 37 mg/dL for assuming glucose stays where it is
(30 unseen patients, forecasts every hour over three days, given the meals and steps that followed).

**Honest limits.** These patients are synthetic and follow steadier routines than real people, so the twin can
partly anticipate meals from the time of day. Real-world accuracy will be lower. Next steps would be validation
on public CGM datasets and a prospective pilot with clinicians.
""")

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("About GlucoTwin")
    st.markdown("""
**The problem.** India has over 100 million adults with diabetes. Blood-sugar spikes after meals damage blood
vessels, nerves and kidneys over time, yet patients usually find out only after the spike has happened.

**What GlucoTwin does.** It keeps a virtual copy of each patient that combines two data streams:

- **Static health record (EHR):** age, BMI, HbA1c, medication, diagnoses, blood pressure, cholesterol and a
  genetic risk marker (TCF7L2).
- **Live wearable data:** continuous glucose every 5 minutes, heart rate, steps, sleep stages, overnight HRV and
  logged meals.

Every 15 minutes the twin estimates the chance that glucose will cross 180 mg/dL in the next two hours, explains
why, and lets a doctor test changes such as a short walk after lunch.

**Data.** All 100 patients are synthetic, generated with physiology-based rules calibrated to clinical benchmarks
(HbA1c to average-glucose formula, time-in-range guidelines). No real patient data is used, in line with the
DPDP Act and HIPAA.

**How the twin thinks — two engines.**

- *Early warning (machine learning):* LightGBM gradient-boosted trees with 53 features from the EHR, wearables
  and the patient's own history, trained on 60 patients, tuned on 10 and tested on 30 unseen patients.
  Explanations use exact tree SHAP values.
- *Scenario simulator (physiology):* a small glucose model with four personal parameters (baseline glucose,
  rise per gram of carbs, clearance speed, and how much walking helps), refitted to each patient's last three
  days. It answers what-if questions the machine-learning model cannot answer reliably on its own.

**Not a medical device.** This is a research proof-of-concept built for the Happiest Health Reimagining and
Reforming Healthcare in India Summit 2026.

*Team I Dont Think We Can Code · XLRI Jamshedpur*
""")
