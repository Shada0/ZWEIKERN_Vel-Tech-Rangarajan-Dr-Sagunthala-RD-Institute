"""Dashboard views A-H + Ravi demo + replay + check-in + limitations."""
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src.config import *
from src.data.preprocessing import FEATURES
from src.data.validation import validate, missingness
from src.models.baseline_model import lr_contributions
from src.models.explainability import twin_sensitivity
from src.simulation.crash_replay import list_events, replay
from src.simulation.heatwave_simulator import change_table, DEMO_TARGET_GAIN_KG
from src.dashboard.common import *


def _pr(D):
    return D.prior if D.prior is not None else ((PRIOR_LOGK[0], PRIOR_LOGS[0]), (PRIOR_LOGK[1], PRIOR_LOGS[1]))


def _paper(E):
    return E["patient_holdout"]["models"]


def home():
    D, F, P, M, E = load_all(); RR = _real()
    st.markdown("""<div class='hero'><h1>NephroTwin AI</h1><p><b>A personalized dialysis digital twin for predicting intradialytic hypotension</b></p>
    <p>Predict the risk. Understand the patient. Plan dialysis more safely.</p>
    <p style='font-size:.85rem;opacity:.85'>Happiest Health Digital Twin Challenge 2026 - research prototype on synthetic data, validated against an open real-world dataset.</p></div>""", unsafe_allow_html=True)
    banner()
    a, b, c = st.columns(3)
    a.markdown("<div class='card'><b>1 &nbsp;Two data streams</b><br><span class='muted'>Static EHR history (profile, comorbidities, labs, past sessions) fused with dynamic signals (home weight, HR/HRV, activity, weather) up to the prediction timestamp.</span></div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><b>2 &nbsp;A physiological twin per patient</b><br><span class='muted'>Two-compartment fluid model + BP response. Patient-specific refill rate and BP sensitivity are estimated from past sessions; a Kalman filter tracks fluid excess.</span></div>", unsafe_allow_html=True)
    c.markdown("<div class='card'><b>3 &nbsp;Risk + what-if simulation</b><br><span class='muted'>Pre-session hypotension probability with uncertainty, and hypothetical fluid-removal scenarios. Simulation only; clinician review required.</span></div>", unsafe_allow_html=True)
    st.markdown("### What the evidence shows (computed, held-out patients)")
    s, r = _paper(E), (_paper(RR) if RR else None)
    k = st.columns(4)
    k[0].metric("Synthetic: twin AUC", f"{s['Personalized digital twin']['auc']:.3f}", f"{s['Personalized digital twin']['auc'] - s['Standard model: logistic regression']['auc']:+.3f} vs standard LR")
    if r:
        k[1].metric("Real HEMOBP: twin AUC", f"{r['Personalized digital twin']['auc']:.3f}", f"{r['Personalized digital twin']['auc'] - r['Standard model: logistic regression']['auc']:+.3f} vs standard LR", delta_color="normal")
        k[2].metric("Real HEMOBP: hybrid AUC", f"{r['Hybrid: standard features + twin outputs']['auc']:.3f}", f"{r['Hybrid: standard features + twin outputs']['auc'] - r['Standard model: logistic regression']['auc']:+.3f} vs standard LR (about on par)", delta_color="off")
        metric_cap(k[3], "Real sessions evaluated", f"{list(r.values())[0]['n']:,}", f"{list(r.values())[0]['prevalence']:.1%} hypotension prevalence")
    st.info("**Honest finding.** On synthetic data the personalized twin beats the standard model. On real HEMOBP data the twin alone does **not**: sparse readings make per-patient parameters hard to pin down. "
            "Combining the twin's outputs with standard features (hybrid) roughly matches the best standard model. Details: *Real-data validation* and *Model evaluation*.")
    st.markdown("**Start here:** open **Ravi demonstration** for the guided 7-step story, then **Historical replay**, **Pre-session card** (what-if planner) and **Real-data validation**.")


def _real():
    import json
    f = MODEL_DIR / "evaluation_real.json"
    return json.load(open(f)) if f.exists() else None


def real_validation():
    D, F, P, M, E = load_all(); RR = _real()
    st.title("Real-data validation - HEMOBP"); banner_real()
    if RR is None:
        st.warning("Run `python -m src.cli real <folder>` to generate real-data results (see docs/real_data.md)."); return
    st.caption("Open dataset: Lin et al., Sci Data 2019 (figshare doi:10.6084/m9.figshare.6260654). 328 patients, 14,638 sessions, ~105k BP readings. No wearables, weather, labs or medications. Hypotension = any SBP < 90 after the first reading.")
    sp = st.radio("Split", ["patient_holdout", "temporal"], horizontal=True, format_func=lambda x: "Patient-grouped hold-out" if x == "patient_holdout" else "Temporal hold-out", key="rsplit")
    R = RR[sp]; mods = R["models"]
    tab = pd.DataFrame({short(m): {c: v[c] for c in ("auc", "brier", "precision", "recall", "f1")} for m, v in mods.items()}).T
    tab.columns = ["ROC-AUC", "Brier", "Precision", "Recall", "F1"]
    st.dataframe(tab.style.format("{:.3f}").background_gradient(subset=["ROC-AUC"], cmap="GnBu"), width="stretch")
    d, h = R["auc_diff_twin_minus_lr"], R["auc_diff_hybrid_minus_lr"]
    c1, c2 = st.columns(2)
    c1.metric("Personalized twin AUC", f"{mods['Personalized digital twin']['auc']:.3f}", f"{d['mean']:+.3f} vs standard LR", delta_color=delta_color_for(d['lo'], d['hi']))
    c1.caption(f"95% patient-cluster bootstrap CI of the difference: {d['lo']:+.3f} to {d['hi']:+.3f}")
    c2.metric("Hybrid AUC", f"{mods['Hybrid: standard features + twin outputs']['auc']:.3f}", f"{h['mean']:+.3f} vs standard LR", delta_color=delta_color_for(h['lo'], h['hi']))
    c2.caption(f"95% CI of the difference: {h['lo']:+.3f} to {h['hi']:+.3f} (includes or touches 0: about on par)")
    st.plotly_chart(roc_fig(mods, "ROC curves (real HEMOBP data)"), width="stretch")
    st.markdown("**How to read this.** The twin alone trails the standard model on real data, but personalization clearly helps relative to a twin with population-average parameters. "
                "The hybrid (standard features + twin outputs) is on par with the best standard model; any gain is small. These are retrospective results on one hospital's data and do not establish clinical benefit.")
    with st.expander("Why might the twin underperform on real data? (hypotheses, untested)"):
        st.markdown("- Median 6 BP readings per session make per-patient parameter estimates noisy (the learned population prior is wide).\n- The standard model can use each patient's past event rate, which carries a lot of signal; the twin's physiology does not.\n"
                    "- No medications, comorbidity detail or heart rate in this dataset; real hypotension has causes outside a fluid-volume mechanism.\n- Possible next steps: richer observation model, hierarchical Bayesian pooling, more frequent BP data.")


def about_data():
    pass


def _tbl(D, P):
    up = P[P.is_upcoming == 1].copy()
    up["category"] = up.p_twin.map(cat_of)
    up["twin_risk"], up["standard_risk"] = up.p_twin, up.p_lr
    up["completeness"] = [D.completeness(p) for p in up.patient_id]
    up["next_session"] = [D.sessions.loc[s, "date"].date() for s in up.session_id]
    up["schedule"] = up.patient_id.map(lambda p: f"{D.patients.loc[p, 'dialysis_per_week']}x/week")
    up["diabetes"] = up.patient_id.map(lambda p: "Yes" if D.patients.loc[p, "diabetes"] else "No")
    up["priority"] = up.p_twin * (1 + 0.5 * np.minimum(up.nadir_sd / 15, 1))
    return up


def overview():
    D, F, P, M, E = load_all(); up = _tbl(D, P); h = D.sessions[D.sessions.is_upcoming == 0]
    st.title("NephroTwin AI - Unit overview")
    st.caption("A personalized dialysis digital twin for predicting intradialytic hypotension (synthetic research demo)")
    c = st.columns(5)
    c[0].metric("Synthetic patients", len(D.patients)); c[1].metric("Upcoming sessions", len(up))
    c[2].metric("Historical sessions", f"{len(h):,}"); c[3].metric("Observed hypotension rate", f"{h.hypotension.mean():.1%}")
    c[4].metric("Mean twin risk (upcoming)", f"{up.p_twin.mean():.1%}")
    l, r = st.columns([1, 2])
    cnt = up.category.value_counts().reindex(["Green", "Amber", "Red"]).fillna(0)
    l.plotly_chart(px.pie(values=cnt.values, names=cnt.index, color=cnt.index, color_discrete_map=COL, hole=.55, title="Risk distribution").update_layout(height=320, margin=dict(t=45, b=0)), width="stretch")
    r.markdown("**Upcoming-session priorities** (calculated risk + uncertainty; does not assign treatment)")
    top = up.sort_values("priority", ascending=False).head(8)
    r.dataframe(top[["patient_id", "next_session", "twin_risk", "standard_risk", "category", "nadir_sd"]].rename(columns={"nadir_sd": "uncertainty (mmHg)"})
                .style.format({"twin_risk": fmt_p, "standard_risk": fmt_p, "uncertainty (mmHg)": "{:.1f}"}), hide_index=True, width="stretch")
    mon = h.assign(m=h.date.dt.to_period("M").astype(str)).groupby("m").hypotension.mean().reset_index()
    st.plotly_chart(px.bar(mon, x="m", y="hypotension", title="Population-level hypotension rate by month (synthetic)", color_discrete_sequence=[TEAL], labels={"m": "Month", "hypotension": "Hypotension rate"}).update_layout(height=280, template="nephro", yaxis_tickformat=".0%", xaxis_title="Month", yaxis_title="Hypotension rate"), width="stretch")
    q = validate(D); st.markdown("**Synthetic data-quality indicators:** " + " | ".join(f"{'OK' if v else 'FAIL'} {k}" for k, v in q.items()))


def patient_list():
    D, F, P, M, E = load_all(); up = _tbl(D, P)
    st.title("Patient list"); st.caption("Synthetic patients with an upcoming session")
    cols = ["patient_id", "age", "diabetes", "schedule", "next_session", "twin_risk", "standard_risk", "category", "completeness"]
    up = up.sort_values("twin_risk", ascending=False)
    st.dataframe(up[cols].style.format({"twin_risk": fmt_p, "standard_risk": fmt_p, "completeness": "{:.0%}"})
                 .map(lambda v: f"background-color:{COL.get(v,'')};color:white" if v in COL else "", subset=["category"]), hide_index=True, width="stretch", height=520)
    pid = st.selectbox("Select patient for twin profile / pre-session card", up.patient_id, format_func=lambda x: label(D, x), key="sel_pid_list")
    if st.button("Open selected patient"):
        st.session_state["pid"] = pid; st.success(f"Selected {pid}. Use the sidebar to open the Twin profile or Pre-session card.")


def _pid():
    D = load_all()[0]
    ids = list(D.patients.index)
    return st.sidebar.selectbox("Patient", ids, index=ids.index(st.session_state.get("pid", "SYN-RAVI")), format_func=lambda x: label(D, x))


def twin_profile(pid):
    D, F, P, M, E = load_all(); p = D.patients.loc[pid]; sid = upcoming_sid(D, pid); ctx = get_ctx(sid, False)
    st.title(f"Digital twin profile - {label(D, pid)}"); banner()
    st.markdown(f"**{p.age} y, {p.sex}**  |  Diabetes: {'Yes' if p.diabetes else 'No'}  |  Hypertension: {'Yes' if p.hypertension else 'No'}  |  CVD: {'Yes' if p.cardiovascular_disease else 'No'}  |  "
                f"{p.dialysis_per_week}x/week  |  Baseline SBP {p.baseline_sbp}  |  Dry weight {p.dry_weight_kg} kg  |  Meds: {p.bp_medication_categories}")
    pm = ctx["twin"].params
    a, b, c, d = st.columns(4)
    metric_cap(a, "Refill rate k (/h)", f"{pm['k_refill']:.2f}", f"population median {np.exp(_pr(D)[0][0]):.2f}")
    metric_cap(b, "BP sensitivity S", f"{pm['S_bp']:.2f}", f"population median {np.exp(_pr(D)[0][1]):.2f}")
    metric_cap(c, "Fluid gain g (kg/day)", f"{pm['g_kg_day']:.2f}", f"\u00b1 {pm['g_sd']:.2f} kg/day")
    metric_cap(d, "HR response (bpm/%)", f"{pm['c_hr']:.2f}", "per % blood-volume drop")
    st.caption(f"Parameter uncertainty (log-sd): k {pm['sd_logk']:.2f}, S {pm['sd_logS']:.2f}; fitted from {pm['n_sessions']} prior sessions ({pm['n_obs']} BP observations) by MAP nonlinear least squares.")
    stt = ctx["state"]
    st.markdown(f"**Estimated current state (Kalman-fused):** fluid excess {stt['pre_excess']:.2f} +- {stt['F_sd']:.2f} kg over recorded dry weight; gain since last session {stt['gain_since_last']:.2f} kg.")
    h = D.sess_by_pid[pid]; h = h[h.is_upcoming == 0].tail(30)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=h.date, y=h.min_sbp, mode="lines+markers", name="Session minimum SBP", marker=dict(color=["#d64545" if v == 1 else NAVY for v in h.hypotension])))
    fig.add_trace(go.Bar(x=h.date, y=h.idwg_kg, name="Interdialytic weight gain (kg)", yaxis="y2", opacity=.3, marker_color=TEAL))
    fig.add_hline(y=HYPO_SBP, line_dash="dot", line_color="#d64545")
    fig.update_layout(title="Historical trends (last 30 sessions; red = hypotension)", height=340, template="nephro", yaxis2=dict(overlaying="y", side="right", title="kg"), legend=dict(orientation="h", y=-.2), margin=dict(t=45, b=10))
    st.plotly_chart(fig, width="stretch")
    s = D.sens_by_pid[pid]; s = s[s.date <= pd.Timestamp("2026-10-23")].tail(10)
    st.markdown("**Dynamic sensor readings (last 10 days; blanks = missing)**")
    show_table(s[["date", "temp_c", "home_weight_kg", "home_sbp", "hr_rest", "hrv_rmssd_ms", "activity_steps", "sleep_h"]], SENSOR_FMT)
    st.markdown("**Prediction explanation** (association, not causality)")
    l, r = st.columns(2)
    row = F.loc[sid]; con = lr_contributions(M["baselines"]["lr"], row).head(8)
    l.plotly_chart(px.bar(con[::-1], orientation="h", title="Standard model (logistic regression): top logit contributions", color_discrete_sequence=[NAVY], labels={"value": "Logit contribution", "index": "Feature"}).update_layout(height=330, showlegend=False, template="nephro", margin=dict(t=45), xaxis_title="Logit contribution", yaxis_title=""), width="stretch")
    base, sens = twin_sensitivity(ctx["twin"], ctx["plan"], M["twin_cal"])
    r.markdown(f"Twin sensitivity around current risk **{fmt_p(base)}** (one-at-a-time, local)")
    r.dataframe(sens.style.format({"risk": fmt_p, "change_vs_current": fmt_d}), hide_index=True, width="stretch")
    with st.expander("Synthetic ground truth (for validation only; never used by models)"):
        t = D.truth.loc[pid]
        st.write(f"True k = {np.exp(t.true_log_k):.2f}, true S = {np.exp(t.true_log_S):.2f}, true g = {t.true_g_kg_day:.2f}  vs estimated k = {pm['k_refill']:.2f}, S = {pm['S_bp']:.2f}, g = {pm['g_kg_day']:.2f}")


def presession(pid):
    D, F, P, M, E = load_all(); sid = upcoming_sid(D, pid)
    st.title(f"Pre-session card - {label(D, pid)}"); banner()
    heat = st.toggle("Heat-wave scenario for this patient (synthetic)", key=f"hw_{pid}")
    ctx = get_ctx(sid, heat); p, r = ctx["p"], ctx["row"]; std = lr_for_ctx(F, M, sid, ctx)[0]
    a, b, c = st.columns([1.2, 1, 2])
    a.markdown(f"<div class='card'><div class='muted'>Twin-estimated hypotension probability</div><div class='big'>{fmt_p(p)}</div>{chip(cat_of(p))}"
               f"<div class='muted'>Demo thresholds (amber {st.session_state.get('amber', AMBER_DEFAULT):.0%} / red {st.session_state.get('red', RED_DEFAULT):.0%}) - configurable, not clinical</div></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='card'><div class='muted'>Standard population model</div><div class='big'>{fmt_p(std)}</div><div class='muted'>re-evaluated for this scenario</div></div>", unsafe_allow_html=True)
    c.markdown(f"**Prediction timestamp:** {r.prediction_timestamp} (scheduled connection time; only pre-timestamp data used)  \n"
               f"**Planned:** {ctx['plan'].uf_volume:.1f} L over {ctx['plan'].duration_h:.2f} h ({ctx['plan'].uf_rate_l_h:.2f} L/h), starting SBP {r.pre_sbp:.0f}, fluid excess {ctx['plan'].fluid_excess:.2f} kg  \n"
               f"**Uncertainty:** simulated nadir SBP {ctx['pred'].nadir_mean:.0f} +- {ctx['pred'].nadir_sd:.0f} mmHg (parameter uncertainty); {ctx['twin'].n_sessions} prior sessions informed the twin.")
    miss = [n for n, v in [("wearable HR/HRV (3d)", F.loc[sid, "hrv_3d"]), ("home weight", D.sens_by_pid[pid].iloc[-4:].home_weight_kg.mean())] if not np.isfinite(v)]
    st.markdown("**Missing information:** " + (", ".join(miss) if miss else "none critical") + f"  |  data completeness {D.completeness(pid):.0%}")
    st.markdown("**Main contributors (twin sensitivity):**")
    base, sens = twin_sensitivity(ctx["twin"], ctx["plan"], M["twin_cal"]); st.dataframe(sens.style.format({"risk": fmt_p, "change_vs_current": fmt_d}), hide_index=True, width="stretch")
    st.markdown("**Considerations / questions for the clinician:** Is the fluid-gain estimate consistent with the clinical picture? Is the planned fluid-removal rate appropriate for this patient's previous responses? Are medications timed relative to the session? *(Prompts only, not orders.)*")
    st.divider(); planner(ctx, f"ps_{pid}")
    st.warning("Review with nephrologist before any treatment decision.")


def priority():
    D, F, P, M, E = load_all(); up = _tbl(D, P).sort_values("priority", ascending=False)
    st.title("Weekly priority list"); st.caption("Patients who may merit earlier clinical review: priority = risk x (1 + 0.5 x normalised parameter uncertainty). A heuristic ordering only; no treatment is assigned.")
    st.dataframe(up[["patient_id", "next_session", "twin_risk", "standard_risk", "category", "nadir_sd", "n_fit", "completeness", "priority"]]
                 .style.format({"twin_risk": fmt_p, "standard_risk": fmt_p, "nadir_sd": "{:.1f}", "completeness": "{:.0%}", "priority": "{:.2f}", "n_fit": "{:,.0f}"}), hide_index=True, width="stretch", height=560)


def replay_view():
    D, F, P, M, E = load_all(); ev = list_events(D, P)
    st.title("Historical replay"); banner()
    pids = sorted(ev.patient_id.unique(), key=lambda x: (x != "SYN-RAVI", x))
    pid = st.selectbox("Patient", pids, format_func=lambda x: label(D, x)); e = ev[ev.patient_id == pid]
    sid = st.selectbox("Historical simulated hypotension event", e.index, format_func=lambda s: f"{e.loc[s, 'date'].date()} ({s})")
    R = replay(D, sid, M["twin_cal"], P.loc[sid]); r = R["row"]
    st.markdown(f"**Prediction timestamp: {R['timestamp']}** - information available before this moment only (prior {R['twin'].n_sessions} sessions, home sensors, pre-dialysis vitals).")
    a, b = st.columns(2)
    a.metric("Standard model risk", fmt_p(R['p_lr'])); b.metric("Personalized twin risk", fmt_p(R['p_twin']))
    flagged = R["p_twin"] >= st.session_state.get("amber", AMBER_DEFAULT)
    st.markdown(f"Twin category at demo thresholds: {chip(cat_of(R['p_twin']))} - {'flagged BEFORE the session as elevated' if flagged else 'not flagged as elevated before the session'}.", unsafe_allow_html=True)
    show_table(pd.DataFrame([{"gap (days)": r.gap_days, "pre-weight (kg)": r.pre_weight_kg, "planned UF (L)": r.planned_uf_l, "duration (h)": r.duration_h, "pre-SBP (mmHg)": r.pre_sbp, "pre-HR (bpm)": r.pre_hr, "temp (\u00b0C)": r.temp_c}]),
               {"gap (days)": "{:.0f}", "pre-weight (kg)": "{:.1f}", "planned UF (L)": "{:.1f}", "duration (h)": "{:.2f}", "pre-SBP (mmHg)": "{:.0f}", "pre-HR (bpm)": "{:.0f}", "temp (\u00b0C)": "{:.1f}"})
    reveal = st.toggle("Reveal subsequent simulated outcome", value=False)
    st.plotly_chart(sbp_fig({"Twin prediction (made before session)": R["pred"]}, observed=(R["t"], R["sbp"], R["keep"]) if reveal else None), width="stretch")
    if reveal:
        st.error(f"Simulated outcome: hypotension occurred (minimum SBP {r.min_sbp:.0f} mmHg at {r.nadir_time_h:.2f} h). Red markers = readings after the event (excluded from twin fitting).")
        st.caption("This replay shows whether the model flagged the event beforehand. It does not show that any alternative plan would have prevented it.")


def evaluation():
    D, F, P, M, E = load_all()
    st.title("Model evaluation"); banner()
    sp = st.radio("Evaluation split", ["patient_holdout", "temporal"], horizontal=True, format_func=lambda x: {"patient_holdout": "Patient-grouped held-out test set", "temporal": "Temporal hold-out (later sessions, training patients)"}[x])
    R = E[sp]; mods = R["models"]
    st.caption(f"{R['n_test_patients']} test patients; sessions with >= 6 prior sessions; threshold for P/R/F1 chosen on training data. Synthetic results - not evidence of clinical effectiveness.")
    tab = pd.DataFrame({short(m): {k: v[k] for k in ("auc", "brier", "precision", "recall", "f1", "n", "prevalence")} for m, v in mods.items()}).T
    tab["n"] = tab["n"].astype(int); tab.columns = ["ROC-AUC", "Brier", "Precision", "Recall", "F1", "Sessions", "Prevalence"]
    st.dataframe(tab.style.format({"ROC-AUC": "{:.3f}", "Brier": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Sessions": "{:,.0f}", "Prevalence": "{:.1%}"}), width="stretch")
    d = R["auc_diff_twin_minus_lr"]
    st.markdown(f"**AUC difference (twin - logistic regression):** {d['mean']:+.3f} (95% patient-cluster bootstrap CI {d['lo']:+.3f} to {d['hi']:+.3f}). "
                + ("The interval excludes 0 in this synthetic setting." if d["lo"] > 0 else "The interval includes 0: no reliable advantage shown on this split."))
    hh = R["auc_diff_hybrid_minus_lr"]; st.markdown(f"**AUC difference (hybrid - logistic regression):** {hh['mean']:+.3f} (95% CI {hh['lo']:+.3f} to {hh['hi']:+.3f}).")
    st.info("Caveat: the twin's structure (two-compartment model) is close to the synthetic data-generating process, which favours it. Real data would not offer this. "
            "The ablation row shows what is lost when personalization is switched off.")
    l, r = st.columns(2)
    l.plotly_chart(roc_fig(mods, "ROC curves (synthetic cohort)"), width="stretch")
    fig = go.Figure()
    for m in ("Standard model: logistic regression", "Personalized digital twin"):
        v = mods[m]; fig.add_trace(go.Scatter(x=v["calib"][0], y=v["calib"][1], mode="lines+markers", name=short(m)))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
    fig.update_layout(title="Calibration (quantile bins)", height=470, template="nephro", xaxis_title="Mean predicted", yaxis_title="Observed fraction", legend=dict(orientation="h", y=-.3))
    r.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    for col, m in ((c1, "Standard model: logistic regression"), (c2, "Personalized digital twin")):
        cm = np.array(mods[m]["cm"]); col.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred -", "Pred +"], y=["Actual -", "Actual +"], color_continuous_scale="Blues", title=f"Confusion matrix: {short(m)}").update_layout(height=300, coloraxis_showscale=False), width="stretch")
    pp = pd.DataFrame(R["per_patient"])
    st.plotly_chart(px.scatter(pp, x="standard_brier", y="twin_brier", size="n", hover_name="patient_id", title="Patient-level Brier score (below diagonal = twin better)", color_discrete_sequence=[TEAL], labels={"standard_brier": "Standard model Brier score", "twin_brier": "Twin Brier score", "n": "Sessions"}).add_shape(type="line", x0=0, y0=0, x1=pp.standard_brier.max(), y1=pp.standard_brier.max(), line=dict(dash="dot")).update_layout(height=380, template="nephro", xaxis_title="Standard model Brier score", yaxis_title="Twin Brier score"), width="stretch")
    pr = E.get("parameter_recovery", dict(corr_logS=float("nan"), corr_logk=float("nan"))); st.markdown(f"**Parameter recovery vs hidden synthetic truth (final fit per patient):** corr(log S) = {pr['corr_logS']:.2f}, corr(log k) = {pr['corr_logk']:.2f}.")


def quality():
    D, F, P, M, E = load_all()
    st.title("Data quality"); st.caption("Synthetic dataset information and prediction eligibility")
    for k, v in validate(D).items(): st.write(("OK  " if v else "FAIL  ") + k)
    m = missingness(D); st.plotly_chart(px.bar(m, y="missing_fraction", title="Missing fraction by sensor channel", color_discrete_sequence=[TEAL], labels={"index": "Sensor channel", "missing_fraction": "Missing fraction"}).update_layout(height=300, template="nephro", yaxis_tickformat=".0%", xaxis_title="Sensor channel", yaxis_title="Missing fraction"), width="stretch")
    comp = pd.Series({p: D.completeness(p) for p in D.patients.index})
    st.plotly_chart(px.histogram(comp, nbins=15, title="Per-patient 14-day data completeness", color_discrete_sequence=[NAVY], labels={"value": "Data completeness", "count": "Patients"}).update_layout(height=280, template="nephro", showlegend=False, xaxis_title="Data completeness (14 days)", yaxis_title="Patients", xaxis_tickformat=".0%"), width="stretch")
    elig = F[F.is_upcoming == 1]; st.write(f"Prediction eligibility: {int((elig.n_prior >= 6).sum())}/{len(elig)} upcoming sessions have >= 6 prior sessions for twin personalization.")
    st.markdown(f"**Dataset:** seed {SEED}; {len(D.patients)} patients; {len(D.sessions):,} sessions; {len(D.sensors):,} patient-days of sensor/weather data; {len(D.obs):,} timed BP/HR observations. Files in `data/` (see docs/data_dictionary.md).")


def checkin():
    D, F, P, M, E = load_all()
    st.title("Simulated daily check-in"); banner()
    st.caption("Local simulation - no WhatsApp or real account. The weight entry updates the twin's Kalman fluid state; symptoms and HR are logged as context and are NOT converted into physiology.")
    pid = "SYN-RAVI"; sid = upcoming_sid(D, pid); base = get_ctx(sid, False); log = st.session_state.setdefault("checkins", [])
    with st.form("ci"):
        c = st.columns(5)
        w = c[0].number_input("Morning weight (kg)", 50.0, 120.0, float(round(base["pre_weight"] - 0.4, 1)), 0.1)
        wb = c[1].slider("Well-being (1-5)", 1, 5, 3); dz = c[2].checkbox("Dizziness"); fa = c[3].checkbox("Fatigue"); hr = c[4].number_input("Heart rate", 40, 160, 78)
        ok = st.form_submit_button("Submit check-in")
    if ok: log.append(dict(weight=w, wellbeing=wb, dizziness=dz, fatigue=fa, hr=hr))
    if log:
        new = get_ctx(sid, False, tuple(x["weight"] for x in log[-1:]))
        a, b = st.columns(2)
        a.metric("Fluid excess before check-in (kg)", f"{base['state']['pre_excess']:.2f}"); b.metric("After check-in (kg)", f"{new['state']['pre_excess']:.2f}", f"{new['state']['pre_excess'] - base['state']['pre_excess']:+.2f}", delta_color="off")
        st.metric("Twin risk before -> after", f"{fmt_p(base['p'])} -> {fmt_p(new['p'])}")
        st.dataframe(pd.DataFrame(log).rename(columns={"weight": "weight (kg)", "wellbeing": "well-being (1-5)", "hr": "heart rate (bpm)"}), hide_index=True, width="stretch")
        st.caption("In this prototype the clinic-measured pre-dialysis weight dominates the Kalman update (low noise), so the effect of a single home weight is small by design.")


def limitations():
    st.title("Clinical limitations & safety"); st.markdown((ROOT / "docs" / "clinical_limitations.md").read_text())


def lr_for_ctx(F, M, sid, ctx):
    row = F.loc[sid].copy(); pl = ctx["plan"]
    row["pre_weight"] = ctx["pre_weight"]; row["excess_est"] = ctx["pre_weight"] - ctx["dry"]; row["planned_uf"] = pl.uf_volume
    row["uf_rate_l_h"] = pl.uf_rate_l_h; row["uf_rate_ml_kg_h"] = 1000 * pl.uf_rate_l_h / ctx["pre_weight"]
    row["temp_c"] = pl.temp_c; row["heatwave_flag"] = int(pl.temp_c >= 38); row["temp_3d_max"] = max(row["temp_3d_max"], pl.temp_c)
    return float(M["baselines"]["lr"].predict_proba(row[FEATURES].to_frame().T.astype(float))[0, 1]), row


def ravi():
    D, F, P, M, E = load_all(); pid = "SYN-RAVI"; sid = upcoming_sid(D, pid); p = D.patients.loc[pid]
    st.title("Guided demonstration - Ravi"); banner()
    top = st.columns([3, 1.4])
    step = top[0].radio("Step", ["1 Patient overview", "2 Dynamic data update", "3 Heat-wave scenario", "4 Digital twin update", "5 Risk flag", "6 What-if planner", "7 Clinical review"], horizontal=True, key="ravi_step")
    # The toggle is rendered on EVERY step so Streamlit never discards its value when the step changes.
    heat = top[1].toggle("Heat-wave scenario (synthetic)", key="ravi_heat", help="Switch on at Step 3; every later step recomputes from it.")
    base = get_ctx(sid, False); hot = get_ctx(sid, True); cur = hot if heat else base; r = base["row"]
    if step.startswith("1"):
        st.markdown(f"**Ravi, {p.age} y** - CKD requiring maintenance hemodialysis, twice weekly (Tue/Fri). Diabetes, hypertension. Baseline SBP {p.baseline_sbp}, estimated dry weight {p.dry_weight_kg} kg.")
        h = D.sess_by_pid[pid]; h = h[h.is_upcoming == 0]
        c = st.columns(4); c[0].metric("Sessions on record", len(h)); c[1].metric("Hypotension events", int(h.hypotension.sum())); c[2].metric("Event rate", f"{h.hypotension.mean():.0%}")
        c[3].metric("Current interdialytic gap", f"{(r.date - base['prev'].date).days} days")
        t = h.tail(12)[["date", "gap_days", "pre_weight_kg", "post_weight_kg", "idwg_kg", "planned_uf_l", "duration_h", "pre_sbp", "min_sbp", "hypotension"]].copy()
        t["hypotension"] = t.hypotension.map({1: "Yes", 0: "No"})
        t.columns = ["Date", "Gap (days)", "Pre-weight (kg)", "Post-weight (kg)", "IDWG (kg)", "Planned UF (L)", "Duration (h)", "Pre-SBP", "Min SBP", "Hypotension"]
        show_table(t.rename(columns={"Date": "date"}), {"Gap (days)": "{:.0f}", "Pre-weight (kg)": "{:.1f}", "Post-weight (kg)": "{:.1f}", "IDWG (kg)": "{:.1f}", "Planned UF (L)": "{:.1f}", "Duration (h)": "{:.2f}", "Pre-SBP": "{:.0f}", "Min SBP": "{:.0f}"})
    elif step.startswith("2"):
        s = D.sens_by_pid[pid]; s = s[(s.date > base["prev"].date) & (s.date <= r.date)]
        st.markdown(f"Ravi approaches the next session ({r.date.date()}). Data since last dialysis ({base['prev'].date.date()}):")
        show_table(s[["date", "temp_c", "home_weight_kg", "home_sbp", "hr_rest", "hrv_rmssd_ms", "activity_steps", "sleep_h"]], SENSOR_FMT)
        metric_cap(st, "Kalman fluid-gain estimate since last session", f"{base['state']['gain_since_last']:.2f} kg", f"\u00b1 {base['state']['F_sd']:.2f} kg state sd")
        st.caption("Wearable HR/HRV/activity are supporting signals; blood pressure here is a simulated clinical/home cuff reading, not a wearable measurement. Dashes mark missing sensor values.")
    elif step.startswith("3"):
        x = ["Last session"] + [f"Day {i+1}" for i in range(len(base["days"]))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[base["state"]["post_excess"]] + [d["excess"] if d["excess"] is not None else np.nan for d in base["days"]], name="Baseline (home weight excess)", mode="lines+markers", connectgaps=True))
        if heat: fig.add_trace(go.Scatter(x=x, y=[hot["state"]["post_excess"]] + [d["excess"] for d in hot["days"]], name="Heat-wave scenario", mode="lines+markers", line=dict(color="#d64545")))
        fig.update_layout(title="Fluid excess over recorded dry weight (kg)", height=340, template="nephro", margin=dict(t=45), xaxis_title="", yaxis_title="Fluid excess (kg)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        if heat:
            st.warning(f"Heat-wave scenario ON: temperature {hot['state']['temp_c']:.0f} C (synthetic). Ravi gains **{DEMO_TARGET_GAIN_KG} kg by day 3** - a DEMONSTRATION INPUT, not a clinically established threshold; weather alone does not prove fluid overload.")
        else: st.info("Use the heat-wave toggle (top right) to simulate a synthetic heat wave; the next steps recompute from it.")
    elif step.startswith("4"):
        st.markdown(f"Scenario shown: **{'heat wave' if heat else 'baseline'}**. Fluid state before vs during the heat wave:")
        ct = change_table(base["state"], hot["state"]); ct = ct.rename(columns={"heat_wave": "heat wave"})
        st.dataframe(ct.style.format({"baseline": "{:.2f}", "heat wave": "{:.2f}", "change": "{:+.2f}"}, na_rep="\u2014"), hide_index=True, width="stretch")
        pm = cur["twin"].params; std, _ = lr_for_ctx(F, M, sid, cur)
        c = st.columns(3)
        metric_cap(c[0], "Twin: refill k", f"{pm['k_refill']:.2f} /h", f"population median {np.exp(_pr(D)[0][0]):.2f}")
        metric_cap(c[1], "Twin: BP sensitivity S", f"{pm['S_bp']:.2f}", f"population median {np.exp(_pr(D)[0][1]):.2f}")
        metric_cap(c[2], "Fluid state (kg)", f"{cur['state']['pre_excess']:.2f}", f"\u00b1 {cur['state']['F_sd']:.2f} kg")
        pop = cur["twin"].predict(cur["plan"], 800, 3, personalize=False).p_raw; popc = float(M["pop_cal"](pop)[0])
        st.dataframe(pd.DataFrame([dict(model="Standard population model (LR)", risk=std), dict(model="Twin with population-average parameters (ablation)", risk=popc), dict(model="Personalized twin", risk=cur["p"])]).style.format({"risk": fmt_p}), hide_index=True, width="stretch")
        st.caption("Parameter uncertainty is small here because 8 recent sessions are informative in this synthetic world; real data would be noisier.")
    elif step.startswith("5"):
        std, row = lr_for_ctx(F, M, sid, cur); p_ = cur["p"]; p_base = base["p"]; p_hot = hot["p"]
        a, b = st.columns(2)
        a.markdown(f"<div class='card'><div class='muted'>Twin-estimated hypotension probability ({'heat wave' if heat else 'baseline'})</div><div class='big'>{fmt_p(p_)}</div>{chip(cat_of(p_))}<div class='muted'>Computed live by the twin; demo thresholds, not clinical.</div></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='card'><div class='muted'>Standard model comparison</div><div class='big'>{fmt_p(std)}</div>{chip(cat_of(std))}</div>", unsafe_allow_html=True)
        st.markdown(f"**Same patient, two scenarios:** baseline **{fmt_p(p_base)}** {chip(cat_of(p_base))} &nbsp;&rarr;&nbsp; heat wave **{fmt_p(p_hot)}** {chip(cat_of(p_hot))} &nbsp;(change {fmt_d(p_hot - p_base)})", unsafe_allow_html=True)
        if heat and cat_of(p_) != "Green": st.error("Elevated simulated risk under the heat-wave scenario. For clinician review only; not a treatment recommendation.")
        base_p, sens = twin_sensitivity(cur["twin"], cur["plan"], M["twin_cal"]); st.markdown("**Main contributing factors (twin sensitivity)**"); st.dataframe(sens.style.format({"risk": fmt_p, "change_vs_current": fmt_d}), hide_index=True, width="stretch")
        st.markdown("**Standard-model linear contributions**"); st.dataframe(lr_contributions(M["baselines"]["lr"], row).head(6).rename("logit contribution").to_frame().style.format("{:+.2f}"), width="stretch")
    elif step.startswith("6"):
        st.markdown(f"Scenario: **{'heat wave' if heat else 'baseline'}**. Original hypothetical plan: {cur['plan'].uf_volume:.1f} L over {cur['plan'].duration_h:.1f} h. Every alternative below is a simulation.")
        planner(cur, f"ravi_{'h' if heat else 'b'}")
    else:
        st.markdown("## Review with nephrologist before any treatment decision.")
        st.markdown("This prototype produces simulated risk estimates and hypothetical scenarios only. It does not order treatment, change dialysis prescriptions or medications, and has not been clinically validated.")
