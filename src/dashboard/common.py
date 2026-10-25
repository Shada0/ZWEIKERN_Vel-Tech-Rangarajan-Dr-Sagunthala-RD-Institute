"""Shared dashboard plumbing: cached resources, styling, charts, planner widget."""
import json
import plotly.io as pio
import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from src.config import *
from src.data.preprocessing import Data, build_features, FEATURES
from src.models.compat import fix_sklearn_compat
from src.models.baseline_model import lr_contributions
from src.models.explainability import twin_sensitivity
from src.physiology.fluid_model import heat_gain_mult, heat_stress
from src.planner.what_if_planner import scenario_table, run_plan, NOTE, wording
from src.simulation.session_simulator import build_context

pio.templates["nephro"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", color="#16324f", size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#16324f", "#0f8b8d", "#e07b39", "#8a5cf6", "#d64545", "#6b7a8c"], xaxis=dict(showgrid=False, linecolor="#c9d4de", zeroline=False),
    yaxis=dict(gridcolor="#e3e9ef", linecolor="#c9d4de", zeroline=False), title=dict(font=dict(size=15)), hoverlabel=dict(font_size=12)))
pio.templates.default = "nephro"
COL = {"Green": "#1f9d6b", "Amber": "#e0a100", "Red": "#d64545"}
NAVY, TEAL, GREY = "#16324f", "#0f8b8d", "#6b7a8c"


@st.cache_resource(show_spinner="Loading synthetic cohort...")
def load_all():
    D = Data(); F = build_features(D)
    P = pd.read_csv(DATA_DIR / "predictions.csv").set_index("session_id", drop=False)
    M = fix_sklearn_compat(joblib.load(MODEL_DIR / "models.joblib")); D.prior = M.get("prior")
    E = json.load(open(MODEL_DIR / "evaluation.json"))
    return D, F, P, M, E


@st.cache_resource(show_spinner="Updating digital twin...")
def get_ctx(sid, heat, checkin_weights=()):
    D, F, P, M, E = load_all()
    return build_context(D, sid, M["twin_cal"], heat=heat, checkins=[dict(weight=w) for w in checkin_weights])


def css():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html,body,[class*="css"]{{font-family:'Inter','Segoe UI',sans-serif}}
    .block-container{{padding-top:2.6rem;max-width:1280px}}
    h1{{color:{NAVY};font-weight:700;letter-spacing:-.02em}} h2,h3,h4{{color:{NAVY};font-weight:600}}
    [data-testid="stSidebar"]{{background:linear-gradient(180deg,#0f2740 0%,#16324f 100%)}}
    [data-testid="stSidebar"] *{{color:#e8f1f8 !important}}
    [data-testid="stSidebar"] .stRadio label{{padding:.15rem 0}}
    [data-testid="stSidebar"] [data-baseweb="select"] *{{color:#16324f !important}}
    [data-testid="stSidebar"] [data-baseweb="select"]>div{{background:#f4f8fb}}
    .banner{{background:#fff4e5;border:1px solid #f0c987;color:#7a4b00;padding:.5rem .9rem;border-radius:10px;font-size:.82rem;margin-bottom:1rem}}
    .card{{background:#fff;border:1px solid #e3e9ef;border-radius:14px;padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(20,40,60,.06);height:100%}}
    .hero{{background:linear-gradient(120deg,#16324f 0%,#0f8b8d 100%);color:#fff;padding:1.8rem 2rem;border-radius:18px;margin-bottom:1.2rem}}
    .hero h1{{color:#fff;margin:0 0 .3rem 0;font-size:2.3rem}} .hero p{{margin:.2rem 0;font-size:1.02rem;opacity:.95}}
    .chip{{display:inline-block;padding:.15rem .8rem;border-radius:999px;color:#fff;font-weight:600;font-size:.9rem}}
    .big{{font-size:2.3rem;font-weight:700;color:{NAVY};line-height:1.1}}
    .kpi{{font-size:1.9rem;font-weight:700;color:{NAVY}}} .muted{{color:{GREY};font-size:.82rem}}
    .good{{color:#1f9d6b;font-weight:600}} .bad{{color:#d64545;font-weight:600}}
    div[data-testid="stMetric"]{{background:#fff;border:1px solid #e3e9ef;border-radius:12px;padding:.6rem .9rem;box-shadow:0 1px 4px rgba(20,40,60,.05)}}
    </style>""", unsafe_allow_html=True)


def fmt_p(p):
    """Display a probability without flat '100%' / '0%' saturation."""
    if p is None or not np.isfinite(p): return "n/a"
    if p >= 0.995: return "\u2265 99%"
    if p < 0.005: return "< 1%"
    return f"{p:.0%}"


def fmt_d(d):
    """Signed probability change in percentage points (never '+100%')."""
    if d is None or not np.isfinite(d): return "n/a"
    d = float(np.clip(d, -0.99, 0.99))
    return f"{d * 100:+.0f} pts"


def delta_color_for(lo, hi):
    """Colour a metric delta only when its CI excludes 0; else neutral grey."""
    return "normal" if (lo >= 0.005 or hi <= -0.005) else "off"


def metric_cap(col, label, value, caption):
    """Metric with a neutral caption instead of a delta (so no misleading arrow)."""
    col.metric(label, value); col.caption(caption)


SHORT = {"Historical reference (prior hypotension rate)": "Prior-rate reference", "Standard model: logistic regression": "Logistic regression",
         "Standard model: random forest": "Random forest", "Twin ablation: population parameters (no personalization)": "Twin, no personalization",
         "Personalized digital twin": "Personalized twin", "Hybrid: standard features + twin outputs": "Hybrid (standard + twin)"}


def short(name):
    return SHORT.get(name, name.split(":")[-1].strip())


def roc_fig(mods, title):
    fig = go.Figure()
    for m, v in mods.items():
        fig.add_trace(go.Scatter(x=v["roc"][0], y=v["roc"][1], mode="lines", name=f"{short(m)} (AUC {v['auc']:.3f})"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="grey"))
    fig.update_layout(title=title, height=470, template="nephro", xaxis_title="False-positive rate", yaxis_title="True-positive rate",
                      margin=dict(l=10, r=10, t=45, b=120), legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font=dict(size=11)))
    return fig


def clean_dates(df, cols=("date",)):
    df = df.copy()
    for c in cols:
        if c in df: df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


def show_table(df, fmt=None, **kw):
    """Dataframe with ISO dates, formatted numbers and an em dash for missing values (formatted to strings so nothing renders as None)."""
    df = clean_dates(df); out = pd.DataFrame(index=df.index)
    for c in df.columns:
        f = (fmt or {}).get(c)
        def one(v, f=f):
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or (not isinstance(v, str) and pd.isna(v)): return "\u2014"
            return f.format(v) if f else str(v)
        out[c] = [one(v) for v in df[c]]
    st.dataframe(out, hide_index=True, width="stretch", **kw)


SENSOR_FMT = {"temp_c": "{:.1f}", "home_weight_kg": "{:.1f}", "home_sbp": "{:.0f}", "hr_rest": "{:.0f}", "hrv_rmssd_ms": "{:.0f}", "activity_steps": "{:,.0f}", "sleep_h": "{:.1f}"}


def banner_real():
    st.markdown("<div class='banner' style='background:#e8f6f6;border-color:#9fd3d4;color:#0b5b5d'><b>Real de-identified open dataset (HEMOBP); retrospective research validation only.</b> "
                "Not a medical device, not a treatment recommendation, and no evidence of clinical benefit.</div>", unsafe_allow_html=True)


def banner():
    st.markdown(f"<div class='banner'><b>SIMULATED PATIENT - NOT REAL CLINICAL DATA.</b> Research prototype, not a medical device. "
                f"All risk categories use configurable demonstration thresholds. Not a treatment recommendation.</div>", unsafe_allow_html=True)


def chip(cat):
    return f"<span class='chip' style='background:{COL[cat]}'>{cat}</span>"


def cat_of(p):
    return risk_category(p, st.session_state.get("amber", AMBER_DEFAULT), st.session_state.get("red", RED_DEFAULT))


def label(D, pid):
    n = D.patients.loc[pid, "display_name"]
    return f"{pid} ({n})" if isinstance(n, str) and n else pid


def upcoming_sid(D, pid):
    s = D.sessions[(D.sessions.patient_id == pid) & (D.sessions.is_upcoming == 1)]
    return s.index[0]


def sbp_fig(preds: dict, observed=None, title="Simulated systolic BP trajectory"):
    fig = go.Figure(); cols = [NAVY, TEAL, "#8a5cf6", "#e07b39"]
    for i, (name, pr) in enumerate(preds.items()):
        c = cols[i % 4]
        fig.add_trace(go.Scatter(x=pr.times * 60, y=pr.p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pr.times * 60, y=pr.p10, fill="tonexty", line=dict(width=0), fillcolor=f"rgba(22,50,79,{0.12 if i == 0 else 0.07})",
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pr.times * 60, y=pr.mean_sbp, name=name, line=dict(color=c, width=3 if i == 0 else 2, dash="solid" if i == 0 else "dash")))
    if observed is not None:
        t, sbp, keep = observed
        fig.add_trace(go.Scatter(x=t * 60, y=sbp, mode="lines+markers", name="Observed (simulated)", line=dict(color="#222"),
                                 marker=dict(size=6, color=["#222" if k else "#d64545" for k in keep])))
    fig.add_hline(y=HYPO_SBP, line_dash="dot", line_color="#d64545", annotation_text="SBP 90 (demo outcome threshold)")
    fig.update_layout(title=title, xaxis_title="Minutes from session start", yaxis_title="SBP (mmHg)", height=380, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=-0.25), template="simple_white")
    return fig


def planner(ctx, key, heat_available=True):
    """Interactive what-if planner. Simulation only."""
    D, F, P, M, E = load_all(); cal = M["twin_cal"]; twin, plan0, st0 = ctx["twin"], ctx["plan"], ctx["state"]
    st.markdown("##### Hypothetical session parameters (simulation only)")
    c1, c2, c3 = st.columns(3)
    dur = c1.slider("Session duration (h)", 2.5, 5.5, float(plan0.duration_h), 0.25, key=f"{key}_d")
    uf = c2.slider("Hypothetical UF volume (L)", 0.5, 5.0, float(min(max(plan0.uf_volume, .5), 5.0)), 0.1, key=f"{key}_uf")
    prof = c3.selectbox("UF-rate profile", ["constant", "front-loaded"], key=f"{key}_pf")
    c4, c5, c6 = st.columns(3)
    sbp0 = c4.slider("Starting SBP (mmHg)", 90, 200, int(plan0.pre_sbp), 1, key=f"{key}_sbp")
    exc = c5.slider("Fluid accumulation / excess (kg)", 0.3, 5.0, float(np.clip(plan0.fluid_excess, .3, 5.0)), 0.1, key=f"{key}_f")
    gap = c6.slider("Extra interdialytic gap (days)", 0, 3, 0, key=f"{key}_gap")
    hw = st.checkbox("Heat-wave scenario (synthetic 42 C; documented fluid-gain and sensitivity rule)", value=plan0.temp_c >= 40, key=f"{key}_hw")
    temp = 42.0 if hw else plan0.temp_c
    extra = gap * twin.g * float(heat_gain_mult(temp)) + twin.g * (float(heat_gain_mult(temp)) - float(heat_gain_mult(plan0.temp_c))) * len(ctx['days'])
    if hw and plan0.temp_c < 40:
        st.caption(f"Heat scenario changes: temperature {plan0.temp_c:.0f}->42 C, heat-stress input {float(heat_stress(plan0.temp_c)):.2f}->1.00, extra fluid {extra:+.2f} kg (documented rule: +5%/day gain per C above 32 C), BP sensitivity +15%.")
    plan = plan0.replace(duration_h=dur, uf_volume=uf, uf_profile=prof, pre_sbp=float(sbp0), fluid_excess=exc + extra, temp_c=temp)
    pr, p = run_plan(twin, plan, cal); pr0, p0 = run_plan(twin, plan0, cal)
    a, b, c = st.columns([1, 1, 2])
    a.markdown(f"<div class='card'><div class='muted'>Simulated risk (this scenario)</div><div class='big'>{fmt_p(p)}</div>{chip(cat_of(p))}</div>", unsafe_allow_html=True)
    b.markdown(f"<div class='card'><div class='muted'>Original plan risk</div><div class='big'>{fmt_p(p0)}</div><div class='muted'>change {fmt_d(p - p0)}</div></div>", unsafe_allow_html=True)
    c.info(wording(p - p0) + "  \n" + NOTE)
    st.plotly_chart(sbp_fig({"This scenario": pr, "Original plan": pr0}), width="stretch")
    fig = go.Figure(go.Scatter(x=pr.times * 60, y=pr.uf_cum, name="Cumulative UF (L)", line=dict(color=TEAL)))
    fig.add_trace(go.Scatter(x=pr.times * 60, y=pr.drop_mean, name="Mean relative blood-volume drop (%)", yaxis="y2", line=dict(color="#e07b39")))
    fig.update_layout(title="Fluid-removal profile", height=300, template="simple_white", yaxis=dict(title="L"), yaxis2=dict(title="%", overlaying="y", side="right"),
                      xaxis_title="Minutes", margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Scenario comparison** (all rows are simulations)")
    df, _ = scenario_table(twin, plan, cal, twin.g * float(heat_gain_mult(temp)))
    show = df.copy(); show["risk"] = show.risk.map(fmt_p); show["delta_vs_A"] = show.delta_vs_A.map(fmt_d)
    show = show.rename(columns={"duration_h": "duration (h)", "uf_volume_l": "UF volume (L)", "uf_rate_l_h": "UF rate (L/h)", "fluid_excess_kg": "fluid excess (kg)", "simulated_nadir_sbp": "nadir SBP (mmHg)", "delta_vs_A": "change vs A"})
    st.dataframe(show.style.format({"duration (h)": "{:.2f}", "UF volume (L)": "{:.1f}", "UF rate (L/h)": "{:.2f}", "fluid excess (kg)": "{:.2f}", "nadir SBP (mmHg)": "{:.0f}"}), hide_index=True, width="stretch")
    pm = twin.params
    st.caption(f"Personalized parameters used: refill k = {pm['k_refill']:.2f}/h (+-{pm['sd_logk']*100:.0f}% log-sd), BP sensitivity S = {pm['S_bp']:.2f} mmHg per % blood-volume drop "
               f"(+-{pm['sd_logS']*100:.0f}%), accumulation g = {pm['g_kg_day']:.2f} kg/day. Review with nephrologist before any treatment decision.")
    return plan, p
