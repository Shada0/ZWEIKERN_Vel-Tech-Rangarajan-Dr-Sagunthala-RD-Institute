"""NephroTwin AI - clinician dashboard (synthetic research prototype). Run: streamlit run app.py"""
import streamlit as st
from src.config import AMBER_DEFAULT, RED_DEFAULT
from src.dashboard import views
from src.dashboard.common import css, banner, load_all

st.set_page_config(page_title="NephroTwin AI", page_icon=":drop_of_blood:", layout="wide")
css()
PAGES = ["Project overview", "Unit overview", "Patient list", "Digital twin profile", "Pre-session card", "Weekly priority list", "Ravi demonstration",
         "Historical replay", "Model evaluation", "Real-data validation", "Data quality", "Daily check-in", "Limitations & safety"]
ICONS = {"Project overview": "\U0001F3E0", "Unit overview": "\U0001F3E5", "Patient list": "\U0001F4CB", "Digital twin profile": "\U0001F9EC", "Pre-session card": "\U0001FA7A",
         "Weekly priority list": "\u2B50", "Ravi demonstration": "\U0001F3AC", "Historical replay": "\u23EE\uFE0F", "Model evaluation": "\U0001F4CA", "Real-data validation": "\U0001F30D",
         "Data quality": "\U0001F9EA", "Daily check-in": "\U0001F4DD", "Limitations & safety": "\u26A0\uFE0F"}
with st.sidebar:
    st.markdown("### NephroTwin AI")
    st.caption("Predict the risk. Understand the patient. Plan dialysis more safely.")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", format_func=lambda p: f"{ICONS[p]}  {p}")
    st.markdown("---")
    st.markdown("**Demo risk thresholds** (configurable, not clinical)")
    st.session_state["amber"] = st.slider("Amber from", 0.05, 0.6, AMBER_DEFAULT, 0.05)
    st.session_state["red"] = st.slider("Red from", 0.2, 0.95, RED_DEFAULT, 0.05)
    pid = views._pid() if page in ("Digital twin profile", "Pre-session card") else None
    st.markdown("---"); st.caption("Research prototype v1.0 - synthetic cohort + open HEMOBP validation. Not a medical device. Review with a nephrologist before any treatment decision.")

if page not in ("Digital twin profile", "Pre-session card", "Project overview", "Real-data validation", "Ravi demonstration"): banner()
{"Project overview": views.home, "Real-data validation": views.real_validation, "Unit overview": views.overview, "Patient list": views.patient_list, "Weekly priority list": views.priority, "Ravi demonstration": views.ravi,
 "Historical replay": views.replay_view, "Model evaluation": views.evaluation, "Data quality": views.quality, "Daily check-in": views.checkin,
 "Limitations & safety": views.limitations}.get(page, lambda: None)()
if page == "Digital twin profile": views.twin_profile(pid)
if page == "Pre-session card": views.presession(pid)
