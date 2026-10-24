"""Data access + leakage-safe feature construction.

Prediction timestamp = scheduled connection time (after chair-side pre-dialysis vitals). Only information
timestamped <= that moment is used: static EHR, sessions strictly BEFORE this one, sensor rows up to the
session-morning, and pre-dialysis vitals. Intradialytic readings of the target session are never used.
"""
import numpy as np
import pandas as pd
from src.config import DATA_DIR, SENS_HEAT
from src.physiology.fluid_model import heat_gain_mult, heat_stress
from src.physiology.twin_engine import PatientTwin

FEATURES = ["age", "sex_m", "diabetes", "hypertension", "cvd", "n_bp_meds", "albumin", "hemoglobin", "potassium",
            "gap_days", "pre_weight", "excess_est", "planned_uf", "duration_h", "uf_rate_l_h", "uf_rate_ml_kg_h",
            "pre_sbp", "pre_dbp", "pre_hr", "hr_rest_3d", "hrv_3d", "activity_3d", "sleep_3d", "temp_c",
            "heatwave_flag", "temp_3d_max", "prior_hypo_rate10", "prior_min_sbp5", "prior_hypo_last", "n_prior"]
FIT_WINDOW = 8


class Data:
    def __init__(self, data_dir=DATA_DIR, prior=None):
        self.prior = prior
        self.patients = pd.read_csv(data_dir / "synthetic_patients.csv").set_index("patient_id", drop=False)
        s = pd.read_csv(data_dir / "synthetic_sessions.csv", parse_dates=["date"]).sort_values(["patient_id", "session_index"])
        self.sessions = s.set_index("session_id", drop=False)
        self.sensors = pd.read_csv(data_dir / "synthetic_sensor_data.csv", parse_dates=["date"]).sort_values(["patient_id", "date"])
        self.obs = pd.read_csv(data_dir / "synthetic_intradialytic_obs.csv")
        self.truth = pd.read_csv(data_dir / "synthetic_ground_truth_params.csv").set_index("patient_id")
        self.obs_by_sid = {k: (g.t_h.values, g.sbp.values, g.hr.values, g.post_event.values == 0) for k, g in self.obs.groupby("session_id")}
        self.sens_by_pid = {k: g.reset_index(drop=True) for k, g in self.sensors.groupby("patient_id")}
        self.sess_by_pid = {k: g for k, g in self.sessions.groupby("patient_id")}
        empty = self.sensors.iloc[0:0].copy(); empty["date"] = pd.to_datetime(empty["date"])
        for pid in self.patients.index:                       # real data (HEMOBP) has no wearable/weather stream
            self.sens_by_pid.setdefault(pid, empty)
        self._gap_stats()
        self.sess_by_pid = {k: g for k, g in self.sessions.groupby("patient_id")}

    def _gap_stats(self):
        """Per-session mean heat multiplier over the interdialytic gap and per-day weight gain (observable)."""
        gm = pd.Series(np.nan, index=self.sessions.index)
        for pid, g in self.sess_by_pid.items():
            sens = self.sens_by_pid[pid]; dates = sens.date.values; mult = heat_gain_mult(sens.temp_c.values)
            prev = None
            for sid, r in g.iterrows():
                if prev is not None:
                    m = (dates > np.datetime64(prev)) & (dates <= np.datetime64(r.date))
                    gm[sid] = mult[m].mean() if m.any() else 1.0
                prev = r.date
        self.sessions["gap_heat_mult"] = gm
        self.sessions["gain_per_day"] = self.sessions.idwg_kg / self.sessions.gap_days / self.sessions.gap_heat_mult

    # ---- twin construction -------------------------------------------------------------
    def fit_rec(self, sid):
        r = self.sessions.loc[sid]
        t, sbp, hr, keep = self.obs_by_sid[sid]
        dry = r.target_dry_weight_kg
        return dict(t=t, sbp=sbp, hr=hr, keep=keep, F0=r.pre_weight_kg - dry, u=(r.uf_delivered_l_h if "uf_delivered_l_h" in r.index and np.isfinite(r.uf_delivered_l_h) else r.uf_rate_l_h), pre_sbp=r.pre_sbp,
                    hr_pre=r.pre_hr, stress=float(heat_stress(r.temp_c)))

    def history(self, pid, idx, n=FIT_WINDOW):
        g = self.sess_by_pid[pid]
        h = g[(g.session_index < idx) & (g.is_upcoming == 0)]
        return h

    def twin_for(self, pid, idx, personalize=True):
        h = self.history(pid, idx)
        recs = [self.fit_rec(s) for s in h.session_id.tail(FIT_WINDOW)]
        return PatientTwin(personalize, prior=self.prior).fit(recs, h.gain_per_day.tail(FIT_WINDOW).values)

    def days_between(self, pid, prev_date, date):
        sens = self.sens_by_pid[pid]; dry = self.patients.loc[pid, "dry_weight_kg"]
        m = sens[(sens.date > prev_date) & (sens.date <= date)]
        return [dict(date=r.date, temp=r.temp_c, excess=(r.home_weight_kg - dry) if np.isfinite(r.home_weight_kg) else None,
                     hr_rest=r.hr_rest, hrv=r.hrv_rmssd_ms, activity=r.activity_steps, sleep=r.sleep_h, home_sbp=r.home_sbp,
                     heatwave=r.heatwave_flag) for r in m.itertuples()]

    def fluid_estimate(self, twin, sid):
        """Kalman-fused fluid-excess state at prediction time for session sid (uses only pre-timestamp data)."""
        r = self.sessions.loc[sid]; g = self.sess_by_pid[r.patient_id]
        prev = g[g.session_index == r.session_index - 1]
        dry = r.target_dry_weight_kg
        if prev.empty:
            return r.pre_weight_kg - dry, 0.3 ** 2
        pv = prev.iloc[0]
        days = self.days_between(r.patient_id, pv.date, r.date)
        return twin.fluid_state(pv.post_weight_kg - dry, days, pre_excess=r.pre_weight_kg - dry)

    def completeness(self, pid, days=14):
        s = self.sens_by_pid[pid]; s = s[s.date <= pd.Timestamp("2026-10-23")].tail(days)
        cols = ["home_weight_kg", "home_sbp", "hr_rest", "hrv_rmssd_ms", "activity_steps", "sleep_h"]
        return float(1 - s[cols].isna().values.mean())


def build_features(D: Data) -> pd.DataFrame:
    P, S = D.patients, D.sessions
    rows = []
    for pid, g in D.sess_by_pid.items():
        p = P.loc[pid]; sens = D.sens_by_pid[pid]
        y = g.hypotension.values; mins = g.min_sbp.values
        for i, (sid, r) in enumerate(g.iterrows()):
            w = sens[(sens.date > r.date - pd.Timedelta(days=3)) & (sens.date <= r.date)]
            past_y = y[:i][~np.isnan(y[:i])]; past_min = mins[:i][~np.isnan(mins[:i])]
            wear = w[["hr_rest", "hrv_rmssd_ms", "activity_steps", "sleep_h"]]
            rows.append(dict(
                session_id=sid, patient_id=pid, date=r.date, session_index=r.session_index, is_upcoming=r.is_upcoming, y=r.hypotension,
                age=p.age, sex_m=int(p.sex == "M"), diabetes=p.diabetes, hypertension=p.hypertension, cvd=p.cardiovascular_disease,
                n_bp_meds=len(str(p.bp_medication_categories).split(";")) if isinstance(p.bp_medication_categories, str) else 0,
                albumin=p.albumin_g_dl, hemoglobin=p.hemoglobin_g_dl, potassium=p.potassium_mmol_l,
                gap_days=r.gap_days, pre_weight=r.pre_weight_kg, excess_est=r.pre_weight_kg - r.target_dry_weight_kg,
                planned_uf=r.planned_uf_l, duration_h=r.duration_h, uf_rate_l_h=r.uf_rate_l_h, uf_rate_ml_kg_h=r.uf_rate_ml_kg_h,
                pre_sbp=r.pre_sbp, pre_dbp=r.pre_dbp, pre_hr=r.pre_hr,
                hr_rest_3d=wear.hr_rest.mean(), hrv_3d=wear.hrv_rmssd_ms.mean(), activity_3d=wear.activity_steps.mean(),
                sleep_3d=wear.sleep_h.mean(), temp_c=r.temp_c, heatwave_flag=r.heatwave_flag, temp_3d_max=w.temp_c.max(),
                prior_hypo_rate10=past_y[-10:].mean() if len(past_y) else np.nan,
                prior_min_sbp5=past_min[-5:].mean() if len(past_min) else np.nan,
                prior_hypo_last=past_y[-1] if len(past_y) else np.nan, n_prior=len(past_y)))
    return pd.DataFrame(rows).set_index("session_id", drop=False)


def split_patients(patients, test_frac=0.3, seed=7):
    rng = np.random.default_rng(seed)
    has = "SYN-RAVI" in patients.index
    ids = [i for i in patients.index if i != "SYN-RAVI"]
    test = set(rng.choice(ids, int(round(test_frac * len(patients))) - int(has), replace=False)) | ({"SYN-RAVI"} if has else set())
    return {pid: ("test" if pid in test else "train") for pid in patients.index}
