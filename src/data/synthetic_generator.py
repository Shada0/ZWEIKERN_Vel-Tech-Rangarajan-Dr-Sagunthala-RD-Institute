"""Reusable synthetic dialysis cohort generator (fixed seed -> reproducible).

Outcome arises from a physiological simulation (two-compartment fluid model + BP response) with
*deliberate misspecification* relative to the twin (quadratic BP response, linear drift, patient dry-weight
recording error, latent autonomic factor linked to HRV) so the twin is not simply the data-generating model.
All patients are synthetic. NO real patient data.
"""
import numpy as np
import pandas as pd
from src.config import *
from src.physiology.fluid_model import drop_pct_closed, heat_gain_mult, heat_stress

SCHEDULES = {3: [(0, 2, 4), (1, 3, 5)], 2: [(0, 3), (1, 4)]}
MED_CATS = ["ACE-inhibitor/ARB", "Beta-blocker", "Calcium-channel blocker", "Alpha-blocker", "Diuretic"]


def _weather(rng, dates):
    doy = dates.dayofyear.values
    temp = 30 + 3 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.5, len(dates))
    n = len(dates)
    for start in rng.choice(np.arange(5, n - 20), 3, replace=False):   # synthetic heat-wave episodes (history only)
        temp[start:start + 4] += 9.0
    return np.round(temp, 1)


def make_patients(rng, n):
    rows = []
    for i in range(n):
        ravi = i == 0
        age = 58 if ravi else int(np.clip(rng.normal(60, 11), 30, 85))
        dm = 1 if ravi else int(rng.random() < 0.30 + 0.004 * (age - 60) + 0.1)
        htn = 1 if ravi else int(rng.random() < 0.80)
        cvd = 0 if ravi else int(rng.random() < 0.25 + 0.10 * dm)
        freq = 2 if ravi else int(rng.choice([2, 3], p=[0.35, 0.65]))
        sched = (1, 4) if ravi else SCHEDULES[freq][rng.integers(2)]      # Ravi: Tue/Fri
        meds = [] if ravi is False and rng.random() < 0.1 else list(rng.choice(MED_CATS, size=rng.integers(1, 4), replace=False))
        if ravi: meds = ["ACE-inhibitor/ARB", "Calcium-channel blocker"]
        alb = float(np.clip(rng.normal(3.7, 0.4), 2.4, 4.6)); hb = float(np.clip(rng.normal(10.5, 1.2), 7, 14))
        lnk = np.log(0.5) + rng.normal(0, 0.35) - 0.20 * dm - 0.20 * cvd + 0.3 * (alb - 3.7)
        lns = np.log(1.7) + rng.normal(0, 0.30) + 0.20 * dm + 0.25 * cvd + 0.005 * (age - 60) + 0.05 * len(meds)
        g = float(np.clip(rng.normal(0.62, 0.15), 0.25, 1.1))
        if ravi: lnk, lns, g = np.log(0.40), np.log(1.9), 0.64
        dry = 72.0 if ravi else float(np.clip(rng.normal(68, 11), 42, 110))
        rows.append(dict(
            patient_id="SYN-RAVI" if ravi else f"SYN-{i:03d}", display_name="Ravi" if ravi else "",
            age=age, sex="M" if ravi else str(rng.choice(["M", "F"], p=[0.6, 0.4])), diabetes=dm, hypertension=htn,
            cardiovascular_disease=cvd, ckd_stage="5D (maintenance hemodialysis)", dialysis_per_week=freq,
            schedule_days=",".join(map(str, sched)), shift_start="07:00" if ravi else str(rng.choice(["07:00", "13:00"])),
            baseline_sbp=int(round(142 if ravi else np.clip(rng.normal(138 + 4 * htn, 14), 105, 185))),
            dry_weight_kg=round(dry, 1), usual_duration_h=4.0 if ravi else float(rng.choice([3.5, 4.0, 4.5], p=[.3, .5, .2])),
            bp_medication_categories=";".join(meds), albumin_g_dl=round(alb, 2), hemoglobin_g_dl=round(hb, 1),
            potassium_mmol_l=round(float(np.clip(rng.normal(5.0, 0.6), 3.5, 6.8)), 1),
            baseline_hr=int(round(78 if ravi else np.clip(rng.normal(76, 8), 55, 100))),
            _lnk=lnk, _lns=lns, _g=g, _cv=float(np.clip(rng.normal(0.8, 0.25), 0.2, 1.5)),
            _dry_err=0.0 if ravi else float(rng.normal(0, 0.4)), _dry_true=dry))
    return pd.DataFrame(rows)


def _gen_sbp(pre, S_eff, drop, t, rng):
    """Generator BP response: quadratic + drift (NOT in twin) + AR(1) physiological noise + measurement noise."""
    e = np.zeros(len(t)); innov = rng.normal(0, AR_SD, len(t))
    for j in range(1, len(t)):
        e[j] = AR_RHO * e[j - 1] + innov[j]
    sbp = pre - S_eff * (drop + 0.015 * drop ** 2) - 1.0 * t + e + rng.normal(0, MEAS_SD, len(t))
    sbp[0] = pre
    return sbp


def generate(seed=SEED, n_patients=N_PATIENTS):
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(CUTOFF_DATE)
    dates = pd.date_range(cutoff - pd.Timedelta(weeks=HISTORY_WEEKS), cutoff + pd.Timedelta(days=7))
    temps = _weather(rng, dates)
    pats = make_patients(rng, n_patients)
    sess_rows, obs_rows, sens_rows = [], [], []

    for _, p in pats.iterrows():
        sched = [int(x) for x in p.schedule_days.split(",")]
        k, S, g, cv = np.exp(p._lnk), np.exp(p._lns), p._g, p._cv
        F = max(0.3, 1.8 + rng.normal(0, 0.4)); idx = 0; prev_date = None; prev_post = None; upcoming_done = False
        dry_rec = p.dry_weight_kg
        for d, T in zip(dates, temps):
            stress = float(heat_stress(T))
            if prev_date is not None:
                F = max(0.2, F + g * float(heat_gain_mult(T)) + rng.normal(0, 0.14))
            # daily sensors (pre-session / morning values; available before prediction timestamp)
            wear_ok = rng.random() > 0.20
            hrv = np.exp(np.log(32) - 0.30 * (p._lns - np.log(1.7)) - 0.25 * p.diabetes - 0.04 * (F - 2) - 0.1 * stress + rng.normal(0, 0.25))
            sens_rows.append(dict(
                patient_id=p.patient_id, date=d.date(), temp_c=T, heatwave_flag=int(T >= 38.0),
                home_weight_kg=round(p._dry_true + F + rng.normal(0, 0.2), 2) if rng.random() > 0.15 else np.nan,
                home_sbp=round(p.baseline_sbp + 4 * (F - 2) + rng.normal(0, 9)) if rng.random() > 0.25 else np.nan,
                hr_rest=round(p.baseline_hr + 1.5 * (F - 2) + 2 * stress + rng.normal(0, 3), 1) if wear_ok else np.nan,
                hrv_rmssd_ms=round(hrv, 1) if wear_ok else np.nan,
                activity_steps=int(np.exp(np.log(4500) - 0.2 * stress + rng.normal(0, 0.4))) if wear_ok else np.nan,
                sleep_h=round(float(np.clip(6.5 - 0.5 * stress + rng.normal(0, 1), 3, 9)), 1) if wear_ok else np.nan))
            if d.weekday() not in sched or upcoming_done:
                continue
            is_up = d > cutoff
            if is_up:
                upcoming_done = True
            dur = p.usual_duration_h - (0.5 if rng.random() < 0.08 else 0)
            pre_w = p._dry_true + F + rng.normal(0, 0.10)
            exc_rec = pre_w - dry_rec
            uf = round(float(np.clip(exc_rec * rng.uniform(0.72, 1.05), 0.5, 5.0)), 1)
            if p.patient_id == "SYN-RAVI" and is_up:
                uf = round(float(np.clip(exc_rec * 0.95, 0.5, 5.0)), 1)
            u = uf / dur
            pre_sbp = float(np.clip(round(p.baseline_sbp + 4 * (F - 2) + rng.normal(0, 12)), 100, 200))
            pre_hr = round(p.baseline_hr + 1.5 * (F - 2) + rng.normal(0, 4))
            t = np.arange(0, dur + 1e-9, OBS_STEP_H)
            drop = drop_pct_closed(t, F, u, k)
            sbp = _gen_sbp(pre_sbp, S * np.exp(rng.normal(0, SESSION_SD)) * (1 + SENS_HEAT * stress), drop, t, rng)
            hr = pre_hr + cv * drop + rng.normal(0, 2, len(t)); hr[0] = pre_hr
            hypo_idx = np.where((sbp < HYPO_SBP) & (t > 0))[0]
            hypo = len(hypo_idx) > 0
            first = hypo_idx[0] if hypo else None
            pause = bool(hypo and rng.random() < 0.8); early = bool(hypo and rng.random() < 0.08)
            removed = uf * (1 - 0.3 * pause - 0.4 * early)
            post_w = p._dry_true + F - removed + rng.normal(0, 0.1)
            sid = f"{p.patient_id}-S{idx:03d}"
            row = dict(session_id=sid, patient_id=p.patient_id, session_index=idx, date=d.date(), start_time=p.shift_start,
                       prediction_timestamp=f"{d.date()} {p.shift_start}",
                       gap_days=(d - prev_date).days if prev_date is not None else np.nan,
                       pre_weight_kg=round(pre_w, 2), post_weight_kg=round(post_w, 2) if not is_up else np.nan,
                       target_dry_weight_kg=dry_rec,
                       idwg_kg=round(pre_w - prev_post, 2) if prev_post is not None else np.nan,
                       planned_uf_l=uf, duration_h=dur, uf_rate_l_h=round(u, 3),
                       uf_rate_ml_kg_h=round(1000 * u / pre_w, 1), pre_sbp=pre_sbp, pre_dbp=round(0.6 * pre_sbp + 5),
                       pre_hr=pre_hr, temp_c=T, heatwave_flag=int(T >= 38.0), is_upcoming=int(is_up))
            if is_up:
                row.update(min_sbp=np.nan, nadir_time_h=np.nan, hypotension=np.nan, saline_given=np.nan,
                           uf_pause=np.nan, early_termination=np.nan)
            else:
                row.update(min_sbp=round(float(sbp[1:].min()), 1), nadir_time_h=float(t[1:][sbp[1:].argmin()]),
                           hypotension=int(hypo), saline_given=int(hypo), uf_pause=int(pause), early_termination=int(early))
                for j in range(len(t)):
                    obs_rows.append(dict(session_id=sid, patient_id=p.patient_id, t_h=t[j], sbp=round(sbp[j], 1),
                                         dbp=round(0.6 * sbp[j] + 5, 1), hr=round(hr[j], 1),
                                         post_event=int(hypo and j > first)))
                F = max(0.0, F - removed)
            sess_rows.append(row)
            prev_date, prev_post, idx = d, (post_w if not is_up else None), idx + 1

    truth = pats[["patient_id", "_lnk", "_lns", "_g", "_cv"]].copy()
    truth.columns = ["patient_id", "true_log_k", "true_log_S", "true_g_kg_day", "true_hr_response"]
    pats_out = pats.drop(columns=[c for c in pats.columns if c.startswith("_")])
    return pats_out, pd.DataFrame(sess_rows), pd.DataFrame(sens_rows), pd.DataFrame(obs_rows), truth


def write_all(out_dir=DATA_DIR, seed=SEED, n_patients=N_PATIENTS):
    out_dir.mkdir(exist_ok=True, parents=True)
    pats, sess, sens, obs, truth = generate(seed, n_patients)
    pats.to_csv(out_dir / "synthetic_patients.csv", index=False)
    sess.to_csv(out_dir / "synthetic_sessions.csv", index=False)
    sens.to_csv(out_dir / "synthetic_sensor_data.csv", index=False)
    obs.to_csv(out_dir / "synthetic_intradialytic_obs.csv", index=False)
    truth.to_csv(out_dir / "synthetic_ground_truth_params.csv", index=False)
    return pats, sess, sens, obs, truth


if __name__ == "__main__":
    pats, sess, sens, obs, _ = write_all()
    h = sess[sess.is_upcoming == 0]
    print(len(pats), "patients", len(h), "sessions; hypotension prevalence", round(h.hypotension.mean(), 3))
    print(h.groupby("patient_id").hypotension.mean().describe().round(3).to_dict())
    print(h[h.patient_id == "SYN-RAVI"].hypotension.mean(), sess[sess.is_upcoming == 1].shape)
