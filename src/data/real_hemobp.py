"""Adapter for the open HEMOBP dataset (Lin et al., Scientific Data 2019; data: figshare doi:10.6084/m9.figshare.6260654.v3;
article licensed CC BY 4.0 - verify the file-level license on figshare before redistributing; this repo does NOT ship the data).

Documented tables (per the data descriptor): Idp (Pid, Gender, Birth year, First_dialysis, Diabetes),
Hemrec_D1 (Pid, Key-in date, Dialysis start/end, Weight start/end, Dry weight, Temperature),
Hemrec_VIP (Pid, Data time, Measure time, SBP, DBP, Dia_temp_value, Conductivity, Ultrafiltration rate, Blood_flow, Dialysis time).

NOT VERIFIED against the real files by the author of this code (no access when written): column spellings, units, formats.
Column names are matched loosely (case/punctuation-insensitive) and a sanity report is printed so mismatches are visible.
Missing in real data (left NaN, never fabricated): labs, hypertension/CVD flags, medications, wearables, weather, heart rate.
"""
import re
from pathlib import Path
import numpy as np, pandas as pd
from src.config import HYPO_SBP

ALIASES = {"pid": ["pid", "patientid"], "gender": ["gender", "sex"], "birthyear": ["birthyear", "birthyr", "birthday"], "diabetes": ["diabetes", "dm"],
           "keyindate": ["keyindate", "keyin"], "dialysisstart": ["dialysisstart", "start"], "dialysisend": ["dialysisend", "end"],
           "weightstart": ["weightstart"], "weightend": ["weightend"], "dryweight": ["dryweight"],
           "datatime": ["datatime"], "measuretime": ["measuretime"], "sbp": ["sbp"], "dbp": ["dbp"],
           "ultrafiltrationrate": ["ultrafiltrationrate", "ufr", "ultrafiltrationrateh", "uf"], "dialysistime": ["dialysistime", "time"]}


def _norm(c): return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _rename(df):
    m = {}
    for c in df.columns:
        for std, al in ALIASES.items():
            if _norm(c) in al and std not in m.values():
                m[c] = std; break
    return df.rename(columns=m)


def _read(raw_dir: Path, key: str):
    hits = [p for p in raw_dir.iterdir() if any(k in p.stem.lower() for k in key.split("|")) and p.suffix.lower() in (".csv", ".txt", ".tsv", ".gz")]
    if not hits:
        raise FileNotFoundError(f"No CSV matching '{key}' in {raw_dir}. Export the figshare table to CSV (see docs/real_data.md).")
    return _rename(pd.read_csv(hits[0], encoding_errors="replace", low_memory=False))


def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["1", "y", "yes", "true", "t", "dm"]).astype(int)


def convert(raw_dir, out_dir, max_patients=120, max_sessions=80, seed=0, min_obs=4):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    idp, d1, vip = _read(raw_dir, "idp"), _read(raw_dir, "hemrec_d1|d1"), _read(raw_dir, "hemrec_vip|vip")
    need = {"d1": (d1, ["pid", "keyindate", "weightstart", "weightend", "dryweight"]), "vip": (vip, ["pid", "datatime", "sbp"])}
    for n, (df, cols) in need.items():
        miss = [c for c in cols if c not in df.columns]
        if miss: raise ValueError(f"Hemrec_{n}: could not find columns {miss}; found {list(df.columns)}")
    rng = np.random.default_rng(seed)
    pids = rng.choice(d1.pid.unique(), min(max_patients, d1.pid.nunique()), replace=False)
    d1 = d1[d1.pid.isin(pids)].copy(); vip = vip[vip.pid.isin(pids)].copy()
    d1["date"] = pd.to_datetime(d1.keyindate, errors="coerce").dt.normalize()
    for c in ("weightstart", "weightend", "dryweight"): d1[c] = pd.to_numeric(d1[c], errors="coerce")
    d1 = d1.dropna(subset=["date", "weightstart", "dryweight"]).drop_duplicates(["pid", "date"])
    ds = pd.to_datetime(d1.date.dt.strftime("%Y-%m-%d") + " " + d1.dialysisstart.astype(str), errors="coerce")
    de = pd.to_datetime(d1.date.dt.strftime("%Y-%m-%d") + " " + d1.dialysisend.astype(str), errors="coerce")
    d1["sched_h"] = ((de - ds).dt.total_seconds() / 3600) % 24                      # handles sessions crossing midnight
    d1["start_hhmm"] = d1.dialysisstart.astype(str)
    for c in ("sbp", "dbp", "ultrafiltrationrate", "dialysistime"):
        vip[c] = pd.to_numeric(vip[c], errors="coerce") if c in vip.columns else np.nan
    vip = vip[vip.sbp.between(30, 250) & vip.dialysistime.between(0, 480)].copy()
    ts = pd.to_datetime(vip.datatime, errors="coerce")
    vip["date"] = (ts - pd.to_timedelta(vip.dialysistime, unit="m")).dt.normalize()   # session START date (robust to midnight crossing)
    vip["t_raw"] = vip.dialysistime / 60.0                                             # elapsed hours since session start (documented 'time', minutes)
    vip = vip.dropna(subset=["date"])
    if "measuretime" in vip.columns:                                                   # 'measuretime' is a reading index; de-duplicate repeats
        vip = vip.drop_duplicates(["pid", "date", "measuretime"])
    v = vip.merge(d1[["pid", "date"]], on=["pid", "date"], how="inner").sort_values(["pid", "date", "t_raw"])
    v["sid"] = v.pid.astype(str) + "-" + v.date.dt.strftime("%Y%m%d")
    sess, obs_all = [], []
    for sid, g in v.groupby("sid"):
        g = g.drop_duplicates("t_raw")
        if len(g) < min_obs or g.t_raw.iloc[0] > 0.5: continue                         # need a pre-dialysis reading near start
        tt = (g.t_raw - g.t_raw.iloc[0]).values; keep = tt <= 6
        g, tt = g[keep], tt[keep]
        if len(g) < min_obs: continue
        sbp = g.sbp.values; hyp = np.where((sbp < HYPO_SBP) & (tt > 0))[0]; first = hyp[0] if len(hyp) else None
        u = g.ultrafiltrationrate.values
        u0 = np.nanmedian(u[tt <= 0.75]) if (tt <= 0.75).any() else np.nan               # initial (pre-event) UF setting = the PLAN
        sess.append(dict(session_id=sid, patient_id=str(g.pid.iloc[0]), date=g.date.iloc[0], _tmax=tt.max(), _ufr=u0,
                         uf_delivered_l_h=float(np.nanmean(u)), pre_sbp=sbp[0], pre_dbp=g.dbp.iloc[0], min_sbp=sbp[1:].min(),
                         nadir_time_h=tt[1:][sbp[1:].argmin()], hypotension=int(len(hyp) > 0)))
        obs_all.append(pd.DataFrame(dict(session_id=sid, patient_id=str(g.pid.iloc[0]), t_h=tt, sbp=sbp, dbp=g.dbp.values, hr=np.nan,
                                         post_event=[int(first is not None and j > first) for j in range(len(tt))])))
    S = pd.DataFrame(sess).merge(d1.assign(patient_id=d1.pid.astype(str))[["patient_id", "date", "weightstart", "weightend", "dryweight", "sched_h", "start_hhmm"]], on=["patient_id", "date"])
    S = S.sort_values(["patient_id", "date"])
    # planned duration = patient's usual duration from PRIOR sessions only (the realised duration can be cut short by events -> leakage)
    S["duration_h"] = S.groupby("patient_id").sched_h.transform(lambda x: x.shift().expanding(min_periods=1).median()).fillna(S.sched_h)
    S["duration_h"] = (S.duration_h * 4).round() / 4
    S = S[S.duration_h.between(2, 6) & S._ufr.notna() & (S._ufr >= 0)].copy()
    S["pre_weight_kg"], S["post_weight_kg"], S["target_dry_weight_kg"] = S.weightstart, S.weightend, S.dryweight
    S = S[(S.pre_weight_kg - S.target_dry_weight_kg).between(-1, 8)]
    S["uf_rate_l_h"] = S._ufr; S["planned_uf_l"] = (S._ufr * S.duration_h).round(2)
    S = S.sort_values(["patient_id", "date"]).groupby("patient_id").tail(max_sessions)
    S = S.groupby("patient_id").filter(lambda x: len(x) >= 10).copy()
    S["session_index"] = S.groupby("patient_id").cumcount()
    prev = S.groupby("patient_id")
    S["gap_days"] = prev.date.diff().dt.days; S["idwg_kg"] = S.pre_weight_kg - prev.post_weight_kg.shift()
    S = S[(S.gap_days.isna()) | (S.gap_days.between(1, 7))]
    S["session_index"] = S.groupby("patient_id").cumcount()
    S["uf_rate_ml_kg_h"] = 1000 * S.uf_rate_l_h / S.pre_weight_kg
    S["start_time"] = S.start_hhmm; S["prediction_timestamp"] = S.date.dt.strftime("%Y-%m-%d") + " " + S.start_hhmm
    S["pre_hr"] = np.nan; S["temp_c"] = 28.0; S["heatwave_flag"] = 0; S["is_upcoming"] = 0
    for c in ("saline_given", "uf_pause", "early_termination"): S[c] = np.nan
    S = S.drop(columns=[c for c in S.columns if c.startswith("_") or c in ("weightstart", "weightend", "dryweight", "sched_h", "start_hhmm")])
    O = pd.concat(obs_all); O = O[O.session_id.isin(S.session_id)]
    # patients
    p = S.groupby("patient_id").agg(baseline_sbp=("pre_sbp", "median"), dry_weight_kg=("target_dry_weight_kg", "median"),
                                    first=("date", "min"), last=("date", "max"), n=("date", "size")).reset_index()
    idp = idp.assign(patient_id=idp.pid.astype(str)).drop_duplicates("patient_id").drop(columns=["pid"])
    P = p.merge(idp, on="patient_id", how="left")
    by = pd.to_numeric(P.get("birthyear"), errors="coerce"); P["age"] = (P["first"].dt.year - by).fillna(60)
    P["sex"] = P["gender"].astype(str).str.upper().str[0].map({"M": "M", "1": "M"}).fillna("F") if "gender" in P else "M"
    P["diabetes"] = _truthy(P["diabetes"]) if "diabetes" in P else 0
    P["dialysis_per_week"] = (P.n / (((P["last"] - P["first"]).dt.days / 7).clip(lower=1))).round(1)
    P = P.assign(display_name="", hypertension=np.nan, cardiovascular_disease=np.nan, ckd_stage="ESRD (HEMOBP)", albumin_g_dl=np.nan,
                 hemoglobin_g_dl=np.nan, potassium_mmol_l=np.nan, bp_medication_categories=np.nan, baseline_hr=np.nan, usual_duration_h=4.0,
                 schedule_days="", shift_start="")
    P = P[["patient_id", "display_name", "age", "sex", "diabetes", "hypertension", "cardiovascular_disease", "ckd_stage", "dialysis_per_week",
           "schedule_days", "shift_start", "baseline_sbp", "dry_weight_kg", "usual_duration_h", "bp_medication_categories", "albumin_g_dl",
           "hemoglobin_g_dl", "potassium_mmol_l", "baseline_hr"]]
    sens = pd.DataFrame(columns=["patient_id", "date", "temp_c", "heatwave_flag", "home_weight_kg", "home_sbp", "hr_rest", "hrv_rmssd_ms", "activity_steps", "sleep_h"])
    truth = pd.DataFrame(columns=["patient_id", "true_log_k", "true_log_S", "true_g_kg_day", "true_hr_response"])
    P.to_csv(out_dir / "synthetic_patients.csv", index=False); S.to_csv(out_dir / "synthetic_sessions.csv", index=False)
    O.to_csv(out_dir / "synthetic_intradialytic_obs.csv", index=False); sens.to_csv(out_dir / "synthetic_sensor_data.csv", index=False)
    truth.to_csv(out_dir / "synthetic_ground_truth_params.csv", index=False)     # filenames kept so the pipeline is reused unchanged
    rep = dict(patients=len(P), sessions=len(S), bp_obs=len(O), hypotension_rate=float(S.hypotension.mean()),
               median_excess_kg=float((S.pre_weight_kg - S.target_dry_weight_kg).median()), median_uf_l=float(S.planned_uf_l.median()),
               median_obs_per_session=float(O.groupby("session_id").size().median()))
    print("HEMOBP conversion sanity report:", rep)
    return rep
