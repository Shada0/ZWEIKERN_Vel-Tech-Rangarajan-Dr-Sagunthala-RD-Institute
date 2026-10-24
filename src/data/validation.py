"""Dataset sanity checks (used by tests and the Data Quality view)."""
import numpy as np
import pandas as pd


def validate(D):
    s = D.sessions; h = s[s.is_upcoming == 0]
    checks = {
        "n_patients >= 100": len(D.patients) >= 100,
        ">= 6 months history per patient": bool((h.groupby("patient_id").date.agg(lambda x: (x.max() - x.min()).days) >= 170).all()),
        "both outcome classes present": bool(0.05 < h.hypotension.mean() < 0.5),
        "upcoming sessions carry no outcome": bool(s[s.is_upcoming == 1].hypotension.isna().all()),
        "no intradialytic obs for upcoming sessions": not D.obs.session_id.isin(s[s.is_upcoming == 1].session_id).any(),
        "all patients synthetic IDs": bool(D.patients.patient_id.str.startswith("SYN-").all()),
    }
    return checks


def missingness(D):
    cols = ["home_weight_kg", "home_sbp", "hr_rest", "hrv_rmssd_ms", "activity_steps", "sleep_h"]
    return D.sensors[cols].isna().mean().rename("missing_fraction").to_frame()
