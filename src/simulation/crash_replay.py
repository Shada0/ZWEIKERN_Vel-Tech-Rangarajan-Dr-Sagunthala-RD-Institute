"""Historical replay: reconstruct the pre-session prediction for a past hypotension session, then reveal the outcome."""
import numpy as np
from src.models.personalized_model import plan_from_row


def list_events(D, pred_df, min_prior=6):
    s = D.sessions[(D.sessions.hypotension == 1) & (D.sessions.session_index >= min_prior)]
    return s.join(pred_df[["p_twin", "p_lr"]], how="left")


def replay(D, sid, cal, pred_row):
    r = D.sessions.loc[sid]
    twin = D.twin_for(r.patient_id, r.session_index)          # fitted ONLY on sessions before this one
    F, Fv = D.fluid_estimate(twin, sid)
    plan = plan_from_row(r, F, Fv)
    pr = twin.predict(plan, 800, seed=3)
    t, sbp, hr, keep = D.obs_by_sid[sid]
    return dict(row=r, twin=twin, plan=plan, pred=pr, p_twin=float(cal(pr.p_raw)[0]), p_lr=float(pred_row["p_lr"]),
                t=t, sbp=sbp, hr=hr, keep=keep, timestamp=r.prediction_timestamp)
