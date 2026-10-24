"""Model B: personalized digital-twin risk estimator (physics-based Monte-Carlo + 1-D Platt calibration)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from src.config import SENS_HEAT
from src.physiology.twin_engine import PatientTwin, Plan, kalman_fluid


def _logit(p):
    p = np.clip(p, 0.005, 0.995); return np.log(p / (1 - p))


class TwinCalibrator:
    """Maps raw simulated probability to a calibrated probability (fit on TRAIN patients only)."""
    def __init__(self):
        self.lr = LogisticRegression(C=100)

    def fit(self, p_raw, y):
        self.lr.fit(_logit(np.asarray(p_raw))[:, None], y); return self

    def __call__(self, p_raw):
        return self.lr.predict_proba(_logit(np.atleast_1d(np.asarray(p_raw, float)))[:, None])[:, 1]


def plan_from_row(r, F_est, F_sd):
    return Plan(duration_h=r.duration_h, uf_volume=r.planned_uf_l, fluid_excess=float(F_est), pre_sbp=r.pre_sbp,
                fluid_sd=float(np.sqrt(F_sd + 0.2 ** 2)), temp_c=r.temp_c)


def compute_twin_table(D, progress=None, pids=None, prior=None):
    """Sequential (temporal) twin predictions for EVERY session using only earlier sessions' data."""
    out = []
    pids = list(D.sess_by_pid) if pids is None else pids
    for n, pid in enumerate(pids):
        g = D.sess_by_pid[pid]; recs, gains = [], []; x0 = None
        for sid, r in g.iterrows():
            twin = PatientTwin(prior=prior).fit(recs[-8:], gains[-8:])
            F_est, F_var = D.fluid_estimate(twin, sid)
            plan = plan_from_row(r, F_est, F_var)
            pr = twin.predict(plan, n_mc=400, seed=1)
            pp = twin.predict(plan, n_mc=400, seed=1, personalize=False)
            prm = twin.params
            out.append(dict(session_id=sid, p_twin_raw=pr.p_raw, p_pop_raw=pp.p_raw, nadir_mean=pr.nadir_mean,
                            nadir_sd=pr.nadir_sd, k_hat=prm["k_refill"], S_hat=prm["S_bp"], sd_logk=prm["sd_logk"],
                            sd_logS=prm["sd_logS"], g_hat=prm["g_kg_day"], c_hr=prm["c_hr"], F_est=F_est, F_sd=np.sqrt(F_var),
                            n_fit=prm["n_sessions"]))
            if not r.is_upcoming:
                recs.append(D.fit_rec(sid)); gains.append(r.gain_per_day)
        if progress and n % 10 == 0:
            progress(n, len(pids))
    return pd.DataFrame(out).set_index("session_id")


def estimate_population_prior(D, train_pids, n_sessions=15, floor_sd=0.12):
    """Empirical-Bayes population prior on (log k, log S) from TRAIN patients only.
    Each train patient is fitted with a broad prior on their first n_sessions; the between-patient spread is
    corrected for estimation noise (deconvolution) and floored. No test-patient or later-session data are used."""
    from src.physiology.parameter_estimation import fit_patient
    broad = (np.array([-0.7, 0.5]), np.array([1.0, 1.0])); est, var = [], []
    for pid in train_pids:
        g = D.sess_by_pid[pid]; g = g[g.is_upcoming == 0].head(n_sessions)
        if len(g) < 5: continue
        th, cov, n = fit_patient([D.fit_rec(s) for s in g.session_id], prior=broad)
        if n >= 20: est.append(th); var.append(np.diag(cov))
    est, var = np.array(est), np.array(var)
    mu = est.mean(0); sd = np.sqrt(np.maximum(est.var(0) - var.mean(0), floor_sd ** 2))
    return (mu.tolist(), sd.tolist()), len(est)
