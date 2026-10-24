"""Patient-specific digital-twin engine.

A PatientTwin holds (i) estimated parameters (k, S, g, c_hr) with uncertainty, (ii) a Kalman-filtered
fluid-excess state, and (iii) a Monte-Carlo simulator that propagates parameter/state uncertainty plus
physiological noise through the two-compartment model to give P(SBP < 90 mmHg at any monitored time).
"""
from dataclasses import dataclass, field
import numpy as np
from src.config import (HYPO_SBP, OBS_STEP_H, MEAS_SD, PRIOR_LOGK, PRIOR_LOGS, PRIOR_G, SESSION_SD)
from src.physiology.fluid_model import (drop_pct_closed, drop_pct_ode, uf_profile_fn, cumulative_uf,
                                        heat_gain_mult, heat_stress)
from src.physiology.blood_pressure_model import sbp_trajectory, ar1_noise
from src.physiology.parameter_estimation import fit_patient, fit_hr_response, estimate_accumulation


@dataclass
class Plan:
    duration_h: float
    uf_volume: float
    fluid_excess: float
    pre_sbp: float
    fluid_sd: float = 0.3
    uf_profile: str = "constant"
    temp_c: float = 28.0

    def replace(self, **kw):
        d = dict(self.__dict__); d.update(kw); return Plan(**d)

    @property
    def uf_rate_l_h(self):
        return self.uf_volume / self.duration_h


@dataclass
class Prediction:
    p_raw: float
    times: np.ndarray
    mean_sbp: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    nadir_mean: float
    nadir_sd: float
    drop_mean: np.ndarray
    uf_cum: np.ndarray
    k: float
    S: float
    extras: dict = field(default_factory=dict)


def kalman_fluid(post_excess, days, g_hat, pre_excess=None, q=0.10, r_home=0.30, r_clinic=0.12):
    """Scalar Kalman filter for fluid excess F (kg above recorded dry weight).
    Predict: F += g * heat_gain_mult(T);  Update: home-scale weight (noisy, may be missing) and clinic weight."""
    x, P = float(post_excess), 0.15 ** 2
    for d in days:
        x += g_hat * float(heat_gain_mult(d["temp"]))
        P += q ** 2 + (0.15 * g_hat) ** 2
        if d.get("excess") is not None and np.isfinite(d["excess"]):
            K = P / (P + r_home ** 2); x += K * (d["excess"] - x); P *= (1 - K)
    if pre_excess is not None:
        K = P / (P + r_clinic ** 2); x += K * (pre_excess - x); P *= (1 - K)
    return x, P


class PatientTwin:
    def __init__(self, personalize=True, prior=None):
        self.personalize = personalize
        self.prior = prior
        self.prior_mu = np.array([PRIOR_LOGK[0], PRIOR_LOGS[0]]) if prior is None else np.asarray(prior[0], float)
        self.prior_sd = np.array([PRIOR_LOGK[1], PRIOR_LOGS[1]]) if prior is None else np.asarray(prior[1], float)
        self.theta, self.cov = self.prior_mu.copy(), np.diag(self.prior_sd ** 2)
        self.g, self.g_sd, self.c_hr, self.n_fit_obs, self.n_sessions = PRIOR_G[0], PRIOR_G[1], 0.8, 0, 0

    # ---- initialisation / updating from historical + new observations
    def fit(self, recs, per_day_gain):
        self.n_sessions = len(recs)
        self.theta, self.cov, self.n_fit_obs = fit_patient(recs, prior=self.prior)
        self.c_hr = fit_hr_response(recs, self.theta) if recs else 0.8
        self.g, self.g_sd = estimate_accumulation(per_day_gain)
        return self

    def update(self, new_rec, new_gain=None, history=None, gains=None):
        """Incorporate a newly completed session: re-estimate parameters with warm start."""
        recs = (history or []) + [new_rec]
        g = (gains or []) + ([new_gain] if new_gain is not None else [])
        th, cov, n = fit_patient(recs, x0=self.theta, prior=self.prior)
        self.theta, self.cov, self.n_fit_obs, self.n_sessions = th, cov, n, len(recs)
        self.g, self.g_sd = estimate_accumulation(g)
        return self

    @property
    def params(self):
        s = np.sqrt(np.diag(self.cov))
        return dict(k_refill=float(np.exp(self.theta[0])), S_bp=float(np.exp(self.theta[1])),
                    sd_logk=float(s[0]), sd_logS=float(s[1]), g_kg_day=self.g, g_sd=self.g_sd,
                    c_hr=self.c_hr, n_sessions=self.n_sessions, n_obs=self.n_fit_obs)

    def fluid_state(self, post_excess, days, pre_excess=None):
        return kalman_fluid(post_excess, days, self.g, pre_excess)

    # ---- simulation
    def predict(self, plan: Plan, n_mc=500, seed=0, personalize=None):
        pers = self.personalize if personalize is None else personalize
        rng = np.random.default_rng(seed)
        mu, cov = (self.theta, self.cov) if pers else (self.prior_mu, np.diag(self.prior_sd ** 2))
        th = rng.multivariate_normal(mu, cov, size=n_mc)
        k, S = np.exp(th[:, 0]), np.exp(th[:, 1] + rng.normal(0, SESSION_SD, n_mc))   # + session-level variability
        F = np.clip(rng.normal(plan.fluid_excess, plan.fluid_sd, n_mc), 0.05, None)
        t = np.arange(0, plan.duration_h + 1e-9, OBS_STEP_H)
        if plan.uf_profile == "constant":
            drop = drop_pct_closed(t[None, :], F[:, None], plan.uf_rate_l_h, k[:, None])
        else:
            drop = drop_pct_ode(t, F, k, uf_profile_fn(plan.uf_volume, plan.duration_h, plan.uf_profile))
        clean = sbp_trajectory(plan.pre_sbp, S[:, None], float(heat_stress(plan.temp_c)), drop)
        noisy = clean + ar1_noise(n_mc, len(t), rng) + rng.normal(0, MEAS_SD, (n_mc, len(t)))
        noisy[:, 0] = plan.pre_sbp
        p = float((noisy[:, 1:].min(1) < HYPO_SBP).mean())
        nad = clean.min(1)
        return Prediction(p_raw=p, times=t, mean_sbp=clean.mean(0), p10=np.percentile(noisy, 10, 0),
                          p90=np.percentile(noisy, 90, 0), nadir_mean=float(nad.mean()), nadir_sd=float(nad.std()),
                          drop_mean=drop.mean(0), uf_cum=cumulative_uf(t, plan.uf_volume, plan.duration_h, plan.uf_profile),
                          k=float(np.exp(mu[0])), S=float(np.exp(mu[1])))
