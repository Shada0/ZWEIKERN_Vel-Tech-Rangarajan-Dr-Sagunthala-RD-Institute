"""MAP (regularised nonlinear least squares) estimation of patient-specific parameters.

theta = (log k, log S). Residuals = (observed SBP - model SBP)/sigma plus Gaussian prior terms.
Posterior covariance via Laplace approximation (J^T J)^-1. A Kalman filter is NOT used for these quasi-static
nonlinear parameters; a scalar Kalman filter IS used for the dynamic fluid-excess state (twin_engine.kalman_fluid).
"""
import numpy as np
from scipy.optimize import least_squares
from src.config import PRIOR_LOGK, PRIOR_LOGS, PRIOR_G, SENS_HEAT
from src.physiology.fluid_model import drop_pct_closed

SIGMA_BP = 4.5


def _stack(recs):
    out = [[] for _ in range(6)]
    for r in recs:
        m = r["keep"] & (r["t"] > 0)
        n = int(m.sum())
        if n == 0:
            continue
        for lst, v in zip(out, (r["t"][m], r["sbp"][m], np.full(n, r["F0"]), np.full(n, r["u"]),
                                np.full(n, r["pre_sbp"]), np.full(n, r["stress"]))):
            lst.append(v)
    return None if not out[0] else [np.concatenate(a) for a in out]


def fit_patient(recs, x0=None, prior=None):
    """Returns theta (2,), cov (2,2), n_obs. prior = (mu(2,), sd(2,)) on (log k, log S); default = config constants."""
    pm, ps = (np.array([PRIOR_LOGK[0], PRIOR_LOGS[0]]), np.array([PRIOR_LOGK[1], PRIOR_LOGS[1]])) if prior is None else (np.asarray(prior[0], float), np.asarray(prior[1], float))
    st = _stack(recs)
    if st is None:
        return pm.copy(), np.diag(ps ** 2), 0
    t, y, F0, u, pre, stress = st

    def resid(th):
        drop = drop_pct_closed(t, F0, u, np.exp(th[0]))
        pred = pre - np.exp(th[1]) * (1 + SENS_HEAT * stress) * drop
        return np.concatenate([(y - pred) / SIGMA_BP, (th - pm) / ps])

    res = least_squares(resid, pm if x0 is None else x0, bounds=([-3, -2], [1.5, 2.5]), max_nfev=40)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        cov = np.diag(ps ** 2)
    return res.x, cov, len(y)


def fit_hr_response(recs, theta):
    """Closed-form regression for HR response c (bpm per % blood-volume drop)."""
    num = den = 0.0
    for r in recs:
        m = r["keep"] & (r["t"] > 0)
        if m.sum() == 0:
            continue
        d = drop_pct_closed(r["t"][m], r["F0"], r["u"], np.exp(theta[0]))
        num += float(np.nansum((r["hr"][m] - r["hr_pre"]) * d)); den += float(np.sum(d * d))
    return num / den if den > 1e-9 and np.isfinite(num) else 0.8


def estimate_accumulation(per_day_gain, prior=PRIOR_G, n0=3.0):
    """Shrinkage estimate of fluid-accumulation rate g (kg/day) and its sd."""
    x = np.asarray([v for v in per_day_gain if np.isfinite(v)], float)
    n = len(x)
    g = (x.sum() + n0 * prior[0]) / (n + n0)
    sd = prior[1] / np.sqrt(n + n0) if n < 2 else np.sqrt((np.var(x, ddof=1) + 1e-4) / (n + n0))
    return float(g), float(sd)
