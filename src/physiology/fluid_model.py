"""Two-compartment fluid model (blood + tissue/interstitial).

State (relative to start of session):  dVb = change in blood volume (L),  Vt = excess tissue fluid (L)
    d(dVb)/dt = k * Vt - u(t)        k: plasma refill rate (1/h), u: ultrafiltration rate (L/h)
    d(Vt)/dt  = -k * Vt
Initial: Vt(0) = (1 - BLOOD_FRAC) * F, F = total fluid excess over dry weight (kg ~ L).
Closed form for constant u:  dVb(t) = Vt0 (1 - e^{-kt}) - u t
Relative blood-volume drop (%): drop = max(0, -100 * dVb / BV0)
Inter-session accumulation: F += g * heat_gain_mult(T) per day (g = patient accumulation rate, kg/day).
"""
import numpy as np
from scipy.integrate import solve_ivp
from src.config import BV0, BLOOD_FRAC, HEAT_ONSET


def heat_gain_mult(temp_c):
    """Documented synthetic rule: +5% daily fluid gain per deg C above 32 C."""
    return 1.0 + 0.05 * np.maximum(0.0, np.asarray(temp_c, float) - HEAT_ONSET)


def heat_stress(temp_c):
    """0..1 environmental stress input (synthetic): linear 32 C -> 42 C."""
    return np.clip((np.asarray(temp_c, float) - HEAT_ONSET) / 10.0, 0.0, 1.0)


def tissue_volume(F):
    return (1.0 - BLOOD_FRAC) * np.asarray(F, float)


def drop_pct_closed(t, F, u, k):
    """Closed-form relative blood-volume drop (%), constant UF rate u. Broadcasts."""
    t = np.asarray(t, float)
    raw = 100.0 * (u * t - tissue_volume(F) * (1.0 - np.exp(-np.asarray(k, float) * t))) / BV0
    return np.maximum(raw, 0.0)


def uf_profile_fn(total_l, duration_h, kind="constant"):
    """UF-rate profile u(t) (L/h) integrating to total_l over duration_h."""
    mean = total_l / duration_h
    if kind == "front-loaded":
        return lambda t: mean * (1.4 - 0.8 * t / duration_h)
    return lambda t: mean


def drop_pct_ode(t_eval, F, k, u_fn):
    """Numerical (SciPy solve_ivp) solution, vectorised over N parameter draws. Returns (N, T)."""
    F, k = np.atleast_1d(np.asarray(F, float)), np.atleast_1d(np.asarray(k, float))
    n = max(F.size, k.size)
    F, k = np.broadcast_to(F, (n,)), np.broadcast_to(k, (n,))

    def rhs(t, y):
        flux = k * y[n:]
        return np.concatenate([flux - u_fn(t), -flux])

    y0 = np.concatenate([np.zeros(n), tissue_volume(F)])
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return np.maximum(-100.0 * sol.y[:n] / BV0, 0.0)


def cumulative_uf(t, total_l, duration_h, kind="constant"):
    fine = np.linspace(0, duration_h, 400)
    fn = uf_profile_fn(total_l, duration_h, kind)
    u = np.array([fn(x) for x in fine])
    cum = np.concatenate([[0], np.cumsum((u[1:] + u[:-1]) / 2 * np.diff(fine))])
    return np.interp(t, fine, cum)
