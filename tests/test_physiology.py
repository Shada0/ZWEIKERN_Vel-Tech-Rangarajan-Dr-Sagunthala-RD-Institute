import numpy as np
from src.physiology.fluid_model import drop_pct_closed, drop_pct_ode, heat_gain_mult
from src.physiology.parameter_estimation import fit_patient
from src.physiology.twin_engine import kalman_fluid

def test_ode_matches_closed_form():
    t = np.arange(0, 4.01, .25); F, k = np.array([2.5]), np.array([.5])
    assert np.abs(drop_pct_closed(t[None], F[:, None], .625, k[:, None]) - drop_pct_ode(t, F, k, lambda x: .625)).max() < 1e-5

def test_faster_removal_bigger_drop():
    t = np.array([4.0]); assert drop_pct_closed(t, 2.5, 0.8, .5) > drop_pct_closed(t, 2.5, 0.5, .5)

def test_slower_refill_bigger_drop():
    t = np.array([4.0]); assert drop_pct_closed(t, 2.5, .6, .2) > drop_pct_closed(t, 2.5, .6, .8)

def test_heat_multiplier():
    assert heat_gain_mult(30) == 1.0 and heat_gain_mult(42) > 1.4

def test_parameter_recovery_noise_free():
    rng = np.random.default_rng(0); recs = []
    k, S = 0.3, 3.0
    for _ in range(10):
        F0 = rng.uniform(1.5, 3.5); u = F0 * 0.95 / 4; t = np.arange(0, 4.01, .25)
        sbp = 130 - S * drop_pct_closed(t, F0, u, k) + rng.normal(0, 1, len(t))
        recs.append(dict(t=t, sbp=sbp, keep=np.ones(len(t), bool), F0=F0, u=u, pre_sbp=130, stress=0.0))
    th, cov, _ = fit_patient(recs)
    assert abs(np.exp(th[1]) - S) / S < 0.25 and abs(np.exp(th[0]) - k) / k < 0.35

def test_kalman_moves_toward_measurement():
    x, P = kalman_fluid(0.3, [dict(temp=28, excess=3.0)], 0.6)
    assert 0.9 < x < 3.0 and P < 0.2
