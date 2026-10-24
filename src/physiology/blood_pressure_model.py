"""Simplified circulating-volume -> blood-pressure relation used by the twin.

SBP(t) = SBP_pre - S * (1 + SENS_HEAT * stress) * drop(t)          [S: mmHg per % blood-volume drop]
HR(t)  = HR_pre + c * drop(t)                                      [c: bpm per %, cardiovascular response]
Residual physiological variability: AR(1) noise (rho, sd) + measurement noise.
"""
import numpy as np
from src.config import SENS_HEAT, AR_RHO, AR_SD


def sbp_trajectory(pre_sbp, S, stress, drop):
    return pre_sbp - S * (1.0 + SENS_HEAT * stress) * drop


def hr_trajectory(pre_hr, c, drop):
    return pre_hr + c * drop


def ar1_noise(n, T, rng, rho=AR_RHO, sd=AR_SD):
    e = np.zeros((n, T))
    for j in range(1, T):
        e[:, j] = rho * e[:, j - 1] + sd * rng.standard_normal(n)
    return e
