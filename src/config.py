"""Global configuration. All thresholds are DEMONSTRATION values, not clinical standards."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

SEED = 42
N_PATIENTS = 120
HISTORY_WEEKS = 26                 # ~6 months of history
CUTOFF_DATE = "2026-10-22"         # "now": history ends here, upcoming sessions follow

# Outcome definition (simplified research label)
HYPO_SBP = 90.0                    # mmHg; session positive if any monitored SBP < 90
OBS_STEP_H = 0.25                  # BP monitored every 15 min

# Physiological model constants (illustrative, NOT validated individual physiology)
BV0 = 5.0                          # nominal blood volume (L)
BLOOD_FRAC = 0.15                  # fraction of fluid excess sitting in blood compartment at session start
HEAT_ONSET = 32.0                  # deg C above which synthetic heat effects start
SENS_HEAT = 0.15                   # +15% BP sensitivity at full heat stress

# Noise model assumed by the twin (and used, with extra misspecification, by the generator)
AR_RHO, AR_SD, MEAS_SD = 0.7, 3.0, 1.5
SESSION_SD = 0.22                  # session-to-session log-variability of BP sensitivity (day-of-session physiology)

# Weakly informative population priors on log-parameters: (mean, sd)
PRIOR_LOGK = (-0.693, 0.40)        # refill rate k, median 0.5 /h
PRIOR_LOGS = (0.531, 0.40)         # BP sensitivity S, median 1.7 mmHg per % blood-volume drop
PRIOR_G = (0.65, 0.18)             # fluid accumulation, kg/day

# Configurable demo risk thresholds
AMBER_DEFAULT, RED_DEFAULT = 0.30, 0.55

DISCLAIMER = "SIMULATED DATA - RESEARCH PROTOTYPE - NOT A MEDICAL DEVICE - NOT A TREATMENT RECOMMENDATION"


def risk_category(p, amber=AMBER_DEFAULT, red=RED_DEFAULT):
    return "Red" if p >= red else ("Amber" if p >= amber else "Green")
