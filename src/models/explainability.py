"""Explanations: LR linear contributions (exact, not causal), twin finite-difference sensitivity. SHAP not used."""
import pandas as pd
from src.models.baseline_model import lr_contributions


def twin_sensitivity(twin, plan, cal, seed=3):
    """One-at-a-time perturbations with common random numbers. Local sensitivity, NOT causal effects."""
    base = float(cal(twin.predict(plan, 800, seed).p_raw)[0])
    tests = {"Fluid excess +0.5 kg": plan.replace(fluid_excess=plan.fluid_excess + 0.5),
             "UF volume -0.5 L": plan.replace(uf_volume=max(0.5, plan.uf_volume - 0.5)),
             "Duration +30 min": plan.replace(duration_h=plan.duration_h + 0.5),
             "Starting SBP +10 mmHg": plan.replace(pre_sbp=plan.pre_sbp + 10),
             "Heat +5 C": plan.replace(temp_c=plan.temp_c + 5)}
    rows = [dict(perturbation=k, risk=float(cal(twin.predict(v, 800, seed).p_raw)[0])) for k, v in tests.items()]
    df = pd.DataFrame(rows); df["change_vs_current"] = df.risk - base
    return base, df
