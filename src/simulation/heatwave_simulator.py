"""Synthetic heat-wave scenario. Uses only the documented synthetic rule heat_gain_mult(T); for the Ravi demo the
2.8 kg gain by day 3 is a DEMONSTRATION INPUT (not a clinical threshold). Weather is synthetic, not real data."""
import pandas as pd
from src.physiology.fluid_model import heat_gain_mult, heat_stress

DEMO_TARGET_GAIN_KG = 2.8     # demonstration input for Ravi (by day 3)
DEMO_HEAT_TEMP_C = 42.0


def heatwave_days(days, post_excess, g_hat, temp=DEMO_HEAT_TEMP_C, target_gain_by_day3=None):
    """Replace inter-session days by a heat-wave scenario. Returns (days, pre_excess)."""
    per_day = target_gain_by_day3 / 3.0 if target_gain_by_day3 else g_hat * float(heat_gain_mult(temp))
    out = []
    for i, d in enumerate(days):
        nd = dict(d); nd.update(temp=temp, excess=post_excess + per_day * (i + 1), heatwave=1)
        out.append(nd)
    return out, post_excess + per_day * len(days)


def change_table(base, heat):
    """Exactly which inputs changed between baseline and heat-wave scenario."""
    rows = []
    for k, label in [("temp_c", "Temperature (C)"), ("heat_stress", "Heat-stress input (0-1)"), ("pre_excess", "Fluid excess at session (kg)"),
                     ("gain_since_last", "Interdialytic weight gain (kg)"), ("S_eff", "Effective BP sensitivity (mmHg per %)"),
                     ("p_hypo", "Twin hypotension probability")]:
        rows.append(dict(variable=label, baseline=base[k], heat_wave=heat[k], change=heat[k] - base[k]))
    return pd.DataFrame(rows)
