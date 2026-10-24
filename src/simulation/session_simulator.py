"""Builds the 'pre-session context' for any session: twin, Kalman fluid state, default hypothetical plan and risk."""
import numpy as np
from src.config import SENS_HEAT
from src.physiology.fluid_model import heat_stress
from src.physiology.twin_engine import Plan
from src.simulation.heatwave_simulator import heatwave_days, DEMO_TARGET_GAIN_KG, DEMO_HEAT_TEMP_C


def build_context(D, sid, cal, heat=False, checkins=None, personalize=True):
    r = D.sessions.loc[sid]; pid = r.patient_id
    twin = D.twin_for(pid, r.session_index, personalize)
    g = D.sess_by_pid[pid]; prev = g[g.session_index == r.session_index - 1].iloc[0]
    dry = r.target_dry_weight_kg; post_ex = prev.post_weight_kg - dry
    days = D.days_between(pid, prev.date, r.date)
    pre_w, temp, uf = r.pre_weight_kg, r.temp_c, r.planned_uf_l
    if heat:
        tg = DEMO_TARGET_GAIN_KG if pid == "SYN-RAVI" else None
        days, pre_ex = heatwave_days(days, post_ex, twin.g, DEMO_HEAT_TEMP_C, tg)
        pre_w, temp = dry + pre_ex, DEMO_HEAT_TEMP_C
        uf = round(float(np.clip(pre_ex * 0.95, 0.5, 5.0)), 1)
    if checkins:                                    # simulated daily check-in weights update the state
        for c in checkins:
            if c.get("weight") and days:
                days[-1] = dict(days[-1], excess=float(c["weight"]) - dry)
    F, Fv = twin.fluid_state(post_ex, days, pre_excess=pre_w - dry)
    plan = Plan(duration_h=r.duration_h, uf_volume=uf, fluid_excess=float(F), pre_sbp=float(r.pre_sbp),
                fluid_sd=float(np.sqrt(Fv + 0.2 ** 2)), temp_c=float(temp))
    pred = twin.predict(plan, n_mc=800, seed=3)
    p = float(cal(pred.p_raw)[0])
    state = dict(temp_c=float(temp), heat_stress=float(heat_stress(temp)), pre_excess=float(F), gain_since_last=float(pre_w - prev.post_weight_kg),
                 S_eff=float(twin.params["S_bp"] * (1 + SENS_HEAT * heat_stress(temp))), p_hypo=p, F_sd=float(np.sqrt(Fv)),
                 post_excess=float(post_ex))
    return dict(row=r, twin=twin, plan=plan, pred=pred, p=p, days=days, state=state, prev=prev, pre_weight=float(pre_w), dry=float(dry))
