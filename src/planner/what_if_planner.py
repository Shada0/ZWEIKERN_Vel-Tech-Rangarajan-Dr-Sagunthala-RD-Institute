"""Hypothetical what-if planner. Produces SIMULATIONS only - never treatment orders."""
import pandas as pd

NOTE = "Simulation only - not a treatment recommendation. Review with the treating nephrologist before any decision."


def wording(delta):
    if delta <= -0.05:
        return "Simulation indicates a potentially lower-risk scenario. Clinical review required."
    if delta >= 0.05:
        return "Simulation indicates a potentially higher-risk scenario. Consider reviewing the planned fluid-removal rate with the treating nephrologist."
    return "No material change in simulated risk. Discuss the hypothetical plan with the treating nephrologist."


def run_plan(twin, plan, cal, seed=3, n_mc=800):
    pr = twin.predict(plan, n_mc=n_mc, seed=seed)
    return pr, float(cal(pr.p_raw)[0])


def scenario_table(twin, plan, cal, heat_gain_per_day):
    scen = [("A: Original hypothetical plan", plan),
            ("B: Longer session (+30 min), same fluid volume", plan.replace(duration_h=plan.duration_h + 0.5)),
            ("C: Lower fluid-removal target (-0.5 L)", plan.replace(uf_volume=max(0.5, plan.uf_volume - 0.5))),
            ("D: Longer gap (+1 day): extra fluid gained, UF target raised to match", plan.replace(fluid_excess=plan.fluid_excess + heat_gain_per_day, uf_volume=round(plan.uf_volume + 0.95 * heat_gain_per_day, 1)))]
    rows, preds = [], {}
    for name, pl in scen:
        pr, p = run_plan(twin, pl, cal); preds[name] = pr
        rows.append(dict(scenario=name, duration_h=pl.duration_h, uf_volume_l=round(pl.uf_volume, 2), uf_rate_l_h=round(pl.uf_rate_l_h, 3),
                         fluid_excess_kg=round(pl.fluid_excess, 2), simulated_nadir_sbp=round(pr.nadir_mean, 1), risk=p))
    df = pd.DataFrame(rows); df["delta_vs_A"] = df.risk - df.risk.iloc[0]
    df["interpretation"] = [("Reference scenario." if i == 0 else wording(d)) for i, d in enumerate(df.delta_vs_A)]
    return df, preds
