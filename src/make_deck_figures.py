"""Render deck/README figures from the shipped models (no Streamlit needed): Ravi baseline vs heat wave, and real-data ROC comparison."""
import json, joblib, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import *
from src.data.preprocessing import Data
from src.simulation.session_simulator import build_context

NAVY, TEAL, RED, AMB, GRN = "#16324f", "#0f8b8d", "#d64545", "#e0a100", "#1f9d6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9d4de", "axes.labelcolor": NAVY, "text.color": NAVY})


def fp(p): return "\u2265 99%" if p >= .995 else ("< 1%" if p < .005 else f"{p:.0%}")
def cat(p): return risk_category(p, AMBER_DEFAULT, RED_DEFAULT)


def ravi():
    D = Data(); from src.models.compat import fix_sklearn_compat; M = fix_sklearn_compat(joblib.load(MODEL_DIR / "models.joblib")); D.prior = M.get("prior")
    sid = D.sessions[(D.sessions.patient_id == "SYN-RAVI") & (D.sessions.is_upcoming == 1)].index[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for heat, col, name in ((False, TEAL, "Baseline"), (True, RED, "Heat wave (synthetic 42 \u00b0C)")):
        c = build_context(D, sid, M["twin_cal"], heat=heat, checkins=[]); pr = c["pred"]; t = pr.times * 60
        ax.fill_between(t, pr.p10, pr.p90, color=col, alpha=.13, lw=0)
        ax.plot(t, pr.mean_sbp, color=col, lw=2.6, label=f"{name}: risk {fp(c['p'])} ({cat(c['p'])})")
    ax.axhline(HYPO_SBP, color=RED, ls=":", lw=1.3); ax.text(2, HYPO_SBP + 1.5, "SBP 90 mmHg (demo outcome threshold)", color=RED, fontsize=8)
    ax.set_xlabel("Minutes from session start"); ax.set_ylabel("Simulated SBP (mmHg)")
    ax.set_title("Ravi (synthetic patient): same plan, two scenarios", fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9); ax.grid(axis="y", color="#e3e9ef"); fig.tight_layout(); fig.savefig(ROOT / "docs/figures/ravi_baseline_vs_heatwave.png", facecolor="white")


def real_roc():
    R = json.load(open(MODEL_DIR / "evaluation_real.json"))["patient_holdout"]["models"]
    pick = [("Standard model: logistic regression", "Logistic regression", GRN, "-"), ("Personalized digital twin", "Personalized twin alone", RED, "-"),
            ("Hybrid: standard features + twin outputs", "Hybrid (standard + twin)", NAVY, "-"), ("Twin ablation: population parameters (no personalization)", "Twin, no personalization", "#6b7a8c", "--")]
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=200)
    for k, lab, col, ls in pick:
        v = R[k]; ax.plot(v["roc"][0], v["roc"][1], color=col, ls=ls, lw=2.4 if "Hybrid" in lab or "Logistic" in lab else 2, label=f"{lab} (AUC {v['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.set_title("Real HEMOBP data, held-out patients", fontsize=12, fontweight="bold", loc="left"); ax.legend(frameon=False, loc="lower right", fontsize=8.5); ax.grid(color="#e3e9ef")
    fig.tight_layout(); fig.savefig(ROOT / "docs/figures/real_roc_comparison.png", facecolor="white")


if __name__ == "__main__":
    ravi(); real_roc(); print("figures written")
