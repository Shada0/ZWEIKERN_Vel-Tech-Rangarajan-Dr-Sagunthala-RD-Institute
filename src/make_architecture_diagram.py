"""Generates docs/architecture_diagram.pdf with matplotlib."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.config import ROOT

fig, ax = plt.subplots(figsize=(13, 8.5)); ax.set_xlim(0, 13); ax.set_ylim(0, 8.5); ax.axis("off")
def box(x, y, w, h, text, c="#e8f1f8", ec="#16324f", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec=ec, lw=1.5)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#16324f")
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#16324f", lw=1.6))
ax.text(6.5, 8.2, "NephroTwin AI - Architecture (synthetic research prototype)", ha="center", fontsize=15, weight="bold", color="#16324f")
box(0.4, 6.7, 3.6, 1.0, "STREAM A: Synthetic historical EHR\nprofile, comorbidities, labs, past sessions,\nUF, IDWG, prior hypotension", "#dff3ee")
box(9.0, 6.7, 3.6, 1.0, "STREAM B: Dynamic simulated sensors\nHR, HRV, activity, sleep, home weight/BP,\nweather + heat-wave flag", "#fdf0dc")
box(0.4, 5.3, 3.6, 0.85, "Clinical feature extraction\n(only sessions BEFORE prediction time)")
box(9.0, 5.3, 3.6, 0.85, "Time-series pre-processing\n(missingness flags, 3-day windows)")
arrow(2.2, 6.7, 2.2, 6.15); arrow(10.8, 6.7, 10.8, 6.15)
box(3.6, 4.0, 5.8, 0.9, "Patient-specific state estimation\nMAP least-squares (k, S) + shrinkage g + scalar Kalman filter (fluid excess)", "#e3e0f7")
arrow(2.2, 5.3, 4.8, 4.9); arrow(10.8, 5.3, 8.2, 4.9)
ax.plot([0.2, 12.8], [3.65, 3.65], ls="--", color="#d64545"); ax.text(6.5, 3.74, "PREDICTION TIMESTAMP = scheduled connection time (no intradialytic data of the target session crosses this line)", ha="center", va="bottom", fontsize=8, color="#d64545")
box(0.2, 2.1, 3.0, 1.1, "Population baseline (Model A)\nlogistic regression / random forest\npre-session features only", "#eef0f3", fs=8)
box(3.4, 2.1, 3.4, 1.1, "Digital-twin physiology engine (Model B)\n2-compartment fluid ODE + BP response\nMonte-Carlo risk + Platt calibration", "#e3e0f7", fs=8)
box(7.0, 2.1, 3.0, 1.1, "Hybrid: standard features\n+ twin outputs (LR)", "#fdf0dc", fs=8)
box(10.2, 2.1, 2.6, 1.1, "Ablation: twin with\npopulation parameters", "#eef0f3", fs=8)
arrow(6.5, 4.0, 5.1, 3.2); arrow(2.2, 5.3, 1.7, 3.2); arrow(8.4, 4.0, 8.5, 3.2); arrow(8.4, 4.0, 11.5, 3.2)
ax.text(12.9, 4.35, "Validated on: synthetic cohort (120 pts)\nand open real data HEMOBP (328 pts)", ha="right", fontsize=8, color="#16324f", style="italic")
box(2.5, 0.9, 8.0, 0.8, "Pre-session hypotension risk (probability, Green/Amber/Red demo thresholds, uncertainty, missing data)", "#fde8e8")
arrow(5.1, 2.1, 6.0, 1.7); arrow(1.7, 2.1, 4.0, 1.7); arrow(8.5, 2.1, 8.0, 1.7)
box(0.4, 0.05, 6.0, 0.6, "What-if simulation (hypothetical scenarios, heat wave, replay)", "#dff3ee")
box(6.8, 0.05, 5.8, 0.6, "Clinician dashboard (Streamlit) - review required", "#dff3ee")
arrow(6.5, 0.9, 3.4, 0.65); arrow(6.5, 0.9, 9.7, 0.65)
fig.savefig(ROOT / "docs" / "architecture_diagram.pdf", bbox_inches="tight"); fig.savefig(ROOT / "docs" / "architecture_diagram.png", dpi=130, bbox_inches="tight")
