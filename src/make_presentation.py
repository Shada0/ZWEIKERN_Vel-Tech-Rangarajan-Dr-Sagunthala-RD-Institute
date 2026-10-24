"""Builds presentation/project_presentation.pdf (all numbers read from models/evaluation*.json)."""
import json
from reportlab.lib.pagesizes import landscape
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas
from src.config import ROOT, MODEL_DIR

W, H = 960, 540
NAVY, TEAL, ORANGE, GREY = HexColor("#16324f"), HexColor("#0f8b8d"), HexColor("#e07b39"), HexColor("#6b7a8c")
S = json.load(open(MODEL_DIR / "evaluation.json")); R = json.load(open(MODEL_DIR / "evaluation_real.json"))
sm, rm = S["patient_holdout"]["models"], R["patient_holdout"]["models"]
LR, TW, HY, AB = "Standard model: logistic regression", "Personalized digital twin", "Hybrid: standard features + twin outputs", "Twin ablation: population parameters (no personalization)"
c = canvas.Canvas(str(ROOT / "presentation/project_presentation.pdf"), pagesize=(W, H)); page = [0]


def slide(title, sub=None):
    if page[0]: c.showPage()
    page[0] += 1
    c.setFillColor(HexColor("#f6f9fb")); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(0, H - 78, W, 78, fill=1, stroke=0)
    c.setFillColor(white); c.setFont("Helvetica-Bold", 28); c.drawString(40, H - 50, title)
    if sub: c.setFont("Helvetica", 13); c.setFillColor(HexColor("#bcd3e6")); c.drawString(40, H - 70, sub)
    c.setFillColor(GREY); c.setFont("Helvetica", 9)
    c.drawString(40, 18, "NephroTwin AI - simulated research prototype, not a medical device. Team ZWEIKERN, Vel Tech, Chennai"); c.drawRightString(W - 40, 18, f"{page[0]}")


def bullets(items, x=50, y=H - 115, size=16, width=860, gap=10, color=NAVY):
    for it in items:
        lines = simpleSplit(it, "Helvetica", size, width - 24); c.setFillColor(TEAL); c.circle(x + 4, y + 5, 3, fill=1, stroke=0); c.setFillColor(color); c.setFont("Helvetica", size)
        for ln in lines: c.drawString(x + 20, y, ln); y -= size + 4
        y -= gap
    return y


def card(x, y, w, h, head, body, col=TEAL):
    c.setFillColor(white); c.setStrokeColor(HexColor("#e3e9ef")); c.roundRect(x, y, w, h, 10, fill=1, stroke=1)
    c.setFillColor(col); c.rect(x, y + h - 6, w, 6, fill=1, stroke=0); c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 15); c.drawString(x + 14, y + h - 30, head)
    yy = y + h - 52; c.setFont("Helvetica", 12.5); c.setFillColor(HexColor("#33475b"))
    for ln in simpleSplit(body, "Helvetica", 12.5, w - 28): c.drawString(x + 14, yy, ln); yy -= 16


def img(path, x, y, w, h):
    ir = ImageReader(str(path)); iw, ih = ir.getSize(); s = min(w / iw, h / ih); c.drawImage(ir, x + (w - iw * s) / 2, y + (h - ih * s) / 2, iw * s, ih * s, mask="auto")


# 1 title
page[0] = 1; c.setFillColor(NAVY); c.rect(0, 0, W, H, fill=1, stroke=0); c.setFillColor(TEAL); c.rect(0, 0, W, 14, fill=1, stroke=0)
c.setFillColor(white); c.setFont("Helvetica-Bold", 54); c.drawString(60, 340, "NephroTwin AI")
c.setFont("Helvetica", 21); c.drawString(60, 300, "A personalized dialysis digital twin for predicting")
c.drawString(60, 272, "intradialytic hypotension and supporting safer fluid-removal planning")
c.setFillColor(HexColor("#9fd6d7")); c.setFont("Helvetica-Oblique", 17); c.drawString(60, 225, "Predict the risk. Understand the patient. Plan dialysis more safely.")
c.setFillColor(HexColor("#bcd3e6")); c.setFont("Helvetica", 13)
c.drawString(60, 130, "Happiest Health Digital Twin Challenge 2026"); c.drawString(60, 108, "Team: ZWEIKERN   |   College: Vel Tech Rangarajan Dr. Sagunthala R&D Institute, Chennai   |   Members: Shada Ali Kuzhikattil, Rushdha V")
c.drawString(60, 70, "Simulated research prototype - synthetic cohort + open real-data validation (HEMOBP). Not a medical device.")

slide("The problem", "Intradialytic hypotension (IDH)")
bullets(["Maintenance hemodialysis removes fluid that accumulates between sessions; removing it too fast can drop blood pressure during treatment.",
         "Risk differs sharply between patients (refill rate, vascular response, comorbidity) and between sessions (fluid gain, heat, gap length).",
         "Population-level scores ignore patient-specific physiology; clinicians need a patient-level, explainable pre-session view.",
         "Our target: estimate the probability of hypotension BEFORE a session starts, show why, and let clinicians explore hypothetical scenarios - never issue orders."])

slide("The digital-twin concept", "A persistent, updating virtual patient - not a renamed classifier")
card(40, 310, 280, 130, "1  Initialise", "From historical sessions: estimate refill rate k, BP sensitivity S, fluid-gain rate g with uncertainty (MAP least squares, empirical-Bayes population prior learned from training patients only).")
card(340, 310, 280, 130, "2  Update", "Kalman filter fuses home and clinic weights into a fluid-excess state; each new session re-fits the parameters with warm start.")
card(640, 310, 280, 130, "3  Simulate", "Two-compartment ODE (SciPy) + BP response + AR(1) noise. Monte-Carlo over parameter and state uncertainty gives P(SBP < 90 mmHg) for the planned session and what-if variants.")
bullets(["Outcome label (demo definition): any monitored SBP < 90 mmHg.  Prediction timestamp = scheduled connection time; no later data are used."], y=270, size=17)

slide("Two data streams, leakage-controlled")
card(40, 320, 430, 120, "Stream A - static / historical", "Age, sex, diabetes, hypertension, CVD, baseline BP, dry weight, labs, medications, 6 months of sessions (UF, duration, IDWG, timed BP/HR, events).", NAVY)
card(490, 320, 430, 120, "Stream B - dynamic", "Daily home weight and BP, wearable HR / HRV / activity / sleep (with missingness), temperature and heat-wave flag, pre-dialysis vitals. BP is a cuff reading, not a wearable measurement.", ORANGE)
bullets(["Only data timestamped before the prediction time feed any model; twin parameters for session n use sessions < n only.",
         "Patient-grouped hold-out + temporal hold-out; automated tests assert that outcome columns never enter features.",
         "Real data (HEMOBP) has no wearables/weather/labs - those streams stay synthetic and are labelled as such."], y=285, size=16)

slide("Architecture"); img(ROOT / "docs/architecture_diagram.png", 60, 40, 840, 400)

slide("Evidence 1 - synthetic cohort (120 patients)", "Held-out patients; AUC difference with 95% patient-cluster bootstrap CI")
d = S["patient_holdout"]["auc_diff_twin_minus_lr"]
bullets([f"Personalized twin AUC {sm[TW]['auc']:.3f} vs standard logistic regression {sm[LR]['auc']:.3f} (difference {d['mean']:+.3f}, CI {d['lo']:+.3f} to {d['hi']:+.3f}).",
         f"Ablation: twin with population parameters only {sm[AB]['auc']:.3f} - personalization is what adds value.",
         "Caveat: the twin's structure resembles the data-generating process (with deliberate misspecification) - results are optimistic and not clinical evidence."], size=15, y=H - 112)
img(ROOT / "docs/figures/auc_comparison.png", 40, 30, 880, 270)

slide("Evidence 2 - real open data (HEMOBP)", "328 patients, 14,638 sessions, ~105k BP readings; hypotension 9.5%")
d2, h2 = R["patient_holdout"]["auc_diff_twin_minus_lr"], R["patient_holdout"]["auc_diff_hybrid_minus_lr"]
bullets([f"Twin alone does NOT beat the standard model on real data: AUC {rm[TW]['auc']:.3f} vs {rm[LR]['auc']:.3f} (difference {d2['mean']:+.3f}, CI {d2['lo']:+.3f} to {d2['hi']:+.3f}).",
         f"Personalization still helps vs the population-parameter twin ({rm[AB]['auc']:.3f} -> {rm[TW]['auc']:.3f}).",
         f"Hybrid (standard features + twin outputs): {rm[HY]['auc']:.3f}, difference vs standard {h2['mean']:+.3f} (CI {h2['lo']:+.3f} to {h2['hi']:+.3f}) - about on par.",
         "We report this negative result openly: sparse readings (median 6/session) limit per-patient identifiability."], size=14, y=H - 112)
img(ROOT / "docs/figures/roc_curves.png", 60, 20, 840, 255)

slide("The clinician dashboard", "Streamlit - 13 views")
bullets(["Project overview, Unit overview, Patient list, Weekly priority list (risk x uncertainty; no treatments assigned).",
         "Digital twin profile: parameters with uncertainty, Kalman fluid state, trends, explanations (LR contributions, twin sensitivity).",
         "Pre-session card + what-if planner: duration, UF volume/profile, start SBP, fluid gain, gap, heat wave; four scenario comparison with non-directive wording.",
         "Ravi guided demo (7 steps, heat-wave toggle, 'what changed' table), Historical replay (prediction first, outcome revealed after).",
         "Model evaluation, Real-data validation, Data quality, simulated daily check-in, Limitations & safety."], size=15)

slide("Safety, limitations and honesty")
bullets(["Not a medical device; synthetic patients labelled everywhere; thresholds are configurable demonstration values.",
         "Planner outputs are simulations: 'Clinical review required', 'Not a treatment recommendation'. No orders for duration, UF, medication or sessions.",
         "Real-data twin underperforms the standard model; synthetic superiority is not evidence of clinical effectiveness; retrospective simulation cannot show treatment benefit.",
         "Heat-wave inputs follow a documented synthetic rule; Ravi's 2.8 kg gain is a demonstration input."])

slide("Future scope")
bullets(["Prospective, ethics-approved validation with richer dialysis data (medications, continuous BP, UF profiles).",
         "Hierarchical Bayesian pooling and a better observation model to improve identifiability from sparse readings.",
         "Clinician usability study; integration of real wearable and weather feeds; calibration monitoring.",
         "Open source (MIT): repository github.com/Shada0/NephroTwin-AI  |  demo video: see README"])
c.save(); print("presentation written")
