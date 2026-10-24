import joblib, numpy as np, pandas as pd
from src.config import DATA_DIR, MODEL_DIR
from src.physiology.twin_engine import PatientTwin, Plan

def test_probabilities_valid():
    P = pd.read_csv(DATA_DIR / "predictions.csv")
    for c in ("p_lr", "p_rf", "p_twin"): assert P[c].between(0, 1).all()

def test_twin_risk_monotone_in_uf():
    tw = PatientTwin(); pl = Plan(4.0, 2.0, 2.5, 130)
    assert tw.predict(pl.replace(uf_volume=3.5), 400, 1).p_raw >= tw.predict(pl, 400, 1).p_raw

def test_evaluation_present_and_honest_fields():
    import json; E = json.load(open(MODEL_DIR / "evaluation.json"))
    for sp in ("patient_holdout", "temporal"):
        assert "Personalized digital twin" in E[sp]["models"] and "lo" in E[sp]["auc_diff_twin_minus_lr"]

def test_test_patients_not_in_training():
    M = joblib.load(MODEL_DIR / "models.joblib"); P = pd.read_csv(DATA_DIR / "predictions.csv")
    assert (P.groupby("patient_id").split.nunique() == 1).all()


def test_hybrid_and_ci_fields():
    import json; E = json.load(open(MODEL_DIR / "evaluation.json"))
    for sp in ("patient_holdout", "temporal"):
        assert "Hybrid: standard features + twin outputs" in E[sp]["models"] and "lo" in E[sp]["auc_diff_hybrid_minus_lr"]


def test_empirical_prior_used_and_train_only():
    M = joblib.load(MODEL_DIR / "models.joblib"); mu, sd = M["prior"]
    assert len(mu) == 2 and all(s > 0 for s in sd)
    from src.models.personalized_model import estimate_population_prior
    from src.data.preprocessing import Data
    D = Data(); train = [p for p, v in M["split"].items() if v == "train"][:15]
    (mu2, sd2), n = estimate_population_prior(D, train); assert n > 5 and all(np.isfinite(mu2))
