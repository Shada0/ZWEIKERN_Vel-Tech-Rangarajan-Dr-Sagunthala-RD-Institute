from src.physiology.twin_engine import PatientTwin, Plan
from src.planner.what_if_planner import scenario_table, wording
import joblib
from src.config import MODEL_DIR

def test_scenarios_recalculate_and_wording():
    cal = joblib.load(MODEL_DIR / "models.joblib")["twin_cal"]
    df, _ = scenario_table(PatientTwin(), Plan(4.0, 2.8, 3.0, 125), cal, 0.7)
    assert len(df) == 4 and df.risk.between(0, 1).all()
    assert df.risk.iloc[2] <= df.risk.iloc[0]          # lower UF target should not raise simulated risk
    txt = " ".join(df.interpretation).lower()
    assert "prescribe" not in txt and "you must" not in txt

def test_wording_is_non_directive():
    for d in (-.2, 0, .2): assert "review" in wording(d).lower() or "discuss" in wording(d).lower()

def test_heat_changes_prediction(D):
    from src.simulation.session_simulator import build_context
    cal = joblib.load(MODEL_DIR / "models.joblib")["twin_cal"]
    sid = D.sessions[(D.sessions.patient_id == "SYN-RAVI") & (D.sessions.is_upcoming == 1)].index[0]
    a, b = build_context(D, sid, cal, False), build_context(D, sid, cal, True)
    assert b["state"]["pre_excess"] > a["state"]["pre_excess"] and b["p"] > a["p"]
