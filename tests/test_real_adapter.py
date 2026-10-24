import numpy as np, pandas as pd
from src.data.real_hemobp import convert
from src.data.preprocessing import Data


def test_hemobp_adapter_roundtrip(D, tmp_path):
    """Writes a small raw dataset in the REAL HEMOBP file format (idp.csv, d1.csv, vip.csv; verified against the figshare files)."""
    pids = list(D.patients.index[:6]); raw = tmp_path / "raw"; raw.mkdir()
    S = D.sessions[D.sessions.patient_id.isin(pids) & (D.sessions.is_upcoming == 0)].reset_index(drop=True)
    P = D.patients.loc[pids].reset_index(drop=True)
    pd.DataFrame({"pid": P.patient_id, "gender": "M", "birthday": 1965, "first_dialysis": "2015-01", "DM": 1}).to_csv(raw / "idp.csv", index=False)
    pd.DataFrame({"pid": S.patient_id, "keyindate": S.date.dt.strftime("%Y-%m-%d") + " 08:00:00", "dialysisstart": "09:00", "dialysisend": "13:00",
                  "weightstart": S.pre_weight_kg, "weightend": S.post_weight_kg, "dryweight": S.target_dry_weight_kg, "temperature": 36.4}).to_csv(raw / "d1.csv", index=False)
    O = D.obs.reset_index(drop=True); O = O[O.session_id.isin(S.session_id)].merge(S[["session_id", "date", "uf_rate_l_h"]], on="session_id")
    O = O[(O.t_h * 4) % 2 == 0].reset_index(drop=True)
    pd.DataFrame({"pid": O.patient_id, "datatime": (O.date + pd.Timedelta(hours=9) + pd.to_timedelta(O.t_h, unit="h")).dt.strftime("%Y-%m-%d %H:%M:%S"),
                  "measuretime": (O.t_h * 2).astype(int) + 1, "sbp": O.sbp, "dbp": O.dbp, "dia_temp_value": 36, "conductivity": 14,
                  "uf": O.uf_rate_l_h, "blood_flow": 200, "time": (O.t_h * 60).astype(int)}).to_csv(raw / "vip.csv", index=False)
    rep = convert(raw, tmp_path / "out", max_patients=6)
    R = Data(tmp_path / "out")
    assert rep["sessions"] > 100 and R.sessions.is_upcoming.sum() == 0
    assert R.sessions.hypotension.isin([0, 1]).all() and (R.sessions.duration_h.between(2, 6)).all()
    assert R.sessions.groupby("patient_id").session_index.apply(lambda x: (x.diff().dropna() == 1).all()).all()
