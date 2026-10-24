import numpy as np
from src.data.synthetic_generator import generate
from src.data.validation import validate

def test_reproducible():
    a = generate(seed=1, n_patients=4)[1]; b = generate(seed=1, n_patients=4)[1]
    assert a.drop(columns=[]).equals(b)

def test_dataset_checks(D):
    assert all(validate(D).values())

def test_no_leakage_columns_in_features(D):
    from src.data.preprocessing import FEATURES
    banned = {"min_sbp", "hypotension", "nadir_time_h", "saline_given", "uf_pause", "early_termination", "post_weight_kg"}
    assert not banned & set(FEATURES)

def test_history_features_use_only_past(D):
    from src.data.preprocessing import build_features
    F = build_features(D); g = F[F.patient_id == "SYN-RAVI"].sort_values("session_index")
    first = g.iloc[0]; assert first.n_prior == 0 and np.isnan(first.prior_hypo_rate10)
