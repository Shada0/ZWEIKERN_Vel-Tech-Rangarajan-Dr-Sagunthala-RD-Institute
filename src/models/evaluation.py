"""Held-out evaluation: patient-grouped and temporal splits, metrics, calibration, bootstrap CI, ablation."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, precision_recall_fscore_support, confusion_matrix
from sklearn.calibration import calibration_curve
from src.models.personalized_model import TwinCalibrator

MIN_PRIOR = 6
HYB = "Hybrid: standard features + twin outputs"


def twin_feats(df):
    """Twin-derived features available at prediction time (computed from earlier sessions only)."""
    p = np.clip(df.p_twin_raw.values, 0.005, 0.995)
    return pd.DataFrame({"tw_logit": np.log(p / (1 - p)), "tw_logk": np.log(df.k_hat.values), "tw_logS": np.log(df.S_hat.values),
                         "tw_sdS": df.sd_logS.values, "tw_nadir": df.nadir_mean.values, "tw_nadir_sd": df.nadir_sd.values, "tw_F": df.F_est.values}, index=df.index)


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.8, 76)
    f1 = [precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1))])


def metrics(y, p, thr):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p >= thr, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y, p); fr, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)), precision=float(pr), recall=float(rc),
                f1=float(f1), threshold=thr, cm=confusion_matrix(y, p >= thr, labels=[0, 1]).tolist(), n=int(len(y)),
                prevalence=float(np.mean(y)), roc=[fpr.tolist(), tpr.tolist()], calib=[mp.tolist(), fr.tolist()])


def evaluate(F: pd.DataFrame, P: pd.DataFrame, baselines, split: dict, temporal_cut=None):
    """F: features, P: twin table (raw) + baseline predictions already in P columns p_lr, p_rf."""
    d = F.join(P, how="inner")
    d["split"] = d.patient_id.map(split)
    d = d[(d.is_upcoming == 0) & (d.n_prior >= MIN_PRIOR)].copy()
    d["p_hist"] = d.prior_hypo_rate10.fillna(d.y.mean())
    temporal_cut = temporal_cut or d.date.quantile(0.7)
    results = {}
    scenarios = {
        "patient_holdout": ((d.split == "train"), (d.split == "test")),
        "temporal": ((d.split == "train") & (d.date < temporal_cut), (d.split == "train") & (d.date >= temporal_cut)),
    }
    from src.models.baseline_model import make_lr, make_rf
    from src.data.preprocessing import FEATURES
    for name, (trm, tem) in scenarios.items():
        tr, te = d[trm], d[tem]
        preds = {"Historical reference (prior hypotension rate)": te.p_hist.values}
        if name == "patient_holdout":
            preds["Standard model: logistic regression"] = te.p_lr.values
            preds["Standard model: random forest"] = te.p_rf.values
        else:   # retrain on earlier sessions only
            preds["Standard model: logistic regression"] = make_lr().fit(tr[FEATURES], tr.y).predict_proba(te[FEATURES])[:, 1]
            preds["Standard model: random forest"] = make_rf().fit(tr[FEATURES], tr.y).predict_proba(te[FEATURES])[:, 1]
        cal_t = TwinCalibrator().fit(tr.p_twin_raw, tr.y); cal_p = TwinCalibrator().fit(tr.p_pop_raw, tr.y)
        preds["Twin ablation: population parameters (no personalization)"] = cal_p(te.p_pop_raw.values)
        preds["Personalized digital twin"] = cal_t(te.p_twin_raw.values)
        Xtr, Xte = pd.concat([tr[FEATURES], twin_feats(tr)], axis=1), pd.concat([te[FEATURES], twin_feats(te)], axis=1)
        hyb = make_lr().fit(Xtr, tr.y); preds[HYB] = hyb.predict_proba(Xte)[:, 1]
        trp = {"Historical reference (prior hypotension rate)": tr.p_hist.values,
               "Standard model: logistic regression": tr.p_lr.values if name == "patient_holdout" else make_lr().fit(tr[FEATURES], tr.y).predict_proba(tr[FEATURES])[:, 1],
               "Standard model: random forest": tr.p_rf.values if name == "patient_holdout" else make_rf().fit(tr[FEATURES], tr.y).predict_proba(tr[FEATURES])[:, 1],
               "Twin ablation: population parameters (no personalization)": cal_p(tr.p_pop_raw.values),
               "Personalized digital twin": cal_t(tr.p_twin_raw.values), HYB: hyb.predict_proba(Xtr)[:, 1]}
        res = {m: metrics(te.y.values, p, best_threshold(tr.y.values, trp[m])) for m, p in preds.items()}
        # patient-cluster bootstrap of AUC difference (twin - LR)
        rng = np.random.default_rng(0); pids = te.patient_id.unique(); diffs = []
        te = te.assign(_tw=preds["Personalized digital twin"], _lr=preds["Standard model: logistic regression"], _hy=preds[HYB])
        grp = {k: v for k, v in te.groupby("patient_id")}
        for _ in range(200):
            s = pd.concat([grp[k] for k in rng.choice(pids, len(pids))])
            if s.y.nunique() < 2: continue
            diffs.append(roc_auc_score(s.y, s._tw) - roc_auc_score(s.y, s._lr))
        pat = te.groupby("patient_id").apply(lambda x: pd.Series(dict(
            n=len(x), observed_rate=x.y.mean(), twin_brier=brier_score_loss(x.y, x._tw), standard_brier=brier_score_loss(x.y, x._lr))), include_groups=False)
        dh = []
        for _ in range(200):
            s_ = pd.concat([grp[k] for k in rng.choice(pids, len(pids))]); 
            if s_.y.nunique() < 2: continue
            dh.append(roc_auc_score(s_.y, s_._hy) - roc_auc_score(s_.y, s_._lr))
        results[name] = dict(models=res, auc_diff_hybrid_minus_lr=dict(mean=float(np.mean(dh)), lo=float(np.percentile(dh, 2.5)), hi=float(np.percentile(dh, 97.5))), auc_diff_twin_minus_lr=dict(mean=float(np.mean(diffs)), lo=float(np.percentile(diffs, 2.5)),
                             hi=float(np.percentile(diffs, 97.5))), n_test_patients=int(len(pids)), per_patient=pat.reset_index().to_dict("records"))
    return results
