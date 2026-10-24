"""Command line: python -m src.cli generate | train | evaluate | all"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
import joblib, numpy as np, pandas as pd
from src.config import DATA_DIR, MODEL_DIR, SEED, N_PATIENTS
from src.data import synthetic_generator
from src.data.preprocessing import Data, build_features, split_patients, FEATURES
from src.models.baseline_model import train_baselines
from src.models.personalized_model import compute_twin_table, TwinCalibrator, estimate_population_prior
from src.models.evaluation import evaluate, MIN_PRIOR


def twin_table_chunked(D, cache, budget_s=1e9, chunk=40, prior=None):
    """Resumable twin fitting: chunks of patients cached on disk (useful for the 300+ patient real cohort)."""
    cache.mkdir(exist_ok=True, parents=True); pids = list(D.sess_by_pid); t0 = time.time(); parts = []
    for i in range(0, len(pids), chunk):
        f = cache / f"chunk_{i:05d}.csv"
        if not f.exists():
            if time.time() - t0 > budget_s: return None
            compute_twin_table(D, pids=pids[i:i + chunk], prior=prior).to_csv(f); print(f"  fitted patients {i}-{i+chunk} of {len(pids)} ({time.time()-t0:.0f}s)", flush=True)
        parts.append(pd.read_csv(f).set_index("session_id"))
    return pd.concat(parts)


def generate():
    pats, sess, sens, obs, _ = synthetic_generator.write_all(DATA_DIR, SEED, N_PATIENTS)
    print(f"generated {len(pats)} patients, {len(sess)} sessions, {len(sens)} sensor-days, {len(obs)} BP observations")


def train(data_dir=DATA_DIR, tag=""):
    MODEL_DIR.mkdir(exist_ok=True)
    D = Data(data_dir); F = build_features(D); split = split_patients(D.patients)
    F["split"] = F.patient_id.map(split)
    tr = F[(F.split == "train") & (F.is_upcoming == 0)]
    base = train_baselines(tr, tr.y.astype(int))
    train_pids = [p for p, v in split.items() if v == "train"]
    prior, n_pr = estimate_population_prior(D, train_pids); D.prior = prior
    print(f"  empirical population prior from {n_pr} train patients: log k ~ N({prior[0][0]:.2f}, {prior[1][0]:.2f}), log S ~ N({prior[0][1]:.2f}, {prior[1][1]:.2f})")
    t0 = time.time(); T = twin_table_chunked(D, data_dir / "_twin_cache", budget_s=float(os.environ.get("TWIN_BUDGET", "1e9")), prior=prior)
    if T is None:
        print("PARTIAL: time budget reached; re-run the same command to resume (chunks are cached)."); return
    out = F.join(T)
    out["p_lr"] = base["lr"].predict_proba(out[FEATURES])[:, 1]; out["p_rf"] = base["rf"].predict_proba(out[FEATURES])[:, 1]
    fit = out[(out.split == "train") & (out.is_upcoming == 0) & (out.n_prior >= MIN_PRIOR)]
    cal = TwinCalibrator().fit(fit.p_twin_raw, fit.y.astype(int)); calp = TwinCalibrator().fit(fit.p_pop_raw, fit.y.astype(int))
    out["p_twin"] = cal(out.p_twin_raw.values); out["p_pop_twin"] = calp(out.p_pop_raw.values)
    out.to_csv(data_dir / f"predictions{tag}.csv", index=False)
    joblib.dump(dict(baselines=base, twin_cal=cal, pop_cal=calp, features=FEATURES, split=split, prior=prior), MODEL_DIR / f"models{tag}.joblib")
    print(f"trained; saved models/models{tag}.joblib and predictions{tag}.csv")


def do_evaluate(data_dir=DATA_DIR, tag=""):
    D = Data(data_dir); F = build_features(D)
    P = pd.read_csv(data_dir / f"predictions{tag}.csv").set_index("session_id")
    from src.models.compat import fix_sklearn_compat; M = fix_sklearn_compat(joblib.load(MODEL_DIR / f"models{tag}.joblib"))
    cols = ["p_lr", "p_rf", "p_twin_raw", "p_pop_raw", "k_hat", "S_hat", "n_fit", "sd_logk", "sd_logS", "nadir_mean", "nadir_sd", "F_est"]
    res = evaluate(F, P[cols], M["baselines"], M["split"])
    # parameter-recovery check against hidden generator truth (final fit per patient)
    res["data_source"] = tag or "synthetic"
    last = P.reset_index().sort_values("session_index").groupby("patient_id").tail(1).set_index("patient_id")
    if len(D.truth):
        tru = D.truth.loc[last.index]
        res["parameter_recovery"] = dict(
            corr_logS=float(np.corrcoef(np.log(last.S_hat), tru.true_log_S)[0, 1]), corr_logk=float(np.corrcoef(np.log(last.k_hat), tru.true_log_k)[0, 1]))
    json.dump(res, open(MODEL_DIR / f"evaluation{tag}.json", "w"), indent=1)
    for sp in ("patient_holdout", "temporal"):
        print(f"== {sp} (n_test_patients={res[sp]['n_test_patients']})")
        for m, v in res[sp]["models"].items():
            print(f"  {m:62s} AUC {v['auc']:.3f} Brier {v['brier']:.3f} P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f}")
        print("  AUC diff twin-LR", res[sp]["auc_diff_twin_minus_lr"])
    print("parameter recovery", res.get("parameter_recovery", "n/a (real data has no ground truth)"))


def real(raw_dir, max_patients=120):
    """python -m src.cli real <folder with Idp/Hemrec_D1/Hemrec_VIP csv> : convert, train, evaluate on real HEMOBP data."""
    from src.data.real_hemobp import convert
    out = DATA_DIR / "real"; convert(raw_dir, out, max_patients=max_patients)
    train(out, "_real"); do_evaluate(out, "_real")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "real": real(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120); sys.exit(0)
    if cmd in ("generate", "all"): generate()
    if cmd in ("train", "all"): train()
    if cmd in ("evaluate", "all"): do_evaluate()
