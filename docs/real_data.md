# Real open data: HEMOBP

**Source:** Wu CJ (2018), *HEMOBP*, figshare, doi:10.6084/m9.figshare.6260654.v3; described in Lin CJ et al., *Sci Data* 6:313 (2019), doi:10.1038/s41597-019-0319-8.
The article is CC BY 4.0; the data are described as de-identified with ethics approval. **Confirm the license on the figshare page before redistributing. The converted data are NOT included in this repository** (`data/real/` is git-ignored); only aggregate results (`models/evaluation_real.json`) are.

**Actual files (verified):** `idp.csv` (pid, gender, birthday [year], first_dialysis, DM), `d1.csv` (pid, keyindate, dialysisstart/end as HH:MM, weightstart, weightend, dryweight, temperature; 165,986 rows),
`vip.csv` (pid, datatime, measuretime [reading index], sbp, dbp, dia_temp_value, conductivity, uf [L/h], blood_flow, time [elapsed minutes]; 4,366,298 rows). Missing values are `NA`.

**Mapping:** session key = pid + (datatime - elapsed time).date (robust to midnight crossing; duplicate readings removed by measuretime). Fluid excess F = weightstart - dryweight. t = elapsed `time`/60 (first reading = pre-dialysis, t=0).
Outcome = any SBP < 90 mmHg after the first reading (same demo label; readings are ~20-40 min apart, median 6 per session in the converted set).
Leakage controls: *planned* UF rate = median `uf` in the first 45 min (the delivered mean is used only to fit PAST sessions, because UF is often reduced after an event); planned duration = patient's median scheduled duration from PRIOR sessions only; post-event readings are excluded from twin fitting.

**Run:** unzip the figshare download to a folder (idp.csv, d1.csv, vip.csv), then `python -m src.cli real <folder> 1100` (use `TWIN_BUDGET=90` to run in resumable ~90 s slices; re-run until it finishes). Converting all 1,072 patients (last 60 sessions each, >= 10 usable sessions) yielded 328 patients, 14,638 sessions, 104,881 readings, 9.5% hypotension.

**Missing in this dataset (left blank, never fabricated):** labs, hypertension/CVD, medications, wearables, weather, heart rate. Ravi and the heat-wave demo remain synthetic.

## Result on real data (this run)
Patient-grouped hold-out, 87 patients, 3,799 sessions (>= 6 prior sessions), prevalence 9.2%:

| Model | ROC-AUC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Historical reference (prior rate) | 0.819 | 0.068 | 0.46 | 0.46 | 0.46 |
| Standard: logistic regression | 0.863 | 0.064 | 0.52 | 0.41 | 0.46 |
| Standard: random forest | 0.869 | 0.063 | 0.49 | 0.44 | 0.46 |
| Twin, population parameters (ablation) | 0.578 | 0.084 | 0.23 | 0.09 | 0.13 |
| Personalized twin | 0.715 | 0.079 | 0.30 | 0.27 | 0.28 |

AUC difference twin - LR = -0.148 (95% patient-cluster CI -0.187 to -0.102). Temporal hold-out: twin 0.871 vs LR 0.930 (difference -0.060, CI -0.099 to -0.033).
**On real data the personalized twin did NOT outperform the standard model**; personalization did help relative to the population-parameter twin (0.578 -> 0.715). The claim in the competition brief is therefore not supported by this real-data experiment.

## Why this may be (hypotheses, untested)
Priors/noise model tuned on the synthetic generator; sparse readings (median 6) make per-patient fits weak; the twin cannot use patient history (prior event rate, which carries much of the signal here) while the baseline can; a UF-rate-vs-fluid mechanism may explain only part of real hypotension; no medications/comorbidity/HR in this dataset.
Possible follow-ups: empirical-Bayes priors from the cohort, hybrid model (baseline features + twin outputs), richer observation model.
