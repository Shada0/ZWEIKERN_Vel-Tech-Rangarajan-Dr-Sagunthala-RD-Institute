# Model card - NephroTwin AI
**Intended use:** research demonstration of patient-specific digital twins for pre-session intradialytic-hypotension risk. **Not for clinical use.**
**Outcome:** any monitored SBP < 90 mmHg (simplified demonstration definition). **Prediction time:** scheduled connection time.

| Model | Description |
|---|---|
| A. Standard | Logistic regression / random forest on 30 pre-session features (static, session, 3-day sensor window, past-session history) |
| B. Personalized twin | Two-compartment ODE + BP response; patient parameters (k, S, g) by MAP least squares with an empirical-Bayes population prior learned from training patients; Kalman fluid state; Monte-Carlo risk; 1-D Platt calibration (training patients) |
| C. Hybrid | Logistic regression on A's features + twin outputs (logit risk, log k, log S, uncertainty, nadir, fluid state) |
| Ablation | Twin with population-average parameters |

**Data:** (1) synthetic cohort, 120 patients, 26 weeks, seed 42; (2) open real dataset HEMOBP (Lin et al., Sci Data 2019; 328 patients / 14,638 sessions after filtering; not bundled).
**Evaluation:** patient-grouped hold-out and temporal hold-out; sessions with >= 6 prior sessions; patient-cluster bootstrap CIs. All numbers: `docs/results.md`.
**Findings:** synthetic - twin > standard (CI excludes 0). Real - twin < standard; hybrid ~ standard. Personalization beats its ablation in both.
**Known issues:** synthetic twin is structurally close to the generator; real data are sparse (median 6 readings/session) with no medications, heart rate or wearables; thresholds are demo values; explanations are associations, not causal; no clinical validation.
