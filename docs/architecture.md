# Architecture
See `architecture_diagram.pdf`. Flow: synthetic EHR (A) -> feature extraction; synthetic sensors/weather (B) -> preprocessing; both -> state estimation (MAP least squares for k, S; shrinkage for g; scalar Kalman filter for fluid excess)
-> twin engine (two-compartment ODE + BP response, Monte-Carlo) and population baseline -> pre-session risk -> what-if simulation -> Streamlit dashboard.

## Equations
Let F = fluid excess (kg ≈ L), Vt(0) = (1−0.15)·F, k = refill rate (1/h), u(t) = UF rate (L/h), BV0 = 5 L.
- d(ΔVb)/dt = k·Vt − u(t);  d(Vt)/dt = −k·Vt  (solved with SciPy `solve_ivp`; closed form ΔVb = Vt0(1−e^{−kt}) − u·t for constant u, tested to agree)
- Relative blood-volume drop: drop(t) = max(0, −100·ΔVb/BV0) %
- SBP(t) = SBP_pre − S·(1 + 0.15·stress)·drop(t) + AR(1)(ρ=0.7, σ=3) + N(0, 1.5²);  HR(t) = HR_pre + c·drop(t)
- Accumulation: F ← F + g·(1 + 0.05·max(0, T−32)) per day;  stress = clip((T−32)/10, 0, 1)
- Estimation: (log k, log S) = argmin Σ((y−ŷ)/4.5)² + prior terms; covariance (JᵀJ)⁻¹ (Laplace). Population prior on (log k, log S): empirical Bayes - each TRAIN patient is fitted with a broad prior, the between-patient mean/sd (noise-deconvolved, floored) becomes the prior; test patients never contribute.
- Risk: draw θ ~ N(θ̂, Σ), session variability σ=0.22 on log S, F ~ N(F̂, σ_F²); P = fraction of simulated trajectories with SBP < 90 at any monitored point; then Platt calibration (train patients only).
- Kalman (scalar): predict F += g·mult(T), P += q²; update with home weight (r=0.30 kg) and clinic weight (r=0.12 kg).

- Hybrid model: logistic regression on the 30 standard features plus twin outputs (logit risk, log k, log S, sd log S, simulated nadir, nadir sd, fluid state), all computed from earlier sessions only.
- Real data: no wearables/weather; fluid excess = weight at start - dry weight; planned UF = initial UF rate; Kalman step reduces to the clinic-weight update.
