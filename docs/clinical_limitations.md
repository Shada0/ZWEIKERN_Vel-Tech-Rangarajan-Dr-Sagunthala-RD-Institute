# Clinical limitations and safety statement

NephroTwin AI is a **simulated research proof-of-concept**. It is **not** a medical device, **not** a clinically validated digital twin,
**not** a replacement for a nephrologist, **not** a treatment-recommendation system, and **not** a validated hypotension prediction model.

* **Synthetic data.** Every patient, session, sensor value and weather observation is generated. Relationships were authored by the developers; the model can only
  rediscover structure that was put in. Performance on synthetic data says nothing about real-world accuracy.
* **Physiological simplification.** Two fluid compartments, a single blood-volume -> BP gain, and fixed noise models. Real hypotension involves autonomic tone, cardiac function,
  vascular resistance, temperature, medications, food intake, dialysate composition and more. Parameter values are illustrative, not validated individual physiology.
* **Real-data evidence.** On the open HEMOBP dataset the twin alone was worse than a standard model (see `docs/results.md`); the hybrid was about equal. Synthetic-data advantages must not be read as real-world benefit.
* **Circularity.** The twin's structure is close to the generating process (with deliberate misspecification: quadratic BP response, drift, dry-weight error). This favours the twin.
* **Outcome definition.** "Hypotension" = simulated SBP < 90 mmHg at any 15-minute monitoring point - a simplified demonstration label, not a consensus clinical definition.
* **Thresholds.** Green/Amber/Red cut-offs are configurable demonstration values, not validated.
* **Prediction uncertainty.** Probabilities reflect parameter uncertainty and modelled noise only; they omit model-structure uncertainty. Missing sensors reduce information and are shown, not hidden.
* **Heat-wave scenario.** A documented synthetic rule (+5 % daily fluid gain per deg C above 32 deg C, +15 % BP sensitivity at full stress). The Ravi "2.8 kg by day 3" is a demonstration input. Weather does not by itself prove dehydration or fluid overload and is not claimed to cause any specific complication.
* **Wearables.** Consumer wearables do not measure BP or dialysis fluid removal here; BP is a simulated clinical/home cuff reading. HR, HRV, activity and sleep are supporting signals only.
* **Potential bias.** A real deployment could be biased by who is monitored, device availability, missing data patterns and unit practice; none of this is modelled.
* **Planner.** Scenarios are simulations. Wording intentionally avoids orders; nothing prescribes duration, UF targets, medications or extra sessions.
* **Retrospective simulation cannot establish treatment benefit.** Only prospective, ethically approved clinical studies could.
* **Expert review.** Any real use requires clinical validation, regulatory review, data-protection review and expert oversight.
