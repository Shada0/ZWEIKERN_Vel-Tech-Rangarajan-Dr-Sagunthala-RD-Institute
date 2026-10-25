# NephroTwin AI: 20-minute demo video script (ZWEIKERN)

**Target length: 20:45** (the brief says *minimum 20 minutes*, so never finish under 20:00).
Speaking pace: about 130 words per minute. Spoken lines are in quotes; `[SCREEN]` is what is visible; `[DO]` is what you click.

## 0. Either the PowerPoint or the YouTube video? Do both with ONE recording

The brief accepts either a PowerPoint-style video or an unlisted YouTube video. Record **one** 20+ minute screen capture that mixes the pitch deck (slides) with the live app, upload it as an **Unlisted** YouTube video, and also keep the deck on Google Drive. That satisfies both readings of the rule.

- Record with OBS Studio or Zoom (screen + microphone, 1080p). Put the deck in slideshow mode, and alt-tab to the app when the script says `[APP]`.
- Slide numbers below refer to `presentation/NephroTwin_AI_Pitch_Deck.pptx` (14 slides).
- After uploading: Visibility = **Unlisted**, paste the link into `README.md` (`[INSERT YOUTUBE UNLISTED VIDEO LINK HERE]`) and into slides 1 and 14. Open it in an incognito window to confirm it plays without login.
- Use the chapter list at the bottom as the first lines of the YouTube description.

## 1. Fact-check before you record (cite on screen)

| Claim you will say | Status | Where to cite |
|---|---|---|
| India: about 2.2 lakh new ESRD patients a year, about 3.4 crore added dialysis sessions of annual demand | Sourced | Ministry of Health and Family Welfare, PM National Dialysis Programme guidelines (mohfw.gov.in) |
| Worldwide: more than 3.8 million people with end-stage kidney disease on dialysis | Team figure used in the deck | Verify against the ISN Global Kidney Health Atlas or USRDS and show the source |
| IDH affects roughly 1 in 5 to 1 in 4 hemodialysis sessions in published studies (range depends on definition) | Widely reported, verify | Cite one review, e.g. a KDOQI or nephrology review, on screen |
| IDH is associated with more cardiovascular events, access problems, early session termination and higher mortality | Widely reported association, not our result | Same review |
| In HEMOBP, 9.5% of 14,638 sessions had a hypotensive reading (any SBP < 90) | **Our computed result** | `docs/real_data.md` |
| Synthetic test cohort: 36 patients, 2,379 sessions, 23.5% prevalence; twin AUC 0.977 vs logistic regression 0.949 | **Our computed result** | `docs/results.md` |
| Real data: hybrid 0.871 vs standard 0.863; twin alone 0.736 (difference -0.127, CI -0.159 to -0.097) | **Our computed result** | `models/evaluation_real.json` |

If you cannot verify a "widely reported" line, soften it to "published studies report" and show the citation, or drop the number.

## 2. Timed script

### 0:00 to 2:00 | Cold open: the impact (Slide 2 as backdrop, then slide 1)
`[SCREEN]` Black screen with one line of text: "Every dialysis session removes about two to three litres of fluid from a person's body in about four hours."
"Imagine that. Three times a week, for years. In India alone, about two point two lakh new patients reach kidney failure every year, and that creates roughly three point four crore additional dialysis sessions of demand, according to the Ministry of Health's national dialysis programme. Worldwide, more than three point eight million people depend on dialysis to stay alive."
"And in a large share of those sessions, something goes wrong. Published studies report that blood pressure drops dangerously in roughly one in five to one in four hemodialysis sessions. This is called intradialytic hypotension. The patient gets cramps, nausea, dizziness, sometimes faints. The nurse stops or slows the machine. The session is cut short, the patient is under-dialysed, and over months and years this pattern is linked to heart injury, blocked vascular access and higher mortality."
"Here is the part that should bother us: today, this is mostly handled **after** it starts. The patient is already crashing. What if the care team knew **before** the patient was connected, who is at risk today, why, and what to adjust?"
`[SCREEN]` Slide 1 (cover).
"I am Shada Ali Kuzhikattil, and with my teammate Rushdha V, we are team ZWEIKERN from Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai. This is NephroTwin AI, our entry to the Happiest Health Digital Twin Challenge 2026."

### 2:00 to 2:45 | What we built, in one minute (Slide 1)
"In one sentence: NephroTwin AI is a personalised dialysis digital twin. It keeps a persistent virtual model of each patient's fluid and blood-pressure physiology, updates it from history and daily signals, and before each session it estimates the probability of hypotension, explains the reasons, shows the uncertainty, and lets a clinician explore what-if scenarios safely."
"I will cover the problem, why a twin and not just another classifier, the data, the models, a live demo with a patient called Ravi, our results on synthetic **and real** data, including where we do not win, the safety design, and the business plan. A clear disclaimer first: this is a research prototype. It is not a medical device, and it does not tell clinicians what to do."

### 2:45 to 4:30 | The problem in depth (Slide 2)
`[SCREEN]` Slide 2.
"Why does blood pressure drop? Dialysis removes fluid from the blood. The blood refills from the tissues around it, but only at a certain speed. If we remove fluid faster than the body can refill, blood volume falls and pressure drops. Removal outruns refill."
"The trouble is that the refill speed is different for every patient. A diabetic patient with heart disease refills slowly and has weak vascular response. Another patient tolerates the same fluid removal easily. And it is different for the same patient from one session to the next: a long weekend gap means more fluid to remove, a heat wave makes patients gain more fluid, and the pressure response changes."
"Routine risk scores treat patients as averages. They are population-level. They do not know this patient's refill rate, this patient's blood-pressure sensitivity, or how much extra fluid this patient gained this week. Our target: the probability of hypotension **before** the session starts, the reasons behind it, and what-if scenarios for clinicians, never orders."
"In the open real-world dataset we validated on, 9.5 percent of 14,638 sessions had a hypotensive reading. In our synthetic cohort the rate is above 20 percent. Either way, this is frequent enough that every unit sees it every week."

### 4:30 to 5:30 | Relevance, the gap and novelty (Slide 12 briefly, then Slide 3)
`[SCREEN]` Slide 12 (competitive landscape).
"Three approaches exist. Static rules, like fixed thresholds: transparent but generic. Standard machine learning: learned from population features, it gives a probability but has no physiology, cannot simulate a scenario, and gives no per-patient uncertainty. NephroTwin keeps a per-patient state with parameters for refill rate, blood-pressure sensitivity and fluid-gain rate, updated every session."
"Our novelty, in four points. One: it is a true twin. The patient state persists and updates; it is not a renamed classifier. Two: it is mechanistic and explainable. The outputs are physiological quantities a nephrologist already understands. Three: it simulates. The same engine that gives the risk also answers what-if questions. Four: it carries uncertainty, from the parameters and from the fluid state, and it is evaluated honestly against standard models on real data."

### 5:30 to 7:15 | The digital twin concept (Slide 3)
`[SCREEN]` Slide 3.
"The twin has three steps. **Initialise**: from past sessions we estimate three parameters for each patient: refill rate k, blood-pressure sensitivity S, and fluid-gain rate g, using maximum a posteriori nonlinear least squares. The prior comes from training patients only, an empirical-Bayes prior, so patients with few readings are pulled toward the population instead of producing wild estimates."
"**Update**: a Kalman filter fuses the home weight and the clinic weight into a running estimate of fluid excess over dry weight, with its own uncertainty. Every new session re-fits the parameters with a warm start."
"**Simulate**: a two-compartment fluid model, solved with SciPy, drives a blood-pressure response with correlated noise. We run a Monte Carlo over the uncertainty in parameters and state. The fraction of simulated sessions where systolic pressure falls below 90 gives the risk, and a Platt calibration makes the probability honest."
"The outcome label is a demo definition: any monitored systolic pressure below 90 millimetres of mercury. The prediction timestamp is the scheduled connection time. Nothing after that moment is used."

### 7:15 to 8:30 | Data streams and leakage control (Slide 4)
`[SCREEN]` Slide 4.
"Two data streams, as the challenge asks. **Stream A, history**: age, sex, diabetes, hypertension, cardiovascular disease, baseline blood pressure, dry weight, labs, medications, six months of sessions with fluid removal, duration, weights, timed blood pressure and prior events. **Stream B, dynamic signals**: daily home weight and cuff readings, wearable heart rate, heart-rate variability, activity and sleep, with realistic gaps, plus temperature and heat-wave flag."
"An important honesty point: blood pressure here is a simulated cuff reading, not something a wearable measures. We never claim a wearable measures blood pressure or fluid removal."
"Leakage is where many health-AI projects fail, so we tested it. Only data stamped before the scheduled connection time reaches any model. The twin for session n uses sessions before n, never later ones. Automated tests assert that outcome columns never enter the features."
"We use two datasets. A synthetic cohort of 120 patients over 26 weeks, about 8,400 sessions, generated with a deliberate mismatch from our own model, so the twin cannot simply read the answer. And a real, de-identified open dataset, HEMOBP: 328 patients, 14,638 sessions, about 105 thousand blood-pressure readings. Real data has no wearables, weather or labs, so those streams stay synthetic and are labelled."

### 8:30 to 9:30 | Architecture and models (Slide 5; show docs/architecture_diagram.pdf)
`[SCREEN]` Slide 5, then docs/architecture_diagram.pdf.
"Reading left to right: both streams are preprocessed with missingness flags and three-day windows, then feed the physiology engine, with MAP estimation and the Kalman filter, the fluid model, the blood-pressure response and the Monte Carlo risk. At the end: risk with uncertainty, Green, Amber and Red demonstration thresholds, a what-if simulator and the Streamlit dashboard."
"We compare four model families: a population baseline, logistic regression and random forest; the personalised twin; a hybrid that gives the standard features plus the twin's outputs to a logistic regression; and an ablation, the twin with population-average parameters, which tells us whether personalisation itself adds value. The full equations are in `docs/architecture.md`."

### 9:30 to 10:30 | Dashboard tour (Slide 7, then `[APP]`)
`[SCREEN]` Slide 7 for ten seconds, then switch to the app.
"The dashboard has 13 views, grouped into unit, patient, story and trust."
`[APP][DO]` **Project overview**: point to the evidence cards and the honest-finding box.
"On the home page the evidence is computed, not typed: synthetic twin AUC 0.977, and on real data the twin alone has a red down arrow, minus 0.127, because we do not hide a loss."
`[DO]` **Unit overview**: "A unit-level view: risk distribution, upcoming-session priorities with uncertainty, and the population hypotension rate by month."
`[DO]` **Patient list**, then **Digital twin profile**: "For one patient: refill rate, blood-pressure sensitivity, fluid gain and their uncertainty next to the population median, the Kalman fluid estimate, 30-session history, the last ten days of sensor data, where blanks are missing values shown as dashes, and explanations: logistic-regression contributions and twin sensitivities. These are associations, not causes."

### 10:30 to 14:00 | Live demo: meet Ravi (Slide 8 as a 10-second preview, then `[APP]`)
`[APP]` Ravi demonstration. Say once: "Ravi is a **simulated** patient. Everything in this demo is synthetic."
`[DO]` **Step 1, Patient overview.** "Ravi, a man with kidney failure on maintenance hemodialysis, twice a week, with diabetes and hypertension. His record: sessions on record, hypotension events, event rate and the current interdialytic gap. This is Stream A."
`[DO]` **Step 2, Dynamic data update.** "Between sessions his home weight, home blood pressure, heart rate, heart-rate variability, steps and sleep arrive. Dashes are missing values. The Kalman filter turns the weights into a fluid-gain estimate, with its uncertainty. This is Stream B."
`[DO]` **Step 3, Heat-wave scenario.** Turn the toggle on at the top right. "Now a synthetic heat wave, 42 degrees. By day three Ravi gains 2.8 kilograms. I want to be clear that 2.8 kilograms is a demonstration input, not a clinical threshold, and weather alone does not prove fluid overload. The chart shows his fluid excess over dry weight, baseline against heat wave."
`[DO]` **Step 4, Digital twin update.** "Here is what changed, row by row: temperature, heat-stress input, fluid excess at the session, weight gain and effective blood-pressure sensitivity. Below it, the twin's learned refill rate and sensitivity next to the population median. The table compares the standard model, the twin with population parameters, and the personalised twin."
`[DO]` **Step 5, Risk flag.** "The twin estimates Ravi's risk at **4 percent, Green** at baseline, and **55 percent, Amber** in the heat wave, a change of 51 points. The standard model sees 12 percent baseline. This number is computed live by the simulation; it is not hard-coded, and you can see it recompute when I toggle the scenario. Under it, the main contributing factors: fluid excess, fluid-removal volume, duration, starting pressure and heat."
`[DO]` **Step 6, What-if planner** (see next section for the narration). **Step 7, Clinical review.** "The last step is the most important: review with a nephrologist before any treatment decision. The tool produces simulated risk estimates and hypothetical scenarios only. It never orders treatment, changes prescriptions or medications."

### 14:00 to 15:00 | What-if planner (Ravi step 6 or Pre-session card)
`[APP][DO]` Move sliders: session duration, hypothetical fluid-removal volume, UF-rate profile, starting pressure, fluid excess, extra gap day, heat-wave checkbox.
"The planner lets a clinician ask: if the session were 30 minutes longer, if the fluid-removal target were half a litre lower, if the gap were one day longer, what does the simulation say about risk? The scenario table compares A to D and uses deliberately non-directive wording: 'Simulation indicates a potentially lower-risk scenario. Clinical review required.' Not 'do this'. Every row is labelled a simulation."
"Because the model is physiological, the answer changes for the right reasons: a longer session reduces the removal rate, so the simulated blood-volume drop is smaller. A standard classifier cannot do this, because it has no concept of removal rate."

### 15:00 to 15:45 | Historical replay
`[APP][DO]` Historical replay, Ravi, pick an event, leave the outcome hidden, then toggle "Reveal".
"We replay a past hypotension event. First the prediction, made with information available **before** that moment only, the prior sessions, home sensors and pre-dialysis vitals. The twin said 51 percent and flagged it Amber **before** the session. Then I reveal the outcome. Be careful what this shows: it shows the model flagged the event beforehand. It does not show that any alternative plan would have prevented it. We say that on the screen."

### 15:45 to 17:30 | Results: synthetic and real (Slide 6, `[APP]` Model evaluation and Real-data validation)
`[SCREEN]` Slide 6.
"On the synthetic cohort, tested on 36 held-out patients, 2,379 sessions: the personalised twin reaches an AUC of 0.977, logistic regression 0.949, random forest 0.913, and the twin without personalisation only 0.879. The difference against logistic regression is plus 0.028, with a patient-cluster bootstrap interval of plus 0.015 to plus 0.043. So personalisation is what adds value. The Brier score, which measures probability quality, improves from 0.074 to 0.050. But we tell you the caveat: the twin's structure resembles the data generator, so this is optimistic."
`[APP][DO]` Real-data validation page.
"This is the test that matters. Real HEMOBP data, 87 held-out patients, 3,799 sessions. The banner says it plainly: real de-identified open dataset, retrospective research validation only."
"Here the twin **alone** does worse than the standard model: AUC 0.736 against 0.863, a difference of minus 0.127, and the interval excludes zero. We are not hiding it, and the arrow is red. Why? Real sessions have a median of only six blood-pressure readings, too few to pin down a patient's physiology, and the standard model can use each patient's past event rate, which carries a lot of signal."
"But the hybrid, standard features plus the twin's physiological outputs, scores 0.871 against 0.863, about on par, with the interval touching zero. We do not claim a win. The honest claim is this: the hybrid matches the best standard models **and** adds explainable physiology and what-if simulation that a standard model cannot give, and personalisation still beats the ablation on real data, 0.736 against 0.607. We also list untested hypotheses for the gap and concrete next steps: hierarchical Bayesian pooling, a richer observation model and more frequent blood-pressure data."
"We believe reporting a loss clearly is a strength, because a clinician will only trust a tool whose authors show where it fails."

### 17:30 to 18:30 | Safety, ethics and limitations (Slide 13)
`[SCREEN]` Slide 13, then the app's Limitations & safety page.
"Safety is built in. Synthetic patients are labelled everywhere. Thresholds are configurable demonstration values, not clinical ones. Planner outputs are marked 'Clinical review required' and 'Not a treatment recommendation'. There are no orders for duration, fluid removal, medication or sessions."
"Limitations we state openly: synthetic data and simplified physiology; circularity between our twin and our generator; one hospital's retrospective real data; missing sensors; and, most important, retrospective results cannot show clinical benefit. Before any real use you would need clinical validation, regulatory review, data-protection review and expert oversight."
"On engineering quality: the project is open source under the MIT licence, with automated tests, a continuous-integration workflow, a Docker image, a model card and a data dictionary."

### 18:30 to 20:00 | Business, market and go-to-market (Slides 9, 10, 11)
`[SCREEN]` Slide 9.
"Who pays and why? Dialysis chains and hospitals. The value is fewer emergency interventions, shorter nursing time per event, fewer early session terminations, and better retention. Pricing: a per-patient-per-month licence, about 25 dollars in the US and Europe, about 10 dollars in India, plus a 10 thousand dollar one-time EHR integration fee per site."
"Here is the ROI framework, with planning assumptions that we will replace with measurements. A 100-patient centre runs about 15,600 sessions a year. If 10 percent need an acute intervention and the tool helps avoid 15 percent of them, that is 234 events avoided. At about 100 dollars of consumables and chair time per event, plus nursing time, plus three avoided emergency admissions valued at 4,000 dollars each, the benefit is about 39,800 dollars against a 30,000 dollar licence: a net of about 9,800 dollars, benefit-to-cost of about 1.3. The cost per acute event prevented is about 128 dollars. Break-even needs roughly 10 percent fewer interventions. These are assumptions, not results, and we claim no savings until the Phase 1 study."
`[SCREEN]` Slide 10.
"The market: globally, a dialysis market of ninety-five billion dollars plus, our total addressable market. Software and AI decision support in North American and European clinics is the serviceable market, about 3.2 billion. Our initial obtainable market, the top 50 regional outpatient chains, is about 120 million. These are team estimates from published market research, to be verified before external use. India is our home-market wedge: about 2.2 lakh new patients a year, with entry through private chains and district dialysis centres."
`[SCREEN]` Slide 11.
"Go-to-market in three phases. **Phase 1, zero to twelve months**: an IRB-approved prospective observational study with partner clinics, collecting continuous blood pressure, medications and fluid-removal profiles, plus a clinician usability study. **Phase 2, twelve to thirty months**: regulation as Software as a Medical Device, a clinical decision support tool, through FDA 510(k) and CE mark, with India CDSCO pilots in parallel; the exact route to be confirmed with regulatory counsel. **Phase 3, thirty to forty-eight months**: integration with dialysis machine makers such as Fresenius and Baxter, and EHRs such as Epic and Cerner, then rollout across the top 50 regional chains. These are target partners; there are no agreements yet."

### 20:00 to 20:45 | Impact vision, roadmap and close (Slide 14)
`[SCREEN]` Slide 14.
"If this works prospectively, the impact is simple to say. Fewer patients crash on the machine. Nurses can plan their attention before the session instead of reacting during it. Sessions are completed, and patients are adequately dialysed. In a country adding more than two lakh dialysis patients a year, with limited nephrologists and dialysis machines, a tool that tells a nurse in advance who needs the most attention is not a luxury. It is leverage."
"Our roadmap: **Now**, prospective ethics-approved validation with richer data. **Next**, hierarchical Bayesian pooling, a better observation model, a usability study, and real wearable and weather feeds. **Later**, calibration monitoring, the SaMD submission and partnerships."
"What we ask: clinic partners for Phase 1, nephrology advisors, and seed funding for the study. The code is open source on GitHub, github.com/Shada0/NephroTwin-AI, the demo and pitch deck links are in the README, and the project is MIT licensed."
"Predict the risk. Understand the patient. Plan dialysis more safely. We are team ZWEIKERN, Shada Ali Kuzhikattil and Rushdha V, from Vel Tech, Chennai. Thank you."

## 3. If you run short or long
- Under 20:00: expand the Ravi demo (open the Pre-session card, run the heat-wave toggle there), show Model evaluation calibration and confusion matrices, and read two limitations aloud from `docs/clinical_limitations.md`.
- Over 21:30: shorten the architecture slide and the market slide, not the real-data or safety sections.

## 4. YouTube upload
- **Title:** NephroTwin AI: A Personalized Dialysis Digital Twin to Predict Intradialytic Hypotension | Team ZWEIKERN | Happiest Health Digital Twin Challenge 2026
- **Visibility:** Unlisted.
- **Description:** one-paragraph summary, GitHub URL, the disclaimer "research prototype on synthetic and open real data, not a medical device", then the chapters:

```
0:00 Impact: dialysis in India and worldwide
2:00 What we built
2:45 The clinical problem
4:30 Relevance, gap and novelty
5:30 The digital twin concept
7:15 Data streams and leakage control
8:30 Architecture and models
9:30 Dashboard tour
10:30 Live demo: Ravi
14:00 What-if planner
15:00 Historical replay
15:45 Results: synthetic and real data
17:30 Safety and limitations
18:30 Business, market and go-to-market
20:00 Impact, roadmap and close
```

Recording is not generated by this repo. Keep the total between 20:00 and 21:00.
