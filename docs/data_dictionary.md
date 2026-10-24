# Data dictionary (all synthetic; seed 42)
**synthetic_patients.csv** - patient_id (SYN-xxx; SYN-RAVI = demo), display_name, age, sex, diabetes, hypertension, cardiovascular_disease, ckd_stage, dialysis_per_week, schedule_days (0=Mon), shift_start, baseline_sbp, dry_weight_kg, usual_duration_h, bp_medication_categories, albumin_g_dl, hemoglobin_g_dl, potassium_mmol_l, baseline_hr.
**synthetic_sessions.csv** - session_id, patient_id, session_index, date, prediction_timestamp, gap_days, pre/post_weight_kg, target_dry_weight_kg, idwg_kg, planned_uf_l, duration_h, uf_rate_l_h, uf_rate_ml_kg_h, pre_sbp/dbp/hr, temp_c, heatwave_flag, is_upcoming, and OUTCOMES (blank for upcoming): min_sbp, nadir_time_h, hypotension (SBP<90), saline_given, uf_pause, early_termination.
**synthetic_intradialytic_obs.csv** - timed (15 min) sbp, dbp, hr per historical session; post_event=1 after the first hypotensive reading.
**synthetic_sensor_data.csv** - daily temp_c, heatwave_flag, home_weight_kg, home_sbp, hr_rest, hrv_rmssd_ms, activity_steps, sleep_h (missing = NaN).
**synthetic_ground_truth_params.csv** - hidden generator parameters (validation only, never model input).
**predictions.csv** - per-session features + model outputs (p_lr, p_rf, p_twin_raw, p_twin, p_pop_twin, twin parameters, split).

**predictions.csv** also contains twin outputs: p_twin_raw (uncalibrated simulation probability), p_twin (calibrated), p_pop_raw/p_pop_twin (ablation), k_hat, S_hat, sd_logk, sd_logS, g_hat, c_hr, F_est, F_sd, nadir_mean, nadir_sd, n_fit.
**Real data (not bundled):** converted into the same schema under `data/real/` (git-ignored); see docs/real_data.md.
