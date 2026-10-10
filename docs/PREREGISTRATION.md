# Pre-registration: final-run analysis (written 2026-10-02, before any sealed-test result exists)

Written before the one-time sealed test of `final_offset` (due 2026-10-02/03) and of `final_centred` is read.
It fixes which results count, how they are judged, and what a negative result means. The criteria are the
constants in `scripts/posthoc_report.py`; the analysis scripts are `scripts/posthoc_*.py`.

## Question
Does a time-resolved SUIT NB03 (Mg II k) sequence contain predictable information about future chromospheric
structure beyond solar rotation and static instrumental/background effects, at horizons of 29 min, 58 min,
1.9 h and 3.9 h?

## Fixed inputs
- Data sets: `final_offset` (pointing offset@1280,600, 10-27 Sep 2026) and `final_centred` (centred@1030,1010,
  18-23 Sep 2026), splits and 4-h embargo as in `configs/datasets/`.
- Models: UNetSmall and ConvLSTM, background-aware inputs, seeds 0, 1, 2 (`configs/phase3.toml`); the primary
  forecast of each architecture is its **seed ensemble** (`unet_bg-ens`, `convlstm_bg-ens`).
- Reference: the **strongest** of the five physical baselines (lowest median MAE per region and horizon, chosen
  on the evaluated windows themselves): B1, B1-avg, B1-avg-LDadd, B1-avg-bgS, B1-avg-clim.
- Region: the disk inside r < 0.9 (the limb ring is excluded). Metric: MAE (primary); RMSE and gradient
  correlation reported.

## Primary criteria (sealed test)
- **P1 skill**: at a horizon, an architecture has skill if the 95 % bootstrap interval (run-hour blocks) of
  its median skill over the strongest baseline has a lower bound > 0.
- **P2 robust**: P1 also holds with whole observing days as bootstrap blocks.
- **P3 solar origin**: P1 holds in BOTH pointing modes for the same architecture and horizon. Skill in the
  offset mode alone is attributed to instrumental effects, not to the Sun.

## Validity controls (validation split; a model failing them is reported as artefact-driven)
- **C1 frozen Sun**: on a non-evolving Sun with real noise, the upper 95 % bound of skill over its own B1 is
  <= 0.01 for every run.
- **C2 corotation**: the share of a model's correction that is one fixed map on the detector grid is <= 0.20
  at every horizon.
- Shuffled context: reported as a diagnostic (how much of the correction does not depend on solar content).

## Secondary analyses (reported, not used for the primary claim)
- **S1 optical-flow baseline** (`posthoc_flow_baseline.py`, validation): if it beats the strongest baseline,
  the models are also compared with it.
- **S2 cross-mode transfer** (`posthoc_cross_mode.py`, validation): models of one pointing mode scored on the
  other; a large drop indicates learned detector structure.
- Per-ring and plage-region skill; seed spread.

## What counts as a negative result
If P1 fails at every horizon in both data sets, the conclusion is: beyond solar rotation and the static
instrument background, NB03 structure is not predictable at 29 min-3.9 h with these models and this data
(an upper bound on predictability, reported with its intervals). This is a result, not a failure.

## Changes
Any later change to these criteria is recorded here with its date and reason; results under the original
criteria are always reported as well.

- 2026-10-03 01:16-01:18: the `final_offset` sealed test was started by the runner and stopped at batch 6/399 at
  the user's request (no metric computed; `eval_test/INTERRUPTED_READ.txt`); it was run once, in full, with the
  same model set on 2026-10-05 together with `final_centred`.

# Addendum A: prospective sealed test on later dates (written 2026-10-05, after the first sealed tests, before any SUIT data after 2026-09-27 was downloaded)

**Status: WITHDRAWN on 2026-10-05 before any data after 2026-09-27 was downloaded** (the user wants a sealed
forecast made without any later data; replaced by Addendum B). Kept for the record.

The first sealed tests (2026-10-05) gave skill over the strongest baseline in both pointing modes (P1, P3) but
failed control C2 in both. This addendum fixes, before the data exist on our disk, how the same frozen models are
tested on SUIT data taken after the final data sets ended. Nothing above is changed.

## Question
Do the frozen final models keep their skill over the strongest baseline on NB03 data from later dates?

## Fixed inputs
- Models: exactly the 12 final models (`final_offset` and `final_centred`, UNetSmall and ConvLSTM, seeds 0-2), as
  recorded in each data set's `eval_test/test_reads.json`. No retraining, fine-tuning, threshold or baseline change.
- Data: SUIT Level-1 NB03 full-disk frames from 2026-09-27 19:36 UT (the `final_offset` test end plus the 4-h
  embargo) up to the last frame of the first download made for this test. The end time is fixed when the data set
  configuration is written and is not extended after any result is seen.
- Frames are grouped by pointing cluster with the same registration rules; each cluster is one prospective data
  set (`prosp_<mode>`), all of it a single sealed test split (no train or validation part).
- Preprocessing: frame-local steps (registration, derotation, masks) run on the new frames. Products fitted on
  data (static background, calibration pattern, noise maps) are taken from the model's own training data set
  where the pipeline allows; any product that has to be recomputed on the new frames is listed in the report.
- Same baselines, region, metric, horizons and bootstrap as above; read once through the seal, read logged.

## Criteria
- **A1 (primary)**: P1 for each architecture ensemble and horizon, scored on new frames of the pointing mode it
  was trained on.
- **A2**: P2 (day blocks) on the same, where at least two observing days exist.
- **A3 (secondary)**: frames of the other or of a new pointing mode are scored by both modes' models, reported
  as a cross-pointing transfer test (a drop again indicates learned detector structure).
- Controls C1 and C2 are not re-run (they are properties of the frozen models); C2's failure stays reported.

## Negative result
If A1 fails at every horizon, the conclusion is: the skill seen in the first sealed tests does not carry over to
later dates with these frozen models; reported with intervals, as a result.

# Addendum B: sealed blind forecasts beyond the data (written 2026-10-05, before the forecasts are made)

The frozen final models are given ONLY data we already have and forecast times after it; the forecasts are
written to disk and sealed (SHA-256 of every file, recorded with the time of sealing) before any real image of
those times is on our disk. Real images are used later only as the answer key.

## Fixed inputs
- Models: the 6 `final_offset` models (UNetSmall and ConvLSTM, seeds 0-2, `best.pt`, background-aware inputs),
  unchanged; their checksums go into the seal. The primary forecast of each architecture is its seed ensemble.
  The `final_centred` models are not used (the last data are in the offset pointing).
- Origin: the last processed frames of `final_offset` (the continuous observing stretch 2026-09-27 03:27-14:58 UT,
  store indices 7216 frames, last 14:58:26 UT), with the `final_offset` static background and frame cache.
- B-short: from each of the last 5 frames as forecast origin, forecasts at the trained horizons 20, 40, 80, 160
  frames (target time = origin time + horizon x 89 s, the median cadence; about 30 min to 3.9 h).
- B-roll (multi-day, experimental): 18 chained jumps of the longest trained horizon (160 frames, about 3.94 h),
  about 71 h in all (to about 2026-09-30 13:50 UT). Jump 1 uses the last 4x18+5 = 77 real frames and forecasts
  73 consecutive frames from their 5-frame contexts; every later jump forecasts from 5-frame contexts made only of
  the previous jump's forecasts, so the chain loses 4 frames per jump and ends with 5. Each seed is rolled on its
  own forecasts; the ensemble at each jump is the mean of its seeds' rollouts. Only the trained horizon 160 is ever used.
  Pixels that rotate in from the east limb have no source and stay invalid (NaN).

## Scoring (later, when real images of those times are on disk; descriptive)
- Each forecast is compared with the real NB03 full-disk frame nearest to its target time, within +-2 min (the
  forecast derotated by the small time difference); same disk region (r < 0.9) and MAE as above.
- References: the five physical baselines made from the same origin frames for the same target times; the
  strongest is chosen on the scored targets, as above.
- B-short: skill of each ensemble over the strongest baseline per horizon (few targets: the 14:07-17:06 UT
  observing gap on 27 Sep leaves mainly the 3.9-h horizon scoreable; reported as single values).
- B-roll: skill against lead time (one realisation, 5 adjacent targets per jump; no confidence interval).
  Expectation registered here: skill decays towards zero (or below) within the first few jumps, because the models
  were trained for 3.9 h at most and errors compound. This part makes no primary claim.
- Targets 2026-09-27 17:06-18:58 UT already exist in the raw archive (never processed or viewed); they are scored
  separately and marked as such. All later targets are truly unseen.

# Addendum C: detector-fixed correction removal (post-hoc, exploratory; written 2026-10-08 before it was run)

Motivated by the C2 failure and by the region breakdown (skill smallest in plage, largest near the limb). For each
final model and horizon, the fixed correction map M(H) is the per-pixel mean of the model's correction (forecast
minus its background-aware B1) over TRAINING samples only. On the VALIDATION split (the test splits are not
re-read), three forecasts are scored against the strongest baseline, per architecture ensemble, region and horizon:
- full model (as before);
- model minus M(H): the part of the skill that depends on the input frames;
- B1-bg plus M(H) alone: what a static detector map gives without any solar information.
Reading: if "model minus M" keeps the skill, it is input-driven (candidate solar); if "M alone" reaches the full
model's skill, the skill is a fixed instrumental correction. No claim is changed by this analysis; it is reported
as a diagnostic next to C2.
- Amendment (2026-10-08, after a 24-sample code check, before the full run): "B1-bg plus M(H)" adds the map to the
  single last frame, while the strongest baseline averages the 5 context frames, so it mainly measures the lost
  averaging. Two references are added and reported next to the registered ones: "B1-avg-bgS plus M(H)" (the
  strongest baseline with the fixed map) and "B1-avg-bgS smoothed" (Gaussian sigma 0.7, 1.0, 1.5 px; what pure
  denoising gives). Reason: methodological (a confound), not the code-check numbers.

# Addendum D: does the input-driven skill need the right Sun, and is it denoising? (post-hoc, exploratory; written 2026-10-08 before it was run)

Follows Addendum C (the advantage over the strongest baseline is input-driven, not the fixed map). Validation split
only; same samples, mask, five baselines, regions and run-hour block bootstrap as the main evaluation (the full-model
rows must reproduce it exactly).
- **D1 correction transplant**: for each validation window, another validation window of the same horizon is drawn
  from the farther half in time (|dt| above the median for that window); each model's correction computed on THAT
  window's context is added to the CORRECT window's background-aware B1. Scored like the models. If the transplant
  keeps the skill, the useful correction does not depend on the solar content; if it falls to <= 0 (or below the
  model minus its fixed map), the skill needs the right Sun. Limitation: large-scale structure changes slowly, so a
  transplant from a few hours away is not a fully independent Sun; the time separation is reported.
- **D2 edge-preserving denoising**: the strongest baseline (B1-avg-bgS) filtered by NaN-aware median filters (3x3,
  5x5) and bilateral filters (5x5, spatial sigma 1 px, range sigma 1 and 2 x the image's robust noise level). The best
  filter is chosen by median MAE on 100 training windows per horizon, then scored on validation. If it reaches the
  models' skill, the advantage is explainable as denoising.
- D3 (scoring the sealed blind forecasts) follows Addendum B as registered there.
- Amendment (2026-10-08, after a 16-window code check, before the full run): a model's correction also cancels the
  pixel noise of its own last frame, so a transplanted correction adds another frame's noise and D1 alone cannot
  separate "needs the right Sun" from "denoises its own frame". Added: <model>-lowpass (the model's own correction,
  Gaussian-smoothed with sigma 2 px) and <model>-transplant-lowpass (the transplanted correction, same smoothing).
  If the own low-pass keeps skill and the transplanted low-pass does not, the large-scale correction needs the
  right Sun. Reason: methodological (a confound), not the code-check numbers.
- Amendment 2 (2026-10-08, before the full run): the low-pass variants above sit on the single last frame, which is
  noisier than the strongest baseline's 5-frame average, so they inherit the same averaging handicap. The decisive
  pair is defined relative to the strongest baseline itself: <model>-avglp = B1-avg-bgS + G2(model - B1-avg-bgS) and
  <model>-transplant-avglp = B1-avg-bgS + G2(other window's model - other window's B1-avg-bgS), G2 = Gaussian sigma
  2 px. All variants are reported. Reason: methodological, not the numbers.

# Addendum E: tuned classical forecasters (post-hoc, exploratory; written 2026-10-09 before it was run)

After Addenda C and D the advantage over the strongest baseline is input-specific, not the fixed map and not
standard denoising. Three classical explanations remain; each becomes a baseline TUNED ON TRAINING WINDOWS ONLY
(300 per horizon) and is scored on the VALIDATION split with the evaluation's samples, mask, regions and block
bootstrap (the test splits are not re-read):
- **E1 weighted persistence**: the 5 background-aware derotated context frames combined with least-squares weights
  and an offset per horizon (instead of the equal-weight mean).
- **E3 limb-darkening ratio**: each derotated context pixel scaled by q(mu_target)/q(mu_source) (q = the
  background file's quiet-Sun profile), then averaged; it applies at any lead, so it is also reported on the
  sealed multi-day targets (descriptive).
- **E2 sharpening**: unsharp masking B + a (B - G_s(B)) of the E1+E3 forecaster, a in {0, 0.1, ..., 1.0},
  s in {0.5, 0.75, 1.0, 1.5} px, chosen by median training MAE (undoes the bilinear derotation blur).
- **Tuned classical forecaster** = E1 + E3 + E2 in that order. Reading: if the models keep a positive interval
  over it, the advantage is not explained by frame weighting, derotation blur or limb darkening; if it vanishes,
  that classical forecaster is the result to report.
- Amendment (2026-10-09, after a 16-window code check and a training-window check, before the full run): the
  background file's profile q(mu) is not a usable limb-darkening law: it rises 17 % from mu 0.9 to 1.0 (the active
  region near disk centre in these data), so scaling by q(mu_t)/q(mu_s) changes frames by ~1.7 % even near disk
  centre, and both the ratio and its inverse worsen persistence on training windows (-13 % and -15 % at 3.9 h). E3
  is therefore defined data-driven, still from training windows only: per horizon, the mean ratio (clipped to
  0.5-2) of the target to the forecast in bins of (mu_target: 12 bins over 0.43-1, mu_target - mu_source: 9 bins
  over -0.04..0.04), applied multiplicatively. E3 alone is fitted on B1-avg-bgS; in the tuned classical forecaster
  the order is E1 weights, then E3 fitted on the E1 output, then E2 fitted on that. The registered q-ratio version
  is still reported (E3-qratio). Reason: methodological, found on training data, not on validation numbers.

# Addendum F: state estimate or forecast? (post-hoc, exploratory; written 2026-10-09 before it was run)

Question: is the models' advantage at longer horizons a better estimate of the CURRENT chromosphere carried forward
by rotation, or does it include predicted evolution? Validation split only, same baselines and block bootstrap.
For every validation window with target horizon H in {80, 160} frames, each frozen model is also run at a SHORTER
horizon h in {20, 40, 80} (< H) from the same context: its forecast for t_last + h x 89 s is then moved to the
real target time by solar rotation alone (background-aware derotation, as in the baseline). Seed ensembles as before.
Reading: if the rotated short-horizon forecast reaches the skill of the model's own H forecast, the advantage is a
better state estimate (no predicted evolution); if the model's own H forecast is clearly better (interval of the
difference above zero), the models predict some evolution between t_last + h and the target.

# Addendum G: an independent later test period (written 2026-10-09 before any of these frames were processed)

The frozen final models are scored on SUIT NB03 data from 2026-09-28 00:00 UT to 2026-10-06 23:59 UT, which no
model, baseline fit or analysis choice has used (the sealed blind forecasts of Addendum B touch 27-30 Sep only through
their own few targets, which are excluded here by starting at 28 Sep 00:00 and are reported separately anyway).
- Frames: 2048 NB03 full-disk frames, registered with the pipeline's method and calibrated with the FROZEN products
  of the matching training data set (scripts/answer_frames.py approach, validated against the pipeline); grouped by
  pointing mode; each mode is scored only with the models of that mode.
- Windows: the pipeline's rules (runs split at gaps > 300 s; context 5 consecutive frames and the target inside one
  run; horizons 20, 40, 80, 160 frames).
- Metrics, regions, five baselines (B1-avg-clim with the training data set's maps) and run-hour block bootstrap as
  in the main evaluation; day-block intervals as P2.
- Criteria: G1 = P1 on this period for each ensemble and horizon; G2 = P2 where at least two days exist.
- Negative result: if G1 fails at every horizon, the skill seen in 10-27 Sep does not carry over to later dates.

# Addendum H: target-validity leak check (written 2026-10-09 before it was run)

Found while running Addendum F: during training and in every evaluation so far, the models' mask input channel was
`isfinite(target) & isfinite(context)` (suitdyn/ml/data.py), so the models saw which TARGET pixels are invalid
(spikes, clipping, seams, the limb at the target's pointing). Those pixels are never scored, but their pattern can
carry information about the target frame (for example where the detector seam falls, i.e. its pointing). Check, on
the VALIDATION split with the evaluation's samples, scoring mask, five baselines and bootstrap: the same frozen models
with the mask channel built from the context only (`isfinite(context)`), next to the standard result. The sealed
blind forecasts (Addendum B) already used the context-only mask. Reading: if the skill is essentially unchanged, the
leak does not drive the results and is reported as a disclosed flaw; if the skill drops materially, every result that
used the target-aware mask is overstated, and the models must be retrained without it before any claim is made.

# Addendum I: retraining without the target-validity leak (written 2026-10-09, before any v2 model was trained)

Outcome of Addendum H (validation, disk, seed ensembles, skill vs the strongest baseline in %, standard mask ->
context-only mask, horizons 20/40/80/160 frames): final_offset U-Net 2.1 -> 1.0, 3.4 -> 0.5, 2.0 -> 0.2, 3.5 -> -0.4;
final_offset ConvLSTM 1.8 -> 1.6, 1.8 -> 1.5, 2.1 -> 1.6, 3.3 -> 2.9; final_centred both models unchanged within 0.1.
The drop is material for the final_offset U-Net, so by Addendum H every result that used the target-aware mask (the
main evaluation and sealed test, Addenda C, D, E, the region analyses) is treated as overstated for these models
("v1"). Size of the leak, measured on 600 windows per split: the target-aware mask switched off 0.42-0.50 %
(final_centred) and 0.55-0.57 % (final_offset) of the context-valid disk pixels, in at least 99.8 % of the windows.
So the centred v1 models received the leaked information too; their skill just does not depend on it.

Change ("v2"): the models' mask channel is built from the context frames only (suitdyn/ml/data.py input_mask). The
loss mask and the scoring masks are unchanged (target and every context frame finite). Nothing else changes:
architectures, hyper-parameters, seeds 0-2, epochs, patience, epoch fraction, data sets, splits, frame caches, static
backgrounds, baselines, regions, bootstrap, criteria. The [thermal] settings only change the speed.

Order (scripts/retrain_v2.py): the 6 final_offset runs first, then the 6 final_centred runs, each trained into
phase3/runs_v2. Before a data set's v2 models are evaluated, its v1 products are kept as phase3/runs_v1_targetmask,
eval_v1_targetmask, posthoc_v1_targetmask, and a copy eval_test_v1_targetmask (the test-read record stays in
eval_test/test_reads.json and keeps every read).

Evaluation of v2, each item run once, in this order, whatever the validation shows:
1. Validation: the main evaluation (phase3_evaluate --split val), with the controls C1/C2.
2. final_offset, primary test: the independent period 28 Sep - 6 Oct 2026 (Addendum G: its cache, windows, scoring,
   criteria G1/G2), v2 ensembles with the context-only mask. No model has seen this period and nothing is chosen on
   it. The v1 models are scored on the same cache with both masks for comparison.
3. The original sealed test split of each data set: one further read with the v2 models (phase3_evaluate --split
   test --allow-new-models, recorded by the guard as a changed model set). Criteria P1-P3 as registered, reported as
   a second read next to the first (v1) read, which is reported as overstated.
Claims are made from v2 only; the leak, the Addendum H numbers and the v1 results are reported as such. Addenda C,
D, E and F are repeated on v2 afterwards with the same scripts (validation only).

Note on Addendum G: its job was started before this change and imports the evaluation code only when it reaches the
scoring step, so its own first summary (newperiod/summary.csv) scores the v1 models with the context-only mask. The
re-scoring in item 2 writes summary_v2.csv, summary_v1_targetmask.csv and summary_v1_ctxmask.csv.

Note to Addendum I (2026-10-10, before the full-period scoring): the new-period cache built on 2026-10-09 silently lost
5056 of its 6453 frames, every frame from 29 Sep (part) to 6 Oct, to read errors on the data share (the frame builder
returned an empty frame on any error; the same files build correctly on 2026-10-10). Item 2 was therefore first scored
on 28-29 Sep only (4067 windows, 2 days); those results are kept as newperiod/*_partialcache.* and reported as such.
The cache is repaired (only the empty frames are built again, with retries and a reported failure count) and the three
scorings of item 2 are repeated unchanged on the repaired cache (scripts/after_v2.py). The repaired scoring is the
item-2 result; nothing else changes.
