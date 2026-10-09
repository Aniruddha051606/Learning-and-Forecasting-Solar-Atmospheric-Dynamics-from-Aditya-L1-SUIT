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
