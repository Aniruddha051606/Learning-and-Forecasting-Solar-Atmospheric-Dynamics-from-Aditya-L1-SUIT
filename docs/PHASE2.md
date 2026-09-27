# SUIT-DYN — Phase 2 report

Data: SUIT Level-1, 2026-09-22 03:27 → 2026-09-25 23:30 UT; 23,172 files, 40.9 GB.
Dataset hash (manifest_sha256): `706c5c3e…`. Everything is produced by the scripts listed at the end.
This is still prototype data. Nothing here is evidence of generalisation.

---

## 0. Data source (updated 2026-09-27)

**Archive of record:** the network share `//192.168.1.2/DATA/pradan1.issdc.gov.in/...` (config
`raw_root`, read-only), reached over Ethernet: 1 Gbps link; 89 MB/s large-file reads; ~19 MB/s when
checksumming many small files over SMB. It is still being downloaded into.
- **Local copy:** the earlier `D:/Data/...` copy is kept untouched (`local_copy_root`). Its manifest
  supplies only files not yet on the share (`source = local_copy`, 1,360 files on 2026-09-27, none of
  them in data set v0).
- **Identity:** all 10,977 files present in both places were byte-identical.
- **Dataset v0 is fully reproducible from the share alone:** `scripts/verify_dataset.py` reports
  2,202 of 2,202 frames identical, all read from the share.
- **New data:** 26 Sep (T26_1489) is processed (119 full-disk, 741 ROI frames). It lies outside the
  v0 splits and is kept for the next data set version.
- **Bug found and fixed on the way:** incremental manifest rebuilds reused rows via `itertuples`,
  which silently renames non-identifier columns. `DATE-OBS` was dropped from 11,837 reused rows. The
  fix reuses rows as records and only reuses complete rows; there is a regression test.

## 1. New instrument findings

### 1.1 Two pointing modes: the Sun moved 480 px on the detector (2026-09-23 ~05:00 UT)

| Mode | NB03 frames | Span | Disk centre (binned px) | Disk on CCD | Quadrant seam |
|---|---|---|---|---|---|
| centred | 177 | 22 Sep 03:27 → 23 Sep 05:00 | (1033, 1012) | 100 % | through disk centre |
| offset | 2,237 | 23 Sep 05:01 → 25 Sep 23:30 | (1282, 598) | 95–97 % | 256 px east of centre |

The change is a slew of about 2 min. The first frame after it is already at (1298, 570). All unbinned
multi-filter bursts fall in the offset mode. The most likely cause is a spacecraft repointing (Sarkar
et al. 2025 describe the SUIT–VELC misalignment). This is not established.

**Decision:** the prototype dataset is offset-mode only. The two modes put different detector regions
under the Sun and cross the seam at different places, so mixing them would build a pointing-dependent
instrument signature into the data. Centred frames are kept in the manifest, labelled. They are used
as a test that the fixed pattern is fixed to the detector (§2).

### 1.2 `MEAS_EXP` does not describe the exposure of the pixel data

`MEAS_EXP` takes quantised values of 286.2–291.6 ms (±1 %) while `CMD_EXPT` = 300 ms and
`EXPTIME` = 0.3 s are constant. `NORM` = 0, so Level-1 did not normalise the exposure. Within runs
the NB03 disk brightness does **not** follow `MEAS_EXP`:
- residual correlation −0.005;
- dividing by `MEAS_EXP` raises the frame-to-frame scatter from 0.24 % to 0.37 %.

**Decision:** exposure normalisation uses the commanded exposure (`CMD_EXPT`). For NB03 this is a
constant. It matters for multi-filter work, where the commanded exposures differ by filter.

### 1.3 Single-frame whole-disk brightness excursions

The Phase 1 "brightness jump" flag, recomputed without the exposure error, still finds single frames
whose disk median is 1–3 % above or below their neighbours. They cluster in program 139 on 23 and
25 Sep. A whole-disk Mg II k change of 2 % for one or two frames is not plausibly solar.
**Decision:** these frames are kept and flagged. The normalisation experiment decides whether a
per-frame level absorbs them. Frames that start a run after a gap are 15–19 % too bright, extending
PHASE1 A8 to gaps inside a program; they are excluded.

### 1.4 Disk brightness follows the pointing oscillation

Within each run, the slow-detrended disk brightness depends on the pointing. It falls by 0.02–0.17 %
per pixel of x-shift, with the same sign in all 8 runs. Pointing explains a median **38 %** of the
detrended variance (R² from 0.11 to 0.51). With the ±6 px oscillation that is roughly a ±0.5 %
instrumental brightness oscillation. A candidate cause is the seam (A5): as the disk moves in +x, more
of it falls on the dimmer side. The seam study (§3) tests this. Whatever part survives calibration is
an argument for frame-level normalisation (§4).

---

## 2. Fixed-pattern calibration

**Method** (`suitdyn/flat.py`, `scripts/phase2_calibration.py`, `scripts/phase2_calibration_followup.py`):
1. Each NB03 frame is divided by a normalised-convolution smoothing of itself (σ = 15 px). The
   smoothing uses only valid disk pixels and never crosses the quadrant seam, so neither the limb nor
   the seam step leaks in.
2. The median of that relative residual is taken per detector pixel over 320 offset-mode frames
   spread over 2.8 days. Solar structure moves across the detector; a detector pattern does not.
3. 120 centred-mode frames are kept separately, as a test.

**Results:**

| Question | Test | Result |
|---|---|---|
| Size | rms of the estimate | 4.7 % of the disk level (143 counts) |
| Which scales are a detector pattern? | Correlation of independent estimates after high-pass | early vs last day: 0.85 (32 px) → 0.95 (8 px) → 0.98 (2 px); offset vs centred pointing: 0.57 → 0.76 → 0.83 |
| Multiplicative or additive? | Pixels with plage passing over them (local level ×1.51) vs quiet | absolute amplitude ratio **1.000**, relative **0.686**. Additive predicts 1.0 and 0.663; multiplicative predicts 1.51 and 1.0 → **additive** |
| Does the correction work? | Plain phase correlation of 149 consecutive offset-mode frames vs expected motion | rms error 1.54 px → **0.053 px**; recovered rotation 0.128 → **0.166** px/frame (predicted 0.167) |

**Adopted:** an additive pattern, high-passed at 8 px (`nb03_pattern_additive_hp8.npy`, rms 105
counts, provenance in `nb03_pattern_adopted.json`). Only scales where independent days agree at
r ≥ 0.95 are corrected. Larger scales carry solar leakage: in the centred-mode estimate, active
regions that did not move far enough in one day are clearly visible.

**Hypotheses:**
- *Refuted:* the pattern is the imprint of solar network in the 2024 flat field. That predicts a
  multiplicative pattern, and it is additive.
- *Open:* Level-1 subtracts a scatter-calibration image (`SCAT_CF` = `NB03_scat_…_2024-06-01`); if
  that image contains solar network, subtracting it gives exactly an additive, network-shaped
  pattern. It is untestable without that file.

**Corrections to our own method:**
- The first multiplicative-vs-additive test regressed bright-frame on dark-frame estimates. That
  cannot separate the two cases: noise in both estimates flattens both slopes, and both hypotheses
  predict the same slope ratio. Its "additive" verdict was not evidence, and it was replaced by the
  plage-lever test above rather than by changing the rule.
- The first rotation test used a run containing the pointing slew. It was redone on an offset-only
  run.
- The first full analysis ran out of memory (25 GB) by taking one median over the whole cache. It was
  rewritten as a bounded single pass.
- The adopted pattern file was first made by an inline command, not by a pipeline script.
  **Fixed:** `phase2_calibration.py` now only measures (relative pattern, level map, subset
  estimates); `phase2_calibration_followup.py` makes the decision (plage-lever test) and writes
  `nb03_pattern_adopted.npy` with provenance. The high-pass scale is a documented setting
  (`[calibration] pattern_highpass_px = 8`).
  **Verified:** rerunning both scripts from the cache reproduces the file used by store v0 byte for
  byte (SHA-256 `92537789…`, maximum difference 0).

## 3. Seam and large-scale detector response

The E–W mirror test (`scripts/phase2_seam.py`) is **invalid** as a seam test. The mirror points west
of the disk centre lie near the CCD's right edge, which is strongly vignetted (Phase 1: the west limb
is about 5× dimmer than the east). Mirror ratios of 0.3–0.8 measure that vignetting, not the seam.
The step across the seam itself is unchanged by the pattern correction: −18 to −28 % in the southern
rows, ~0 near row 1000 and +3 % above.

The broader point: **the large-scale detector response is strongly non-uniform** (vignetting towards
the west edge, the seam, and a horizontal feature near detector rows 290–490; see §4.1). All of it is
fixed on the CCD, while the Sun moves ±10 px. How much it matters for forecasting is measured
directly by the noise-floor and pointing-sensitivity maps (§4.1).

**Self-calibration attempted: a negative result** (`suitdyn/largescale.py`,
`scripts/phase2_largescale.py`).
- **Model:** log I_k(x) = r(x) + q(mu) + a_k, fitted on quiet-disk pixels. r is the log response
  (32-px bilinear splines, separate on each side of the seam), q is a shared limb-darkening profile
  and a_k is a per-frame level. Robust fit.
- **Synthetic test:** passed. The response is recovered to 2 % rms and the seam step to 3 %.
- **Validation on real data**, against measurements the fit never saw:

| Fit on | V1: fitted gradient vs measured pointing sensitivity (val) | Variance explained (x / y) | V2: seam step vs measured |
|---|---|---|---|
| training offset frames + centred frames | r = 0.14 / 0.41 | none / 30 % | r = 0.56, rms difference 15 pp |
| training offset frames only | r = 0.38 / 0.57 | 14 % / 32 % | r = 0.89, rms difference 8 pp |

- **Reading:** combining the two pointing modes, the step meant to make the response identifiable,
  makes the fit worse. The likely cause is that the model's key assumption (one detector-fixed,
  multiplicative response for both modes) does not hold. Level-1 over-subtracts scattered light
  (off-limb −400 counts, ~13 % of the disk level); scattered light moves with the Sun, not the
  detector, and is additive. Even the offset-only fit explains little of the measured sensitivity.
- **Not adopted.** The empirical first-order correction (§4.2), estimated directly from the one-frame
  pointing sensitivity, is better validated and stays. An 8-pp error on a step of up to 28 % is not
  good enough to correct the seam, so it stays masked.

**Handling in the data set:**
- SEAM mask bits per frame.
- The first-order response correction.
- A *trusted region*: pixels whose pointing sensitivity, measured on the **training** split after
  the response correction, is below the core median + 3 robust σ. It covers 91.8 % of the disk
  (`outputs/phase2/noise_maps/noise_maps_v0_train_g2_resp.npz`). It excludes the seam band, the
  vignetted south-west edge and **parts of the plage belts**: plage streaks along the rotation
  direction, and its steep gradients amplify residual misregistration. Phase 3 therefore reports
  metrics on the trusted region, the full valid disk and plage only, not on the trusted region
  alone.

## 4. Noise floor, normalisation, baselines

### 4.1 Noise floor and where the instrument dominates (`scripts/phase2_noise_maps.py`)

The noise floor is the error of rotation-corrected persistence (B1) one frame ahead (~87 s, where
the Sun barely changes). The pointing sensitivity is the per-pixel slope of that error against the
pointing change between the two frames. Both are measured on all one-frame windows, at 768².

| Store / correction | Floor, disk-core median | Floor, p95 | Pointing sensitivity, core | Sensitivity, p95 |
|---|---|---|---|---|
| v0raw (no calibration) | 2.80 % | 3.61 % | 0.29 %/px | 0.61 %/px |
| v0 (fixed pattern corrected) | **1.95 %** | **2.14 %** | 0.17 %/px | 0.35 %/px |
| v0 + pointing-response correction | 1.94 % | 2.12 % | **0.11 %/px** | **0.24 %/px** |

(validation split, 498 windows; the training split gives the same numbers to ±0.1 %.)

After the pattern correction, the maps show large-scale detector-fixed structure: an east–west
gradient, bands with sharp edges, and the seam line. At about 0.3 %/px, the ±10 px pointing
oscillation turns this into up to ±3 % brightness modulation. That is the dominant instrument
effect at 30–90 min horizons.

### 4.2 Pointing-response correction (`suitdyn/response.py`, `scripts/phase2_response.py`)

In a registered frame I(u) = S(u)·R(u + c), where c is the disk centre on the detector. The one-frame
error slope against the pointing change equals −∇ln R, so every frame is brought to a reference
pointing by the factor exp(s·(c − c̄)). The slope maps come **from the training split only**
(smoothed, σ = 6 px at 768²; reference pointing = training median).

It is judged on the validation split:
- the per-pixel pointing sensitivity falls by 36 % (table above);
- whole-disk level vs pointing within runs falls from R² 0.51 / 0.83 (median 0.67) to
  0.28 / 0.41 (median 0.34).

The correction is real but first order: about half of the level modulation remains. Possible
reasons are a non-linear or time-varying response, or noise in the smoothed maps.

### 4.3 Normalisation experiment (store v0raw, validation split, 120 windows per horizon, native 1536²)

All variants are scored on the same windows. The forecast is always normalised with statistics of
the frame it came from, never the truth.

| Variant | B1 relative MAE vs global | Pattern of plage-excess change kept (corr. with global) | Whole-disk level scatter | Verdict |
|---|---|---|---|---|
| global | — | 1.00 | 0.19–0.37 % | reference |
| **per-frame median** | identical (within CI) | **0.98–0.99** | **0.05–0.20 %** | **adopted** |
| quiet-Sun contrast | +8–10 % | 0.50–0.80 | 0.11–0.20 % | rejected: distorts plage evolution |
| robust percentile | +50 % | 0.57–0.82 | 0.43–1.1 % | rejected |

The per-frame median removes the whole-disk level jitter (§1.3, and half of §1.4) while keeping the
spatial pattern of solar change. Any genuine whole-disk NB03 brightening is removed too. It is at
most a few 0.1 % here and was shown to be largely instrumental, and the trade-off is recorded.

### 4.4 Baselines (store v0raw, global normalisation, validation, native resolution)

| Horizon (measured) | 1.4 min | 7.1 min | 14 min | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|---|---|---|
| B1 relative MAE | 2.83 % | 3.22 % | 3.35 % | 3.45 % | 3.69 % | 3.94 % | 4.81 % |
| skill B1 vs B0 | 0.01 | 0.05 | 0.08 | 0.17 | 0.32 | 0.47 | 0.51 |
| skill B2 vs B1 | −0.12 | −0.86 | −1.48 | −2.09 | −2.67 | −3.79 | −4.51 |

(95 % intervals from a bootstrap over run×hour blocks are about ±0.05 percentage points on B1.)

- **Most of the error at every horizon up to ~4 h is the noise floor.** B1 rises only from 2.8 % to
  4.8 % over 4 h. The part of the target a forecaster could learn beyond B1 is, at pixel level, a
  small signal on a large floor.
- **B2 (optical-flow extrapolation) is dropped.** It is worse than B1 at every horizon, even one
  frame ahead: the estimated flow is dominated by noise and residual pointing jitter. The OpenCV
  Farneback parameters were not tuned, and that assumption is recorded.
- **Rotation correction matters more with horizon** (skill over B0: 0.17 at 28 min, 0.5 at 2–4 h).

### 4.5 Effect of calibration on the baselines (same validation windows, native 1536², B1, global)

| Horizon | 1.4 min | 7.1 min | 14 min | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|---|---|---|
| v0raw (uncorrected) | 2.83 % | 3.22 % | 3.35 % | 3.45 % | 3.69 % | 3.94 % | 4.81 % |
| v0 (fixed pattern) | 1.86 % | 1.96 % | 2.06 % | 2.16 % | 2.52 % | 2.83 % | 3.88 % |
| **v0 + pointing response** | **1.85 %** | **1.91 %** | **1.99 %** | **2.10 %** | **2.33 %** | **2.74 %** | **3.75 %** |
| B0 persistence (v0 + resp.) | 1.88 % | 2.06 % | 2.32 % | 2.99 % | 4.39 % | 6.77 % | 9.16 % |

- The fixed-pattern correction removes a third of the error at every horizon. SSIM rises from 0.88 to
  0.97 and gradient correlation from 0.57 to 0.84: the uncorrected pattern dominated the image edges.
- The response correction helps where predicted. It has no effect at one frame and the largest
  effect at about 1 h, where the pointing oscillation acts (2.52 → 2.33 %; the 95 % intervals do not
  overlap). Its intervals are tighter too, because it removes instrumental variance.
- Skill of B1 over B0 reaches 0.6 at 2–4 h: rotation is the dominant predictable signal.

### 4.6 Resolution study (v0 + response, B1, global, validation)

| Horizon | 1.4 min | 7.1 min | 14 min | 28 min | 57 min | 1.9 h | 3.8 h | growth 1 fr → 3.8 h |
|---|---|---|---|---|---|---|---|---|
| 1536² (1.41″/px) | 1.85 % | 1.91 % | 1.99 % | 2.10 % | 2.33 % | 2.74 % | 3.75 % | 1.90 pp |
| 768² (2.82″/px) | 1.62 % | 1.75 % | 1.77 % | 1.93 % | 2.16 % | 2.62 % | 3.67 % | 2.05 pp |
| 384² (5.64″/px) | 1.46 % | 1.61 % | 1.73 % | 1.78 % | 2.10 % | 2.52 % | 3.58 % | 2.12 pp |

- **The noise floor is mostly not pixel noise.** Photon noise would halve with every 2× binning,
  but the floor falls only 10–12 % per step. It is dominated by spatially correlated terms:
  registration residuals (≈0.5–1 px), derotation error, residual detector response, and plausibly
  real fast chromospheric variation. Mg II k shows ~3-min oscillations, and 87 s is about half a
  period. These cannot be separated yet.
- **The forecastable change is almost independent of resolution:** B1 grows by about 2 percentage
  points from one frame to 3.8 h at every resolution. At hour horizons the evolution B1 misses is
  large-scale.
- Gradient correlation rises from 0.84 (1536) to 0.96 (768) and 0.98 (384) as noise edges drop out.

### 4.6b What the one-frame floor is made of (`scripts/phase2_floor_origin.py`)

The structure function D(τ) is the median relative |B1(F(t)) − F(t+τ)| on the disk, with the
response correction and per-frame normalisation applied (768², quiet pixels unless stated).

| τ (s) | 21 | 43 | 64 | 85 | 107 | 128 | 149 | 170 | 213 | 256 | 298 | 341 | 405 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 21-s cadence stretch (train), quiet | 0.87 % | 1.16 | 1.41 | 1.60 | 1.69 | **1.71** | 1.67 | 1.59 | 1.41 | **1.34** | 1.38 | 1.48 | 1.62 |
| 87-s cadence (val), quiet | | | | 1.54 | | | | 1.55 | | **1.31** | | 1.46 | 1.62 |

- **D is not monotonic.** It peaks near 2 min and returns to a minimum near 4.3 min, the signature of
  an oscillation with a period of about 4–5 min (largest difference at half a period, smallest near a
  full period). Instrument noise and slow evolution can only make D grow with τ. The independent
  87-s validation data agree: the error three frames ahead (1.31 %) is lower than one frame ahead
  (1.54 %).
- **The dip is spatially selective:** −15 % in quiet Sun, −4 % in plage. Suppression of
  chromospheric oscillations in strong-field regions is known. A registration or pointing artefact
  would do the opposite, largest where the image has most structure. Whole-frame (global) instrument
  oscillations are removed by the per-frame normalisation.
- **Interpretation (hedged):** a large part of the "noise floor" at 87 s is **solar chromospheric
  oscillation**, sampled near its half period. Instrument plus registration noise is at most ~0.9 %
  (the 21-s value, which still contains some oscillation). A check of the pointing series for a
  ~4-min periodicity was inconclusive, because the high-pass used creates negative autocorrelation at
  those lags. An instrumental periodicity is therefore not excluded, but it would have to survive
  registration and normalisation and be weaker where the image has more structure.
- **Consequences:**
  1. The skill ceilings in §4.7 treat the one-frame error as irreducible. At short horizons they are
     too pessimistic, since part of that error is an oscillation that may be partly predictable from
     a few frames of context. Short-horizon skill must be reported separately as possible
     "oscillation skill".
  2. At horizons ≥ 30 min the oscillation is effectively random. Averaging the target over about one
     period (3 frames) should remove it, which is an ablation for Phase 3.

### 4.7 What a forecaster could gain over B1 (pixel level), corrected in Phase 3

**Correction.** The first version of this section treated the whole one-frame error as irreducible
and used (err_B1 − floor) / err_B1 as a ceiling (e.g. 18 % at 28 min, 384²). That was wrong. The
one-frame error |F(t+1) − F(t)| contains the noise (and oscillation) of **both** frames. A
forecaster cannot remove the target's own noise, but it can remove the noise of its input, for
example by averaging the context. The irreducible part is therefore about floor / √2 (independent,
equal noise in the two frames), and the ceiling is 1 − (floor / √2) / err_B1:

| Horizon | 7 min | 14 min | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|---|---|
| 1536² | 32 % | 34 % | 38 % | 44 % | 52 % | 65 % |
| 768² | 35 % | 35 % | 41 % | 47 % | 56 % | 69 % |
| 384² | 36 % | 40 % | 42 % | 51 % | 59 % | 71 % |

These are approximate upper bounds. They assume the target's noise is independent of everything a
model sees and that the oscillation part of the floor is not predictable, which is not guaranteed
at short horizons (§4.6b). With the first version, a model at ~20 % skill at 28 min would have been
wrongly flagged as "fitting noise". A large part of the room between B1 and these ceilings can be
taken by plain temporal averaging of the context, so Phase 3 adds that as a baseline (B1-avg) that
every model must also beat.

---

## 6. Experiment matrix (updated with Phase 2 measurements)

| ID | Input | Target | Horizons | Model | Loss | Primary metric | Expected | Failure criterion |
|---|---|---|---|---|---|---|---|---|
| N0 | frame t | frame t+1 | 1 fr | — | — | rel. MAE | **1.85 / 1.62 / 1.46 %** (1536/768/384) | — (the floor) |
| B0 | frame t | t+H | 1–160 fr | persistence | — | rel. MAE | measured §4.5 | — |
| B1 | frame t | t+H | 1–160 fr | rotation-corrected persistence | — | rel. MAE, SSIM, grad. corr. | measured §4.5–4.6 | — (reference) |
| ~~B2~~ | — | — | — | optical flow | — | — | — | dropped: worse than B1 at every horizon |
| A1 | frames t−K..t | t+H | 20, 40, 80, 160 fr | ConvLSTM (small) | masked L1 on B1 residual | skill vs B1 **and vs B1-avg** (block CI) | above B1-avg at ≥ 1 horizon | CI of skill vs B1-avg includes 0 at every H, **or** skill above the corrected §4.7 ceiling (leakage) |
| A2 | same | same | same | CNN/ViT encoder + temporal model | same | same | ≈ A1 at this data size | no gain over A1 → keep the smaller one |
| M1–M4 | as in PHASE1 §8 | | ≥ 1 h | | | | | M3 needs a burst-to-burst photometric scatter (±1–3 %) below the signal |

The targets are **residuals relative to B1** (the model predicts what rotation-corrected
persistence misses). Horizons below 20 frames (~30 min) are not model targets, because §4.7 leaves
too little room above the floor.

## 7. Model recommendation (now data-backed, still prototype-scale)

| Setting | Recommendation | Evidence |
|---|---|---|
| Input resolution | **384²** for the first models (5.6″/px); 768² only if a model shows it uses fine scale | Floor lowest and forecastable growth unchanged at 384 (§4.6); the 6 GB GPU |
| Horizons | **20, 40, 80, 160 frames** (0.5, 1, 2, 4 h), reported separately | §4.7: below ~30 min the ceiling is < 15 % |
| Target | residual after B1 (F(t+H) − B1(t+H)) on the valid, trusted, response-corrected disk | Rotation is the dominant signal; forcing a model to relearn it wastes capacity and inflates apparent skill |
| Context | start with K = 5 frames (≈ 7 min), then test 20 | The one-step floor decorrelates quickly; a longer context is an ablation, not a default |
| Normalisation | per-frame disk median, after pattern and response correction | §4.3 |
| Model size | ConvLSTM ≈ 1–3 M parameters; encoder + temporal ≤ 10 M | ~35 h of training data across 5 runs (the number of independent hours, not frames, bounds capacity) |
| Batch / memory | 384², K = 5, batch 8, mixed precision: well within 6 GB | — |
| Optimiser | AdamW, lr 3e-4 with cosine decay, early stopping on validation skill vs B1 | standard; the test split stays sealed |

Before any of this: this is still 2.8 days of one pointing mode, with a validation split of three
runs and a test split of one. A model result on it is a pipeline check, not a scientific claim.

## 8. Limitations and open items

- **Seam:** masked, not corrected. The mirror test is invalid because of vignetting, and the
  self-calibration reproduces the step only to 8 pp (§3).
- **Large-scale response:** only a first-order, pointing-gradient correction. About half the
  whole-disk modulation remains. The self-calibration failed validation (§3). A model with a
  Sun-fixed additive scattered-light term, fitted per pointing mode, is the next thing to try; so is
  asking the instrument team for the Level-1 scatter and flat files.
- **The floor's origin:** now tested (§4.6b). Much of it is probably chromospheric oscillation
  (4–5 min). An instrumental periodicity is not fully excluded.
- **Pattern origin:** additive. The scatter-calibration-file hypothesis is untested without the file.
- **Pipeline B:** the per-filter fixed patterns and the burst photometric scatter are not calibrated
  yet (18 bursts is too few for a pattern estimate).
- **Data volume:** the download stopped at 23,172 files. Every result here needs to be repeated on
  more data before it means anything beyond this prototype.

## 9. Reproduce (from the raw manifest)

```
python scripts/build_manifest.py                   # manifest + dataset hash (incremental)
python scripts/process_frames.py                   # per-frame limb/QC/artefact measurements (incremental)
python scripts/registration_study.py               # pointing modes, registration, validation
python scripts/phase2_calibration.py --reuse-cache # fixed pattern: relative pattern, level, subset estimates
python scripts/phase2_calibration_followup.py      # scale / additive / offset-run tests -> nb03_pattern_adopted.npy
python scripts/build_sequences.py                  # data set v0: frames, splits, windows, bursts, test seal
python scripts/build_store.py --name v0raw --pattern none
python scripts/build_store.py --name v0          # default: the adopted pattern, mode from its provenance
python scripts/phase2_noise_maps.py --store v0 --split train
python scripts/phase2_response.py --store v0
python scripts/phase2_noise_maps.py --store v0 --split train --response outputs/phase2/response/response_v0.npz --tag resp
python scripts/phase2_floor_origin.py --store v0   # structure function (oscillation test)
python scripts/phase2_largescale.py [--centred-frames 0]   # self-calibration (diagnostic; not adopted)
python scripts/phase2_baselines.py --store v0 --split val --no-b2 --variants per_frame_median --response outputs/phase2/response/response_v0.npz --tag resp [--grid-factor 1|2|4]
python -m pytest tests
```

Long runs were launched detached (Win32_Process Create) so they survive the session ending. Every
JSON output records the git commit, the dirty flag and the config hashes.

---

## 5. Dataset v0

Built by `scripts/build_sequences.py` (settings `configs/phase2.toml`):

| Split | Frames | Hours | Runs | Span |
|---|---|---|---|---|
| train | 1,440 | 34.7 | 5 | 23 Sep 05:01 → 24 Sep 23:30 |
| val | 501 | 11.8 | 3 | 25 Sep 00:08 → 14:07 |
| test | 261 | 6.4 | 1 | 25 Sep 17:06 → 23:30 (**sealed**, hash `3ca4a17d…`) |

Frames excluded:
- 177 in the centred pointing mode;
- 18 limb-fit outliers;
- 13 around the slew;
- 12 run-first frames;
- 3 with unusual spike rates.

Windows with a 20-frame context, per horizon:

| Horizon | Measured time | Train | Val | Test |
|---|---|---|---|---|
| 1 | 1.5 min | 1,300 | 446 | 241 |
| 5 | 7.4 min | 1,280 | 438 | 237 |
| 10 | 14.8 min | 1,255 | 428 | 232 |
| 20 | 29.7 min | 1,205 | 408 | 222 |
| 40 | 59.3 min | 1,105 | 368 | 202 |
| 80 | 1.98 h | 919 | 293 | 162 |
| 160 | 3.94 h | 599 | 213 | 82 |

Multi-filter bursts in the offset mode: 18, each with all 11 filters present. Missing filters would
appear as explicit rows.

**Leakage rules (tested in `tests/test_sequences.py`):**
- split boundaries must fall in observing gaps, and the build refuses otherwise;
- a window never crosses a run or a split;
- elapsed times are stored per window;
- reading test windows needs an explicit unseal with a reason, the read is logged, and it fails if the
  test frame list has changed.

**Stores** (`outputs/phase2/stores/`, Zarr, float16 image + uint8 QC mask, grid 1536, r_ref 690):
- `v0raw`: no pattern correction. 2,202 frames, 5.7 GB.
- `v0`: with the pattern correction (pending).
