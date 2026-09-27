# SUIT-DYN — project brief (for a reader new to the project)

Status as of 2026-09-27. This brief summarises the project, its findings so far, and what every file
in the repository does. The detailed reports are `docs/DESIGN.md`, `docs/PHASE1.md`, `docs/PHASE2.md`
and `docs/PHASE3.md`.

---

## 1. What the project is

**Goal:** learn and forecast the evolution of the solar chromosphere from images taken by **SUIT**
(Solar Ultraviolet Imaging Telescope) on ISRO's **Aditya-L1** spacecraft at the Sun–Earth L1 point. The
approach is rigorous and falsifiable: every model must beat strong physical baselines, and every claim
must survive instrument negative controls.

- **Main data stream: NB03 filter,** Mg II k line at 279.6 nm (chromosphere). Full-disk images, 2048×2048
  binned pixels (~1.4″/px), taken every ~87 s. This is the only dense time series.
- **Other data:**
  - multi-filter full-disk "snapshots": ~5-min bursts in 11 filters, 6 per day. These are for a later
    "Pipeline B", not started yet.
  - Region-of-interest (ROI) frames chosen by the operators. They are kept out of the main line
    because they are not an unbiased sample.
- **Data level:** Level-1 FITS from ISRO's PRADAN archive. Level 1 includes flat, scatter and PRNU
  corrections, but not distortion or PSF correction. Pixels are int16 with BZERO 30000, so the
  clipped-value floor is −2768.
- **Data volume so far:** 22 Sep 03:27 → 26 Sep 02:18 UT (~24,000 files). That is only a few days,
  so **no claim about generalisation is possible yet**.

## 2. Working rules (the project's scientific stance)

- **No random splits.** Splits are by time, cut only inside observing gaps. The **test split is
  sealed** (hash `3ca4a17d…`); reading it needs an explicit unseal, which has never been done.
- **Baselines first.**
  - B0: persistence.
  - B1: persistence rotated by solar differential rotation.
  - B1-avg: the mean of the context frames, each rotated to the target time.
  - A model gets credit only for skill over the **strongest** baseline.
- **Artefacts are not physics.** Instrument artefacts must not be learned as "solar dynamics". Claims
  of solar skill need negative controls.
- **No causality from correlation.** Operator-chosen ROIs are not an unbiased sample.
- **Workflow:** every intermediate run is a small **code test**. There will be **one full final run**
  at the end.
- **Reproducibility:** every output records the git commit, a dirty flag and config hashes. The raw
  manifest holds a SHA-256 for every file.

## 3. Environment

- **Machine:** Windows 11 laptop, RTX 3050 6 GB laptop GPU, Python 3.14, PyTorch 2.14 (CUDA 13).
- **Raw data:** read directly from a network share,
  `//192.168.1.2/DATA/pradan1.issdc.gov.in/al1/protected/downloadData/suit/level1`.
  - The share is read-only for the pipeline and is still being downloaded into.
  - An older local copy under `D:/Data/...` serves as a fallback for files not yet on the share.
- **Outputs:** everything generated goes to `outputs/`, which git ignores. It is rebuilt from the
  manifest, config and code.
- **Thermal safety:** the laptop shut down once under full GPU load. Training therefore has resumable
  checkpoints and a thermal guard (pause at 80 °C, resume below 72 °C).
- **Detached launches:** long runs are launched detached through WMI with new-process-group flags.

## 4. Pipeline and data flow

```
raw FITS (share) ──► manifest (headers + SHA-256) ──► per-frame measurements (limb fits, QC, motion)
   ──► registration (disk-centred, solar-north-up grid 1536², r = 690 px)
   ──► data set v0: frame list, time splits, (context, horizon) windows, sealed test
   ──► Zarr store: calibrated (fixed pattern), registered frames + QC masks
   ──► pointing-response correction, noise maps, baselines (Phase 2)
   ──► Phase 3 cache: 384² frames, per-frame median normalised; samples (5 context frames derotated
       to the target time, target frame) for horizons 20/40/80/160 frames (28 min / 57 min / 1.9 h / 3.8 h)
   ──► static background S ──► background-aware samples
   ──► models (UNet, ConvLSTM) predict the residual from B1 ──► evaluation and diagnostics
```

## 5. Findings so far, by phase

**Phase 0 (design and audit):**
- **Only dense series:** NB03 is the only dense series.
- **Detector artefacts:**
  - The Sun is off-centre on the detector.
  - Pixels off the limb are negative, because Level 1 over-subtracts scattered light.
  - There are unremoved spikes and a quadrant seam.
- **QVAL is useless:** the quality flag is constant.
- **Model size:** the design argues for small models; 80–120M parameters is not justified.

**Phase 1 (audit, QC, registration):**
- **Pointing:**
  - ~1.4 px per-frame jitter;
  - a ~1.5–2 h oscillation of ±10 px;
  - occasional 5–9 px jumps.
- **Registration:** image-motion phase correlation (with the fixed pattern removed), plus a
  circle-limb anchor per segment; 0.5–0.9 px rms.
- **Fixed pattern:** NB03 carries a detector-fixed pattern (~4 % rms).
- **Seam and first frames:**
  - The vertical quadrant seam has a step of up to −28 %.
  - The first frame of each observing block is 16–19 % too bright, so it is dropped.

**Phase 2 (calibration, dataset, baselines):**
- **Two pointing modes:**
  - "centred" until 23 Sep 05:00 UT;
  - "offset" afterwards, with the Sun about 480 px off detector centre.
  - Dataset v0 uses offset mode only.
- **Dataset v0:**

  | Split | Frames | Span |
  |---|---|---|
  | train | 1,440 | 23 Sep 05:01 → 24 Sep 23:30 |
  | val | 501 | 25 Sep 00:08 → 14:07 |
  | test | 261 | sealed |

- **Calibration:** the fixed pattern is additive and is corrected at scales below 8 px. A
  first-order pointing-response correction is applied. Each frame is normalised to its own median.
  Optical flow (B2) was dropped: it is worse than B1.
- **Noise floor:** a large part of the one-frame error is a ~4–5 min chromospheric oscillation, not
  instrument noise.
- **Large-scale response:** strongly non-uniform, with heavy vignetting toward the west CCD edge.
  Self-calibration failed validation, so it was not adopted.

**Phase 3 (first ML, all numbers provisional test runs):**
- **Models:**
  - UNetSmall, 212k parameters;
  - ConvLSTM, 104k parameters.

  Both start as B1 (zero-initialised output). Loss: masked L1. Early stopping on the last training
  run ("hold-out"); validation is used only for reporting.
- **First result:** the models beat B1-avg by 7 % at 28 min and 23 % at 3.8 h, a gain that grew with
  horizon. That was suspicious.
- **Diagnosis:** most of that gain is an artefact of the baseline, not solar forecasting.
  - B1 derotates the whole image, including things fixed on the grid: the strong offset-mode
    vignetting (east/west brightness ratio ~3 at equal μ; μ is the cosine of the angle from disk
    centre), limb darkening and the seams.
  - That error grows with horizon, and the models learned to undo it.
  - A single static mean-residual map per horizon, learned from training data only, reproduces most
    of the gain.
- **Remaining skill over the strongest static baseline:** 1.5–4 % up to 2 h, 5–8.5 % at 3.8 h, and
  nothing in plage at 28 min. It is not yet shown to be solar.
- **Other effects:**
  - The fitted rotation rate differs from the Snodgrass & Ulrich magnetic rate by
    −0.18 − 0.29 sin²(latitude) deg/day. Correcting it gains only 1.4 %.
  - Contrast "damping" gives nothing.
- **Fix in progress: a static background S,** solved from derotation residuals of training pairs (an
  estimate in the style of Kuhn–Lin–Loranz, using solar rotation as the known shift).
  - Background-aware derotation is rot(F − S) + S.
  - On the hold-out run it gains 18 % over B1-avg at 3.8 h.
  - Models retrained on background-aware inputs are being tested now.
- **Bugs found and fixed:**
  - a limb-darkening profile truncated by a fixed brightness window;
  - an incremental manifest rebuild that silently dropped a header column;
  - thermal-guard gaps.

## 6. Repository map

### Top level
| File | Purpose |
|---|---|
| `.gitignore` | Keeps generated data out of git: `outputs/`, `data/`, `*.fits`, `*.zarr/`, caches. |
| `requirements.txt` | Pinned dependencies: astropy, numpy, pandas, pyarrow, scipy, scikit-image, OpenCV, sunpy, zarr, matplotlib, torch. Known gaps: pytest is missing, and the CUDA torch build needs the PyTorch package index. |

### `configs/`
| File | Purpose |
|---|---|
| `phase1.toml` | Paths (share = `raw_root`, local-copy fallback, `out`), worker count, limb-fit settings, QC thresholds (spikes, seam), calibration choice (fixed pattern high-passed at 8 px), registration grid (1536², r_ref = 690, smoothing window). |
| `phase2.toml` | Dataset rules: offset pointing only, QC flags that exclude a frame, drop first frame of each block; split dates (train/val/test); run break at gaps > 300 s; context lengths and horizons to index. |

Phase 3 has no config file yet. Its settings are constants in the scripts; this is a known item to
fix.

### `docs/`
| File | Purpose |
|---|---|
| `DESIGN.md` | Phase 0 technical design: literature, data audit, architecture critique, experiment plan, open decisions. Its status line is outdated (still "Phase 0"). |
| `PHASE1.md` | Phase 1 report: audit, QC, registration study, EDA. |
| `PHASE2.md` | Phase 2 report: data source, pointing modes, fixed-pattern calibration, seam and large-scale response (negative self-calibration result), noise floor and oscillation, pointing-response correction, normalisation, baselines, resolution study, skill ceilings (with Phase 3 correction notes), dataset v0, reproduce commands. |
| `PHASE3.md` | Phase 3 report (provisional): setup, first result, why skill grows with horizon, what survives, corrections to Phase 2, next steps. |
| `PROJECT_BRIEF.md` | This file. |

### `suitdyn/` (the Python package; reusable logic)
| File | Purpose |
|---|---|
| `__init__.py` | Package marker. |
| `config.py` | Loads `configs/phase1.toml` (+ `phase2.toml`), records config hashes and the git state (commit, dirty) in every output's `_meta`, resolves the output folder. |
| `io.py` | Reads a SUIT FITS file (data + header) and its scale. |
| `filters.py` | Table of the SUIT science filters (names, wavelengths) from Tripathi et al. 2025. |
| `manifest.py` | Builds the raw manifest: one row per FITS file with its full header, SHA-256, parsed file name and frame type. Incremental (reuses unchanged rows); skips files still being written. |
| `geometry.py` | Solar-limb fitting independent of the header: edge points along rays, circle fit plus position-angle harmonics (Level-1 limbs are distorted). |
| `qc.py` | Per-pixel artefact mask (spikes, seam, off-limb, clipped, no source) and per-frame quality statistics. |
| `motion.py` | Frame-to-frame image motion by phase correlation, after removing the detector's fixed pattern, which would otherwise lock the correlation at zero shift. |
| `solar.py` | Solar differential rotation (Snodgrass & Ulrich 1990 magnetic rate, converted to synodic as seen from L1) and the expected disk-centre motion in pixels. |
| `register.py` | Registration transform following the FITS WCS (CROTA2, CDELT): maps a frame onto the common disk-centred, north-up grid, using the fitted limb centre and radius instead of the header's. |
| `flat.py` | Estimates and corrects the NB03 detector fixed pattern from the frames themselves: median of the relative residual per detector pixel, with smoothing kept from leaking across the seam. |
| `largescale.py` | Self-calibration model of the large-scale detector response (log response splines, separate across the seam, plus a limb-darkening profile and per-frame levels). Diagnostic only: it failed validation and is not used. |
| `response.py` | First-order pointing-response correction: per-pixel sensitivity of brightness to pointing, built on training data and applied as a factor per frame. |
| `normalize.py` | Normalisation variants (global, per-frame median (adopted), robust percentile, quiet-Sun contrast) and the μ map. |
| `sequences.py` | Leakage-safe splits (boundaries only inside gaps), (context, horizon) window index, test-split seal, multi-filter burst snapshots. |
| `store.py` | Zarr training store: calibration, per-frame processing into the registered grid, native QC mask, provenance (file hashes, commit). |
| `baselines.py` | Forecast baselines on the registered grid: heliographic coordinates, derotation coordinates (optionally with a fitted rotation-rate offset), B0 persistence, B1 rotated persistence, B2 optical-flow extrapolation (dropped). |
| `metrics.py` | Error metrics on valid pixels: MAE, RMSE, PSNR, SSIM, gradient correlation, bright-region scores. |
| `ml/__init__.py` | Marks the Phase 3 ML subpackage. |
| `ml/models.py` | The two forecasters of the B1 residual. UNetSmall stacks frames as channels (differences to the last frame, last frame, mask, μ, horizon). ConvLSTM processes the frames as a sequence. Both have zero-initialised output, so they start as B1. |

### `scripts/` (entry points, in pipeline order)
| File | Phase | Purpose |
|---|---|---|
| `phase0_audit.py` | 0 | First audit of an archive: inventory of every file, contiguous NB03 segments, pixel statistics of a sample. |
| `build_manifest.py` | 1 | Builds `outputs/phase1/manifest.parquet` from the share; adds local-copy rows for files not yet on the share (`source` column) and checks checksum conflicts. |
| `process_frames.py` | 1 | Per-frame measurements: limb fits, artefact counts, seam profiles, image statistics, NB03 frame-to-frame motion, spike persistence. |
| `registration_study.py` | 1 | Registration study and the adopted per-frame transforms (pointing modes, segments at jumps, limb anchor, smoothing), validation figures and the frame-level QC decision. |
| `eda.py` | 1 | Exploratory tables and plots: per-filter audit, timelines, cadence, intensity, pointing, geometry, artefacts, sample frames. |
| `build_sequences.py` | 2 | Dataset v0: frame list with exclusions, time splits, window index, test seal, burst snapshots, summary. |
| `phase2_calibration.py` | 2 | Fixed-pattern study: stability across halves/days/instrument change, additive vs multiplicative; writes a residual cache. |
| `phase2_calibration_followup.py` | 2 | Settles the open calibration questions (which spatial scales are a detector pattern, …) and writes the adopted pattern `nb03_pattern_adopted.npy` reproducibly. |
| `build_store.py` | 2 | Builds a Zarr store (`v0raw` without pattern correction, `v0` with it) of calibrated, registered frames; parallel and resumable. |
| `phase2_seam.py` | 2 | Seam study (whole-side gain vs local step). Its east–west mirror test turned out invalid because vignetting dominates; this is documented. |
| `phase2_noise_maps.py` | 2 | Maps of where the instrument sets the one-frame error: noise floor, per-pixel pointing sensitivity, trusted region. |
| `phase2_response.py` | 2 | Builds the pointing-response correction from training data and validates it on the validation split. |
| `phase2_floor_origin.py` | 2 | Structure function of the one-frame error; found the ~4–5 min chromospheric oscillation dip. |
| `phase2_largescale.py` | 2 | Large-scale self-calibration attempt and its validation (negative result, not adopted). |
| `phase2_baselines.py` | 2 | Noise floor and baselines B0/B1/B2 across horizons and normalisation variants, with block-bootstrap intervals; resolution study. |
| `verify_dataset.py` | 2 | Checks that every frame of a dataset is present and byte-identical in the current archive (via manifest SHA-256). |
| `sync_archive.py` | — | One-way checksum-verified mirror of the share to a local copy. Written but unused: we read the share directly. |
| `phase3_prepare.py` | 3 | Phase 3 cache: 384² frames (response-corrected, median-normalised) and samples (5 derotated context frames + target) for horizons 20/40/80/160, split into train / hold-out (last training run) / val. |
| `phase3_train.py` | 3 | Trains one model (`--model unet\|convlstm --seed N [--inputs plain\|bg]`): masked L1 on the B1 residual, AdamW + OneCycle, bf16, early stopping on the hold-out; resumable checkpoints; thermal guard. |
| `phase3_evaluate.py` | 3 | First evaluation on validation (B1, B1-avg, LD and blur variants, linear trend, models, seed ensembles) by region with block-bootstrap intervals. Stale: it predates the limb-darkening fix and needs a rebuild around the background-aware baseline. |
| `phase3_why_skill.py` | 3 | Diagnostic study of why skill grows with horizon: rotation-rate fit, damping, static mean-residual map, limb-darkening and background variants, background S baseline, scale decomposition, mean correction maps; CPU only. |
| `phase3_background.py` | 3 | Solves the static background S from derotation residuals of training pairs (sparse least squares with a gradient penalty λ chosen on the hold-out). |
| `phase3_prepare_bg.py` | 3 | Background-aware samples: each context frame derotated as rot(F − S) + S; checks that without S it reproduces the plain samples. |

### `tests/` (pytest; 21 tests, all for Phase 1–2 code)
| File | What it checks |
|---|---|
| `test_baselines.py` | Disk-centre rotation speed; pixels rotating in from behind the limb are invalid; B1 beats B0 on a synthetic rotating Sun; metric properties. |
| `test_flat.py` | Seam-aware smoothing does not leak across the seam; a multiplicative pattern is recovered from a moving scene. |
| `test_geometry.py` | Harmonic limb fit recovers the centre of a distorted, truncated disk; a plain circle is biased by partial coverage. |
| `test_largescale.py` | The self-calibration recovers a synthetic response with a seam step from two pointings. |
| `test_manifest.py` | Incremental rebuild keeps every column; files still being written are skipped. |
| `test_motion.py` | Fixed-pattern removal recovers a sub-pixel shift. |
| `test_qc.py` | Spikes found without flagging real features; seam step measured. |
| `test_register.py` | Transform matches astropy WCS; registration centres the disk and normalises the radius. |
| `test_sequences.py` | Split boundaries inside runs are refused; windows never cross runs or splits; test split sealed; bursts list every filter. |

## 7. Generated outputs (`outputs/`, not in git)

| Folder | Contents |
|---|---|
| `phase0/` | Audit tables. |
| `phase1/` | `manifest.parquet`; per-frame measurements; `registration.parquet`; EDA figures; `calibration/` (adopted fixed pattern). |
| `phase2/` | `sequences/` (dataset v0), `stores/` (Zarr v0raw, v0), `noise_maps/`, `response/`, baseline results. |
| `phase3/cache/` | 384² frame cache; samples `X_384.npy` (plain) and `X_384_bg.npy` (background-aware); targets `Y_384.npy`; sample metadata; μ map; trusted region. |
| `phase3/runs/` | One folder per training run (`best.pt`, `last.pt`, `log.csv`, `run.json` with provenance), plus logs. |
| `phase3/eval/`, `phase3/why/`, `phase3/background/` | Evaluation tables, diagnostics, figures, the static background S. |

## 8. Known weaknesses and next steps

1. **Adopt the background-aware B1 as the reference.**
   - Score it on validation.
   - Finish the quick retraining test on background-aware inputs.
2. **Engineering clean-up:**
   - a `configs/phase3.toml`;
   - move the shared code out of scripts into the `suitdyn` package (scripts currently import other
     scripts);
   - tests for the Phase 3 pieces;
   - a README and fixed requirements.
3. **Better background:**
   - S fixed on the detector rather than the registered grid; the vignetting is detector-fixed and
     pointing drifts ~8 px/day at 384².
   - Validate it with the centred-mode frames.
4. **Rebuild the final evaluation:**
   - strongest baselines and seed ensembles;
   - a plage definition not biased by vignetting;
   - negative controls (content shuffle, co-rotation).
5. **Recompute the Phase 2 numbers** (error growth, ceilings) against the background-aware B1.
6. **Dataset v1:** add 26 Sep and later as the download proceeds.
7. **One end-to-end pipeline command and an in-repo detached launcher,** then the final full run.
8. **Later:** Pipeline B (multi-filter snapshots).

## 9. Glossary

| Term | Meaning |
|---|---|
| NB03 | SUIT narrow-band filter at Mg II k 279.6 nm (chromosphere). |
| B0 / B1 / B1-avg | Persistence / persistence rotated by solar differential rotation / mean of the K context frames each rotated to the target time. |
| Horizon H | Frames ahead: 20 / 40 / 80 / 160 ≈ 28 min / 57 min / 1.9 h / 3.8 h at 87 s cadence. |
| μ | Cosine of the angle from disk centre (1 at centre, 0 at the limb). Limb darkening is brightness vs μ. |
| Hold-out | The last training run, used only for early stopping and for tuning baselines. |
| M(H) | Mean residual (target − B1-avg) over the training pairs of horizon H: the static part of the error. |
| S | Static background solved from rotation residuals; background-aware derotation is rot(F − S) + S. |
| Trusted region | Disk pixels with low pointing sensitivity (excludes the seam band and the vignetted edge; 92 % of the disk). |
| Skill | 1 − MAE_method / MAE_reference, per window; median with 95 % intervals from a bootstrap over (run, hour) blocks. |
