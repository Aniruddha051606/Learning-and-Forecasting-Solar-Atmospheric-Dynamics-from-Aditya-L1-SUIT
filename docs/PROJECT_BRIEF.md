# SUIT-DYN — project brief (for a reader new to the project)

Status as of 2026-09-28 (pipeline rebuilt; all generated outputs cleared for a clean final run). This brief summarises the project, its findings so far, and what every file
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
- **Data volume so far:** 19 → 26 Sep 2026 on the share (~55,600 files). Centred pointing until 23 Sep
  ~05:00 UT, offset pointing after. That is about a quarter of one solar rotation, so **no claim about
  generalisation is possible yet**.

## 2. Working rules (the project's scientific stance)

- **No random splits.** Splits are by time, cut only inside observing gaps. The **test split is
  sealed**: reading it needs an explicit unseal, which has never been done, and a rebuild can never
  silently replace the seal.
- **Self-contained data sets.** Registration, QC statistics, calibration and every later product use only
  frames inside the data set's own time span; the calibration uses only its training split.
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
  `//192.168.1.2/DATA/pradan1.issdc.gov.in/al1/protected/downloadData/suit/level1` (read-only for the
  pipeline, still being downloaded into). Every file of the old local copy (`D:/Data/...`) is on the share.
- **Outputs:** everything generated goes to `outputs/`, which git ignores and the pipeline rebuilds.
- **Thermal safety:** the laptop shut down once under full GPU load. Training uses a duty-cycle
  controller (target 75 °C, hard stop 85 °C); this laptop sustains only ~6-10 % GPU duty, so training
  is slow.
- **Detached runs:** `python -m suitdyn run ... --detach` restarts the runner outside the session (WMI,
  new process group, no window).

## 4. Pipeline and data flow

One command, `python -m suitdyn run --dataset <name>`, runs every stage in order (`suitdyn/pipeline.py`):

```
raw FITS (share)
 ─ archive (frame-local, grows as data sets need it) ─────────────────────────────────────────────
   manifest: headers + SHA-256 of the files in the data set's span   (scripts/build_manifest.py)
   frames:   per-frame limb fit, QC, seams, image motion             (scripts/process_frames.py)
 ─ data set <name> (only frames inside its span) ──────────────────────────────────────────────────
   registration → sequences (splits, windows, sealed test) → calibration (fixed pattern, training
   split) → store (calibrated, registered 1536² frames) → noise maps → pointing response → trusted region
   → samples (384² frame cache + sample index) → background S → train:<model>:<seed> ... → evaluate
```
- **Fingerprints:** each stage has one, covering its code, configs, inputs and upstream stages. A stage
  is skipped when up to date and reruns when anything it depends on changed.
- **Crash and failure handling:** interrupted stages resume (store, training). Only transient
  network/share failures are retried.
- **Preflight checks:** disk space, share, GPU, and a lock against two runners on one data set.
- **Smoke mode:** `--smoke` runs a quick end-to-end test whose outputs are kept apart from the real ones.
- **On-the-fly samples:** assembled on the GPU from the frame cache (`suitdyn/ml/data.py`). Each context
  frame is derotated whole (plain B1) or as rot(F − S) + S (background-aware B1).

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
- **Static background S:** solved from derotation residuals of training pairs, an estimate in the
  style of Kuhn–Lin–Loranz with solar rotation as the known shift. Background-aware derotation is
  rot(F − S) + S. In offset mode it gains 18 % over B1-avg at 3.8 h on the hold-out run and 18.3 % on
  validation.
- **Instrument control, the centred pointing mode (data set c0):**
  - With no vignetting, the static background is worth only 2.5 % at 3.8 h.
  - The model's skill no longer grows with horizon.
  - What it adds over the strongest static baseline is small and flat: 1.0–1.8 % on the disk, and
    about 0.5–1.5 % in plage, where the two shortest horizons are not significant.
  - Whether even that is solar is open: the negative controls are now built into the evaluation.
- **Reproducibility finding:** registration run over a larger archive changed an existing data set (up
  to 0.8/1.3 px, 13 new limb outliers). Everything is now scoped per data set.
- **Bugs found and fixed:**
  - a limb-darkening profile truncated by a fixed brightness window;
  - an incremental manifest rebuild that silently dropped a header column;
  - thermal-guard gaps;
  - in the rebuild, a loop variable that overwrote the output folder, and retries of deterministic errors;
  - the background S is unreliable in the outer disk ring (r > 0.9), so all metrics now exclude it.

## 6. Repository map

### Top level
| File | Purpose |
|---|---|
| `README.md` | The question, enforced rules, how to run, pipeline stages, layout. |
| `requirements.txt` | Pinned dependencies, including pytest and the dashboard build tools; torch from PyTorch's CUDA 13.0 wheel index. |
| `.gitignore` | Keeps generated data and build products out of git: `outputs/`, `*.fits`, `*.zarr/`, `dashboard/build`, `dashboard/dist`. |

### `configs/`
| File | Purpose |
|---|---|
| `phase1.toml` | Archive paths (share), workers, limb fit, QC thresholds, calibration choices (additive pattern high-passed at 8 px, frames used, split-half quality gate), registration grid (1536², r_ref = 690). |
| `datasets/<name>.toml` | One data set: pointing mode, excluded QC flags, first-frame rule, scope margin, time splits (train/val/test), windows. `v0` offset 23-25 Sep, `c0` centred 19-22 Sep, `smoke` ~1.5 days for quick tests. |
| `phase3.toml` | Learning: samples (grid, context, horizons, hold-out), background (lambda scan, fit/score radii), models, training (inputs plain/bg, seeds, epochs, optimiser), thermal controller, evaluation (regions, plage rule, baselines, bootstrap), negative controls. |

### `docs/`
| File | Purpose |
|---|---|
| `DESIGN.md` | Phase 0 design: literature, data audit, architecture critique, plan (status line outdated). |
| `PHASE1.md`, `PHASE2.md`, `PHASE3.md` | Phase reports; PHASE2 carries Phase 3 correction notes; PHASE3 §4b-4c the background and centred-mode control. |
| `PROJECT_BRIEF.md` | This file. |

### `suitdyn/` (library)
| File | Purpose |
|---|---|
| `__main__.py`, `pipeline.py` | `python -m suitdyn plan/run/status`: stage graph, fingerprints, skip/resume, preflight, lock, transient-only retries, keep-awake, smoke mode, detach. |
| `config.py` | Settings files, the current data set (`SUITDYN_DATASET`), data-set time span, provenance (git state, config hashes). |
| `paths.py` | Where every product lives (`outputs/archive`, `outputs/datasets/<name>/...`, `outputs/pipeline/<name>`). |
| `atomic.py` | Crash-safe writes (temporary file, then rename). |
| `progress.py` | Live heartbeats for the dashboard (never raises). |
| `manifest.py` | Raw manifest: header, SHA-256, frame type per file; incremental; span-limited. |
| `io.py`, `filters.py` | FITS reading; the SUIT filter table. |
| `geometry.py`, `register.py`, `motion.py`, `qc.py` | Limb fitting, registration transform, image motion with the fixed pattern removed, pixel/frame QC. |
| `flat.py` | Detector fixed-pattern estimation and correction. |
| `response.py`, `normalize.py` | Pointing-response correction; normalisation variants and the μ map. |
| `sequences.py` | Leakage-safe splits, windows, test seal, burst snapshots. |
| `store.py` | Zarr store of calibrated, registered frames with provenance. |
| `solar.py`, `baselines.py` | Differential rotation; B0/B1 (NumPy), heliographic geometry, optional rate offset. |
| `metrics.py` | Error metrics on valid pixels. |
| `largescale.py` | Large-scale self-calibration (research; failed validation, not used). |
| `ml/geometry.py` | Derotation on the GPU (tested against the NumPy version). |
| `ml/data.py` | The frame bank: samples assembled on the fly (plain and background-aware context). |
| `ml/background.py` | Static background S: mean residual maps, sparse solver, hold-out scoring. |
| `ml/models.py` | UNetSmall (212k parameters) and ConvLSTM (104k); zero-initialised output = B1. |
| `ml/thermal.py` | GPU temperature (NVML) and the duty-cycle thermal controller. |

### `scripts/`
Pipeline stages: `build_manifest.py`, `process_frames.py`, `registration_study.py`, `build_sequences.py`,
`calibrate_pattern.py`, `build_store.py`, `phase2_noise_maps.py`, `phase2_response.py`,
`phase3_prepare.py`, `phase3_background.py`, `phase3_train.py`, `phase3_evaluate.py` (each also runs alone
with `SUITDYN_DATASET=<name>`).
Checks: `verify_dataset.py` (frames byte-identical in the archive), `compare_registration.py` (a rerun
leaves a data set's registration unchanged).
Research studies from Phases 0-2 (not in the pipeline): `phase0_audit.py`, `eda.py`,
`phase2_calibration.py`, `phase2_calibration_followup.py`, `phase2_seam.py`, `phase2_floor_origin.py`,
`phase2_largescale.py`, `phase2_baselines.py`, `sync_archive.py` (unused).

### `tests/` (pytest, 36 tests)
Phase 1-2: baselines, flat, geometry, largescale, manifest (incl. span limits), motion, qc, register,
sequences. Pipeline and learning: `test_ml_geometry.py` (GPU derotation = NumPy), `test_ml_data_background.py`
(sample bank; the solver recovers a planted background), `test_ml_thermal.py` (controller with a fake
sensor), `test_pipeline.py` (skip/rerun logic, upstream propagation, selection, smoke separation, retry
policy).

### `dashboard/`
`suitdyn_dashboard.py` (two windows: pipeline control and live feed), `build_exe.cmd`, `make_icon.py`;
builds `dashboard/dist/SUIT-DYN Dashboard.exe`.

## 7. Generated outputs (`outputs/`, not in git)

| Folder | Contents |
|---|---|
| `archive/` | Manifest and per-frame measurements (frame-local, shared by every data set). |
| `datasets/<name>/` | `phase1/` registration, `sequences/`, `calibration/`, `stores/`, `phase2/` noise maps and response, `phase3/` (`cache/` frame cache and sample index, `background/`, `runs/`, `eval/`, smoke variants). |
| `pipeline/<name>/` | Runner state (one JSON per stage) and per-stage logs. |
| `logs/progress/` | Live heartbeats for the dashboard. |

## 8. Next steps

1. **Pass the smoke test:** the real-data end-to-end run on the `smoke` data set.
2. **Freeze the method:** run the negative controls on c0 and v0, and settle the background model
   (detector-fixed S; the outer ring).
3. **Data cut-off:** choose the final data sets (offset and centred mode, separately) and a cut-off date.
4. **The final run:** `python -m suitdyn run --dataset <name> --detach` for each final data set,
   including a one-time controlled unseal of the test split.
5. **Later:** the short-horizon oscillation experiment, and Pipeline B (multi-filter snapshots).

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
