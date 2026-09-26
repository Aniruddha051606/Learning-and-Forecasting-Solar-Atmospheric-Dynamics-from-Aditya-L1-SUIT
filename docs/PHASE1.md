# SUIT-DYN — Phase 1 report: audit, QC, registration, EDA

Data: SUIT Level-1, **2026-09-23 18:00 → 2026-09-25 23:30 UT (≈ 2.2 days)**, 17,252 FITS files, 30.5 GB.
Every number below is produced by the scripts listed at the end. It is a prototype data set:
it is enough to understand the instrument and to build the pipeline, and **not** enough to support any
claim about generalisation.

---

## 1. Data audit

### 1.1 What is in the archive

| Frame type | Filters | Frames | Cadence (median / p95) | Largest gap | Volume |
|---|---|---|---|---|---|
| Full disk, 2×2 binned, 2048² (1.40″/px) | **NB03 only** | 1,772 | **87.3 s / 89.6 s** | 3.0 h | 14.9 GB |
| Full disk, unbinned, 4096² (0.70″/px) | all 11 | 14 each (NB04: 28) | 2.3 h (one burst) | 7.9 h | 5.2 GB |
| ROI 560² (0.70″/px), operator-chosen | 9 (not BB01, BB03) | ~1,700 each | 87.5 s | 3.0 h | 9.9 GB |

- **Read errors:** 0. **Filter code in the file name vs `FTR_NAME`:** 0 mismatches.
- **Calibration:** one flat field (`master_flat_2024-05-24`) and one version (`V1`) across the archive.
- **Quality keywords:** `QVAL = 100`, `QDESC = "Complete Image"` and `NSPIKES = 0` on every file.
  They carry no information and are kept as metadata only.
- **Flare flags** (`SOLX1TR`, `SOLX2TR`, `HELIOSTR`, `NORM_FLR`, `PROM_FLR`) are 0 everywhere: no flare
  mode in these days.
- **Multi-filter "snapshots" are bursts, not simultaneous observations.** All 11 filters are taken one
  after another within ~5 min, and NB04 is taken twice (start and end). There are 6 bursts per day,
  and none during program 147 (00:08–02:18 UT).
- **The daily schedule repeats.** Program 151 runs 17:06–23:30, program 147 runs 00:08–02:18 and
  program 139 runs 03:27–14:07 UT. The gaps between them fall at the same UT times every day.
- **NB03 continuous runs** (a gap over 5 min ends a run): 10.7, 9.5, 6.4, 6.4, 4.5, 2.2, 2.2 h, plus
  fragments.

### 1.2 Usable / invalid frames (Phase 1 QC, §3)

All 1,940 full-disk frames pass the reject rules: no failed limb fit, and at least 0.9 of the disk
is on the CCD in every frame. **Frames flagged but kept:**

| Flag | NB03 frames | Meaning |
|---|---|---|
| `pointing_jump` | 51 | 5–9 px discrete pointing change since the previous frame |
| `brightness_jump` | 19 | disk median > 8 robust σ from its 30-min running median |
| `limb_outlier` | 12 | limb-fit centre far from the adopted pointing |
| `spike_rate` | 3 | unusually many spikes |

**Frames that must be rejected before training** (found in the EDA, rule to be added in Phase 2):
the **first NB03 frame of every program 147 and 151 block is 16–19 % too bright**. That is an
instrumental warm-up or exposure effect, not the Sun.

### 1.3 Storage and cost

The rate is about 12 GB/day for everything, of which binned NB03 is 6.8 GB/day and ROI about
4.6 GB/day. D: had 105 GB free at the last check. The per-frame processing (limb fits, masks,
statistics) took 44 min for 17k frames on 6 CPU workers.

---

## 2. Instrument artefacts (measured, not assumed)

| # | Artefact | Size | Evidence | Handling |
|---|---|---|---|---|
| A1 | **Pointing jitter** | ~1.2 px (x) / 1.5 px (y) rms **per frame** | image motion and limb agree, r = 0.99 (x) | registration (§4) |
| A2 | **Pointing oscillation** | ±6 px x, ±10 px y, period ~1.5–2 h | circle fit, header and image motion all show it | registration |
| A3 | **Pointing jumps** | 5–9 px, 51 in 2.2 days | phase correlation (reliable above 1 px) | registration; windows are not broken |
| A4 | **Fixed detector pattern** in NB03 | rms 126 counts ≈ **4 % of the disk level**; split-half r = 0.98 | median of 240 high-passed frames in detector coordinates | **must be corrected before training (Phase 2)**; removed for motion estimation already |
| A5 | **Vertical quadrant seam** at x = 1023/1024 (binned) | **−20 to −28 %** in the southern rows, 0 near row 1000, +3 % above; plus a +4 % strip at x = 1020–1023 | per-frame seam profiles vs control columns | masked (±2 px + strip); correction decided in Phase 2 |
| A6 | Horizontal seam at y = 2047/2048 in unbinned frames; seam and ghost ring in NB08 | visible | EDA fig. 08 | masked; NB08 ghost region to be masked |
| A7 | **Program-dependent level** | 147: −1 %, 151: −2.7 % vs 139 (disk median) | per-block medians, repeated on 2 days | normalisation experiment (Phase 2) |
| A8 | **First frame of program blocks** | +16 to +19 % | per-block first frames | reject |
| A9 | Off-limb over-subtraction | median −300 to −450 counts; **83 % of off-limb pixels < 0** (binned) vs 0–25 % in unbinned frames | off-limb statistics | off-limb excluded (`OFF_LIMB` bit) |
| A10 | Limb distortion | ±5–15 px harmonic departures from a circle; the pattern changes with time | limb residuals | not corrected (Level 2 does it); a circle on a fixed ray set keeps the bias constant |
| A11 | **Instrument-state change**, 24 Sep ~12–18 UT | disk radius +2 px (697 → 699), disk on CCD 0.97 → 0.95, usable limb 0.72 → 0.64, distortion pattern changed | EDA fig. 06, registration fig. | recorded per segment; a covariate and a candidate split boundary |
| A12 | Spikes | ~35 per million px (binned), ~10 (unbinned); 0 of the checked spikes persist 87 s later | spatial detector + temporal persistence | masked (`SPIKE` bit) |
| A13 | Clipped pixels | encoding floor −2768 and ceiling 62767 (BZERO 30000); 0–10 px/frame at the ceiling | exact values | masked |
| A14 | Burst photometric scatter | ±1–3 % per filter between bursts, not coherent across filters | EDA fig. 04 | limits any multi-filter ratio (e.g. an Mg II index); quantify before M3 |

A4 is new and important. The pattern **looks like chromospheric network**: 20–30 px cells with dark
lanes, fixed on the detector over three days while the real network rotated hundreds of pixels.
**Hypothesis:** it is solar structure imprinted by the 2024 master flat, which was built from
off-pointed solar images (Sarkar et al. 2025). We cannot confirm this without the flat file. Whatever
its origin, in registered sequences it moves with the ~1.4 px pointing jitter, so an uncorrected
model would see a static "network" that shimmers from frame to frame.

---

## 3. Scientific QC mask

Per-pixel bits (`suitdyn/qc.py`):

| Bit | Name | Rule |
|---|---|---|
| 1 | `CLIP_LO` | value at the lowest encodable value (BZERO + BSCALE·−32768) |
| 2 | `CLIP_HI` | value at the highest encodable value |
| 4 | `SPIKE` | residual from a 5×5 median > 10 robust σ and > 0.5 × local level, connected area ≤ 6 px |
| 8 | `SEAM` | vertical seam ± 2 px plus the bright 4-px strip before it; horizontal boundary ± 2 px |
| 16 | `OFF_LIMB` | r > 1 of the fitted limb |
| 32 | `EDGE` | within 40 px (binned) of the detector edge (vignetting) |

The spike detector is tested on synthetic data: > 95 % of injected spikes are recovered, and < 2 % of
4-px bright features are flagged. On real data none of the checked spikes persist into the next frame.

Frame QC (`scripts/registration_study.py::qc_decision`): reject on a failed limb fit or < 0.9 of the
disk on the CCD. Flag on robust z > 8 of spike count, brightness jump or limb outlier, and on pointing
jumps. **Phase 2 adds:** reject the first frame of a program block.

---

## 4. Registration

### 4.1 Method adopted

For each NB03 frame:
1. **Image motion.** Phase correlation of a high-passed detector box (inside the disk, clear of the
   seam), **with the fixed pattern subtracted**. Every frame is correlated with its nearest keyframe
   (one every 10 frames). Keyframes are linked to each other. Predicted solar rotation (Snodgrass &
   Ulrich 1990, disk-centre rate) is removed. Frames are never chained one to the next.
2. **Absolute anchor.** Each jump-free segment is shifted so that its positions agree on average with
   a plain circle fit to the limb. The circle is fitted on the same 450 of 720 rays in every frame,
   so the distortion bias is the same in all of them.
3. **Radius.** Robust mean of the circle radius per segment, scaled by the Sun–spacecraft distance.
4. **Transform to the common grid.** 1536², disk centre at the grid centre, r = 690 px, solar north
   up via CROTA2. It follows the FITS WCS convention and is tested against astropy to < 0.05 px. It
   is a similarity transform only; no warping. It is stored per frame in `registration.parquet`
   with its provenance.

The unbinned burst frames use their own circle fit, on their own common ray set.

### 4.2 Validation

60 frame pairs, each straddling a pointing jump so that the two frames have independent anchors. The
measured image motion minus rotation is compared with the difference of the adopted centres:

| Method | rms x | rms y | median \|error\| |
|---|---|---|---|
| **Adopted** (image motion + segment anchor) | **0.51 px** | **0.90 px** | **0.64 px** |
| Per-frame circle fit | 0.43 px | 1.22 px | 0.78 px |
| Smoothed circle fit (first attempt, rejected) | 1.35 px | 1.67 px | 1.64 px |
| Header CRPIX | 1.07 px | 2.80 px | 1.98 px |

These errors are upper bounds: they include the error of the validation measurement itself. Against
the adopted centre, the per-frame circle differs by 0.24 px (x) and 0.55 px (y) rms.

### 4.3 What went wrong on the way (kept on record)

1. **A limb model with m = 2–4 distortion harmonics** passed the synthetic tests but, on real frames,
   gave a centre 3–4× noisier than a plain circle, because the harmonics trade off against the
   centre over a 65–75 % limb. **Rejected.**
2. **Chaining consecutive-frame phase correlation** gave exactly 0 median motion, where solar rotation
   alone predicts 0.17 px per frame. The correlation locks on the fixed pattern (A4) whenever the
   true shift is sub-pixel, so the chain accumulated a fake drift of about −60 px per 10-h run.
   **Rejected;** that finding led to A4.
3. **Smoothing the circle-fit centre** assumed its frame-to-frame scatter was noise. It is real
   jitter (A1): image motion and limb agree to r = 0.99 in x. **Rejected**, as the validation shows.

### 4.4 Consequences for forecasting

- **Frame-to-frame registration error is about 0.5 px (x) and 0.9 px (y) at 1.4″/px.** That is
  larger than the ~0.17 px a feature moves by rotation in one 87-s step. At horizons of 1–5 frames
  the persistence error will be dominated by registration and pattern noise, not by the Sun. This
  must be measured in the horizon study (Phase 2) before any short-horizon skill is interpreted.
- **y is worse than x** because the southern limb is off the CCD.

---

## 5. Pipeline architecture

Implemented stages are marked ✅; ◻ means Phase 2.

```
PIPELINE A (NB03, fast)                         PIPELINE B (multi-filter, slow)
raw FITS ─► manifest (header + SHA-256) ✅       same manifest ✅
   ▼                                               ▼
per-frame: limb circle fit on fixed rays ✅      per-frame circle fit (own ray set) ✅
   QC mask bits + frame metrics ✅                 QC mask bits + frame metrics ✅
   ▼                                               ▼
fixed-pattern estimate (box ✅, full frame ◻)    per-filter fixed pattern ◻ (needs more bursts)
   ▼                                               ▼
image-motion registration + anchor ✅            per-filter offsets vs NB03 pointing (measured ✅)
   ▼                                               ▼
program-level / first-frame handling ◻           burst grouping: S_k = {filter → obs, t_obs} ◻
   ▼                                               ▼
normalisation (experiment) ◻                     normalisation (same experiment) ◻
   ▼                                               ▼
Zarr store + sequence index ◻                   Zarr store + snapshot index ◻
```

**Pipeline B keeps asynchrony explicit.** A snapshot is the set of burst frames with each frame's own
`t_obs`. It is never snapped to a common time. A missing filter is an explicit mask entry, never a
filled value. The measured per-filter centre offsets relative to 2·NB03+0.5 are within
±5 px (x) and 0–15 px (y), with 2–7 px scatter. That scatter is limb-fit noise at 4096², so each
burst frame keeps its own fit.

---

## 6. Dataset schema (for the Phase 2 store)

```
suitdyn_v{N}.zarr/
  nb03/image        float16 [T, G, G]   chunks (1, G, G)   registered, fixed-pattern-corrected counts/s
  nb03/mask         uint8   [T, G, G]   chunks (1, G, G)   QC bits after resampling (255 = no source)
  nb03/time         int64   [T]         ns since epoch (actual t_obs)
  nb03/frame_id     str     [T]         → manifest.file
  burst/{FILTER}/image, mask, time, frame_id     same layout, T_filter entries
  burst/index       table: burst_id, filter, frame_id, t_obs, dt_from_burst_start
  grid              attrs: G, r_ref, plate scale, orientation (solar north up)
  attrs             pipeline git commit, config sha256, manifest sha256, registration.parquet sha256,
                    calibration files, normalisation variant
sequences/nb03_k{K}_h{H}.parquet   window index: start frame, context K, horizon H, split, dt of each
                                    step, flags (pointing_jump in window, program change in window)
```

- G is decided in Phase 2: 1536 native, or 768/512 after the resolution study.
- Raw FITS stay under `D:\Data\...` untouched. Nothing is deleted until a Zarr version has been
  rebuilt twice bit-identically and every source checksum re-verified. The deletion itself is the
  user's decision.

---

## 7. Phase 2 plan

1. **Fixed-pattern correction on the full NB03 frame.** Estimate in detector coordinates. Decide
   subtractive vs divisive by testing which one removes the pattern from registered difference
   images. Check stability across days and across the A11 state change.
2. **Seam:** test a row-dependent multiplicative correction against masking. Criterion: the
   registered difference images show no seam line.
3. **Reject block-first frames** (A8).
4. **Normalisation experiment:** global, per-filter, per-frame and robust-percentile, plus a
   μ-dependent quiet-Sun reference. Judged on two things: (a) how much of the program-level step
   (A7) remains, and (b) how much of the disk-integrated and regional variability *within* a block is
   preserved. That within-block variability is the signal we want to forecast.
5. **Zarr store and sequence generator** (NB03 windows; burst snapshots with explicit missing
   filters). Leakage tests: no frame in two splits; no window crosses a split boundary.
6. **Horizon discovery and baselines B0** (persistence), **B1** (rotation-corrected persistence) and
   **B2** (optical flow, if reliable). The **noise floor** comes first: the error between registered
   frames 1 step apart, after removing rotation, is the best any forecaster can do given the
   registration and pattern residuals.

**Prototype split.** Splits are cut at natural gaps, never inside a run:

| Split | Span | Frames |
|---|---|---|
| Train | 23 Sep 18:00 → 24 Sep 23:30 | 1,004 |
| Val | 25 Sep 00:08 → 14:07 | 506 |
| Test | 25 Sep 17:06 → 23:30 | 262 |

Limitations:
- The test set is one evening of one program (151).
- All of val and test fall after the A11 state change, while train spans both states.

Windows available (context 20 frames):

| Horizon | Train | Val | Test |
|---|---|---|---|
| 1 frame (1.5 min) | 908 | 461 | 242 |
| 10 frames (15 min) | 872 | 443 | 233 |
| 40 frames (58 min) | 752 | 383 | 203 |
| 80 frames (1.9 h) | 600 | 305 | 163 |
| 160 frames (3.9 h) | 360 | 225 | 83 |
| 240 frames (5.8 h) | 196 | 145 | **3 → not testable** |

Overlapping windows are not independent: the test set holds roughly six independent hours.

## 8. Experiment matrix (draft; horizons fixed after the Phase 2 noise-floor study)

| ID | Input | Target | Horizon | Model | Loss | Metrics | Purpose | Expected | Failure criterion |
|---|---|---|---|---|---|---|---|---|---|
| N0 | NB03 frame t | NB03 frame t+1 | 1 frame | none (noise floor) | — | MAE on valid pixels | Registration + pattern noise | ≈ pixel noise + residual pattern | — (defines the floor) |
| B0 | NB03 t | NB03 t+Δ | 1–160 fr | persistence | — | MAE, RMSE, SSIM, region error | Reference | error grows with Δ | — |
| B1 | NB03 t | NB03 t+Δ | 1–160 fr | rotation-corrected persistence | — | same | Removes known rotation | beats B0 for Δ ≳ 10 fr | B1 ≤ B0 at Δ ≥ 40 fr → rotation model or registration wrong |
| B2 | NB03 t−k..t | NB03 t+Δ | 1–80 fr | optical-flow extrapolation | — | same | Advection | beats B1 at short Δ, if at all | flow field dominated by noise → drop B2 |
| A1 | NB03 t−k..t | NB03 t+Δ | from N0/B study | ConvLSTM (small) | masked L1 | same + skill vs B1 | First learned model | small gain at intermediate Δ | CI of skill vs B1 includes 0 at all Δ |
| A2 | NB03 t−k..t | NB03 t+Δ | same | CNN/ViT encoder + temporal | masked L1 | same | Architecture comparison | ≈ A1 on this data size | no gain over A1 → keep the smaller one |
| M1 | NB03 history | NB03 t+Δ | 1–4 h | best of A1/A2 | masked L1 | same | Multi-filter reference | = A1/A2 | — |
| M2 | NB03 history + last burst (with its age) | NB03 t+Δ | 1–4 h | M1 + burst encoder | masked L1 | same, stratified by burst age | Does sparse multi-filter information help? | small or none | gain only for recent bursts → age confound |
| M3 | NB03 history | next burst NB02/04/05/08 | to next burst | M1 encoder + filter heads | masked L1 | per-filter error, cross-filter consistency | Cross-filter predictability (not causality) | NB04 ≈ NB03-like; continua harder | error ≥ burst photometric scatter (A14) → not measurable |
| M4 | fast + slow streams, time-aware | future multi-filter | multi | joint model | multi-task | all | Joint state | — | only if M2/M3 show signal |

## 9. Model recommendation — deliberately deferred

The data do not support recommending sequence length, resolution, patch size, encoder, parameter
count, batch size or learning rate yet. Each one depends on a Phase 2 measurement:

- **Horizons:** the N0 noise floor and where B1's error departs from it.
- **Resolution / patch size:** whether downsampling 1536 → 768 → 512 changes B1's skill curve.
- **Context length:** the autocorrelation of the B1 residual.
- **Model size:** the number of *independent* training hours (about 30 in the prototype), which
  already argues for small models (≲ 10M parameters) until the archive grows.

What is already fixed:
- losses are masked, on valid pixels only;
- the test split is used once;
- every result is reported as skill relative to B1 with a bootstrap over blocks of hours.

## 10. Reproduce

```
python scripts/build_manifest.py        # manifest.parquet (+ SHA-256)       ~4 min
python scripts/process_frames.py        # frames_full/roi, seam profiles      ~45 min, 6 workers
python scripts/registration_study.py    # registration.parquet, validation    ~10 min
python scripts/eda.py                   # eda/*.png, audit_filters.csv
python -m pytest tests                  # geometry, registration, QC, motion
```

Settings: `configs/phase1.toml`. Outputs: `outputs/phase1/`. Each JSON output records the git commit,
the dirty flag and the config SHA-256.
