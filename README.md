# SUIT-DYN

Forecasting full-disk Mg II k images (NB03, 279.6 nm) from **SUIT on Aditya-L1** 30 min to 4 h ahead, and
testing whether small neural networks beat physical baselines built from solar rotation.

**Question.** How much of the change in SUIT chromospheric images over the next few hours can be predicted beyond
what solar rotation and fixed instrumental structure already explain?

**Answer so far.** A little, and reliably: the networks beat the strongest of five physical baselines by about
1.5-3.5 % (mean absolute error) on sealed test data in both pointing modes. All rules were fixed in advance
(`docs/PREREGISTRATION.md`).

## How it works

### 1. Data
SUIT Level-1 NB03 full-disk FITS files from ISSDC PRADAN, read from a network share (`configs/phase1.toml`).
A **data set** is a time span in one pointing (`configs/datasets/<name>.toml`): `final_offset` (10-27 Sep 2026,
disk off-centre on the detector) and `final_centred` (18-23 Sep 2026). Mixing pointings is refused, because the
detector background moves with the pointing.

### 2. Pipeline (`python -m suitdyn run --dataset <name>`)
Each stage is skipped when its code, configs and inputs are unchanged, and resumes after a crash
(`suitdyn/pipeline.py`).

| Stage | Script | What it does |
|---|---|---|
| manifest | `build_manifest.py` | header and SHA-256 of every raw file |
| frames | `process_frames.py` | per-frame limb fit, quality checks, image motion |
| registration | `registration_study.py` | disk-centred, solar-north-up position of every frame |
| sequences | `build_sequences.py` | frame list; time splits (train / validation / sealed test) cut in observing gaps with a 4-h embargo |
| calibration | `calibrate_pattern.py` | fixed detector pattern, from the training split only |
| store | `build_store.py` | calibrated, registered 1536 x 1536 frames (Zarr) |
| noise maps, response | `phase2_noise_maps.py`, `phase2_response.py` | pointing-dependent response correction, trusted region |
| samples | `phase3_prepare.py` | 384 x 384 frame cache and the forecast windows (5 context frames, target 20/40/80/160 frames later) |
| background | `phase3_background.py` | static detector background S that rotation must not move |
| train | `phase3_train.py` | U-Net and ConvLSTM, 3 seeds each |
| evaluate | `phase3_evaluate.py` | validation; `--split test` reads the sealed test (logged, once per model set) |

### 3. Baselines and models
- **Baselines** move the context frames to the target time with differential rotation: last frame (B1), mean of
  five frames, limb-darkening corrected, background-aware (`D(I - S) + S`, the detector background stays in place)
  and climatology. The reference for each region and lead time is the **strongest** of them.
- **Models** (`suitdyn/ml/models.py`): a U-Net (212 k parameters) and a ConvLSTM (104 k) predict a correction to
  the background-aware last frame; the ensemble is the mean of three seeds. Their validity mask is built from the
  input frames only.

### 4. Evaluation
- Skill = 1 - MAE(model) / MAE(strongest baseline), median over windows, 95 % intervals from a block bootstrap
  over observing runs and hours (`scripts/phase3_evaluate.py`).
- Regions: full disk (r < 0.9), trusted area, plage, three rings.
- Pre-registered criteria: P1 interval above zero, P2 also with whole days resampled, P3 in both pointings;
  controls C1 (a frozen Sun gives no skill) and C2 (the correction is not one fixed map).

### 5. Pre-registered analyses (`docs/PREREGISTRATION.md`, run after the main evaluation)
| Addendum | Script | Question |
|---|---|---|
| B | `sealed_forecast.py`, `answer_frames.py`, `score_sealed.py` | blind forecasts sealed before the verifying images existed |
| C | `posthoc_fixed_removal.py` | is the advantage the fixed detector map? |
| D | `posthoc_shuffle_denoise.py` | a correction from another time? plain denoising? |
| E | `posthoc_classical.py` | does a tuned classical forecaster do as well? |
| F | `posthoc_state_estimate.py` | forecast of evolution, or a cleaner picture of the present? |
| G | `posthoc_new_period.py` | 28 Sep - 6 Oct 2026, data no model had seen |
| H | `posthoc_mask_check.py`, `posthoc_mask_difference.py` | did the validity mask leak the target? |
| S1, S2 | `posthoc_flow_baseline.py`, `posthoc_cross_mode.py` | optical-flow baseline; models used across pointings |

`posthoc_report.py` collects everything into `outputs/tests/report.json`.

### 6. The leak and the v2 retraining (Addenda H and I)
The first models' validity mask also used the target frame. The offset U-Net relied on it. All 12 models were
retrained with an input-only mask (`scripts/retrain_v2.py`); the old models and results are kept as
`*_v1_targetmask`. `scripts/after_v2.py` then repeats the new-period test and every analysis on the retrained
models and rebuilds the paper numbers.

### 7. Paper
`scripts/paper/make_suit_paper.py` (tables and figures), `make_tex.py` (every number of the manuscript as a LaTeX
macro), `check_tex.py` and `verify_claims.py` (each stated conclusion tested against the numbers). They write to
`paper/`, which is not part of this repository.

## Running it
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m suitdyn plan   --dataset final_offset      # what would run and why
python -m suitdyn run    --dataset final_offset --detach
python -m suitdyn status --dataset final_offset
python -m suitdyn run    --dataset smoke --smoke     # quick end-to-end check
python -m pytest tests
```
- **Laptop GPU:** a duty-cycle controller keeps the GPU near its target temperature (`[thermal]` in
  `configs/phase3.toml`, `suitdyn/ml/thermal.py`). Run one GPU job at a time.
- **Dashboard:** `dashboard/build_exe.cmd` builds `SUIT-DYN Dashboard.exe` (pipeline, live feed, results,
  retraining).
- **Tools:** `tools/buzzer` (status alerts on the data machine), `tools/data_terminal` (data-machine console),
  `tools/make_nb03_video.py`.

## Layout
```
suitdyn/      library: I/O, geometry, registration, calibration, baselines, metrics, pipeline; ml/ data, models, thermal
scripts/      pipeline stages, studies, pre-registered analyses, retraining, paper/ scripts
configs/      archive paths, learning settings, data sets
tests/        unit and end-to-end tests
dashboard/    desktop monitor
docs/         PREREGISTRATION.md: every rule and analysis, written before it was run
outputs/      (not in git) archive/, datasets/<name>/, pipeline/<name>/, logs/: rebuilt from the raw data and this code
```
Every product records the git commit, config hashes and the SHA-256 of its inputs.
