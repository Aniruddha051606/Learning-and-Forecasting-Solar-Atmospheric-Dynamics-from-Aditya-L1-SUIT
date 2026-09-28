# SUIT-DYN

Learning and forecasting the solar chromosphere from **Aditya-L1 / SUIT** NB03 (Mg II k, 279.6 nm)
full-disk images: a rigorous, falsifiable pipeline from the raw Level-1 archive to evaluated models.

**Question.** Does a time-resolved SUIT Mg II k sequence contain predictable information about the future
of chromospheric structure beyond solar rotation and static instrumental/background effects? A "no" with a
measured ceiling is a result too.

**Status (2026-09-28).** Prototype data (days, not rotations). Every run so far is a code test; one full run
comes at the end. Findings so far are in `docs/PHASE1.md` … `docs/PHASE3.md`; the project brief is
`docs/PROJECT_BRIEF.md`.

## Rules the code enforces
- Splits by time, cut only inside observing gaps, with an **embargo** (default 4 h, longer than the longest
  horizon) between splits and before the hold-out run. The **test split is sealed** (hash + unseal log); a
  rebuild can never silently change it, and it is evaluated only with `--with-test`: the first read records
  the model set, and reading it again with different models is refused unless explicitly overridden (logged).
- **One pointing per data set:** the Level-1 background depends on where the Sun sits on the detector, so a
  data set that mixes pointing clusters is refused (`scripts/pointing_modes.py` lists them per day).
- Every data set is **self-contained**: registration, QC statistics, calibration and every later product use
  only frames inside the data set's time span, and the calibration only its training split.
- Models are scored against the **strongest** of several physical baselines, including a background-aware
  B1 that does not drag the static instrument background with the Sun, and must pass **negative controls**
  (shuffled context, a frozen Sun). All regions exclude the unreliable limb ring (r > 0.9).
- Every product records the git commit, a dirty flag, config hashes and the SHA-256 of its raw inputs.

## Run
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m suitdyn plan   --dataset c0            # what would run and why
python -m suitdyn run    --dataset c0 --detach   # the whole pipeline, detached from the terminal
python -m suitdyn status --dataset c0
python -m suitdyn run    --dataset smoke --smoke # quick end-to-end check on ~1.5 days of real data
python -m suitdyn run    --dataset c0 --with-test --detach   # the final run: also the one-time test evaluation
python scripts/pointing_modes.py --scan --from 2026-09-10 --to 2026-09-28   # pointings per day, before defining data sets
python scripts/thermal_benchmark.py --note "cooling pad"     # sustainable training speed on this laptop
python -m pytest tests
```
Data sets: `configs/datasets/<name>.toml` (`v0` offset pointing 23-25 Sep, `c0` centred pointing 19-22 Sep,
`smoke`). Learning settings: `configs/phase3.toml`. Archive paths: `configs/phase1.toml`.
Live view: `dashboard/build_exe.cmd` builds `dashboard/dist/SUIT-DYN Dashboard.exe` (pipeline window + live
feed of the SUIT file being processed).

## The pipeline (`suitdyn/pipeline.py`)
| Stage | Script | What |
|---|---|---|
| manifest | `build_manifest.py` | headers + SHA-256 of the raw files in the data set's span |
| frames | `process_frames.py` | per-frame limb fit, QC, seams, image motion (frame-local, incremental) |
| registration | `registration_study.py` | disk-centred, north-up transforms; frame QC decision |
| sequences | `build_sequences.py` | frame list, time splits, windows, sealed test |
| calibration | `calibrate_pattern.py` | detector fixed pattern from the training split (split-half gate) |
| store | `build_store.py` | calibrated, registered 1536² frames + QC masks (Zarr) |
| noise_maps, response, noise_maps_resp | `phase2_noise_maps.py`, `phase2_response.py` | pointing-response correction, trusted region |
| samples | `phase3_prepare.py` | 384² frame cache + sample index (samples assembled on the GPU on the fly) |
| background | `phase3_background.py` | static background S that derotation must not move |
| train:&lt;model&gt;:&lt;seed&gt; | `phase3_train.py` | UNet / ConvLSTM on the residual over the background-aware B1 |
| evaluate | `phase3_evaluate.py` | strongest baselines, block bootstrap, negative controls |

Each stage is skipped when its code, configs, inputs and upstream stages are unchanged, resumes after a crash,
and retries only transient (network/share) failures. Outputs: `outputs/archive`, `outputs/datasets/<name>`,
`outputs/pipeline/<name>` (state and logs); see `suitdyn/paths.py`. Laptop GPU safety: a duty-cycle thermal
controller (`suitdyn/ml/thermal.py`).

## Layout
`suitdyn/` library (data, geometry, calibration, baselines, ML, pipeline) · `scripts/` stage entry points
and Phase 1-2 research studies · `configs/` · `tests/` · `docs/` reports · `dashboard/` desktop monitor.
