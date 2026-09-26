# SUIT-DYN — Phase 2 report (in progress)

Data: SUIT Level-1, 2026-09-22 03:27 → 2026-09-25 23:30 UT; 23,172 files, 40.9 GB.
Dataset hash (manifest_sha256): `706c5c3e…`. Everything is produced by the scripts listed at the end.
This is still prototype data. Nothing here is evidence of generalisation.

---

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

_(results pending: `scripts/phase2_calibration.py`)_

## 3. Seam

_(pending: `scripts/phase2_seam.py`, run after the pattern correction)_

## 4. Normalisation experiment and baselines

_(pending: `scripts/phase2_baselines.py` on stores v0raw and v0)_

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
