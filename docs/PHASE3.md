# SUIT-DYN — Phase 3 report (in progress)

Data: dataset v0 (Phase 2 §5): train 23 Sep 05:01 → 24 Sep 23:30, validation 25 Sep 00:08 → 14:07,
test sealed (`3ca4a17d…`). Everything is read from the network share (PHASE2 §0).

**Status of the numbers.** Every run in this phase is a code test. The one full run (all seeds, final
settings, all data then available) is done at the end. The numbers below are provisional: they show
that the code works and what the problems are. They are not results to quote. Three days of data are
not evidence of generalisation.

---

## 1. Setup

- **Samples** (`scripts/phase3_prepare.py`):
  - Frames: 384² (5.64″/px), disk-centred, response-corrected, each normalised to its disk median.
  - Inputs: K = 5 consecutive context frames, each derotated to the target time (Snodgrass & Ulrich
    1990 magnetic rate, orthographic).
  - Horizons: 20 / 40 / 80 / 160 frames (28 min / 57 min / 1.9 h / 3.8 h).
  - Target: the residual from B1 (the last context frame derotated), so a model that outputs zero
    reproduces B1.
  - The context spacing is not uniform. Two context frames can be as close as 18 s, so "5 frames"
    spans 5–7 min.

| Horizon | train | early-stop hold-out (run 5) | validation |
|---|---|---|---|
| 20 | 542 | 236 | 449 |
| 40 | 502 | 216 | 409 |
| 80 | 422 | 176 | 329 |
| 160 | 300 | 96 | 239 |

- **Models** (`suitdyn/ml/models.py`):
  - UNetSmall: 212,273 parameters.
  - ConvLSTM: 103,969 parameters.
  - Both have a zero-initialised output layer, so they start as B1.
- **Training** (`scripts/phase3_train.py`):
  - Masked L1 loss on valid disk pixels; AdamW (3e-4, weight decay 1e-4), OneCycle schedule, bf16,
    batch 8.
  - Early stopping on the hold-out run with patience 6. Validation is never seen in training.
- **Robustness:**
  - The laptop shut down under full GPU load (2026-09-27 13:08). Runs now resume from an atomic
    `last.pt` checkpoint.
  - A thermal guard reads the GPU temperature in-process (NVML) before every batch. It pauses at
    80 °C and resumes below 72 °C.
  - The earlier guard, which ran nvidia-smi every 5 batches, overshot an 84 °C trip point to
    88–96 °C: the same range as the shutdown.
  - Never run a second GPU job next to training. Evaluation and diagnostics run on the CPU while
    training runs.
- **Test runs done:**

| Run | Best epoch / epochs run | Hold-out skill vs B1 |
|---|---|---|
| unet_s0 | 4 / 11 | 0.192 |
| unet_s1 | 4 / 11 | 0.195 |
| convlstm_s0 | 25 / 30 | 0.182 |

  The multi-seed chain was stopped on purpose (see the status note above).

## 2. First evaluation: skill that grows with horizon

Validation, whole disk (r < 0.95), relative MAE (%):

| Method | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|
| B1 (last frame, derotated) | 1.84 | 2.09 | 2.48 | 3.58 |
| B1-avg (mean of the 5 derotated context frames) | 1.65 | 1.92 | 2.35 | 3.49 |
| unet_s0 | 1.53 | 1.76 | 2.07 | 2.67 |
| convlstm_s0 | 1.54 | 1.78 | 2.06 | 2.68 |

Against B1-avg, both models gained 6–9 % at 28–57 min and 23 % at 3.8 h. A gain that grows with
horizon is what an instrument artefact would produce. So before believing it, each non-solar
explanation was turned into a baseline fitted without validation data
(`scripts/phase3_why_skill.py`, outputs in `outputs/phase3/why/`).

## 3. Why the skill grows with horizon

Skill over B1-avg, validation, whole disk. Median of the per-window skill; 95 % intervals are from a
bootstrap over (run, hour) blocks.

| Baseline (fitted on train + hold-out only) | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|
| B1-avg-rot: rotation rate fitted to the data | 0.0 % | 0.2 % | 0.9 % | 1.4 % |
| B1-avg-damp: contrast shrunk toward a local mean, β(H) tuned on hold-out | 0.0 | 0.0 | 0.0 | 0.3 |
| B1-avg-LD: limb-darkening correction, multiplicative | 0.6 | 1.7 | 4.7 | 8.9 |
| B1-avg-LDadd: limb-darkening correction, additive | 4.9 | 4.6 | 7.4 | 13.1 |
| B1-avg-clim: + static mean-residual map M(H) | 2.8 | 3.6 | 9.4 | 16.7 |
| B1-avg-rot-clim | 2.7 | 3.7 | 9.6 | **17.2** |
| unet_s0 | 7.0 | 8.6 | 12.1 | 23.1 |
| convlstm_s0 | 6.4 | 7.1 | 11.9 | 22.7 |

### 3.1 Main finding: derotation moves what does not rotate

B1 derotates the whole image, including structure that is fixed on the registered grid:
- the limb-darkening background (NB03 spans ~0.67 → 1.30 of the disk median within r < 0.95);
- the quadrant seams;
- the large-scale detector pattern that Phase 2 left uncorrected, because its self-calibration
  failed validation (PHASE2 §3).

The error this adds does not depend on solar content. It grows with the rotation shift (~7 px at
384² after 3.8 h at disk centre), so a model that is told the horizon can learn to undo it.

Evidence:
- **A single static map explains most of the gain.** M(H) is the mean of (target − B1-avg) over the
  training pairs of each horizon, so solar evolution averages out and only static structure remains.
  - It is worth 17 % at 3.8 h, three-quarters of the models' 23 %.
  - At 1.9 h it is worth 9.6 % of the models' 12 %.
- **The maps look instrumental.** `outputs/phase3/why/mean_maps.png` shows the same pattern in the
  training map, the validation residual and both models' mean corrections:
  - dipole lines at the vertical and horizontal seams;
  - an east/west (left/right) split that strengthens with horizon.

  Spatial correlation with M(H):

  | | 28 min | 57 min | 1.9 h | 3.8 h |
  |---|---|---|---|---|
  | validation mean residual | 0.52 | 0.40 | 0.63 | 0.52 |
  | UNet mean correction | 0.49 | 0.40 | 0.70 | 0.59 |
  | ConvLSTM mean correction | 0.51 | 0.37 | 0.64 | 0.53 |

- **Per window, the static map accounts for much of the models' correction:** 38 % of each UNet
  correction at 1.9–3.8 h (29 % for the ConvLSTM), against 14–19 % of the true residual.
- **The growth sits in the large-scale error.** At 3.8 h, the error RMS above a 4-px (384²)
  Gaussian scale is:
  - 0.0305 for B1-avg;
  - 0.0191 with M(H);
  - 0.0175 for the UNet.

  The error below that scale is 0.0378, 0.0379 and 0.0306.

### 3.2 Limb darkening, and a bug

- **Bug.** The first limb-darkening baseline (committed in 391dded) estimated the profile q(μ) only
  from pixels between 0.7 and 1.3 of the disk median. The true range is 0.67–1.30, so the window cut
  off both ends and left a flattened, non-monotonic profile. The conclusion drawn from it ("LD makes
  B1-avg worse, so LD does not explain the gain") was wrong.
- **Fix:** the per-annulus median of all pixels (fd9689c).
- **Additive beats multiplicative.** The multiplicative correction (I = S·q(μ), rotate S) helps the
  quiet disk but hurts plage (−1.5 % at 3.8 h), while the additive one does not hurt it. Plage
  brightness evidently does not scale with the quiet-Sun limb darkening. This is plausible for a
  chromospheric line, but here it is only observed, not explained.
- **Static background from the median frame** (B1-avg-bg: the per-pixel median of the training
  frames used as the non-rotating background): mixed results, and worse in plage (−4 to −10 %). Plage
  that persists through the median contaminates the background.

### 3.3 Rotation rate, damping, registration drift

- **Rotation rate.** Least squares on the linearised residual, per latitude band, with a per-pair
  translation as nuisance, fitted on train + hold-out:
  - The offset from the Snodgrass & Ulrich magnetic rate is
    −0.18 − 0.29 sin²φ deg/day (95 % CI a ∈ [−0.30, −0.09], b ∈ [−0.57, −0.02]).
  - Without the translation term the fit gives −0.29 − 0.39 sin²φ, so the rate is partly degenerate
    with pointing.
  - Applying the fitted rate gains 1.4 % at 3.8 h. It is real but small. The models do not learn much
    of it. At 3.8 h the median per-window rate offset implied by their corrections is −0.06 to
    −0.10 deg/day, against −0.19 in the validation residuals themselves.
- **Damping** (regression to the mean): the hold-out chose β = 1, i.e. no damping, up to 1.9 h, and
  β = 0.9 at 3.8 h (+0.3 %). Uniform contrast shrinkage explains nothing. The models do shrink
  small-scale contrast more than that: their effective β is 0.92 at 28 min and 0.76–0.78 at 3.8 h.
  They do it selectively, which a uniform β cannot copy (§4).
- **Registration drift.** The per-pair translation grows from 0.09 px (28 min) to 0.20 px (3.8 h) at
  384², i.e. ~0.8 px at 1536². That is within the Phase 1 registration budget (0.5–0.9 px rms) and
  not a main effect.

## 4. What the models add beyond the static effects

Skill of each model over the strongest static baseline (chosen per region and horizon, 95 % block
CI):

| Region | Model | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|---|
| disk | unet_s0 | 2.1 [1.5, 3.0] | 4.4 [3.5, 5.3] | 2.7 [2.3, 3.3] | 7.4 [6.7, 8.1] |
| disk | convlstm_s0 | 1.5 [1.0, 2.0] | 2.9 [2.2, 3.5] | 2.6 [2.3, 3.0] | 6.5 [5.9, 6.9] |
| trusted | unet_s0 | 1.3 [0.9, 2.1] | 2.9 [2.1, 4.2] | 1.7 [1.0, 2.2] | 5.6 [4.8, 6.5] |
| trusted | convlstm_s0 | 0.7 [0.3, 1.0] | 1.9 [1.3, 2.8] | 1.8 [1.5, 2.2] | 5.6 [4.9, 6.2] |
| plage | unet_s0 | 0.2 [−0.9, 1.4] | 2.5 [1.7, 3.5] | 3.7 [2.3, 4.9] | 8.5 [6.9, 10.5] |
| plage | convlstm_s0 | −0.4 [−0.9, 0.3] | 1.1 [0.9, 1.8] | 1.7 [1.2, 3.0] | 5.0 [4.3, 5.8] |

(The strongest baseline is B1-avg-LDadd at 28–57 min and B1-avg-rot-clim at 1.9–3.8 h; in plage it
is B1-avg-clim or B1-avg-rot-clim.)

- **What survives is small:**
  - 1–4 % up to 2 h;
  - 5–8.5 % at 3.8 h;
  - nothing in plage at 28 min.
- **Most of what survives is small-scale** (§3.1: below the 4-px scale at 3.8 h the error falls
  19 %, above it only 8 % after M(H)). It goes with a selective contrast reduction (effective
  β ≈ 0.77 at 3.8 h).
- **Open question:** is this forecasting skill (knowing which structures fade) or noise suppression
  beyond what averaging five frames gives? It is not yet separated from further instrument effects.
- **Tests before any of it is called solar:**
  - Train against a baseline that no longer contains the static artefact (§6).
  - Content-shuffle control: context from another time, same horizon. The correction that survives
    is content-independent.
  - Co-rotation control: does the correction follow the Sun or the detector?

## 4b. Background-aware B1 (first test, 2026-09-27)

- **What the background is.** The offset-mode Level-1 frames carry a large non-solar brightness
  gradient: at equal μ, east/west ≈ 0.97/0.31 and south/north ≈ 0.54/0.77. In centred mode
  (22–23 Sep) the disk is nearly symmetric (0.80/0.83). This is the west-edge vignetting of
  PHASE2 §3. Derotating moves the solar image across it.
- **The fix.** `scripts/phase3_background.py` solves a static background S on the grid from the
  derotation residuals of the train pairs.
  - λ (the gradient penalty) is chosen on the hold-out run; λ = 3 is an interior optimum.
  - S is then refit on train + hold-out.
- **The baseline.** B1-avg-bgS = mean_k [rot_k(F_k − S) + S].
- **The retrained model.** `phase3_prepare_bg.py` builds the samples with that derotation, and one
  quick UNet test (`--inputs bg`, 7 epochs, best epoch 3) was trained on them.

Validation, every 2nd sample (a code test), skill over B1-avg, whole disk:

| | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|
| B1-avg-bgS (no learning) | 1.0 % | 3.3 % | 9.0 % | **18.3 %** |
| unet_s0 (plain inputs) | 7.0 | 8.6 | 12.0 | 23.1 |
| unet_s0_bg (background-aware inputs) | 7.6 | 7.8 | 12.6 | 24.5 |
| unet_s0_bg vs the strongest static baseline | 2.7 | 3.4 | 3.0 | 7.2 |

- **S generalises from training to validation.** Its hold-out gain (18.5 % at 3.8 h) carries over to
  validation, where it is the strongest static baseline at 3.8 h.
- **Retraining on background-aware inputs leaves the models' edge about where it was.** It is 2–7 %
  on the disk, 1.9–6.1 % in the trusted region and 1.1–7.7 % in plage.
- **S is still incomplete:**
  - At 28 min the limb-darkening-only baseline beats it (4.9 % vs 1.0 %): the gradient penalty
    smooths the steep limb-darkening near the limb.
  - The background-aware UNet still removes more large-scale error than S (RMS 0.0163 vs 0.0196 at
    3.8 h).
  - 21–44 % of its corrections still correlate with the static map M(H).

  Part of the remaining edge may therefore still be background.
- **Next refinements:** fit S on top of the analytic limb-darkening profile (penalise only the
  remainder); anchor S to the detector and shift it by each frame's pointing; validate S on
  centred-mode frames.
- **Thermal.** Training now uses a duty-cycle controller (`Thermal` in `phase3_train.py`).
  - On/off pausing let the GPU jump from under 80 °C to 95–96 °C at every restart.
  - A synthetic full-power test with the CPU idle held 63–69 °C at ~6–8 % duty.
  - This laptop's cooling sustains only a small fraction of full GPU power, which limits the speed of
    the final run.

## 4c. Instrument control: the centred pointing mode (data set c0, 2026-09-27/28)

- **The control.** New downloads added 19–22 Sep, all in the CENTRED pointing mode, where the Level-1
  disk is nearly free of the offset-mode vignetting (east/west 0.80/0.83 at μ = 0.4, against
  0.97/0.31). Data set c0 (`configs/phase2_c0.toml`, `SUITDYN_DATASET=c0`) is built from it with the
  same rules as v0:

  | Split | Frames | Hours | Runs |
  |---|---|---|---|
  | train | 2,425 | 50.8 | 12 |
  | val | 451 | 10.8 | 3 |
  | test (sealed) | 278 | 6.7 | 4 |

- **The prediction.** If the long-horizon skill in v0 was mostly the models undoing the derotation
  of the vignetting, then in c0 (a) the static-background baselines should gain much less and (b) the
  models' skill should stop growing with horizon.
- **Pipeline.** Everything was run as test runs:
  - the manifest update (55,577 files, 19–26 Sep, 0 missing from the share, 0 checksum conflicts);
  - Phase 1 on the new frames;
  - the c0 data set, store, pointing response, noise maps, Phase 3 samples and background S;
  - one quick UNet (plain inputs, 8 epochs at most, best epoch 2 of 5, hold-out skill vs B1 = 0.127);
  - the diagnostics on every 2nd validation sample.

Validation, whole disk, skill over B1-avg:

| | 28 min | 57 min | 1.9 h | 3.8 h |
|---|---|---|---|---|
| **v0 (offset)**: static background S | 1.0 % | 3.3 % | 9.0 % | 18.3 % |
| v0: UNet | 7.0 | 8.6 | 12.0 | 23.1 |
| **c0 (centred)**: static background S | −0.4 | 0.3 | 1.3 | 2.5 |
| c0: limb darkening only (additive) | 4.3 | 2.8 | 3.0 | 4.8 |
| c0: UNet | 5.3 | 4.4 | 4.8 | 6.3 |
| **c0: UNet vs the strongest static baseline** | **1.0 [0.9, 1.2]** | **1.8 [1.3, 2.0]** | **1.8 [1.6, 2.1]** | **1.5 [1.3, 1.8]** |

- **Both predictions hold:**
  - In centred mode the static background is worth 2.5 % at 3.8 h instead of 18.3 %. Its hold-out
    gain was 2.8 % instead of 18.5 %.
  - The UNet's skill no longer grows with horizon.
- **What is left over the strongest static baseline (limb darkening only) is small and flat:**
  - 1.0–1.8 % on the disk and 0.9–1.6 % in the trusted region;
  - in plage 0.8 % (not significant), 0.5 % (not significant), 1.2 % and 1.5 %.
- **B1-avg error growth** from 28 min to 3.8 h is 1.21 points in c0 (1.90 → 3.11 %) against 1.84 in
  v0 (1.64 → 3.48 %). About a third of the offset-mode growth was the vignetting artefact, matching §5.
- **Reading (provisional: one model, one seed, a test run):**
  - At 28 min – 3.8 h, SUIT NB03 carries little predictable structure beyond rotation and static
    effects: about 1–2 % of MAE on this data.
  - Whether even that is solar (for example selective decay) or a residual effect (the additive
    limb-darkening profile is itself an approximation) needs the negative controls.
- **Not interpretable yet:** c0 errors are higher at short horizons (B1-avg 1.90 % vs 1.64 % at
  28 min). It could be different solar conditions, the other pointing mode, or noise.

**Reproducibility problem found in the same run.**
- Re-running `registration_study.py` on the larger archive changed the registration of dataset v0
  frames. 2,019–2,087 of 2,202 frames moved by more than 0.01 px, by up to 0.76 px in x and 1.26 px
  in y (`scripts/compare_registration.py`, `outputs/phase2/sequences/registration_check.json`).
- It also added `limb_outlier` to 13 v0 frames; a rebuild of v0 would drop them.
- Registration and QC depend on the other frames processed with them, so Phase 1 is not frame-local.
- v0's results are unaffected: its stores keep the registration they were built with, and the old
  tables are kept in `outputs/phase1/snapshot_before_c0`.
- **Before the final run, freeze Phase 1 per data set.**

## 5. Consequences for Phase 2 (corrections)

- **PHASE2 §4.6** says "at hour horizons the evolution B1 misses is large-scale". Part of that
  large-scale growth is the static artefact of §3.1, not solar evolution. Between 28 min and 3.8 h:
  - B1-avg error grows by 1.84 percentage points (1.65 → 3.49 %);
  - B1-avg-rot-clim grows by 1.28 points (1.60 → 2.88 %).

  About 30 % of the growth is the artefact.
- **The skill ceilings in PHASE2 §4.7** are computed against B1. B1 contains this artefact, so they
  overstate the room for a solar forecaster at long horizons. They have to be recomputed against a
  background-aware B1.
- The Phase 2 baselines table (§4.4–4.5) is unchanged as a measurement of B1. B1 is simply not the
  right reference.

## 6. Next (all as small code tests; the full run comes last)

1. **Background-aware B1** (the root-cause fix): derotate only the solar part. Remove a static
   background before derotation and add it back at the target: the additive LD profile plus a static
   instrument map that includes the seams. Estimate it from training frames without plage
   contamination.
   - Rebuild the Phase 3 samples with it.
   - Recompute the Phase 2 error growth and the ceilings.
   - This baseline becomes the reference every model must beat.
2. Retrain against it; one seed and a few epochs is enough for a code test.
3. The instrument negative controls of §4.
4. Dataset v1: 26 Sep onward, as the share download progresses.
5. One end-to-end command for the final run: raw manifest → dataset → stores → calibration →
   samples → training (all seeds) → evaluation → diagnostics.

## 7. Limitations

- Three days of data; validation is half of one day. One seed per model type for the diagnostics.
- M(H) is empirical and horizon-specific. It assumes the static pattern is stable on the registered
  grid, but pointing drifts ~8 px/day at 384², so the pattern smears. That makes M(H) a conservative
  estimate of the artefact.
- The rate fit and the translation are partly degenerate near disk centre.
- The 4-px scale split is a choice, not a physical boundary.

## 8. Reproduce

```
python scripts/phase3_prepare.py                          # frame cache + samples (384², K = 5)
python scripts/phase3_train.py --model unet --seed 0      # resumable; thermal guard 80/72 C
python scripts/phase3_train.py --model convlstm --seed 0
python scripts/phase3_evaluate.py                         # GPU; do not run next to training
python scripts/phase3_why_skill.py [--runs unet_s0,convlstm_s0] [--threads 4]   # CPU only
python scripts/phase3_why_skill.py --stride 25 --out outputs/phase3/why_smoke   # quick code test
```

`phase3_evaluate.py` still uses the first baseline set. Its summary predates the limb-darkening fix,
so `phase3_why_skill.py` supersedes it until the evaluation is rebuilt around the background-aware B1.
