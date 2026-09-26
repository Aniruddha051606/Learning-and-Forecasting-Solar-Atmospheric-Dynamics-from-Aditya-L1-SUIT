"""Phase 2, step 1: NB03 detector fixed-pattern calibration study.

    python scripts/phase2_calibration.py [--frames 320]

Questions answered, each with a number in <out>/calibration/calibration_summary.json:
  Q1  Is the pattern stable?  Split-half, day-to-day, and across the 24 Sep instrument-state change
      (PHASE1.md A11) correlations of independent estimates.
  Q2  Multiplicative or additive?  At each pixel the local brightness changes as solar structure
      passes. Estimates from the frames where that pixel is bright vs dark: a multiplicative pattern
      keeps its relative size, an additive one keeps its absolute size.
  Q3  Does the correction work on frames it was not estimated from?  Pattern re-estimated from
      held-out corrected frames (should fall towards the noise level), and the Phase 1 failure test:
      plain frame-to-frame phase correlation must measure the ~0.17 px solar rotation instead of 0.
Writes the adopted pattern to <out>/calibration/nb03_pattern_<mode>.npy with its provenance.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, flat, io, motion, solar  # noqa: E402

CFG = config.load()
OUT = config.out_dir(CFG)
CAL = config.out_dir(CFG, "calibration")
CACHE = config.out_dir(CFG, "cache")
SHAPE = (2048, 2048)
STATE_CHANGE = pd.Timestamp("2026-09-24 15:00")  # PHASE1.md A11, between ~12 and ~18 UT


def _fit(r):
    return {"x0": r.reg_x0, "y0": r.reg_y0, "R": r.reg_R, "harm": []}


def _job(args):
    k, path, fit, clip_lo, clip_hi, rel_path, sm_path, n = args
    im, _ = io.read(path)
    rel, ab = flat.frame_residuals(im, fit, clip_lo, clip_hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(np.isfinite(rel), im / (1 + rel), np.nan)
    mm = np.memmap(rel_path, dtype=np.float16, mode="r+", shape=(n,) + SHAPE)
    mm[k] = rel.astype(np.float16)
    mm.flush()
    ms = np.memmap(sm_path, dtype=np.float16, mode="r+", shape=(n,) + SHAPE)
    ms[k] = sm.astype(np.float16)
    ms.flush()
    return k


def select_frames(n, n_centred=120):
    """n offset-mode frames spread over time (the dataset), plus up to n_centred centred-mode frames used
    only to test that the pattern is fixed to the detector."""
    reg = pd.read_parquet(OUT / "registration.parquet")
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path", "clip_lo", "clip_hi"])
    nb = reg[(reg.frame == "full_binned") & reg.qc_usable].merge(man, on="file").sort_values("t")
    bad = nb.qc_reasons.fillna("").str.contains("brightness_jump|limb_outlier|spike_rate|pointing_mode_change")
    first = nb.OBS_MODE.ne(nb.OBS_MODE.shift())  # first frame of every program block (A8)
    nb = nb[~bad & ~first].reset_index(drop=True)
    parts = []
    for mode, m in (("offset", n), ("centred", n_centred)):
        g = nb[nb.pointing_mode == mode]
        parts.append(g.iloc[np.linspace(0, len(g) - 1, min(m, len(g))).astype(int)] if len(g) else g)
    return pd.concat(parts).reset_index(drop=True)


def corr(a, b, m):
    return float(np.corrcoef(a[m], b[m])[0, 1])


def masked_median(mm, sel, rows=128):
    """Per-pixel median over the frames in `sel` (bool per frame, or a per-pixel (n, H, W) bool array
    is handled by the caller)."""
    out = np.full(SHAPE, np.nan, np.float32)
    idx = np.nonzero(sel)[0]
    import warnings
    for y0 in range(0, SHAPE[0], rows):
        blk = mm[idx, y0:y0 + rows, :].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[y0:y0 + rows] = np.nanmedian(blk, 0)
    return out


def bright_dark(rel_mm, sm_mm, n, rows=64):
    """Per pixel: median relative and absolute residual over the frames where the pixel's smoothed
    brightness is above / below its own median."""
    res = {k: np.full(SHAPE, np.nan, np.float32) for k in ("rel_b", "rel_d", "abs_b", "abs_d", "sm_b", "sm_d")}
    import warnings
    for y0 in range(0, SHAPE[0], rows):
        r = rel_mm[:, y0:y0 + rows, :].astype(np.float32)
        s = sm_mm[:, y0:y0 + rows, :].astype(np.float32)
        a = r * s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(s, 0)
            hi = s > med
            lo = s <= med
            for key, v in (("rel", r), ("abs", a), ("sm", s)):
                res[f"{key}_b"][y0:y0 + rows] = np.nanmedian(np.where(hi, v, np.nan), 0)
                res[f"{key}_d"][y0:y0 + rows] = np.nanmedian(np.where(lo, v, np.nan), 0)
    return res


def rotation_test(pattern, mode, n_pairs=120):
    """Plain phase correlation of consecutive frames (no pattern subtraction in the correlator) inside the
    longest run: median x-motion should equal the predicted rotation (~0.17 px/frame) once the pattern is
    corrected, and ~0 when it is not (the Phase 1 failure)."""
    reg = pd.read_parquet(OUT / "registration.parquet")
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path", "HGLT_OBS"]).set_index("file")
    nb = reg[reg.frame == "full_binned"].sort_values("t").reset_index(drop=True)
    run = nb.run.value_counts().idxmax()
    g = nb[nb.run == run].reset_index(drop=True)
    zero = np.zeros(motion.highpass(np.zeros(SHAPE)).shape)
    rows = []
    prev = {}
    for i in range(1, min(n_pairs + 1, len(g))):
        if (g.jump_px.iloc[i] or 0) > 1:
            continue
        for tag, pat in (("uncorrected", None), ("corrected", pattern)):
            ims = []
            for j in (i - 1, i):
                key = (tag, j)
                if key not in prev:
                    im = io.read(man.loc[g.file[j], "path"])[0]
                    prev[key] = motion.prepared(im if pat is None else flat.correct(im, pat, mode), zero)
                ims.append(prev[key])
            dx, dy, _ = motion.shift(ims[0], ims[1])
            rows.append({"i": i, "method": tag, "dx": dx, "dy": dy})
        prev = {k: v for k, v in prev.items() if k[1] >= i}
        rot = solar.disk_centre_motion_px(float(g.cc_R.iloc[i]), float(man.loc[g.file[i], "HGLT_OBS"]),
                                          (g.t[i] - g.t[i - 1]).total_seconds(), float(g.CROTA2.iloc[i]))
        rows[-1]["pred_rot_dx"] = rows[-2]["pred_rot_dx"] = rot[0]
    d = pd.DataFrame(rows)
    # expected x-motion = rotation + change of pointing (from the Phase 1 adopted registration)
    dpx = (g.reg_x0.diff()).reindex(d.i).values
    d["expected_dx"] = d.pred_rot_dx + np.asarray(dpx)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=320)
    a = ap.parse_args()
    t0 = time.time()
    sel = select_frames(a.frames)
    n = len(sel)
    rel_path, sm_path = CACHE / "nb03_rel.f16", CACHE / "nb03_sm.f16"
    for p in (rel_path, sm_path):
        np.memmap(p, dtype=np.float16, mode="w+", shape=(n,) + SHAPE).flush()
    jobs = [(k, r.path, _fit(r), r.clip_lo, r.clip_hi, rel_path, sm_path, n) for k, r in sel.iterrows()]
    with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
        list(ex.map(_job, jobs, chunksize=2))
    rel = np.memmap(rel_path, dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    sm = np.memmap(sm_path, dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    print(f"residuals of {n} frames in {time.time() - t0:.0f} s", flush=True)

    off = (sel.pointing_mode == "offset").values
    k = np.cumsum(off) - 1
    halves = {"A": off & (k % 2 == 0), "B": off & (k % 2 == 1)}
    days = sel.t.dt.floor("D")
    last_day = days[off].max()
    groups = {"early_days": off & (days < last_day).values, "last_day": off & (days == last_day).values,
              "before_state_change": off & (sel.t < STATE_CHANGE).values,
              "after_state_change": off & (sel.t >= STATE_CHANGE).values,
              "centred_mode": ~off}
    P = {name: masked_median(rel, m) for name, m in {**halves, **groups, "all": off}.items() if m.any()}
    cnt = np.isfinite(P["all"])
    disk = cnt & (np.abs(P["all"]) < 0.5)

    # Q2: multiplicative vs additive, from bright/dark frames at each pixel
    bd = bright_dark(rel, sm, n)  # all frames: a pixel's brightness varies most across pointing modes
    m2 = disk & np.isfinite(bd["rel_b"]) & np.isfinite(bd["rel_d"]) & (np.abs(P["all"]) > 2 * np.nanstd(P["all"][disk]))
    ratio_sm = np.nanmedian((bd["sm_b"] / bd["sm_d"])[m2])
    slope_abs = np.polyfit(bd["abs_d"][m2], bd["abs_b"][m2], 1)[0]
    slope_rel = np.polyfit(bd["rel_d"][m2], bd["rel_b"][m2], 1)[0]
    mode = "multiplicative" if abs(slope_rel - 1) < abs(slope_abs - 1) else "additive"

    pattern = P["all"] if mode == "multiplicative" else P["all"] * np.nanmedian(sm, 0)
    np.save(CAL / f"nb03_pattern_{mode}.npy", pattern.astype(np.float32))

    # Q3: held-out check. Correct half B with the pattern from half A, re-estimate: residual pattern rms.
    resid = P["B"] - P["A"] if mode == "multiplicative" else None
    rot = rotation_test(P["all"] if mode == "multiplicative" else pattern, mode)
    rsum = {}
    for tag in ("uncorrected", "corrected"):
        d = rot[rot.method == tag]
        rsum[tag] = {"median_dx": float(d.dx.median()), "median_dy": float(d.dy.median()),
                     "rms_dx_minus_expected": float(np.sqrt(np.nanmean((d.dx - d.expected_dx) ** 2))),
                     "corr_dx_expected": float(np.corrcoef(d.dx, d.expected_dx)[0, 1])}
    rsum["predicted_rotation_dx_median"] = float(rot.pred_rot_dx.median())

    level = float(np.nanmedian(sm[:, 700:900, 1200:1400]))
    summary = {
        "frames": n, "span": [str(sel.t.min()), str(sel.t.max())],
        "pattern_rms_relative": float(np.nanstd(P["all"][disk])),
        "pattern_rms_counts_at_disk_level": float(np.nanstd(P["all"][disk]) * level),
        "q1_stability_r": {
            "split_half_A_vs_B": corr(P["A"], P["B"], disk & np.isfinite(P["A"]) & np.isfinite(P["B"])),
            "early_days_vs_last_day": corr(P["early_days"], P["last_day"],
                                           disk & np.isfinite(P["early_days"]) & np.isfinite(P["last_day"])),
            "before_vs_after_state_change": corr(P["before_state_change"], P["after_state_change"],
                                                 disk & np.isfinite(P["before_state_change"])
                                                 & np.isfinite(P["after_state_change"])),
            "offset_vs_centred_pointing": corr(P["all"], P["centred_mode"],
                                               disk & np.isfinite(P["centred_mode"])) if "centred_mode" in P else None,
            "centred_frames": int((~off).sum()),
        },
        "q2_mode": {"brightness_ratio_bright_over_dark": float(ratio_sm),
                    "slope_relative_pattern_bright_vs_dark": float(slope_rel),
                    "slope_absolute_pattern_bright_vs_dark": float(slope_abs),
                    "pixels_used": int(m2.sum()), "adopted": mode,
                    "rule": "the representation whose bright-vs-dark slope is closer to 1 is the invariant one"},
        "q3_heldout": {"residual_after_correcting_B_with_A_rms": float(np.nanstd(resid[disk])) if resid is not None else None,
                       "pattern_rms_before": float(np.nanstd(P["B"][disk])),
                       "rotation_test": rsum},
        "seconds": round(time.time() - t0, 1), **CFG["_meta"],
    }
    (CAL / "calibration_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    rot.to_csv(CAL / "rotation_test.csv", index=False)

    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    v = 3 * np.nanstd(P["all"][disk])
    ax[0, 0].imshow(P["all"], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0, 0].set_title(f"NB03 relative fixed pattern (all {n} frames), rms {summary['pattern_rms_relative']:.3f}")
    ax[0, 1].imshow(P["all"][500:800, 1200:1500], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0, 1].set_title("zoom 300×300 px")
    ax[0, 2].imshow((P["B"] - P["A"]), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0, 2].set_title(f"half B − half A (r = {summary['q1_stability_r']['split_half_A_vs_B']:.3f})")
    ax[1, 0].plot(bd["rel_d"][m2][::20], bd["rel_b"][m2][::20], ".", ms=1)
    ax[1, 0].plot([-.2, .2], [-.2, .2], "k-", lw=.5)
    ax[1, 0].set_xlabel("relative pattern, dark frames")
    ax[1, 0].set_ylabel("bright frames")
    ax[1, 0].set_title(f"relative: slope {slope_rel:.2f}")
    ax[1, 1].plot(bd["abs_d"][m2][::20], bd["abs_b"][m2][::20], ".", ms=1)
    lim = np.nanpercentile(np.abs(bd["abs_d"][m2]), 99)
    ax[1, 1].plot([-lim, lim], [-lim, lim], "k-", lw=.5)
    ax[1, 1].set_xlabel("absolute pattern, dark frames (counts)")
    ax[1, 1].set_title(f"absolute: slope {slope_abs:.2f} (brightness ratio {ratio_sm:.2f})")
    for tag, c in (("uncorrected", "0.5"), ("corrected", "r")):
        d = rot[rot.method == tag]
        ax[1, 2].plot(d.expected_dx, d.dx, "o", ms=3, color=c, label=tag)
    ax[1, 2].plot([-3, 3], [-3, 3], "k-", lw=.5)
    ax[1, 2].set_xlabel("expected x-motion: rotation + pointing change (px)")
    ax[1, 2].set_ylabel("measured, plain phase correlation (px)")
    ax[1, 2].legend()
    ax[1, 2].set_title("consecutive frames")
    fig.tight_layout()
    fig.savefig(CAL / "fixed_pattern_study.png", dpi=70)
    plt.close(fig)
    del rel, sm
    for p in (rel_path, sm_path):  # regenerable cache
        try:
            p.unlink()
        except OSError:
            pass
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
