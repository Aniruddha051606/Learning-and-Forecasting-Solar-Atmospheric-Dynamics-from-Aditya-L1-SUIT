"""Phase 2, step 1: NB03 detector fixed-pattern calibration study.

    python scripts/phase2_calibration.py [--frames 320]

Questions answered, each with a number in <out>/calibration/calibration_summary.json:
  Q1  Is the pattern stable?  Split-half, day-to-day, and across the 24 Sep instrument-state change
      (PHASE1.md A11) correlations of independent estimates.
  Q2  (diagnostic only) slopes of bright-frame vs dark-frame estimates. This cannot separate a
      multiplicative from an additive pattern (noise in both estimates flattens both slopes, and both
      hypotheses predict the same slope ratio; docs/PHASE2.md §2). The decision is made by the
      plage-lever test in phase2_calibration_followup.py, which also adopts the pattern.
  Q3  split-half residual in both representations.
Writes, for the follow-up: nb03_pattern_relative.npy (median relative residual, offset mode),
nb03_level.npy (median smoothed level), nb03_group_patterns.npz (every subset estimate).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, flat, io  # noqa: E402

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


def block_pass(rel_mm, sm_mm, groups, abs_groups, off, rows=32):
    """One pass over row blocks: every group median (relative residual), absolute-residual medians for
    the held-out halves, the per-pixel bright/dark medians, and the median smoothed level. Each block is
    read once; memory stays at a few hundred MB (the first version took the median of the whole cache
    in one call and ran out of memory)."""
    P = {k: np.full(SHAPE, np.nan, np.float32) for k in groups}
    Pabs = {k: np.full(SHAPE, np.nan, np.float32) for k in abs_groups}
    bd = {k: np.full(SHAPE, np.nan, np.float32) for k in ("rel_b", "rel_d", "abs_b", "abs_d", "sm_b", "sm_d")}
    level = np.full(SHAPE, np.nan, np.float32)
    for y0 in range(0, SHAPE[0], rows):
        sl = slice(y0, y0 + rows)
        r = np.asarray(rel_mm[:, sl, :], dtype=np.float32)
        s = np.asarray(sm_mm[:, sl, :], dtype=np.float32)
        a = r * s
        for k, m in groups.items():
            P[k][sl] = flat.nanmedian0(r[m])
        for k, m in abs_groups.items():
            Pabs[k][sl] = flat.nanmedian0(a[m])
        level[sl] = flat.nanmedian0(s[off])
        med = flat.nanmedian0(s)
        hi, lo = s > med, s <= med
        for key, v in (("rel", r), ("abs", a), ("sm", s)):
            bd[f"{key}_b"][sl] = flat.nanmedian0(np.where(hi, v, np.nan))
            bd[f"{key}_d"][sl] = flat.nanmedian0(np.where(lo, v, np.nan))
    return P, Pabs, bd, level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--reuse-cache", action="store_true",
                    help="reuse the residual cache if it was made from exactly this frame selection")
    a = ap.parse_args()
    t0 = time.time()
    sel = select_frames(a.frames)
    n = len(sel)
    rel_path, sm_path, sel_path = CACHE / "nb03_rel.f16", CACHE / "nb03_sm.f16", CACHE / "nb03_selection.json"
    nbytes = n * SHAPE[0] * SHAPE[1] * 2
    files = sel.file.tolist()
    reuse = (a.reuse_cache and rel_path.exists() and sm_path.exists() and rel_path.stat().st_size == nbytes
             and (not sel_path.exists() or json.loads(sel_path.read_text()) == files))
    if not reuse:
        for p in (rel_path, sm_path):
            np.memmap(p, dtype=np.float16, mode="w+", shape=(n,) + SHAPE).flush()
        jobs = [(k, r.path, _fit(r), r.clip_lo, r.clip_hi, rel_path, sm_path, n) for k, r in sel.iterrows()]
        with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
            list(ex.map(_job, jobs, chunksize=2))
    sel_path.write_text(json.dumps(files))
    rel = np.memmap(rel_path, dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    sm = np.memmap(sm_path, dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    state = "reused" if reuse else "computed"
    print(f"residuals of {n} frames {state} ({time.time() - t0:.0f} s)", flush=True)

    off = (sel.pointing_mode == "offset").values
    k = np.cumsum(off) - 1
    halves = {"A": off & (k % 2 == 0), "B": off & (k % 2 == 1)}
    days = sel.t.dt.floor("D")
    last_day = days[off].max()
    groups = {**halves, "all": off, "early_days": off & (days < last_day).values,
              "last_day": off & (days == last_day).values,
              "before_state_change": off & (sel.t < STATE_CHANGE).values,
              "after_state_change": off & (sel.t >= STATE_CHANGE).values, "centred_mode": ~off}
    groups = {g: m for g, m in groups.items() if m.any()}
    P, Pabs, bd, level_map = block_pass(rel, sm, groups, halves, off)
    print(f"block pass done ({time.time() - t0:.0f} s)", flush=True)
    disk = np.isfinite(P["all"]) & (np.abs(P["all"]) < 0.5)

    # Q2: multiplicative vs additive, from the frames where each pixel is bright vs dark
    strong = np.abs(P["all"]) > 2 * np.nanstd(P["all"][disk])
    m2 = disk & np.isfinite(bd["rel_b"]) & np.isfinite(bd["rel_d"]) & strong
    ratio_sm = np.nanmedian((bd["sm_b"] / bd["sm_d"])[m2])
    slope_abs = np.polyfit(bd["abs_d"][m2], bd["abs_b"][m2], 1)[0]
    slope_rel = np.polyfit(bd["rel_d"][m2], bd["rel_b"][m2], 1)[0]
    np.save(CAL / "nb03_pattern_relative.npy", P["all"].astype(np.float32))
    np.save(CAL / "nb03_level.npy", level_map.astype(np.float32))
    np.savez_compressed(CAL / "nb03_group_patterns.npz", **{k: v.astype(np.float32) for k, v in P.items()})

    # Q3: held-out. What is left in half B after removing half A's pattern, in each representation
    # (absolute residuals divided by the level so both are in the same units).
    lev = np.nanmedian(level_map[disk])
    resid_rel = float(np.nanstd((P["B"] - P["A"])[disk]))
    resid_abs = float(np.nanstd(((Pabs["B"] - Pabs["A"]) / level_map)[disk]))

    def c(a_, b_):
        if a_ not in P or b_ not in P:
            return None
        return corr(P[a_], P[b_], disk & np.isfinite(P[a_]) & np.isfinite(P[b_]))

    summary = {
        "frames": n, "offset_frames": int(off.sum()), "centred_frames": int((~off).sum()),
        "span": [str(sel.t.min()), str(sel.t.max())],
        "pattern_rms_relative": float(np.nanstd(P["all"][disk])),
        "pattern_rms_counts_at_disk_level": float(np.nanstd(P["all"][disk]) * lev),
        "q1_stability_r": {"split_half_A_vs_B": c("A", "B"), "early_days_vs_last_day": c("early_days", "last_day"),
                           "before_vs_after_state_change": c("before_state_change", "after_state_change"),
                           "offset_vs_centred_pointing": c("all", "centred_mode")},
        "q2_diagnostic_not_a_decision": {"brightness_ratio_bright_over_dark": float(ratio_sm),
                                         "slope_relative_pattern_bright_vs_dark": float(slope_rel),
                                         "slope_absolute_pattern_bright_vs_dark": float(slope_abs),
                                         "pixels_used": int(m2.sum()),
                                         "note": "cannot separate the cases; see followup C2"},
        "q3_heldout": {"half_pattern_rms_relative": float(np.nanstd(P["B"][disk])),
                       "residual_B_minus_A_relative_repr": resid_rel,
                       "residual_B_minus_A_absolute_repr_over_level": resid_abs},
        "seconds": round(time.time() - t0, 1), **CFG["_meta"],
    }
    (CAL / "calibration_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    v = 3 * np.nanstd(P["all"][disk])
    ax[0, 0].imshow(P["all"], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0, 0].set_title(f"NB03 relative fixed pattern ({int(off.sum())} frames), rms {summary['pattern_rms_relative']:.3f}")
    ax[0, 1].imshow(P["all"][500:800, 1200:1500], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax[0, 1].set_title("zoom 300x300 px")
    if "centred_mode" in P:
        ax[0, 2].imshow(P["centred_mode"], origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        r_oc = summary["q1_stability_r"]["offset_vs_centred_pointing"]
        ax[0, 2].set_title(f"centred-pointing frames only (r vs offset = {r_oc:.3f})")
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
    ax[1, 2].imshow(level_map, origin="lower", cmap="gray")
    ax[1, 2].set_title("median smoothed level (offset mode)")
    fig.tight_layout()
    fig.savefig(CAL / "fixed_pattern_study.png", dpi=70)
    plt.close(fig)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
