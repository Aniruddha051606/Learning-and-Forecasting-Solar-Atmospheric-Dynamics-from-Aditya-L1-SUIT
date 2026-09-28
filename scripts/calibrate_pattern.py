"""Pipeline calibration step: the NB03 detector fixed pattern of ONE data set, from its training frames.

    python scripts/calibrate_pattern.py [--frames 320] [--keep-cache]

The Phase 2 study (phase2_calibration.py, phase2_calibration_followup.py) established how to correct the
pattern: it is fixed on the detector, additive, and is corrected below an 8-px scale. This step applies
that recipe to one data set without looking outside it:
  1. frames: the data set's TRAINING split only (validation and test never inform the calibration),
     evenly spread in time, without frames flagged brightness_jump / limb_outlier / spike_rate;
  2. per frame, the relative high-pass residual and the smoothed level (suitdyn.flat.frame_residuals);
  3. per pixel, the median over frames of both; pattern = high-pass(relative x level) for 'additive'
     (high-pass(relative) for 'multiplicative'), NaN-aware, at [calibration] pattern_highpass_px;
  4. quality gates: the same pattern from two interleaved halves of the frames must agree (correlation on
     the disk >= [calibration] pattern_min_split_half_r: the estimate is not noise-limited), and from the EARLY
     and the LATE half of the training span (>= pattern_min_time_r: the pattern did not change over the data
     set, e.g. at an instrument-state change like the one on 24 Sep, PHASE1 A11); else the step fails.
Writes outputs/datasets/<name>/calibration/nb03_pattern.npy and nb03_pattern.json (mode, frames and
their hash, split-half r, rms, provenance). The per-frame residual stacks are temporary (~17 MB per
frame) and are removed at the end unless --keep-cache.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, flat, io, paths, progress  # noqa: E402

CFG = config.load_dataset()
SHAPE = (2048, 2048)
EXCLUDE = "brightness_jump|limb_outlier|spike_rate|pointing_mode_change"


def nan_highpass(p, sigma):
    w = np.isfinite(p).astype(np.float32)
    num = gaussian_filter(np.nan_to_num(p), sigma)
    den = gaussian_filter(w, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, p - num / den, np.nan)


def _job(args):
    k, path, fit, clip_lo, clip_hi, rel_path, sm_path, n = args
    im, _ = io.read(path)
    rel, _ = flat.frame_residuals(im, fit, clip_lo, clip_hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(np.isfinite(rel), im / (1 + rel), np.nan)
    for p, a in ((rel_path, rel), (sm_path, sm)):
        mm = np.memmap(p, dtype=np.float16, mode="r+", shape=(n,) + SHAPE)
        mm[k] = a.astype(np.float16)
        mm.flush()
    return k


def medians(path, n, groups, rows=64):
    """Per-pixel median over frames for each group of frame indices, row block by row block."""
    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    out = {g: np.full(SHAPE, np.nan, np.float32) for g in groups}
    for y0 in range(0, SHAPE[0], rows):
        blk = np.asarray(mm[:, y0:y0 + rows, :], dtype=np.float32)
        for g, idx in groups.items():
            out[g][y0:y0 + rows] = flat.nanmedian0(blk[idx])
        progress.report(f"calibrate_pattern: medians {Path(path).stem}", item=f"rows {y0}-{y0 + rows}", i=y0 // rows,
                        n=SHAPE[0] // rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=CFG["calibration"].get("pattern_frames", 320))
    ap.add_argument("--keep-cache", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cal = paths.calibration()
    seq = pd.read_parquet(paths.sequences("frames.parquet"))
    man = pd.read_parquet(paths.archive("manifest.parquet"), columns=["file", "path", "clip_lo", "clip_hi"])
    tr = seq[(seq.split == "train") & ~seq.qc_reasons.fillna("").str.contains(EXCLUDE)]
    tr = tr.merge(man, left_on="frame_id", right_on="file").sort_values("t").reset_index(drop=True)
    if len(tr) < 40:
        sys.exit(f"only {len(tr)} usable training frames: too few for a pattern estimate")
    sel = tr.iloc[np.linspace(0, len(tr) - 1, min(a.frames, len(tr))).astype(int)].reset_index(drop=True)
    n = len(sel)
    rel_path, sm_path = cal / "cache_rel.f16", cal / "cache_sm.f16"
    for p in (rel_path, sm_path):
        np.memmap(p, dtype=np.float16, mode="w+", shape=(n,) + SHAPE).flush()
    jobs = [(k, r.path, {"x0": r.reg_x0, "y0": r.reg_y0, "R": r.reg_R, "harm": []}, r.clip_lo, r.clip_hi,
             str(rel_path), str(sm_path), n) for k, r in sel.iterrows()]
    with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
        for k, _ in enumerate(ex.map(_job, jobs, chunksize=2)):
            progress.report("calibrate_pattern: residuals", item=sel.frame_id.iloc[k], i=k, n=n, path=sel.path.iloc[k])
    print(f"residuals of {n} training frames ({time.time() - t0:.0f} s)", flush=True)

    idx = np.arange(n)
    groups = {"all": idx, "A": idx[0::2], "B": idx[1::2], "early": idx[:n // 2], "late": idx[n // 2:]}
    rel = medians(rel_path, n, groups)
    level = medians(sm_path, n, {"all": idx})["all"]
    hp = float(CFG["calibration"]["pattern_highpass_px"])
    mode = CFG["calibration"].get("pattern_mode", "additive")
    make = (lambda r: nan_highpass(r * level, hp)) if mode == "additive" else (lambda r: nan_highpass(r, hp))
    pattern = make(rel["all"]).astype(np.float32)
    pa, pb = make(rel["A"]), make(rel["B"])
    m = np.isfinite(pa) & np.isfinite(pb) & (np.abs(rel["all"]) < 0.5)
    r_half = float(np.corrcoef(pa[m], pb[m])[0, 1])
    pe, pl = make(rel["early"]), make(rel["late"])
    mt = np.isfinite(pe) & np.isfinite(pl) & (np.abs(rel["all"]) < 0.5)
    r_time = float(np.corrcoef(pe[mt], pl[mt])[0, 1])

    path = atomic.save_npy(cal / "nb03_pattern.npy", pattern)
    info = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": mode,
            "highpass_px": hp, "units": "counts (Level-1 NB03 binned)" if mode == "additive" else "relative",
            "frames": n, "frames_from": "training split only",
            "frames_sha256": hashlib.sha256("\n".join(sel.frame_id).encode()).hexdigest(),
            "span": [str(sel.t.min()), str(sel.t.max())], "split_half_r": r_half,
            "min_split_half_r": CFG["calibration"].get("pattern_min_split_half_r", 0.8),
            "early_late_r": r_time, "min_early_late_r": CFG["calibration"].get("pattern_min_time_r", 0.8),
            "early_late_split": str(sel.t.iloc[n // 2]),
            "rms": float(np.nanstd(pattern)), "valid_pixels": int(np.isfinite(pattern).sum()),
            "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    atomic.write_json(cal / "nb03_pattern.json", info)
    if not a.keep_cache:
        for p in (rel_path, sm_path):
            p.unlink(missing_ok=True)
    print(json.dumps({k: info[k] for k in ("mode", "frames", "split_half_r", "early_late_r", "rms", "seconds")}, indent=1))
    if r_half < info["min_split_half_r"]:
        sys.exit(f"split-half agreement {r_half:.3f} < {info['min_split_half_r']}: pattern not reliable")
    if r_time < info["min_early_late_r"]:
        sys.exit(f"early/late agreement {r_time:.3f} < {info['min_early_late_r']}: the pattern changed within the "
                 f"training span (split near {info['early_late_split']}); calibrate per period or shorten the data set")


if __name__ == "__main__":
    main()
