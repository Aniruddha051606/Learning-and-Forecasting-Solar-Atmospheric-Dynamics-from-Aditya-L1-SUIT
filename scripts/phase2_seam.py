"""Phase 2, step 2: nature of the NB03 vertical quadrant seam (PHASE1.md A5).

    python scripts/phase2_seam.py [--frames 240]

The step across x = 1023|1024 (binned) reaches −28 % in the southern rows. Two hypotheses need
opposite corrections:
  H_half   the whole region right of the seam (or left of it) has a different gain, varying with row
           → rescale that whole side;
  H_local  the error is confined near the boundary → only a local correction is right; rescaling a
           whole side would create an error far from the seam.
Test: east-west symmetry of the quiet Sun. For each detector point left of the seam, its mirror
about the disk centre (same row, same distance from centre) lies far right of the seam. The median,
over many frames, of I(mirror)/I(point) as a function of row and of the point's distance d from the
seam is flat in d under H_half and returns to 1 away from the seam under H_local. Medians over frames
spanning several days of rotation suppress individual active regions; the ratio is also taken
within each frame and then the median over frames, so program-level brightness steps cancel.

Also measured: the step itself across the seam, from the same frames after the fixed-pattern
correction (it should be unchanged: the pattern estimate never smooths across the seam).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import map_coordinates, median_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, flat, io  # noqa: E402

CFG = config.load()
OUT = config.out_dir(CFG)
SEAM = config.out_dir(CFG, "seam")
CAL = OUT / "calibration"
C = 1024
ROW_BINS = np.arange(0, 1344, 64)
D_BINS = np.array([0, 4, 8, 16, 32, 64, 128, 256, 400])


def _frame(args):
    path, x0, y0, R, pattern_path = args
    im, _ = io.read(path)
    if pattern_path:
        im = flat.correct(im, np.load(pattern_path), "multiplicative")
    im = median_filter(im, 3)  # suppress spikes and single-pixel pattern residue
    yy, xx = np.mgrid[0:1344, 560:1020].astype(np.float64)
    xm = 2 * x0 - xx  # east-west mirror about the disk centre, same row
    r = np.hypot(xx - x0, yy - y0) / R
    ok = (r < 0.93) & (xm < 2040) & (yy >= 0)
    ratio = map_coordinates(im, [yy, xm], order=1) / im[yy.astype(int), xx.astype(int)]
    ratio[~ok] = np.nan
    d = C - 1 - xx
    out = np.full((len(ROW_BINS) - 1, len(D_BINS) - 1), np.nan)
    for i in range(len(ROW_BINS) - 1):
        rs = (yy >= ROW_BINS[i]) & (yy < ROW_BINS[i + 1])
        for j in range(len(D_BINS) - 1):
            m = rs & (d >= D_BINS[j]) & (d < D_BINS[j + 1]) & np.isfinite(ratio)
            if m.sum() > 50:
                out[i, j] = np.median(ratio[m])
    # the step across the seam itself, per row band
    step = np.full(len(ROW_BINS) - 1, np.nan)
    for i in range(len(ROW_BINS) - 1):
        rows = slice(ROW_BINS[i], ROW_BINS[i + 1])
        L, Rr = im[rows, C - 11:C - 5], im[rows, C:C + 6]
        rr = np.hypot(np.arange(C - 11, C + 6)[None, :] - x0, np.arange(ROW_BINS[i], ROW_BINS[i + 1])[:, None] - y0) / R
        if (rr < 0.95).mean() > 0.9:
            step[i] = np.median(Rr) / np.median(L) - 1
    return out, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    a = ap.parse_args()
    reg = pd.read_parquet(OUT / "registration.parquet")
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path"])
    nb = reg[(reg.frame == "full_binned") & reg.qc_usable].merge(man, on="file").sort_values("t")
    nb = nb[~nb.OBS_MODE.ne(nb.OBS_MODE.shift())]
    sel = nb.iloc[np.linspace(0, len(nb) - 1, min(a.frames, len(nb))).astype(int)]
    pattern = CAL / "nb03_pattern_multiplicative.npy"
    pp = str(pattern) if pattern.exists() else ""
    with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
        res = list(ex.map(_frame, [(r.path, r.reg_x0, r.reg_y0, r.reg_R, pp) for r in sel.itertuples()]))
    ratios = np.array([r for r, _ in res])
    steps = np.array([s for _, s in res])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(ratios, 0)
        q = np.nanpercentile(ratios, [25, 75], axis=0)
        step_med = np.nanmedian(steps, 0)
    rowc = (ROW_BINS[:-1] + ROW_BINS[1:]) / 2
    dc = (D_BINS[:-1] + D_BINS[1:]) / 2
    # verdict per row band: compare the mirror ratio next to the seam (d < 8) with far from it (d > 128)
    near = np.nanmean(med[:, :2], 1)
    far = np.nanmean(med[:, -2:], 1)
    table = pd.DataFrame({"row": rowc, "step_across_seam": step_med, "mirror_ratio_near_seam": near,
                          "mirror_ratio_far_from_seam": far})
    table.to_csv(SEAM / "seam_table.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i in range(0, len(rowc), 2):
        if np.isfinite(med[i]).sum() > 3:
            ax[0].plot(dc, med[i], "o-", ms=3, label=f"rows {ROW_BINS[i]}-{ROW_BINS[i + 1]}")
    ax[0].set_xscale("symlog", linthresh=4)
    ax[0].axhline(1, color="k", lw=.5)
    ax[0].set_xlabel("distance of the left point from the seam (px)")
    ax[0].set_ylabel("median I(mirror, right of seam) / I(left point)")
    ax[0].set_title("E-W mirror ratio: flat in distance = whole-side gain; →1 = local error")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].plot(rowc, step_med * 100, "r-o", ms=3, label="step across seam (after pattern correction)")
    ax[1].plot(rowc, (1 / near - 1) * 100, "b--", label="left-near-seam vs mirror (%)")
    ax[1].plot(rowc, (1 / far - 1) * 100, "g--", label="left-far-from-seam vs mirror (%)")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_xlabel("row (binned px)")
    ax[1].set_ylabel("%")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SEAM / "seam_study.png", dpi=80)
    plt.close(fig)
    summary = {"frames": len(sel), "pattern_corrected": bool(pp), "table": table.round(4).to_dict("records"),
               **CFG["_meta"]}
    (SEAM / "seam_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
