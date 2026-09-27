"""Phase 2: where on the disk does the instrument, not the Sun, set the forecast error?

    python scripts/phase2_noise_maps.py --store v0 [--split val] [--grid-factor 2]

Over every 1-frame window of the split (87 s: the Sun barely changes), in registered coordinates:
  noise floor map     mean |B1(t+1) − F(t+1)| / disk level
  pointing map        per-pixel slope of that error against the pointing change between the two
                      frames (x and y, % per detector pixel). A detector-fixed response (vignetting,
                      seam, residual pattern) seen through the ±1-10 px pointing motion makes this large;
                      the Sun does not.
A "trusted region" mask is derived from pointing sensitivity only (the instrument signature): disk pixels
below median + 3 robust sigma of the disk-centre distribution. The noise floor is NOT used: plage has a
higher floor because it is brighter and evolving, and excluding it would remove the science.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import baselines, config, normalize, paths, progress, response  # noqa: E402

CFG = config.load_dataset()
SEQ = paths.sequences()
STORES = paths.stores()
OUT = paths.phase2("noise_maps")


def load(g, i, f):
    img = g["nb03/image"][i].astype(np.float32)
    img[g["nb03/mask"][i] != 0] = np.nan
    if f > 1:
        n = img.shape[0] // f
        img = np.nanmean(img[:n * f, :n * f].reshape(n, f, n, f), axis=(1, 3))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=config.DATASET)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--grid-factor", type=int, default=2)
    ap.add_argument("--response", default="", help="response model npz (phase2_response.py) to apply")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    resp = response.load(a.response) if a.response else None
    OUT.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(STORES / f"{a.store}.zarr"), mode="r")
    r_ref = float(g.attrs["r_ref"]) / a.grid_factor
    fr = pd.read_parquet(STORES / f"{a.store}.frames.parquet")
    win = pd.read_parquet(SEQ / "windows.parquet")
    win = win[(win.split == a.split) & (win.horizon == 1) & (win.context == 1)]
    seqf = pd.read_parquet(SEQ / "frames.parquet")
    s_of = pd.Series(fr.store_index.values, index=fr.frame_id)
    man = pd.read_parquet(paths.archive("manifest.parquet"), columns=["file", "HGLT_OBS"]).set_index("file")
    grid = g["nb03/image"].shape[1] // a.grid_factor
    mu = normalize.mu_map(grid, r_ref)
    disk = mu > np.sqrt(1 - 0.95 ** 2)
    acc = {k: np.zeros((grid, grid)) for k in ("n", "abs", "e", "dx", "dy", "edx", "edy", "dx2", "dy2", "dxdy")}
    coords = None
    for wi, (_, w) in enumerate(win.iterrows()):
        a_, b_ = seqf.iloc[w["last"]], seqf.iloc[w.target]
        progress.report(f"noise_maps {a.store} {a.split}{' ' + a.tag if a.tag else ''}", item=b_.frame_id, i=wi,
                        n=len(win))
        i, j = s_of[a_.frame_id], s_of[b_.frame_id]
        last, truth = load(g, i, a.grid_factor), load(g, j, a.grid_factor)
        if resp is not None:
            last = last * response.factor(resp, last.shape[0], a_.reg_x0, a_.reg_y0)
            truth = truth * response.factor(resp, truth.shape[0], b_.reg_x0, b_.reg_y0)
        dt = (b_.t - a_.t).total_seconds()
        if coords is None or abs(dt - coords[1]) > 5:
            coords = (baselines.derotation_coords(grid, r_ref, float(man.loc[a_.frame_id, "HGLT_OBS"]), dt), dt)
        pred = baselines.rotated_persistence(last, r_ref, None, dt, coords=coords[0])
        lvl = np.nanmedian(truth[disk])
        e = (pred - truth) / lvl
        m = np.isfinite(e) & disk
        # pointing change between the frames, in detector pixels of the binned frame
        dx, dy = b_.reg_x0 - a_.reg_x0, b_.reg_y0 - a_.reg_y0
        acc["n"][m] += 1
        acc["abs"][m] += np.abs(e[m])
        acc["e"][m] += e[m]
        for k, v in (("dx", dx), ("dy", dy)):
            acc[k][m] += v
            acc[f"e{k}"][m] += e[m] * v
            acc[f"{k}2"][m] += v * v
        acc["dxdy"][m] += dx * dy
    n = np.where(acc["n"] > 20, acc["n"], np.nan)
    floor = acc["abs"] / n
    # per-pixel least squares e ≈ c + sx·dx + sy·dy (centred sums)
    mx, my, me = acc["dx"] / n, acc["dy"] / n, acc["e"] / n
    sxx = acc["dx2"] / n - mx ** 2
    syy = acc["dy2"] / n - my ** 2
    sxy = acc["dxdy"] / n - mx * my
    sxe = acc["edx"] / n - mx * me
    sye = acc["edy"] / n - my * me
    det = sxx * syy - sxy ** 2
    slope_x = (syy * sxe - sxy * sye) / det * 100
    slope_y = (sxx * sye - sxy * sxe) / det * 100
    sens = np.hypot(slope_x, slope_y)
    core = disk & (mu > np.sqrt(1 - 0.5 ** 2)) & np.isfinite(floor) & np.isfinite(sens)

    def thresh(v):
        c = v[core]
        med = np.median(c)
        return med + 3 * 1.4826 * np.median(np.abs(c - med))

    t_floor, t_sens = thresh(floor), thresh(sens)
    trusted = disk & np.isfinite(sens) & (sens <= t_sens)
    np.savez_compressed(OUT / f"noise_maps_{a.store}_{a.split}_g{a.grid_factor}{('_' + a.tag) if a.tag else ''}.npz", floor=floor, slope_x=slope_x,
                        slope_y=slope_y, trusted=trusted, n=acc["n"])
    frac = float(trusted.sum() / disk.sum())
    summary = {"store": a.store, "split": a.split, "windows": int(len(win)), "grid": grid,
               "noise_floor_rel_mae": {"disk_median": float(np.nanmedian(floor[disk])),
                                       "core_median": float(np.median(floor[core])),
                                       "p95_disk": float(np.nanpercentile(floor[disk], 95))},
               "pointing_sensitivity_pct_per_px": {"core_median": float(np.median(sens[core])),
                                                   "disk_p95": float(np.nanpercentile(sens[disk], 95))},
               "thresholds": {"floor": float(t_floor), "sensitivity": float(t_sens)},
               "trusted_fraction_of_disk": frac, **CFG["_meta"]}
    (OUT / f"summary_{a.store}_{a.split}_g{a.grid_factor}{('_' + a.tag) if a.tag else ''}.json").write_text(json.dumps(summary, indent=1, default=float))
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    im0 = ax[0].imshow(floor * 100, origin="lower", cmap="magma", vmin=0, vmax=np.nanpercentile(floor[disk] * 100, 99))
    ax[0].set_title("noise floor: mean |B1 error| at 1 frame (% of level)")
    plt.colorbar(im0, ax=ax[0], fraction=.046)
    lim = np.nanpercentile(np.abs(slope_x[disk]), 99)
    im1 = ax[1].imshow(slope_x, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1].set_title("error slope vs x-pointing change (% per px)")
    plt.colorbar(im1, ax=ax[1], fraction=.046)
    im2 = ax[2].imshow(slope_y, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_title("error slope vs y-pointing change (% per px)")
    plt.colorbar(im2, ax=ax[2], fraction=.046)
    ax[3].imshow(np.where(disk, trusted, np.nan), origin="lower", cmap="Greens", vmin=0, vmax=1)
    ax[3].set_title(f"trusted region: {100 * frac:.0f} % of the disk (r<0.95)")
    for x in ax:
        x.set_xlabel("registered x (west →)")
    fig.suptitle(f"store {a.store}, {a.split}, {len(win)} one-frame windows; solar north up")
    fig.tight_layout()
    fig.savefig(OUT / f"noise_maps_{a.store}_{a.split}_g{a.grid_factor}{('_' + a.tag) if a.tag else ''}.png", dpi=70)
    plt.close(fig)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
