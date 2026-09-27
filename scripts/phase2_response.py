"""Phase 2: build the pointing-response correction from the TRAINING split and test it on validation.

    python scripts/phase2_response.py --store v0

1. Slope maps from outputs/phase2/noise_maps/noise_maps_<store>_train_g2.npz (run phase2_noise_maps.py
   --split train first), smoothed; reference pointing = median training pointing.
2. Test on the validation split (never used to build it):
   a) whole-disk level vs pointing: R² of the detrended disk median on the pointing within each run,
      before and after correction (PHASE2.md §1.4 found a median R² of 0.38 before);
   b) the validation noise maps are recomputed with the correction applied by phase2_noise_maps.py
      --response (run separately), and their pointing sensitivity is compared.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, normalize, paths, response  # noqa: E402

CFG = config.load_dataset()
STORES = paths.stores()
NM = paths.phase2("noise_maps")
OUT = paths.phase2("response")


def level_r2(g, fr, model, split, factor_grid=2, every=1):
    """Per run: detrended log disk median vs pointing, R², with and without the correction."""
    grid = g["nb03/image"].shape[1] // factor_grid
    r_ref = float(g.attrs["r_ref"]) / factor_grid
    mu = normalize.mu_map(grid, r_ref)
    core = mu > np.sqrt(1 - 0.8 ** 2)
    rows = []
    sub = fr[fr.split == split].iloc[::every]
    for _, r in sub.iterrows():
        img = g["nb03/image"][int(r.store_index)].astype(np.float32)
        img[g["nb03/mask"][int(r.store_index)] != 0] = np.nan
        n = img.shape[0] // factor_grid
        img = np.nanmean(img[:n * factor_grid, :n * factor_grid].reshape(n, factor_grid, n, factor_grid), axis=(1, 3))
        f = response.factor(model, grid, r.reg_x0, r.reg_y0)
        rows.append({"run": r.run, "t": r.t, "x0": r.reg_x0, "y0": r.reg_y0,
                     "raw": float(np.nanmedian(img[core])), "corrected": float(np.nanmedian((img * f)[core]))})
    d = pd.DataFrame(rows)
    out = {}
    for col in ("raw", "corrected"):
        r2s = []
        for _, gg in d.groupby("run"):
            if len(gg) < 40:
                continue
            t = (gg.t - gg.t.iloc[0]).dt.total_seconds().values / 3600
            y = np.log(gg[col].values)
            X0 = np.c_[np.ones_like(t), t, t ** 2]
            res = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
            X = np.c_[gg.x0 - gg.x0.mean(), gg.y0 - gg.y0.mean()]
            pred = X @ np.linalg.lstsq(X, res, rcond=None)[0]
            r2s.append(1 - np.var(res - pred) / np.var(res))
        out[col] = {"median_r2": float(np.median(r2s)), "per_run": [float(v) for v in r2s]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=config.DATASET)
    ap.add_argument("--sigma", type=float, default=6.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(STORES / f"{a.store}.frames.parquet")
    tr = fr[fr.split == "train"]
    model = response.build(NM / f"noise_maps_{a.store}_train_g2.npz", tr.reg_x0.median(), tr.reg_y0.median(), a.sigma)
    path = OUT / f"response_{a.store}.npz"
    response.save(model, path, {"built_from": f"noise_maps_{a.store}_train_g2.npz (training split only)",
                                "store": a.store, **CFG["_meta"]})
    g = zarr.open_group(str(STORES / f"{a.store}.zarr"), mode="r")
    val = level_r2(g, fr, response.load(path), "val")
    summary = {"model": str(path), "ref_pointing": [model["ref_x0"], model["ref_y0"]], "sigma": a.sigma,
               "val_level_vs_pointing_r2": val, **CFG["_meta"]}
    (OUT / f"summary_{a.store}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("ref_pointing", "val_level_vs_pointing_r2")}, indent=1, default=float))


if __name__ == "__main__":
    main()
