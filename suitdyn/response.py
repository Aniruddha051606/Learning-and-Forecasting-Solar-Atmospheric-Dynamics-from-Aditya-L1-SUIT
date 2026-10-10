"""First-order correction of the large-scale detector response seen through the pointing motion."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def nan_smooth(a, sigma):
    w = np.isfinite(a).astype(np.float32)
    num = gaussian_filter(np.nan_to_num(a), sigma)
    den = gaussian_filter(w, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.3, num / den, np.nan)


def build(noise_npz, ref_x0, ref_y0, sigma=6.0):
    d = np.load(noise_npz)
    sx = nan_smooth(d["slope_x"], sigma)
    sy = nan_smooth(d["slope_y"], sigma)
    return {"slope_x": sx.astype(np.float32), "slope_y": sy.astype(np.float32), "ref_x0": float(ref_x0),
            "ref_y0": float(ref_y0), "sigma": float(sigma)}


def save(model, path, meta):
    np.savez_compressed(path, slope_x=model["slope_x"], slope_y=model["slope_y"])
    Path(str(path) + ".json").write_text(json.dumps({k: model[k] for k in ("ref_x0", "ref_y0", "sigma")} | meta,
                                                    indent=1, default=str))


def load(path):
    d = np.load(path)
    meta = json.loads(Path(str(path) + ".json").read_text())
    return {"slope_x": d["slope_x"], "slope_y": d["slope_y"], **meta}


def factor(model, grid, x0, y0):
    """Multiplicative correction for a frame at detector pointing (x0, y0), on a grid of size `grid`."""
    sx, sy = model["slope_x"], model["slope_y"]
    if sx.shape[0] != grid:
        idx = (np.arange(grid) * sx.shape[0] / grid).astype(int)
        sx, sy = sx[np.ix_(idx, idx)], sy[np.ix_(idx, idx)]
    arg = (np.nan_to_num(sx) * (x0 - model["ref_x0"]) + np.nan_to_num(sy) * (y0 - model["ref_y0"])) / 100.0
    return np.exp(arg).astype(np.float32)
