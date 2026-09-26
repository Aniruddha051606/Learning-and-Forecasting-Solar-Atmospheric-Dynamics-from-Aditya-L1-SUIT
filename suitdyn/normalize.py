"""Normalisation variants for registered NB03 frames (the Phase 2 experiment compares them).

All take a registered frame (NaN = invalid) and a disk mask (valid on-disk pixels, r < 0.9 by default)
and return an image in the variant's units. NB03 is a single filter, so "per-filter" normalisation is
the same as "global" here; it matters only for Pipeline B.
"""
import numpy as np

from . import baselines


def global_(img, disk, level):
    """Calibrated counts divided by one constant (the training-set median disk level)."""
    return img / level


def per_frame_median(img, disk, level=None):
    return img / np.nanmedian(img[disk])


def robust_percentile(img, disk, level=None, lo=1.0, hi=99.0):
    a, b = np.nanpercentile(img[disk], [lo, hi])
    return (img - a) / (b - a)


def mu_map(grid, r_ref):
    x, y, _ = baselines._unit(grid, r_ref)
    return np.sqrt(np.clip(1 - x ** 2 - y ** 2, 0, None))


def quiet_sun_contrast(img, disk, level=None, mu=None, n_bins=20, bright=1.3):
    """Divide by the quiet-Sun centre-to-limb profile of this frame: median in μ bins after removing
    bright (plage) pixels, interpolated in μ. The result is contrast: 1 = quiet Sun at that μ."""
    edges = np.linspace(mu[disk].min(), 1.0, n_bins + 1)
    centres, prof = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = disk & (mu >= a) & (mu < b) & np.isfinite(img)
        if m.sum() < 200:
            continue
        v = img[m]
        med = np.median(v)
        q = v[v < bright * med]
        centres.append(0.5 * (a + b))
        prof.append(np.median(q))
    qs = np.interp(mu, centres, prof)
    return img / qs


VARIANTS = {"global": global_, "per_frame_median": per_frame_median, "robust_percentile": robust_percentile,
            "quiet_sun_contrast": quiet_sun_contrast}
