"""Detector fixed-pattern estimation and correction for NB03 binned frames.

Phase 1 found a pattern fixed on the detector (rms ~4 % of the disk level, network-like; PHASE1.md A4).
It is estimated here from the frames themselves: each frame is divided by a smoothed version of itself,
and the median of that relative residual is taken per detector pixel over many frames. Solar
structure moves across the detector (rotation, the ±10 px pointing oscillation, jumps), so it
averages out; the pattern does not.

Two details keep other artefacts out of the estimate:
  * the smoothing is a normalised convolution restricted to valid pixels (on the disk, not clipped,
    not spikes) and never crosses a quadrant seam, so neither the limb nor the seam step leaks in;
  * pixels are used only where they are well inside the disk (r < r_max) in that frame.

Whether the pattern is multiplicative (flat-field-like) or additive is tested, not assumed
(scripts/phase2_calibration.py): both are estimated on one half of the frames and judged on the other.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from . import geometry, qc


def seam_regions(shape, seam_px=1024):
    """Label image of the regions separated by the quadrant boundaries (0..3); the smoothing never
    mixes regions."""
    c = int(seam_px * shape[0] / 2048)
    yy, xx = np.indices(shape)
    return (xx >= c).astype(np.int8) + 2 * (yy >= c).astype(np.int8)


def smooth_within(im, valid, regions, sigma):
    """Normalised Gaussian smoothing of `im` over `valid` pixels, separately in each seam region."""
    out = np.full(im.shape, np.nan, np.float32)
    for k in np.unique(regions):
        w = (valid & (regions == k)).astype(np.float32)
        if w.sum() == 0:
            continue
        num = gaussian_filter(np.where(w > 0, im, 0).astype(np.float32), sigma)
        den = gaussian_filter(w, sigma)
        sel = (regions == k) & (den > 0.2)
        out[sel] = num[sel] / den[sel]
    return out


def frame_residuals(im, fit, clip_lo, clip_hi, *, sigma=15.0, r_max=0.95, seam_px=1024, spike_k=10.0):
    """Relative and absolute high-pass residuals of one frame (NaN where not usable)."""
    r = geometry.r_map_model(im.shape, fit)
    disk = r < r_max
    spk, _ = qc.spikes(im, r < 0.9, k=spike_k)
    valid = disk & (im > clip_lo) & (im < clip_hi) & ~spk
    sm = smooth_within(im, valid, seam_regions(im.shape, seam_px), sigma)
    ok = valid & np.isfinite(sm) & (sm > 0)
    rel = np.where(ok, im / np.where(ok, sm, 1) - 1, np.nan).astype(np.float32)
    ab = np.where(ok, im - np.where(ok, sm, 0), np.nan).astype(np.float32)
    return rel, ab


def nanmedian0(a):
    """Median over axis 0 ignoring NaN, by sorting (NaN sorts last, so the median of the c finite values
    sits at positions (c-1)//2 and c//2). Identical to np.nanmedian(a, 0) and ~25 % faster on
    (440, 64, 512) float32 stacks with 30 % NaN."""
    s = np.sort(a, axis=0)
    c = np.isfinite(a).sum(0)
    lo = np.clip((c - 1) // 2, 0, a.shape[0] - 1)
    hi = np.clip(c // 2, 0, a.shape[0] - 1)
    v = 0.5 * (np.take_along_axis(s, lo[None], 0)[0] + np.take_along_axis(s, hi[None], 0)[0])
    return np.where(c > 0, v, np.nan)


def blockwise_nanmedian(stack_path, shape, n, rows=128):
    """Median over frames of a (n, H, W) float16 memmap, row block by row block (bounded memory)."""
    mm = np.memmap(stack_path, dtype=np.float16, mode="r", shape=(n,) + shape)
    out = np.full(shape, np.nan, np.float32)
    cnt = np.zeros(shape, np.int32)
    for y0 in range(0, shape[0], rows):
        blk = mm[:, y0:y0 + rows, :].astype(np.float32)
        cnt[y0:y0 + rows] = np.isfinite(blk).sum(0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[y0:y0 + rows] = np.nanmedian(blk, 0)
    return out, cnt


def correct(im, pattern, mode):
    """Apply a pattern: 'multiplicative' divides by (1 + p), 'additive' subtracts p. NaN pattern = no change."""
    p = np.nan_to_num(pattern, nan=0.0)
    if mode == "multiplicative":
        return im / (1.0 + p)
    if mode == "additive":
        return im - p
    raise ValueError(mode)
