"""Scientific QC: per-pixel artifact mask and per-frame quality metrics.

The Level-1 quality keywords (QVAL, QDESC, NSPIKES) are constant across the archive and carry no
information (DESIGN §0), so they are kept as metadata only and never used to accept a frame.

Pixel mask bits (uint8):
"""
import numpy as np
from scipy.ndimage import label, median_filter

CLIP_LO = 1      # at the lowest encodable value (BZERO + BSCALE * -32768): not a measurement
CLIP_HI = 2      # at the highest encodable value: saturated or overflowed
SPIKE = 4        # isolated positive outlier (cosmic ray / hot pixel)
SEAM = 8         # CCD quadrant boundary band
OFF_LIMB = 16    # outside the fitted limb (r > 1): the Level-1 scatter correction makes it unreliable
EDGE = 32        # within the vignetted CCD margin
BITS = {"clip_lo": CLIP_LO, "clip_hi": CLIP_HI, "spike": SPIKE, "seam": SEAM, "off_limb": OFF_LIMB, "edge": EDGE}


def spikes(im, disk, k=10.0, rel=0.5, max_area=6, size=5):
    """Isolated positive outliers: residual from a size×size median above k robust sigma (estimated on
    the disk) and above `rel` times the local level, in connected groups of at most max_area pixels.

    The area limit keeps small solar features (network grains are larger than a few 1.4″ pixels) out;
    tests/test_qc.py and the temporal-persistence check in the registration study verify this."""
    med = median_filter(im, size)
    res = im - med
    r = res[disk]
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    cand = (res > k * sig) & (res > rel * np.maximum(np.abs(med), sig))
    lab, n = label(cand)
    if n == 0:
        return cand, sig
    area = np.bincount(lab.ravel())
    ok = area <= max_area
    ok[0] = False
    return ok[lab], sig


def seam_boundaries(shape, seam_px):
    """Candidate quadrant boundaries (first pixel after the boundary), scaled to the frame size."""
    c = int(seam_px * shape[0] / 2048)
    return c, c  # column and row


def seam_profile(im, disk, c, axis, band=64, gap=(11, 5), width=6):
    """Relative step across a boundary at index c, per band of `band` pixels along the boundary.

    axis=1: vertical boundary between columns c-1 and c; axis=0: horizontal, between rows c-1 and c.
    Compares the median of `width` pixels just after the boundary with the median of the pixels
    c-gap[0]..c-gap[1] before it, skipping the bright strip found right before the boundary.
    Returns (band centres, step ratios after/before - 1), NaN where the band is off the disk."""
    a = im if axis == 1 else im.T
    m = disk if axis == 1 else disk.T
    lo, hi = c - gap[0], c - gap[1]
    out_pos, out = [], []
    for s in range(0, a.shape[0], band):
        rows = slice(s, s + band)
        L, Lm = a[rows, lo:hi], m[rows, lo:hi]
        Rr, Rm = a[rows, c:c + width], m[rows, c:c + width]
        out_pos.append(s + band / 2)
        if Lm.mean() > 0.9 and Rm.mean() > 0.9:
            out.append(float(np.median(Rr) / np.median(L) - 1))
        else:
            out.append(np.nan)
    return np.array(out_pos), np.array(out)


def pixel_mask(im, r_map, clip_lo, clip_hi, spike_mask, seam_px, seam_halfwidth, edge_px):
    m = np.zeros(im.shape, np.uint8)
    m[im <= clip_lo] |= CLIP_LO
    m[im >= clip_hi] |= CLIP_HI
    m[spike_mask] |= SPIKE
    c, _ = seam_boundaries(im.shape, seam_px)
    hw = int(seam_halfwidth * im.shape[0] / 2048)
    # the vertical seam plus the bright strip just before it (DESIGN §0 #5, measured in Phase 1)
    m[:, c - 5 - hw:c + hw] |= SEAM
    m[c - hw:c + hw, :] |= SEAM
    m[r_map > 1.0] |= OFF_LIMB
    e = edge_px
    m[:e, :] |= EDGE
    m[-e:, :] |= EDGE
    m[:, :e] |= EDGE
    m[:, -e:] |= EDGE
    return m


def stats(v, prefix):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    p = np.percentile(v, [1, 5, 50, 95, 99, 99.9])
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std()), f"{prefix}_min": float(v.min()),
            f"{prefix}_max": float(v.max()), f"{prefix}_p1": p[0], f"{prefix}_p5": p[1], f"{prefix}_median": p[2],
            f"{prefix}_p95": p[3], f"{prefix}_p99": p[4], f"{prefix}_p999": p[5]}
