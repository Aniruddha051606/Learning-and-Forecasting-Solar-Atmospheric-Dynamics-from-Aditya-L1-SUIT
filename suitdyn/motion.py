"""Image-content motion between NB03 frames, with the detector's fixed pattern removed.

Phase 1 findings (docs/PHASE1.md, Registration) that shape this module:
  * Level-1 NB03 frames carry a pattern fixed on the CCD (residual flat field / PRNU / dust), rms
    ~126 counts in the high-passed disk, reproducible between independent halves of the archive
    (r = 0.98). Correlating two raw frames then peaks at zero shift whenever the true shift is
    sub-pixel, which hides the real pointing jitter.
  * The pattern is estimated as the median of high-passed crops of many frames in detector
    coordinates: solar structure moves across the detector (rotation + pointing), the pattern does
    not. It is subtracted before correlating.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

# Detector box (2048 frame) used for motion: inside the disk for all observed pointings, clear of the
# vertical quadrant seam at x = 1024 and of the limb.
BOX = (300, 876, 1040, 1616)
HIGHPASS_SIGMA = 15.0


def highpass(im, box=BOX, sigma=HIGHPASS_SIGMA):
    y0, y1, x0, x1 = box
    c = im[y0:y1, x0:x1].astype(np.float64)
    c = c - gaussian_filter(c, sigma)
    m = np.median(c)
    s = 1.4826 * np.median(np.abs(c - m))
    return np.clip(c, -6 * s, 6 * s)


def fixed_pattern(images):
    """Median of high-passed crops; also returns a split-half correlation as a reproducibility check."""
    st = np.array([highpass(im) for im in images])
    fp = np.median(st, 0)
    a, b = np.median(st[::2], 0), np.median(st[1::2], 0)
    return fp, float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), float(st.std(axis=(1, 2)).mean())


def prepared(im, fp):
    c = highpass(im) - fp
    return c * np.outer(np.hanning(c.shape[0]), np.hanning(c.shape[1]))


def shift(ref_prepared, mov_prepared, upsample=50):
    """Motion (dx, dy) of image content in `mov` relative to `ref`, detector px (+x = solar west)."""
    s, err, _ = phase_cross_correlation(ref_prepared, mov_prepared, upsample_factor=upsample, normalization=None)
    return float(-s[1]), float(-s[0]), float(err)
