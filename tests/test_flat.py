import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

from suitdyn import flat


def test_smooth_within_does_not_leak_across_seam():
    im = np.full((256, 256), 1000.0, np.float32)
    im[:, 128:] = 700.0  # a 30 % step, like the NB03 seam
    valid = np.ones(im.shape, bool)
    # seam_px is given in 2048-frame units and scaled to the frame size: boundary at column 128 here
    sm = flat.smooth_within(im, valid, flat.seam_regions((2048, 2048), seam_px=1024)[::8, ::8], 15.0)
    rel = im / sm - 1
    assert np.nanmax(np.abs(rel)) < 1e-4  # the step does not appear as a high-pass residual


def test_multiplicative_pattern_recovered_from_moving_scene():
    """A scene drifting across the detector under a fixed multiplicative pattern: the per-pixel median of
    relative residuals recovers the pattern.
    """
    rng = np.random.default_rng(0)
    H = W = 384
    g = gaussian_filter(rng.normal(0, 1, (H + 80, W + 80)), 3)
    scene = 3000 * (1 + 0.10 * g / g.std())
    p = gaussian_filter(rng.normal(0, 1, (H, W)), 1.2)
    pattern = 0.04 * p / p.std()
    fit = {"x0": W / 2, "y0": H / 2, "R": 10 * H, "harm": []}  # disk covers everything
    rels = []
    for k in range(120):
        dy, dx = rng.uniform(-30, 30, 2)
        s = ndshift(scene, (dy, dx), order=1)[40:40 + H, 40:40 + W]
        im = (s * (1 + pattern)).astype(np.float32)
        rel, _ = flat.frame_residuals(im, fit, -1e9, 1e9, sigma=15.0, seam_px=10 ** 6)
        rels.append(rel)
    est = np.nanmedian(np.array(rels), 0)
    inner = (slice(20, -20), slice(20, -20))
    r = np.corrcoef(est[inner].ravel(), pattern[inner].ravel())[0, 1]
    assert r > 0.9
