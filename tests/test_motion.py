import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

from suitdyn import motion


def test_fixed_pattern_removal_recovers_subpixel_shift():
    """A scene moving by 0.3 px over a strong fixed pattern: raw correlation locks near 0, with the
    pattern removed the shift is recovered (the Phase 1 failure mode, reproduced)."""
    rng = np.random.default_rng(0)
    H, W = 1200, 2048
    scene = gaussian_filter(rng.normal(0, 1, (H, W)), 3) * 3000 + 3000
    pattern = gaussian_filter(rng.normal(0, 1, (H, W)), 1.0) * 400
    frames = []
    for k in range(60):  # scene drifts across the detector, pattern stays
        frames.append(ndshift(scene, (rng.uniform(-15, 15), rng.uniform(-15, 15)), order=1) + pattern)
    fp, split_r, _ = motion.fixed_pattern(frames)
    assert split_r > 0.8
    a = scene + pattern
    b = ndshift(scene, (0.0, 0.3), order=3) + pattern
    raw = motion.shift(motion.prepared(a, 0 * fp), motion.prepared(b, 0 * fp))
    fixed = motion.shift(motion.prepared(a, fp), motion.prepared(b, fp))
    assert abs(raw[0]) < 0.15            # locked on the pattern
    assert abs(fixed[0] - 0.3) < 0.08    # recovered
