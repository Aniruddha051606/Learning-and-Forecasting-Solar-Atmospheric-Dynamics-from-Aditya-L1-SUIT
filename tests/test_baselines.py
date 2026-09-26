import numpy as np
from scipy.ndimage import gaussian_filter

from suitdyn import baselines, metrics, solar


def test_disk_centre_moves_by_the_phase1_rotation_model():
    """At disk centre, derotation must agree with suitdyn.solar (the model validated on NB03 in Phase 1)."""
    grid, r_ref, b0, dt = 512, 230.0, 7.0, 3600.0
    src, ok = baselines.derotation_coords(grid, r_ref, b0, dt)
    c = (grid - 1) / 2
    ic = int(round(c))
    dx_expected, _ = solar.disk_centre_motion_px(r_ref, b0, dt, 0.0)
    # source column of the target pixel at disk centre = that pixel's column − motion
    assert abs((ic - src[1, ic, ic]) - dx_expected) < 0.05
    assert abs(src[0, ic, ic] - ic) < 0.2
    assert ok[ic, ic]


def test_pixels_rotating_in_from_behind_the_east_limb_are_invalid():
    grid, r_ref = 512, 230.0
    frame = np.ones((grid, grid))
    out = baselines.rotated_persistence(frame, r_ref, 7.0, 24 * 3600.0)
    c = (grid - 1) / 2
    # 2 px inside the east limb (longitude ≈ −83°): 24 h ago (≈ 13° of rotation) it was behind the limb
    assert np.isnan(out[int(c), int(c - r_ref + 2)])
    assert np.isfinite(out[int(c), int(c)])


def test_b1_beats_b0_on_a_rotating_synthetic_sun():
    grid, r_ref, b0 = 384, 170.0, 7.0
    rng = np.random.default_rng(0)
    base = 1 + 0.2 * gaussian_filter(rng.normal(0, 1, (grid, grid)), 2) / 0.1
    lat, lon = baselines.heliographic(grid, r_ref, b0)
    f0 = np.where(np.isfinite(lat), base, np.nan)
    truth = baselines.rotated_persistence(f0, r_ref, b0, 4 * 3600.0)  # the "future" is the rotated past
    b1 = baselines.rotated_persistence(f0, r_ref, b0, 4 * 3600.0)
    m = np.isfinite(truth)
    assert metrics.mae(b1, truth, m) < 1e-9
    assert metrics.mae(f0, truth, m) > 0.05


def test_metrics_basic_properties():
    rng = np.random.default_rng(1)
    t = 3000 + 300 * gaussian_filter(rng.normal(0, 1, (200, 200)), 2) / 0.1
    assert metrics.mae(t, t) == 0 and abs(metrics.ssim(t, t) - 1) < 1e-9
    assert abs(metrics.gradient_correlation(t, t) - 1) < 1e-9
    noisy = t + rng.normal(0, 200, t.shape)
    assert metrics.ssim(noisy, t) < 0.9
