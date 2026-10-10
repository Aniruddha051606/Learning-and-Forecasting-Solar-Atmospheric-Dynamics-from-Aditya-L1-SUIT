import numpy as np

from suitdyn import geometry


def synthetic_disk(shape=(2048, 2048), x0=1280.3, y0=598.7, R=697.0, harm=((2, 6.0, 1.0), (3, -4.0, 5.0)), seed=0):
    """Limb-darkened disk whose edge carries m=2,3 distortion, placed so the south limb is off the CCD, like
    SUIT Level-1 NB03.
    """
    yy, xx = np.indices(shape, dtype=np.float64)
    phi = np.arctan2(yy - y0, xx - x0)
    redge = R + sum(a * np.cos(m * phi) + b * np.sin(m * phi) for m, a, b in harm)
    r = np.hypot(xx - x0, yy - y0) / redge
    mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))
    img = np.where(r < 1, 3000 * (0.4 + 0.6 * mu), -400.0)
    # soften the edge over ~2 px, as the PSF does
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, 1.5)
    img += np.random.default_rng(seed).normal(0, 30, shape)
    return img.astype(np.float32)


def test_harmonic_fit_recovers_centre_of_distorted_truncated_disk():
    x0, y0, R = 1280.3, 598.7, 697.0
    img = synthetic_disk(x0=x0, y0=y0, R=R)
    out = geometry.limb(img, 1290.0, 610.0, 690.0, harmonics=4)  # guess ~15 px off, like the header
    fit = out["limb"]
    assert abs(fit["x0"] - x0) < 0.3 and abs(fit["y0"] - y0) < 0.3
    assert abs(fit["R"] - R) < 0.5


def test_plain_circle_is_biased_by_partial_coverage():
    """Documents why harmonics are fitted: with the south limb missing, a circle's centre is pulled off."""
    x0, y0 = 1280.3, 598.7
    out = geometry.limb(synthetic_disk(x0=x0, y0=y0), 1290.0, 610.0, 690.0, harmonics=4)
    c = out["circle"]
    assert np.hypot(c["x0"] - x0, c["y0"] - y0) > 1.0


def test_undistorted_disk_circle_and_harmonics_agree():
    img = synthetic_disk(harm=())
    out = geometry.limb(img, 1285.0, 605.0, 690.0, harmonics=4)
    assert abs(out["limb"]["x0"] - out["circle"]["x0"]) < 0.3
    assert abs(out["limb"]["y0"] - out["circle"]["y0"]) < 0.3
