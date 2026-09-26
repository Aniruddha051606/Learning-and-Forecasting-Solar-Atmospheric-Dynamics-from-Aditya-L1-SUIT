import numpy as np

from suitdyn import qc


def _scene(seed=0, n=512):
    rng = np.random.default_rng(seed)
    im = 3000 + rng.normal(0, 40, (n, n))
    yy, xx = np.indices((n, n))
    # bright "network" blobs a few pixels across (FWHM ~4 px), as in NB03 at 1.4″/px
    feats = rng.integers(20, n - 20, (60, 2))
    for y, x in feats:
        im += 1500 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.7 ** 2))
    return im.astype(np.float32), feats


def test_spikes_recovered_and_features_not_flagged():
    im, feats = _scene()
    rng = np.random.default_rng(1)
    pos = rng.integers(10, 502, (200, 2))
    for y, x in pos:
        im[y, x] += rng.uniform(3000, 30000)
    disk = np.ones(im.shape, bool)
    spk, _ = qc.spikes(im, disk, k=10, rel=0.5, max_area=6)
    recovered = spk[pos[:, 0], pos[:, 1]].mean()
    assert recovered > 0.95
    # no detection at the centre of a solar-like feature
    assert spk[feats[:, 0], feats[:, 1]].mean() < 0.02


def test_seam_profile_measures_known_step():
    n = 512
    im = np.full((n, n), 3000.0)
    im[:, 256:] *= 0.87
    disk = np.ones((n, n), bool)
    _, step = qc.seam_profile(im, disk, 256, axis=1, band=64)
    assert np.allclose(step, -0.13, atol=1e-6)
