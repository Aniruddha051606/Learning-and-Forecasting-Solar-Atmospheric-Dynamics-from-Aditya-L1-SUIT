import numpy as np

from suitdyn import largescale


def test_recovers_response_with_seam_step_from_two_pointings():
    """Synthetic detector response (smooth gradient + a row-dependent step at the seam) under a limb-darkened
    disk observed at two pointings 480 px apart plus jitter: the fit must recover the response (up to the
    constant fixed by the constraints) and the seam step.
    """
    rng = np.random.default_rng(0)
    size = 2048

    def true_r(x, y):
        grad = -0.25 * ((x - 1024) / 1024) ** 2 + 0.05 * (y - 1024) / 1024
        step = np.where(x >= 1024, -0.2 * np.clip((900 - y) / 900, 0, 1), 0.0)
        return grad + step

    def q(mu):
        return np.log(0.3 + 0.7 * mu)

    centres = [(1033, 1012)] * 40 + [(1282, 598)] * 60
    X, Y, MU, F, L = [], [], [], [], []
    for k, (cx, cy) in enumerate(centres):
        cx, cy = cx + rng.normal(0, 5), cy + rng.normal(0, 5)
        x = rng.uniform(0, size, 6000)
        y = rng.uniform(0, size, 6000)
        rr = np.hypot(x - cx, y - cy) / 697
        ok = rr < 0.98
        x, y, rr = x[ok], y[ok], rr[ok]
        mu = np.sqrt(1 - rr ** 2)
        a = rng.normal(0, 0.02)
        L.append(true_r(x, y) + q(mu) + a + rng.normal(0, 0.01, len(x)))
        X.append(x), Y.append(y), MU.append(mu), F.append(np.full(len(x), k))
    X, Y, MU, F, L = map(np.concatenate, (X, Y, MU, F, L))
    b = largescale.Basis(size=size, knot=64)
    coef, res = largescale.fit(b, X, Y, MU, F, L, len(centres), smooth=1.0)
    r = b.evaluate_r(coef, step=16)
    yy, xx = np.mgrid[0:size:16, 0:size:16]
    t = true_r(xx, yy)
    covered = np.zeros_like(r, bool)
    for cx, cy in set(centres):
        covered |= np.hypot(xx - cx, yy - cy) < 650
    d = (r - t)[covered]
    d -= d.mean()
    assert np.sqrt(np.mean(d ** 2)) < 0.02
    # seam step at row 300: right minus left just across x = 1024
    row = 300 // 16
    step_fit = r[row, 1024 // 16] - r[row, 1024 // 16 - 1]
    step_true = t[row, 1024 // 16] - t[row, 1024 // 16 - 1]
    assert abs(step_fit - step_true) < 0.03
