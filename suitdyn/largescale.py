"""Self-calibration of the NB03 large-scale detector response from the frames themselves.

Model, fitted in log space on sampled quiet-disk pixels of many frames:
    log I_k(x) = r(x) + q(mu_k(x)) + a_k
  r(x)   large-scale log response on the detector (2048 binned grid): bilinear splines on a coarse
         knot grid, with SEPARATE knot sets left and right of the vertical quadrant seam (x = 1024) so a
         row-dependent seam step is allowed;
  q(mu)  one quiet-Sun limb-darkening profile, shared by all frames (linear spline in mu);
  a_k    per-frame log level (program steps, exposure, whole-disk variation).
What makes r and q separable: the Sun sits at two detector positions 480 px apart (the two pointing
modes) plus the ±10 px pointing oscillation, so the same mu falls on different detector pixels.
Degeneracies (one global constant between r, q and a) are removed by constraint rows; knots the data
do not reach are held by a smoothness penalty.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

SEAM_X = 1024.0


class Basis:
    def __init__(self, size=2048, knot=32, mu_knots=np.linspace(0.15, 1.0, 35)):
        self.size, self.knot = size, knot
        self.ny = int(np.ceil(size / knot)) + 1
        # left side covers x in [0, SEAM_X], right side [SEAM_X, size]; each has its own knot columns
        self.nxl = int(np.ceil(SEAM_X / knot)) + 1
        self.nxr = int(np.ceil((size - SEAM_X) / knot)) + 1
        self.n_r = self.ny * (self.nxl + self.nxr)
        self.mu_knots = np.asarray(mu_knots)
        self.n_q = len(self.mu_knots)

    def _r_index(self, x, y):
        """Bilinear weights of r at detector points: returns (cols, weights) each shaped (n, 4)."""
        right = x >= SEAM_X
        xs = np.where(right, x - SEAM_X, x) / self.knot
        ys = y / self.knot
        ix, iy = np.floor(xs).astype(int), np.floor(ys).astype(int)
        nx = np.where(right, self.nxr, self.nxl)
        ix = np.clip(ix, 0, nx - 2)
        iy = np.clip(iy, 0, self.ny - 2)
        fx, fy = xs - ix, ys - iy
        off = np.where(right, self.ny * self.nxl, 0)

        def col(i, j):
            return off + j * nx + i

        cols = np.stack([col(ix, iy), col(ix + 1, iy), col(ix, iy + 1), col(ix + 1, iy + 1)], 1)
        w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
        return cols, w

    def _q_index(self, mu):
        k = self.mu_knots
        i = np.clip(np.searchsorted(k, mu) - 1, 0, len(k) - 2)
        f = np.clip((mu - k[i]) / (k[i + 1] - k[i]), 0, 1)
        return np.stack([i, i + 1], 1), np.stack([1 - f, f], 1)

    def design(self, x, y, mu, frame, n_frames):
        rc, rw = self._r_index(x, y)
        qc, qw = self._q_index(mu)
        n = len(x)
        rows = np.repeat(np.arange(n), 7)
        cols = np.concatenate([rc, self.n_r + qc, (self.n_r + self.n_q + frame)[:, None]], 1).ravel()
        vals = np.concatenate([rw, qw, np.ones((n, 1))], 1).ravel()
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, self.n_r + self.n_q + n_frames))

    def smoothness(self, n_frames, weight):
        """Second differences of r along x and y within each side, and of q along mu."""
        rows, cols, vals = [], [], []
        r = 0
        for off, nx in ((0, self.nxl), (self.ny * self.nxl, self.nxr)):
            for j in range(self.ny):
                for i in range(1, nx - 1):
                    rows += [r] * 3
                    cols += [off + j * nx + i - 1, off + j * nx + i, off + j * nx + i + 1]
                    vals += [1, -2, 1]
                    r += 1
            for i in range(nx):
                for j in range(1, self.ny - 1):
                    rows += [r] * 3
                    cols += [off + (j - 1) * nx + i, off + j * nx + i, off + (j + 1) * nx + i]
                    vals += [1, -2, 1]
                    r += 1
        for i in range(1, self.n_q - 1):
            rows += [r] * 3
            cols += [self.n_r + i - 1, self.n_r + i, self.n_r + i + 1]
            vals += [1, -2, 1]
            r += 1
        return sparse.csr_matrix((np.array(vals, float) * weight, (rows, cols)),
                                 shape=(r, self.n_r + self.n_q + n_frames))

    def constraints(self, n_frames, weight=1e3):
        """q(mu = 1) = 0 and mean(r over knots) = 0: fixes the constants shared by r, q and a."""
        n = self.n_r + self.n_q + n_frames
        c = sparse.lil_matrix((2, n))
        c[0, self.n_r + self.n_q - 1] = weight
        c[1, :self.n_r] = weight / self.n_r
        return c.tocsr()

    def evaluate_r(self, coef, size=None, step=1):
        """r on the full detector grid (optionally every `step` pixels)."""
        size = size or self.size
        yy, xx = np.mgrid[0:size:step, 0:size:step].astype(np.float64)
        cols, w = self._r_index(xx.ravel(), yy.ravel())
        return (coef[cols] * w).sum(1).reshape(xx.shape)

    def evaluate_q(self, coef, mu):
        qc, qw = self._q_index(np.asarray(mu, float))
        return (coef[self.n_r + qc] * qw).sum(1)


def fit(basis, x, y, mu, frame, logi, n_frames, smooth=3.0, huber=0.03, irls=3):
    """Robust (Huber-weighted) least squares for [r, q, a]. Returns coefficients and final residuals."""
    A = basis.design(x, y, mu, frame, n_frames)
    S = basis.smoothness(n_frames, smooth)
    C = basis.constraints(n_frames)
    w = np.ones(len(logi))
    coef = None
    for _ in range(irls):
        Aw = sparse.diags(np.sqrt(w)) @ A
        M = sparse.vstack([Aw, S, C]).tocsr()
        b = np.concatenate([np.sqrt(w) * logi, np.zeros(S.shape[0] + C.shape[0])])
        coef = lsqr(M, b, atol=1e-8, btol=1e-8, iter_lim=4000, x0=coef)[0]
        res = logi - A @ coef
        a = np.abs(res)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
    return coef, logi - A @ coef
