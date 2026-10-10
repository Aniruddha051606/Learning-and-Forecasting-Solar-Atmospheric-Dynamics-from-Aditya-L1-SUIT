"""Solar-limb fitting, independent of the header pointing."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares


def edge_points(im, cx, cy, R, rays=720, window=0.07, smooth_px=2.0, edge_margin_px=40, valid=None):
    """Steepest intensity drop along each ray from a guessed centre, with sub-pixel refinement."""
    if valid is None:
        valid = np.isfinite(im)
    im = np.where(valid, im, np.nanmedian(im[valid]) if valid.any() else 0.0)
    sm = gaussian_filter(im, smooth_px)
    gy, gx = np.gradient(sm)
    th = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    rr = np.arange(R * (1 - window), R * (1 + window), 0.25)
    X = cx + np.outer(np.cos(th), rr)
    Y = cy + np.outer(np.sin(th), rr)
    H, W = im.shape
    m = edge_margin_px
    usable = ((X > m) & (X < W - 1 - m) & (Y > m) & (Y < H - 1 - m)).all(1)
    # dilate the invalid region by the smoothing scale before sampling it along the rays
    bad = gaussian_filter((~valid).astype(np.float32), smooth_px) > 0.01
    usable &= ~(map_coordinates(bad.astype(np.float32), [Y, X], order=0, mode="constant", cval=1) > 0.5).any(1)
    dr = (map_coordinates(gx, [Y, X], order=1, mode="nearest") * np.cos(th)[:, None]
          + map_coordinates(gy, [Y, X], order=1, mode="nearest") * np.sin(th)[:, None])
    i = np.clip(np.argmin(dr, 1), 1, len(rr) - 2)
    k = np.arange(rays)
    y0, y1, y2 = dr[k, i - 1], dr[k, i], dr[k, i + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0).clip(-1, 1)
    r_edge = rr[i] + frac * (rr[1] - rr[0])
    return th, cx + r_edge * np.cos(th), cy + r_edge * np.sin(th), usable


def _residuals(p, x, y, harmonics):
    phi = np.arctan2(y - p[1], x - p[0])
    model = p[2] + sum(p[3 + 2 * (m - 2)] * np.cos(m * phi) + p[4 + 2 * (m - 2)] * np.sin(m * phi)
                       for m in range(2, harmonics + 1))
    return np.hypot(x - p[0], y - p[1]) - model


def fit_limb(x, y, harmonics=4, clip_sigma=3.0, iters=8):
    """Robust joint least-squares fit of centre, mean radius and distortion harmonics to edge points."""
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    x0, y0, k = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    p = np.r_[x0, y0, np.sqrt(k + x0 ** 2 + y0 ** 2), np.zeros(2 * max(harmonics - 1, 0))]
    keep = np.ones(len(x), bool)
    for _ in range(iters):
        p = least_squares(_residuals, p, args=(x[keep], y[keep], harmonics), loss="soft_l1", f_scale=2.0).x
        res = _residuals(p, x, y, harmonics)
        sig = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) < max(clip_sigma * sig, 0.3)
        if (new == keep).all():
            break
        keep = new
    J = least_squares(_residuals, p, args=(x[keep], y[keep], harmonics), max_nfev=1).jac
    cov = np.linalg.pinv(J.T @ J) * np.var(res[keep])
    return {"x0": float(p[0]), "y0": float(p[1]), "R": float(p[2]), "harm": p[3:].tolist(),
            "sd_x0": float(np.sqrt(cov[0, 0])), "sd_y0": float(np.sqrt(cov[1, 1])),
            "rms": float(np.sqrt(np.mean(res[keep] ** 2))), "n_used": int(keep.sum()), "n_in": int(len(x))}


def limb(im, cx, cy, R, *, rays=720, window=0.07, smooth_px=2.0, edge_margin_px=40, harmonics=4, clip_sigma=3.0,
         scale=1.0, valid=None):
    """Edge detection plus fits, from a header guess."""
    th, ex, ey, usable = edge_points(im, cx, cy, R, rays, window, smooth_px * scale, edge_margin_px * scale, valid)
    x, y = ex[usable], ey[usable]
    if len(x) < 50:
        return None
    fit = fit_limb(x, y, harmonics, clip_sigma)
    # second pass from the fitted centre (the header guess can be ~10 px off)
    th, ex, ey, usable = edge_points(im, fit["x0"], fit["y0"], fit["R"], rays, window, smooth_px * scale,
                                     edge_margin_px * scale, valid)
    x, y = ex[usable], ey[usable]
    fit = fit_limb(x, y, harmonics, clip_sigma)
    circ = fit_limb(x, y, 0, clip_sigma)
    fit["angular_coverage"] = float(usable.mean())
    return {"limb": fit, "circle": circ, "edges": {"theta": th, "x": ex, "y": ey, "usable": usable}}


def disk_radius_map(shape, x0, y0, R):
    yy, xx = np.indices(shape, dtype=np.float32)
    return np.hypot(xx - x0, yy - y0) / R


def r_map_model(shape, fit):
    """Distance from the fitted centre in units of the fitted (distorted) limb radius at that angle, so r = 1
    lies on the measured limb everywhere.
    """
    yy, xx = np.indices(shape, dtype=np.float32)
    dx, dy = xx - fit["x0"], yy - fit["y0"]
    phi = np.arctan2(dy, dx)
    rad = np.full(shape, fit["R"], np.float32)
    h = fit["harm"]
    for j, m in enumerate(range(2, 2 + len(h) // 2)):
        rad += h[2 * j] * np.cos(m * phi) + h[2 * j + 1] * np.sin(m * phi)
    return np.hypot(dx, dy) / rad
