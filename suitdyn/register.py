"""Registration of full-disk frames to a common disk-centred, solar-north-up grid."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def transform(x0, y0, R_fit, crota2_deg, grid, r_ref):
    """Affine map output-grid (u, v) → source pixel (x, y) (0-based), as (A, b) with src = A @ [u, v] + b."""
    c = (grid - 1) / 2
    A = (R_fit / r_ref) * rot(-np.deg2rad(crota2_deg))
    b = np.array([x0, y0]) - A @ np.array([c, c])
    return A, b


def apply(im, A, b, grid, order=1, cval=np.nan):
    """Resample `im` onto the output grid."""
    scale = np.sqrt(abs(np.linalg.det(A)))
    src = gaussian_filter(im, 0.5 * (scale - 1)) if scale > 1.2 else im
    v, u = np.indices((grid, grid), dtype=np.float64)
    x = A[0, 0] * u + A[0, 1] * v + b[0]
    y = A[1, 0] * u + A[1, 1] * v + b[1]
    return map_coordinates(src, [y, x], order=order, mode="constant", cval=cval)


def apply_mask(mask, A, b, grid):
    """Nearest-neighbour resampling of a bit mask; pixels with no source get the EDGE-like value 255."""
    v, u = np.indices((grid, grid), dtype=np.float64)
    x = A[0, 0] * u + A[0, 1] * v + b[0]
    y = A[1, 0] * u + A[1, 1] * v + b[1]
    out = map_coordinates(mask, [y, x], order=0, mode="constant", cval=255)
    return out.astype(np.uint8)
