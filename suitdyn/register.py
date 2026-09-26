"""Registration of full-disk frames to a common disk-centred, solar-north-up grid.

The transform follows the FITS WCS convention used in the SUIT headers (CTYPE HPLN-TAN, CROTA2):

    helioprojective (arcsec) = CDELT · Rot(CROTA2) · (pixel − CRPIX)

with two substitutions measured from the image, not taken from the header:
  CRPIX → the fitted limb centre (suitdyn.geometry), because the header centre is off by up to ~12 px;
  CDELT → RSUN_OBS / R_fit, so every frame's disk has radius r_ref on the output grid.
The rotation is the header CROTA2 (= P_ANGLE); it cannot be checked from the limb, so it is recorded
as an assumption. Output grid pixel (u, v) maps to helioprojective (u − c, v − c) · RSUN_OBS / r_ref
with c = (grid − 1) / 2.

Nothing is warped beyond this similarity transform: the Level-1 distortion (±5-10 px limb harmonics)
is left in place and recorded, not corrected.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def transform(x0, y0, R_fit, crota2_deg, grid, r_ref):
    """Affine map output-grid (u, v) → source pixel (x, y) (0-based), as (A, b) with src = A @ [u, v] + b.

    Derivation: hp = (rsun/r_ref)·([u,v] − c) and src = centre + Rot(−ρ)·hp / cdelt_fit, where
    cdelt_fit = rsun / R_fit, so src = centre + (R_fit / r_ref)·Rot(−ρ)·([u,v] − c)."""
    c = (grid - 1) / 2
    A = (R_fit / r_ref) * rot(-np.deg2rad(crota2_deg))
    b = np.array([x0, y0]) - A @ np.array([c, c])
    return A, b


def apply(im, A, b, grid, order=1, cval=np.nan):
    """Resample `im` onto the output grid. When the output pixel is coarser than the source (4096
    frames onto the common grid), the source is first smoothed to avoid aliasing."""
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
