"""Forecast baselines on the registered grid (disk centred, radius r_ref, solar north up, +x = west).

B0 persistence:                  F(t+Δ) = F(t)
B1 rotation-corrected persistence: F(t) moved by solar differential rotation over Δ
B2 optical-flow extrapolation:   B1 plus the residual motion measured between the two most recent
                                 frames (after derotation), extrapolated linearly over Δ
The disk is treated as a sphere seen orthographically (the perspective correction at L1 is ~0.3° at
the limb and is neglected); pixels that were not on the visible disk at t are invalid (NaN).
"""
import numpy as np
from scipy.ndimage import map_coordinates

from . import solar


def _unit(grid, r_ref):
    c = (grid - 1) / 2
    v, u = np.indices((grid, grid), dtype=np.float64)
    return (u - c) / r_ref, (v - c) / r_ref, c


def heliographic(grid, r_ref, b0_deg):
    """Latitude and longitude (radians, longitude from the central meridian, + west) of every pixel;
    NaN off the disk."""
    x, y, _ = _unit(grid, r_ref)
    rho2 = x ** 2 + y ** 2
    z = np.sqrt(np.clip(1 - rho2, 0, None))
    b = np.deg2rad(b0_deg)
    lat = np.arcsin(np.clip(y * np.cos(b) + z * np.sin(b), -1, 1))
    lon = np.arctan2(x, z * np.cos(b) - y * np.sin(b))
    off = rho2 >= 1
    lat[off] = np.nan
    lon[off] = np.nan
    return lat, lon


def derotation_coords(grid, r_ref, b0_deg, dt_s):
    """Source pixel coordinates (rows, cols) in the frame at t for every pixel of the frame at t+dt_s,
    and a mask of pixels whose source was on the visible disk."""
    lat, lon = heliographic(grid, r_ref, b0_deg)
    w = np.deg2rad(solar.synodic_deg_per_day(np.rad2deg(lat))) / 86400.0
    lon_s = lon - w * dt_s
    b = np.deg2rad(b0_deg)
    xs = np.cos(lat) * np.sin(lon_s)
    ys = np.sin(lat) * np.cos(b) - np.cos(lat) * np.cos(lon_s) * np.sin(b)
    zs = np.sin(lat) * np.sin(b) + np.cos(lat) * np.cos(lon_s) * np.cos(b)
    c = (grid - 1) / 2
    ok = np.isfinite(zs) & (zs > 0)
    return np.stack([c + r_ref * ys, c + r_ref * xs]), ok


def persistence(frame):
    return frame.copy()


def rotated_persistence(frame, r_ref, b0_deg, dt_s, coords=None):
    """B1. `coords` (from derotation_coords) can be passed when many frames share the same Δ and B0."""
    grid = frame.shape[0]
    if coords is None:
        coords = derotation_coords(grid, r_ref, b0_deg, dt_s)
    src, ok = coords
    out = map_coordinates(np.nan_to_num(frame, nan=0.0), src, order=1, mode="constant", cval=np.nan)
    valid_src = map_coordinates(np.isfinite(frame).astype(np.float32), src, order=0, mode="constant", cval=0) > 0.5
    out[~(ok & valid_src)] = np.nan
    return out


def optical_flow_extrapolation(prev, cur, r_ref, b0_deg, dt_prev_s, dt_s, win=31, levels=3):
    """B2. Residual flow between `prev` derotated to the time of `cur` and `cur` (Farneback, OpenCV),
    scaled by dt_s / dt_prev_s and applied on top of B1. Returns (forecast, flow)."""
    import cv2
    p = rotated_persistence(prev, r_ref, b0_deg, dt_prev_s)
    both = np.isfinite(p) & np.isfinite(cur)
    lo, hi = np.nanpercentile(cur[both], [1, 99.5])
    to8 = lambda a: np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)  # noqa: E731
    flow = cv2.calcOpticalFlowFarneback(to8(p), to8(cur), None, 0.5, levels, win, 3, 5, 1.1, 0)
    flow[~both] = 0.0
    b1 = rotated_persistence(cur, r_ref, b0_deg, dt_s)
    k = dt_s / dt_prev_s
    grid = cur.shape[0]
    v, u = np.indices((grid, grid), dtype=np.float64)
    # backward warp: the value at x came from x − k·flow
    src = np.stack([v - k * flow[..., 1], u - k * flow[..., 0]])
    out = map_coordinates(np.nan_to_num(b1, nan=0.0), src, order=1, mode="constant", cval=np.nan)
    ok = map_coordinates(np.isfinite(b1).astype(np.float32), src, order=0, mode="constant", cval=0) > 0.5
    out[~ok] = np.nan
    return out, flow
