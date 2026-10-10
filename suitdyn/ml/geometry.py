"""Derotation on the GPU: the torch counterpart of suitdyn.baselines.derotation_coords / rotated_persistence."""
import math

import torch
import torch.nn.functional as F

from .. import solar

_BASE = {}


def _base(G, r_ref, device):
    key = (G, float(r_ref), str(device))
    if key not in _BASE:
        c = (G - 1) / 2
        v, u = torch.meshgrid(torch.arange(G, device=device, dtype=torch.float32),
                              torch.arange(G, device=device, dtype=torch.float32), indexing="ij")
        x, y = (u - c) / r_ref, (v - c) / r_ref
        _BASE[key] = (x, y, torch.sqrt(torch.clamp(1 - x * x - y * y, min=0.0)), x * x + y * y >= 1)
    return _BASE[key]


def derotation_grid(G, r_ref, b0_deg, dt_s, rate_offset=None):
    """b0_deg: (B,) tensor, dt_s: (B, K) tensor of seconds (target time minus source time)."""
    dev = dt_s.device
    B, K = dt_s.shape
    x, y, z, off = _base(G, r_ref, dev)
    b = torch.deg2rad(b0_deg.float()).view(B, 1, 1)
    lat = torch.asin(torch.clamp(y * torch.cos(b) + z * torch.sin(b), -1, 1))       # (B, G, G)
    lon = torch.atan2(x.expand(B, G, G), z * torch.cos(b) - y * torch.sin(b))
    s2 = torch.sin(lat) ** 2
    a, b2, c4 = solar.SU90
    rate = a + b2 * s2 + c4 * s2 ** 2 - solar.ORBIT                                   # deg/day, synodic
    if rate_offset is not None:
        rate = rate + rate_offset[0] + rate_offset[1] * s2
    w = torch.deg2rad(rate) / 86400.0
    lon_s = lon.unsqueeze(1) - w.unsqueeze(1) * dt_s.float().view(B, K, 1, 1)         # (B, K, G, G)
    lat, b = lat.unsqueeze(1), b.unsqueeze(1)
    xs = torch.cos(lat) * torch.sin(lon_s)
    ys = torch.sin(lat) * torch.cos(b) - torch.cos(lat) * torch.cos(lon_s) * torch.sin(b)
    zs = torch.sin(lat) * torch.sin(b) + torch.cos(lat) * torch.cos(lon_s) * torch.cos(b)
    c = (G - 1) / 2
    cols, rows = c + r_ref * xs, c + r_ref * ys
    ok = (zs > 0) & ~off.view(1, 1, G, G)
    grid = torch.stack([2 * cols / (G - 1) - 1, 2 * rows / (G - 1) - 1], -1).view(B * K, G, G, 2)
    return grid, ok


def warp(frames, grid, ok):
    """frames (B, K, G, G) (NaN = invalid) sampled at grid; NaN where the source is invalid or off-disk."""
    B, K, G, _ = frames.shape
    f = frames.reshape(B * K, 1, G, G).float()
    fin = torch.isfinite(f)
    out = F.grid_sample(torch.nan_to_num(f, nan=0.0), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    vs = F.grid_sample(fin.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True) > 0.5
    out = torch.where(vs & ok.reshape(B * K, 1, G, G), out, torch.full_like(out, math.nan))
    return out.view(B, K, G, G)


def warp_static(field, grid, ok, B, K):
    """A static (G, G) field (a background, a limb-darkening map) sampled like the context frames."""
    G = field.shape[-1]
    return warp(field.view(1, 1, G, G).expand(B, K, G, G), grid, ok)
