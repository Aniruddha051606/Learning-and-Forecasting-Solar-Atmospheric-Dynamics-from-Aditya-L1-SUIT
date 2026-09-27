import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from suitdyn import baselines
from suitdyn.ml import geometry


def _disk_image(G, r_ref, seed=0):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=(G, G)), 2.0).astype(np.float32) + 1.0
    c = (G - 1) / 2
    v, u = np.indices((G, G))
    img[np.hypot(u - c, v - c) / r_ref >= 0.95] = np.nan
    img[10:30, 40:60] = np.nan  # an invalid patch inside the grid
    return img


def test_torch_derotation_matches_numpy():
    G, r_ref, b0 = 128, 57.5, 7.1
    img = _disk_image(G, r_ref)
    dts = [600.0, 3600.0, 13700.0]
    grid, ok = geometry.derotation_grid(G, r_ref, torch.tensor([b0]), torch.tensor([dts]))
    out = geometry.warp(torch.from_numpy(np.stack([img] * len(dts)))[None], grid, ok)[0].numpy()
    for k, dt in enumerate(dts):
        ref = baselines.rotated_persistence(img, r_ref, b0, dt)
        both = np.isfinite(ref) & np.isfinite(out[k])
        # the same pixels are valid, apart from a thin rim where nearest-neighbour rounding differs
        assert (np.isfinite(ref) != np.isfinite(out[k])).mean() < 0.005
        assert np.abs(ref[both] - out[k][both]).max() < 1e-3


def test_static_field_and_batching():
    G, r_ref = 64, 28.75
    field = torch.from_numpy(_disk_image(G, r_ref, 1))
    b0 = torch.tensor([7.0, 7.2])
    dt = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    grid, ok = geometry.derotation_grid(G, r_ref, b0, dt)
    w = geometry.warp_static(field, grid, ok, 2, 2)
    assert w.shape == (2, 2, G, G)
    # zero elapsed time leaves the image unchanged
    g0, ok0 = geometry.derotation_grid(G, r_ref, torch.tensor([7.0]), torch.zeros(1, 1))
    same = geometry.warp_static(field, g0, ok0, 1, 1)[0, 0].numpy()
    f = field.numpy()
    m = np.isfinite(f) & np.isfinite(same)
    assert np.abs(same[m] - f[m]).max() < 1e-4
