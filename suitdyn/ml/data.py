"""Samples assembled on the fly from the frame cache: no precomputed sample arrays.

The cache (scripts/phase3_prepare.py) holds every frame once (frames_<G>.npy, float16, NaN = invalid) and a
sample index (samples_<G>.parquet: split set, horizon, run, the K context frame indices, the target index,
the elapsed seconds from each context frame to the target, B0). A batch is built on the device:

    x_plain = derotate(context)                       every pixel moved by solar rotation to the target time
    x_bg    = derotate(context - S) + S               the background-aware version: the static background S
            = x_plain - derotate(S) + S               (fixed on the grid) is not moved (PHASE3 §3.1, §4b)
    y       = target frame,   valid = y and all K context frames finite

Precomputed sample arrays took 13-17 GB per data set (each frame stored dozens of times); the frame cache
is 0.6-1 GB. derotate() is suitdyn.ml.geometry (checked against the numpy baseline in the tests).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import geometry


class Bank:
    def __init__(self, cache_dir, device="cuda", frames_on_device="auto", max_device_gb=2.0):
        cache_dir = Path(cache_dir)
        self.meta = json.loads((cache_dir / "prepare_meta.json").read_text())
        self.G, self.r_ref = int(self.meta["grid"]), float(self.meta["r_ref"])
        self.device = torch.device(device)
        frames = np.load(cache_dir / f"frames_{self.G}.npy", mmap_mode="r")
        on_dev = frames_on_device is True or (frames_on_device == "auto" and frames.nbytes / 1e9 <= max_device_gb
                                             and self.device.type == "cuda")
        self.frames = torch.from_numpy(np.array(frames))  # one read into memory (0.6-1 GB); writable copy
        self.frames = self.frames.to(self.device) if on_dev else self.frames.pin_memory() if self.device.type == "cuda" \
            else self.frames
        self.mu = torch.from_numpy(np.load(cache_dir / f"mu_{self.G}.npy")).to(self.device)
        self.trusted = torch.from_numpy(np.load(cache_dir / f"trusted_{self.G}.npy").astype(bool)).to(self.device)
        idx = pd.read_parquet(cache_dir / f"samples_{self.G}.parquet")
        self.index = idx
        self.ctx = np.stack(idx.ctx.values).astype(np.int64)
        self.dt = np.stack(idx.dt_context_s.values).astype(np.float32)
        self.tgt = idx.tgt.values.astype(np.int64)
        self.b0 = idx.b0.values.astype(np.float32)
        self.horizon = idx.horizon.values.astype(np.int64)
        self.set = idx.set.values
        self.K = self.ctx.shape[1]
        self.S = None

    def ids(self, subset):
        return np.flatnonzero(self.set == subset)

    def set_background(self, S):
        self.S = None if S is None else torch.as_tensor(np.asarray(S, np.float32)).to(self.device)

    def _frames(self, idx):
        f = self.frames[torch.as_tensor(idx)]
        return f.to(self.device, non_blocking=True).float()

    def batch(self, ids, inputs="plain", ctx_override=None):
        """ids: sample indices. inputs: 'plain' or 'bg' (x is then the background-aware context).
        ctx_override: (len(ids), K) frame indices to use as context instead (negative controls)."""
        ids = np.asarray(ids)
        B = len(ids)
        ctx = self.ctx[ids] if ctx_override is None else np.asarray(ctx_override)
        dt = torch.from_numpy(self.dt[ids]).to(self.device)
        b0 = torch.from_numpy(self.b0[ids]).to(self.device)
        grid, ok = geometry.derotation_grid(self.G, self.r_ref, b0, dt)
        x_plain = geometry.warp(self._frames(ctx.reshape(-1)).view(B, self.K, self.G, self.G), grid, ok)
        out = {"x_plain": x_plain, "y": self._frames(self.tgt[ids]).unsqueeze(1),
               "h": torch.from_numpy(self.horizon[ids]).to(self.device), "grid": grid, "ok": ok}
        if self.S is not None:
            s_rot = geometry.warp_static(self.S, grid, ok, B, self.K)
            out["x_bg"] = x_plain - s_rot + self.S
        if inputs == "bg" and self.S is None:
            raise ValueError("inputs='bg' needs the static background (Bank.set_background)")
        out["x"] = out["x_bg"] if inputs == "bg" else x_plain
        out["valid"] = torch.isfinite(out["y"]) & torch.isfinite(out["x"]).all(1, keepdim=True)
        return out


def model_inputs(b):
    """(x with NaN -> 0, validity mask as float), the form the models take."""
    return torch.nan_to_num(b["x"], nan=0.0), b["valid"].float()
