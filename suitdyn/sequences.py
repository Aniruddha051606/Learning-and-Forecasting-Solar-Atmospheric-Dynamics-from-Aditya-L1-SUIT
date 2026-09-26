"""Leakage-safe split assignment and window indices for NB03 sequences and multi-filter bursts.

Rules (tests/test_sequences.py checks each):
  * frames are assigned to a split by time; every split boundary must fall inside an observing gap,
    so no run of consecutive frames is ever cut by a boundary;
  * a window (context frames t-K+1..t, target t+H) lies entirely inside one run and one split;
  * each window stores the elapsed time of every context frame and of the target relative to t, so
    irregular cadence is visible to every consumer, never hidden behind a frame count;
  * the test split is sealed: `seal()` writes a hash of its frame list, `windows(..., split="test")`
    refuses to return test windows unless `unseal=True`, and every unsealed read is logged.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .filters import ORDER


def runs(t, max_gap_s):
    """Run id per frame (t sorted ascending): a gap longer than max_gap_s starts a new run."""
    dt = pd.Series(t).diff().dt.total_seconds().values
    return np.cumsum(np.r_[True, dt[1:] > max_gap_s])


def assign_splits(t, bounds, max_gap_s):
    """Split name per frame from {name: [start, end]}; raises if a boundary cuts a run."""
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    split = pd.Series([None] * len(t), dtype=object)
    for name, (a, b) in bounds.items():
        split[(t >= pd.Timestamp(a)) & (t < pd.Timestamp(b))] = name
    r = runs(t, max_gap_s)
    for run_id in np.unique(r):
        s = split[r == run_id].dropna().unique()
        if len(s) > 1 or (len(s) == 1 and split[r == run_id].isna().any()):
            raise ValueError(f"split boundary cuts run {run_id} ({t[r == run_id].min()} .. {t[r == run_id].max()})")
    return split.values, r


def window_index(frames, contexts, horizons, max_gap_s):
    """All windows (K context frames ending at position i, target at i+H) inside one run and one split.

    `frames`: DataFrame sorted by t with columns frame_id, t, split. Returns one row per window with
    positions into `frames`, the split, and elapsed seconds of the first context frame and the target."""
    f = frames.reset_index(drop=True)
    r = runs(f.t, max_gap_s)
    ts = (f.t - f.t.iloc[0]).dt.total_seconds().values
    rows = []
    for K in contexts:
        for H in horizons:
            for i in range(K - 1, len(f) - H):
                j0, jt = i - K + 1, i + H
                if r[j0] != r[jt] or f.split[j0] is None or f.split[j0] != f.split[jt]:
                    continue
                rows.append((K, H, j0, i, jt, f.split[i], ts[j0] - ts[i], ts[jt] - ts[i]))
    return pd.DataFrame(rows, columns=["context", "horizon", "first", "last", "target", "split",
                                       "first_dt_s", "target_dt_s"])


def seal(frames, path):
    """Record the test split's frame list and its hash; later reads are checked against it."""
    test_ids = sorted(frames.loc[frames.split == "test", "frame_id"])
    h = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()
    Path(path).write_text(json.dumps({"test_frames": len(test_ids), "sha256": h,
                                      "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    return h


def windows(index, frames, split, seal_path, unseal=False, reason=""):
    """Windows of one split. Test windows need unseal=True and a reason; the read is appended to the
    seal file's log, and the test frame list must still match the sealed hash."""
    if split == "test":
        if not unseal:
            raise PermissionError("test split is sealed; pass unseal=True with a reason")
        rec = json.loads(Path(seal_path).read_text())
        test_ids = sorted(frames.loc[frames.split == "test", "frame_id"])
        if hashlib.sha256("\n".join(test_ids).encode()).hexdigest() != rec["sha256"]:
            raise RuntimeError("test split changed since it was sealed")
        rec.setdefault("unsealed_reads", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason})
        Path(seal_path).write_text(json.dumps(rec, indent=1))
    return index[index.split == split].reset_index(drop=True)


def burst_snapshots(full_frames, max_span_s=600):
    """Group unbinned multi-filter frames into bursts. One row per (burst, filter) for ALL filters in
    ORDER: a filter not observed in a burst is an explicit row with present=False and no frame, never a
    filled value. Each present row keeps its own observation time."""
    f = full_frames.sort_values("t").reset_index(drop=True)
    dt = f.t.diff().dt.total_seconds().fillna(np.inf).values
    burst = np.cumsum(dt > max_span_s)
    rows = []
    for b in np.unique(burst):
        g = f[burst == b]
        t0 = g.t.min()
        for flt in ORDER:
            h = g[g.FTR_NAME == flt]
            if len(h):
                for _, x in h.iterrows():  # NB04 is taken twice per burst: keep both
                    rows.append({"burst": int(b), "burst_start": t0, "filter": flt, "present": True,
                                 "frame_id": x.frame_id, "t_obs": x.t, "dt_from_start_s": (x.t - t0).total_seconds()})
            else:
                rows.append({"burst": int(b), "burst_start": t0, "filter": flt, "present": False,
                             "frame_id": None, "t_obs": pd.NaT, "dt_from_start_s": np.nan})
    out = pd.DataFrame(rows)
    out["present"] = out.present.astype(bool)
    return out
