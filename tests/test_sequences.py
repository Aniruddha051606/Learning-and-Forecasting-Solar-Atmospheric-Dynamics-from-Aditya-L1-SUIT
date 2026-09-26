import json

import numpy as np
import pandas as pd
import pytest

from suitdyn import sequences


def _frames():
    """Three runs of 87-s frames separated by gaps, like the NB03 schedule."""
    t = []
    for start, n in (("2026-09-23 06:00", 60), ("2026-09-23 12:00", 40), ("2026-09-23 18:00", 50)):
        t += list(pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * 87, unit="s"))
    f = pd.DataFrame({"t": t})
    f["frame_id"] = [f"f{i:04d}" for i in range(len(f))]
    return f


BOUNDS = {"train": ["2026-09-23 00:00", "2026-09-23 11:00"], "val": ["2026-09-23 11:00", "2026-09-23 17:00"],
          "test": ["2026-09-23 17:00", "2026-09-24 00:00"]}


def test_boundary_inside_a_run_is_refused():
    f = _frames()
    bad = {"train": ["2026-09-23 00:00", "2026-09-23 06:30"], "val": ["2026-09-23 06:30", "2026-09-24 00:00"]}
    with pytest.raises(ValueError):
        sequences.assign_splits(f.t, bad, 300)


def test_windows_never_cross_runs_or_splits():
    f = _frames()
    f["split"], _ = sequences.assign_splits(f.t, BOUNDS, 300)
    idx = sequences.window_index(f, [1, 5], [1, 10], 300)
    r = sequences.runs(f.t, 300)
    assert len(idx) > 0
    assert (r[idx["first"]] == r[idx.target]).all()
    assert (f.split.values[idx["first"]] == f.split.values[idx.target]).all()
    # elapsed time of the target is the real one (87 s cadence here)
    assert np.allclose(idx.target_dt_s, idx.horizon * 87)
    # no frame of one split is used by windows of another split
    for s in ("train", "val", "test"):
        w = idx[idx.split == s]
        used = set(np.r_[w["first"], w["last"], w.target])
        assert set(f.split.values[list(used)]) == {s}


def test_test_split_is_sealed(tmp_path):
    f = _frames()
    f["split"], _ = sequences.assign_splits(f.t, BOUNDS, 300)
    idx = sequences.window_index(f, [1], [1], 300)
    p = tmp_path / "seal.json"
    sequences.seal(f, p)
    with pytest.raises(PermissionError):
        sequences.windows(idx, f, "test", p)
    w = sequences.windows(idx, f, "test", p, unseal=True, reason="unit test")
    assert len(w) and json.loads(p.read_text())["unsealed_reads"][0]["reason"] == "unit test"
    f.loc[f.split == "test", "frame_id"] = "changed"
    with pytest.raises(RuntimeError):
        sequences.windows(idx, f, "test", p, unseal=True, reason="after tampering")


def test_bursts_list_every_filter_and_mark_missing():
    t0 = pd.Timestamp("2026-09-24 05:38")
    rows = [("NB04", 0), ("NB03", 20), ("NB02", 40), ("NB04", 280)]  # a burst missing most filters
    full = pd.DataFrame({"t": [t0 + pd.Timedelta(seconds=s) for _, s in rows], "FTR_NAME": [n for n, _ in rows],
                         "frame_id": [f"b{i}" for i in range(len(rows))]})
    b = sequences.burst_snapshots(full)
    assert set(b["filter"]) == set(sequences.ORDER)
    assert (b[b["filter"] == "NB04"].present.sum()) == 2
    missing = b[~b.present]
    assert len(missing) == 11 - 3 and missing.frame_id.isna().all() and missing.t_obs.isna().all()
