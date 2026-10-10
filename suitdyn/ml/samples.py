"""The sample index: one row per (context, horizon) window, pointing into the frame cache."""
import numpy as np
import pandas as pd


def build(win, seqf, store_frames, man_b0, set_labels):
    """win: windows (one context length) with columns first, last, target, horizon; seqf: the data set's
    frame list (positions used by the windows); store_frames: store frame table (frame_id, store_index,
    reg_x0, reg_y0); man_b0: Series file -> HGLT_OBS; set_labels: one label per window row.
    """
    win = win.reset_index(drop=True)
    pos2store = pd.Series(store_frames.store_index.values, index=store_frames.frame_id).reindex(seqf.frame_id).values
    if np.isnan(pos2store.astype(float)).any():
        raise ValueError("a data-set frame is missing from the store")
    pos2store = pos2store.astype(np.int64)
    ptg = store_frames.set_index("frame_id")[["reg_x0", "reg_y0"]].reindex(seqf.frame_id).values
    K = int(win["last"].iloc[0] - win["first"].iloc[0] + 1) if len(win) else 0
    ts = seqf.t.values
    first = win["first"].values.astype(np.int64)
    pos = first[:, None] + np.arange(K)[None, :]
    tgt = win.target.values.astype(np.int64)
    dt = (ts[tgt][:, None] - ts[pos]).astype("timedelta64[ns]").astype(np.float64) / 1e9
    idx = pd.DataFrame({
        "set": np.asarray(set_labels), "horizon": win.horizon.values.astype(int),
        "run": seqf.run.values[win["last"].values].astype(int),
        "ctx": list(pos2store[pos]), "tgt": pos2store[tgt], "dt_context_s": list(dt),
        "b0": man_b0.reindex(seqf.frame_id.values[tgt]).values.astype(float),
        "t_last": ts[win["last"].values], "t_target": ts[tgt], "dt_target_s": dt[:, -1] if K else [],
        "target_frame": seqf.frame_id.values[tgt], "px": ptg[tgt, 0], "py": ptg[tgt, 1]})
    if not np.isfinite(idx.b0).all():
        raise ValueError("B0 missing for a target frame")
    return idx


def embargo_before(idx, later_set, earlier_set, hours):
    """Drop rows of `earlier_set` whose TARGET is within `hours` before the first context frame of any
    `later_set` row (the hold-out run must not be predicted from almost the same solar state it contains).
    """
    later = idx[idx.set == later_set]
    if not len(later) or hours <= 0:
        return idx
    first_ctx = pd.to_datetime(later.t_target) - pd.to_timedelta(np.stack(later.dt_context_s.values)[:, 0], unit="s")
    start = first_ctx.min()
    t_tgt = pd.to_datetime(idx.t_target)
    drop = (idx.set == earlier_set) & (t_tgt > start - pd.Timedelta(hours=hours))
    return idx[~drop].reset_index(drop=True)
