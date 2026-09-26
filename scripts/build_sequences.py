"""Phase 2: data-set frame list, splits, window index, test seal, burst snapshots.

    python scripts/build_sequences.py

Reads outputs/phase1/{manifest,registration,frames_full}.parquet; writes outputs/phase2/sequences/:
  frames.parquet     NB03 frames in the data set (one row each: frame_id, t, split, run, flags)
  excluded.parquet   NB03 frames left out, with the reason
  windows.parquet    every (context, horizon) window, all splits
  bursts.parquet     multi-filter snapshots, every filter listed, missing ones explicit
  test_seal.json     hash of the test frame list (reading test windows needs an explicit unseal)
  summary.json       counts per split / horizon, independent-hours estimate, provenance
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, sequences  # noqa: E402

CFG = config.load_phase2()
P1 = config.out_dir(CFG)
OUT = config.ROOT / "outputs" / "phase2" / "sequences"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    D, S, Q = CFG["dataset"], CFG["split"], CFG["sequences"]
    reg = pd.read_parquet(P1 / "registration.parquet")
    man = pd.read_parquet(P1 / "manifest.parquet", columns=["file", "sha256"])
    nb = reg[reg.frame == "full_binned"].merge(man, on="file").sort_values("t").reset_index(drop=True)
    nb = nb.rename(columns={"file": "frame_id"})

    reasons = pd.Series("", index=nb.index)
    reasons[nb.pointing_mode != D["pointing_mode"]] += f"pointing_mode!={D['pointing_mode']};"
    for flag in D["exclude_flags"]:
        reasons[nb.qc_reasons.fillna("").str.contains(flag)] += f"{flag};"
    reasons[~nb.qc_usable] += "qc_rejected;"
    if D["drop_program_first_frame"]:
        # first frame of every program block, including blocks after a gap
        gap = nb.t.diff().dt.total_seconds() > Q["max_gap_s"]
        first = nb.OBS_MODE.ne(nb.OBS_MODE.shift()) | gap
        reasons[first] += "program_block_first_frame;"
    keep = reasons == ""
    nb.assign(reason=reasons)[~keep][["frame_id", "t", "OBS_MODE", "pointing_mode", "reason"]].to_parquet(
        OUT / "excluded.parquet", index=False)
    f = nb[keep].reset_index(drop=True)

    split, run = sequences.assign_splits(f.t, S, Q["max_gap_s"])
    f["split"], f["run"] = split, run
    f = f[f.split.notna()].reset_index(drop=True)
    f["run"] = sequences.runs(f.t, Q["max_gap_s"])
    cols = ["frame_id", "sha256", "t", "split", "run", "OBS_MODE", "pointing_mode", "segment", "reg_x0", "reg_y0",
            "reg_R", "CROTA2", "jump_px", "qc_reasons"]
    f[cols].to_parquet(OUT / "frames.parquet", index=False)

    idx = sequences.window_index(f[["frame_id", "t", "split"]], Q["contexts"], Q["horizons"], Q["max_gap_s"])
    idx.to_parquet(OUT / "windows.parquet", index=False)
    seal_hash = sequences.seal(f, OUT / "test_seal.json")

    full = reg[(reg.frame == "full") & (reg.pointing_mode == D["pointing_mode"])].rename(columns={"file": "frame_id"})
    bursts = sequences.burst_snapshots(full[["frame_id", "t", "FTR_NAME"]])
    bursts.to_parquet(OUT / "bursts.parquet", index=False)

    counts = idx.groupby(["horizon", "split"]).size().unstack(fill_value=0)
    counts = counts[counts.index.isin(Q["horizons"])]
    k1 = idx[idx.context == max(Q["contexts"])]
    summary = {
        "frames": {s: int((f.split == s).sum()) for s in ("train", "val", "test")},
        "hours_of_data": {s: float(f[f.split == s].groupby("run").t.agg(lambda t: (t.max() - t.min()).total_seconds())
                                   .sum() / 3600) for s in ("train", "val", "test")},
        "runs": {s: int(f[f.split == s].run.nunique()) for s in ("train", "val", "test")},
        "excluded": nb.assign(reason=reasons)[~keep].reason.str.split(";").explode().replace("", np.nan).dropna()
        .value_counts().to_dict(),
        "windows_context_max": {int(h): g.split.value_counts().to_dict() for h, g in k1.groupby("horizon")},
        "horizon_minutes_median": {int(h): float(g.target_dt_s.median() / 60) for h, g in idx.groupby("horizon")},
        "bursts": {"count": int(bursts.burst.nunique()),
                   "missing_filter_rows": int((~bursts.present).sum())},
        "test_seal_sha256": seal_hash, **CFG["_meta"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
