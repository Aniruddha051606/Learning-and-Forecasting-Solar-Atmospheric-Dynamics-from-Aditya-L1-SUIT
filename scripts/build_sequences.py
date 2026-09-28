"""Phase 2: data-set frame list, splits, window index, test seal, burst snapshots.

    python scripts/build_sequences.py [--reseal]

Reads the archive manifest and the data set's registration (outputs/datasets/<name>/phase1); writes
outputs/datasets/<name>/sequences/:
  frames.parquet     NB03 frames in the data set (one row each: frame_id, t, split, run, flags)
  excluded.parquet   NB03 frames left out, with the reason
  windows.parquet    every (context, horizon) window, all splits
  bursts.parquet     multi-filter snapshots, every filter listed, missing ones explicit
  test_seal.json     hash of the test frame list (reading test windows needs an explicit unseal)
  embargoed.parquet  frames dropped so each split starts >= [split] embargo_h after the previous one ends
  summary.json       counts per split / horizon, independent-hours estimate, provenance

The test seal is never silently replaced. If a rebuild gives the same test frames, the existing seal (with
its log of every unseal) is kept as it is. If the test frames changed (for example files for those dates
arrived later), the build stops unless --reseal is given; the old seal is then kept as
test_seal_superseded_<time>.json.
"""
import argparse
import hashlib
import time
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, paths, sequences  # noqa: E402

CFG = config.load_dataset()
ARC = paths.archive()
P1 = paths.phase1()
OUT = paths.sequences()


def seal_test(f, reseal):
    path = OUT / "test_seal.json"
    ids = sorted(f.loc[f.split == "test", "frame_id"])
    new = hashlib.sha256("\n".join(ids).encode()).hexdigest()  # the same hash as suitdyn.sequences.seal
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("sha256") == new:
            return new  # unchanged: keep the file and its unseal log
        if not reseal:
            sys.exit(f"The test split of data set {config.DATASET} changed ({old.get('test_frames')} -> {len(ids)} "
                     f"frames). Refusing to replace the seal; rerun with --reseal if this is intended.")
        path.rename(OUT / f"test_seal_superseded_{time.strftime('%Y%m%dT%H%M%S')}.json")
    return sequences.seal(f, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reseal", action="store_true", help="accept a changed test split (keeps the old seal)")
    a = ap.parse_args()
    D, S, Q = CFG["dataset"], CFG["split"], CFG["sequences"]
    reg = pd.read_parquet(P1 / "registration.parquet")
    man = pd.read_parquet(ARC / "manifest.parquet", columns=["file", "sha256"])
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

    bounds = {k: v for k, v in S.items() if k in ("train", "val", "test")}
    split, run = sequences.assign_splits(f.t, bounds, Q["max_gap_s"])
    f["split"], f["run"] = split, run
    f = f[f.split.notna()].reset_index(drop=True)
    # embargo: each split starts >= embargo_h after the previous one ends (independent solar states)
    emb = float(S.get("embargo_h", 4.0))
    dropped = sequences.embargo(f.t, f.split, emb)
    embargoed = f[dropped][["frame_id", "t", "split"]].assign(reason="embargo")
    f = f[~dropped].reset_index(drop=True)
    f["run"] = sequences.runs(f.t, Q["max_gap_s"])
    # one pointing per data set: a new pointing in the archive must not be merged silently (PHASE3 §4b)
    cl = f.pointing_cluster.value_counts() if "pointing_cluster" in f else pd.Series(dtype=int)
    want = D.get("pointing_cluster")
    if want:
        f = f[f.pointing_cluster == want].reset_index(drop=True)
        f["run"] = sequences.runs(f.t, Q["max_gap_s"])
    elif len(cl) > 1 and not D.get("allow_mixed_pointing", False):
        detail = "; ".join(f"{k}: {v} frames, {f.t[f.pointing_cluster == k].min()} .. {f.t[f.pointing_cluster == k].max()}"
                           for k, v in cl.items())
        sys.exit(f"data set {config.DATASET} mixes {len(cl)} pointing clusters ({detail}). Set [dataset] "
                 f"pointing_cluster to one of them, or allow_mixed_pointing = true on purpose.")
    cols = ["frame_id", "sha256", "t", "split", "run", "OBS_MODE", "pointing_mode", "pointing_cluster", "segment",
            "reg_x0", "reg_y0", "reg_R", "CROTA2", "jump_px", "qc_reasons"]
    cols = [c for c in cols if c in f]
    seal_hash = seal_test(f, a.reseal)  # before anything is written: a refused reseal leaves the data set as it was
    atomic.to_parquet(f[cols], OUT / "frames.parquet")
    atomic.to_parquet(embargoed, OUT / "embargoed.parquet")

    idx = sequences.window_index(f[["frame_id", "t", "split"]], Q["contexts"], Q["horizons"], Q["max_gap_s"])
    atomic.to_parquet(idx, OUT / "windows.parquet")

    full = reg[(reg.frame == "full") & (reg.pointing_mode == D["pointing_mode"])].rename(columns={"file": "frame_id"})
    bursts = sequences.burst_snapshots(full[["frame_id", "t", "FTR_NAME"]])
    atomic.to_parquet(bursts, OUT / "bursts.parquet")

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
        "embargo_h": emb, "embargoed_frames": embargoed.split.value_counts().to_dict(),
        "pointing_clusters": {str(k): int(v) for k, v in cl.items()}, "pointing_cluster_used": want,
        "test_seal_sha256": seal_hash, **CFG["_meta"],
    }
    atomic.write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
