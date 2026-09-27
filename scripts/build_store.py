"""Phase 2: build a training store (Zarr) of calibrated, registered NB03 frames.

    python scripts/build_store.py --name v0 [--grid 1536] [--pattern PATH|none] [--splits train,val,test]

Frames and splits come from outputs/phase2/sequences/frames.parquet. The store is written to
outputs/phase2/stores/<name>.zarr with <name>.frames.parquet (row i = store index i) and a
provenance record: git commit, config hashes, manifest hash, calibration file hashes, test seal.
Each worker writes its own frames (one chunk per frame), so the build is parallel and resumable
(--resume skips frames already written).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, store  # noqa: E402

CFG = config.load_phase2()
P1 = config.out_dir(CFG)
SEQ = config.ROOT / "outputs" / "phase2" / "sequences"
STORES = config.ROOT / "outputs" / "phase2" / "stores"
_PATTERN = None
_RESP = None


def _init(pattern_path, response_path=None):
    global _PATTERN, _RESP
    _PATTERN = np.load(pattern_path) if pattern_path else None
    _RESP = np.load(response_path) if response_path else None


def _write(args):
    i, row, zpath, grid, r_ref, mode = args
    g = zarr.open_group(zpath, mode="r+")
    img, m = store.process(row, grid, r_ref, CFG["qc"], int(CFG["limb"]["edge_margin_px"]), _PATTERN, mode,
                           log_response=_RESP)
    g["nb03/image"][i] = img
    g["nb03/mask"][i] = m
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v0")
    ap.add_argument("--grid", type=int, default=CFG["register"]["grid"])
    ap.add_argument("--pattern", default=str(P1 / "calibration" / "nb03_pattern_adopted.npy"),
                    help="pattern file, or 'none'; the default is the one adopted by phase2_calibration_followup.py")
    ap.add_argument("--pattern-mode", default="",
                    help="additive / multiplicative; default: read from the pattern's .json provenance")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--response-map", default="", help="log large-scale response (phase2_largescale.py) to divide out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    a = ap.parse_args()
    pattern = None if a.pattern.lower() == "none" else a.pattern
    if pattern and not a.pattern_mode:
        pj = Path(pattern).with_suffix(".json")
        a.pattern_mode = json.loads(pj.read_text())["mode"] if pj.exists() else "multiplicative"
    STORES.mkdir(parents=True, exist_ok=True)
    zpath = str(STORES / f"{a.name}.zarr")
    r_ref = CFG["register"]["r_ref"] * a.grid / CFG["register"]["grid"]

    frames = pd.read_parquet(SEQ / "frames.parquet")
    frames = frames[frames.split.isin(a.splits.split(","))].reset_index(drop=True)
    if a.limit:
        frames = frames.head(a.limit)
    man = pd.read_parquet(P1 / "manifest.parquet", columns=["file", "path", "clip_lo", "clip_hi", "CMD_EXPT", "MEAS_EXP"])
    rows = frames.merge(man, left_on="frame_id", right_on="file", how="left")
    assert rows.path.notna().all(), "frame missing from manifest"

    T = len(rows)
    comp = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    done = set()
    if a.resume and Path(zpath).exists():
        g = zarr.open_group(zpath, mode="r+")
        img = g["nb03/image"]
        done = {i for i in range(T) if not np.isnan(float(img[i, a.grid // 2, a.grid // 2]))}
    else:
        g = zarr.open_group(zpath, mode="w")
        g.create_array("nb03/image", shape=(T, a.grid, a.grid), chunks=(1, a.grid, a.grid), dtype="float16",
                       compressors=comp, fill_value=np.nan)
        g.create_array("nb03/mask", shape=(T, a.grid, a.grid), chunks=(1, a.grid, a.grid), dtype="uint8",
                       compressors=comp, fill_value=255)
    meta1 = json.loads((P1 / "manifest_meta.json").read_text())
    seal = json.loads((SEQ / "test_seal.json").read_text())
    g.attrs.update(store.provenance(CFG, {
        "name": a.name, "grid": a.grid, "r_ref": r_ref, "orientation": "solar north up, +x = west",
        "units": f"counts at {store.EXPOSURE_REF_MS:.0f} ms commanded exposure",
        "calibration": {"pattern": pattern, "pattern_mode": a.pattern_mode,
                        "pattern_sha256": store.file_sha256(pattern) if pattern else None, "seam": "masked (SEAM bit)",
                        "log_response": a.response_map or None,
                        "log_response_sha256": store.file_sha256(a.response_map) if a.response_map else None},
        "manifest_sha256": meta1.get("manifest_sha256"), "test_seal_sha256": seal["sha256"],
        "frames": T, "splits": a.splits}))
    rows.drop(columns=["path"]).assign(store_index=np.arange(T)).to_parquet(STORES / f"{a.name}.frames.parquet",
                                                                           index=False)
    jobs = [(i, r.to_dict(), zpath, a.grid, r_ref, a.pattern_mode) for i, r in rows.iterrows() if i not in done]
    t0 = time.time()
    with ProcessPoolExecutor(CFG["run"]["workers"], initializer=_init, initargs=(pattern, a.response_map or None)) as ex:
        for k, _ in enumerate(ex.map(_write, jobs, chunksize=2)):
            if k % 200 == 0:
                print(f"{k}/{len(jobs)} frames, {time.time() - t0:.0f} s", flush=True)
    size = sum(p.stat().st_size for p in Path(zpath).rglob("*") if p.is_file())
    summary = {"store": zpath, "frames": T, "written_now": len(jobs), "gigabytes": round(size / 1e9, 2),
               "seconds": round(time.time() - t0, 1), **dict(g.attrs)}
    (STORES / f"{a.name}.summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("store", "frames", "written_now", "gigabytes", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
