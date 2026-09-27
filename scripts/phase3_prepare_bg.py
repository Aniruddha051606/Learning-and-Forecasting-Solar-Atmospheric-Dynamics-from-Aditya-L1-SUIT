"""Phase 3: background-aware samples. Derotate only the solar part of each context frame.

    python scripts/phase3_prepare_bg.py [--background outputs/phase3/background/static_bg_384.npz]

Same windows, targets and splits as scripts/phase3_prepare.py (samples_<G>.parquet, Y_<G>.npy). Only
the inputs change: each context frame F_k becomes rot_k(F_k - S) + S, where S is the static background
from scripts/phase3_background.py. S is fitted on train + hold-out pairs only, never on validation. The
last derotated context frame is then the background-aware B1, and the models' residual target no
longer contains the artefact of moving the non-rotating background with the Sun (PHASE3 §3.1).
Writes outputs/phase3/cache/X_<G>_bg.npy and prepare_bg_meta.json (S checksum, provenance).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase3_why_skill as wy  # noqa: E402  (context_store)
from suitdyn import baselines, config  # noqa: E402

CFG = config.load_phase2()
CACHE = config.ROOT / "outputs" / "phase3" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", default=str(config.ROOT / "outputs" / "phase3" / "background" / "static_bg_384.npz"))
    ap.add_argument("--grid", type=int, default=384)
    a = ap.parse_args()
    t0 = time.time()
    G = a.grid
    S = np.load(a.background)["S"].astype(np.float32)
    meta = pd.read_parquet(CACHE / f"samples_{G}.parquet")
    frames = np.load(CACHE / f"frames_{G}.npy", mmap_mode="r")
    X = np.load(CACHE / f"X_{G}.npy", mmap_mode="r")
    prep = json.loads((CACHE / "prepare_meta.json").read_text())
    r_ref = float(prep["r_ref"])
    store_fr = pd.read_parquet(config.ROOT / "outputs" / "phase2" / "stores" / f"{prep['store']}.frames.parquet")
    man = pd.read_parquet(config.out_dir(CFG) / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file")
    b0_of = store_fr.set_index("store_index").frame_id.map(man.HGLT_OBS).astype(float)
    t_store = store_fr.set_index("t").store_index.sort_index()
    Xb = np.lib.format.open_memmap(CACHE / f"X_{G}_bg.npy", mode="w+", dtype=np.float16, shape=X.shape)
    cache, worst = {}, 0.0
    for n, (i, row) in enumerate(meta.iterrows()):
        b0 = float(b0_of[row.store_target])
        stores = wy.context_store(row, t_store)
        for k, (s, dt) in enumerate(zip(stores, row.dt_context_s)):
            key = (round(dt / 2) * 2, round(b0, 1))  # the same 2-s binning as phase3_prepare.py
            if key not in cache:
                if len(cache) > 3000:
                    cache.clear()
                cache[key] = baselines.derotation_coords(G, r_ref, key[1], key[0])
            f = frames[s].astype(np.float32)
            if n < 20:  # without the background this must reproduce the cached plain sample
                plain = baselines.rotated_persistence(f, r_ref, b0, dt, coords=cache[key])
                worst = max(worst, float(np.nanmax(np.abs(plain - X[i, k].astype(np.float32)))))
            Xb[i, k] = (baselines.rotated_persistence(f - S, r_ref, b0, dt, coords=cache[key]) + S).astype(np.float16)
        if n % 500 == 0:
            print(f"{n}/{len(meta)} samples ({time.time() - t0:.0f} s)", flush=True)
    Xb.flush()
    assert worst < 0.01, f"plain reconstruction differs from the cached samples by {worst}"
    info = {"grid": G, "background": a.background,
            "background_sha256": hashlib.sha256(Path(a.background).read_bytes()).hexdigest(),
            "background_meta": json.loads((Path(a.background).parent / "background_meta.json").read_text())
            .get("lambda_chosen"), "plain_reconstruction_max_abs_diff": worst, "samples": len(meta),
            "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    (CACHE / "prepare_bg_meta.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("samples", "plain_reconstruction_max_abs_diff", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
