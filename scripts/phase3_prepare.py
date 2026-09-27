"""Phase 3 data preparation: frame cache and derotated samples (train, early-stop hold-out, val).

    python scripts/phase3_prepare.py [--grid-factor 4] [--train-stride 2]

1. Frame cache (outputs/phase3/cache/frames_<G>.npy, float16): every store frame, QC-masked, block-averaged
   to G = 1536 / grid_factor, pointing-response corrected (Phase 2 §4.2), divided by its own disk median
   (Phase 2 §4.3). NaN = invalid.
2. Samples, from the Phase 2 window index (context 5; horizons 20/40/80/160 frames): the five context
   frames each rotated to the target time (so persistence of the last one IS baseline B1), the target,
   and per-sample metadata (times, run, horizon, elapsed seconds). Splits:
     train    training runs except the last one
     holdout  the last training run: early stopping only (the validation split is never used to choose
              a model, so it stays a clean reporting set)
     val      the validation split, every window
   The test split is sealed and not read.
Everything is recorded in outputs/phase3/cache/prepare_meta.json with the git commit and config hashes.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import baselines, config, normalize, response  # noqa: E402

CFG = config.load_phase2()
STORES = config.ROOT / "outputs" / "phase2" / "stores"
SEQ = config.seq_dir()
OUT = config.phase3_dir("cache")
HORIZONS = (20, 40, 80, 160)
CONTEXT = 5
_G = {}


def _init(zpath, f, resp_path, pointing):
    g = zarr.open_group(zpath, mode="r")
    _G.update(img=g["nb03/image"], mask=g["nb03/mask"], f=f, resp=response.load(resp_path), pointing=pointing)


def _frame(i):
    img = _G["img"][i].astype(np.float32)
    img[_G["mask"][i] != 0] = np.nan
    f = _G["f"]
    n = img.shape[0] // f
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            img = np.nanmean(img.reshape(n, f, n, f), axis=(1, 3))
    x0, y0 = _G["pointing"][i]
    img = img * response.factor(_G["resp"], n, x0, y0)
    return i, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=config.DATASET)
    ap.add_argument("--grid-factor", type=int, default=4)
    ap.add_argument("--train-stride", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = str(STORES / f"{a.store}.zarr")
    g = zarr.open_group(zpath, mode="r")
    fr = pd.read_parquet(STORES / f"{a.store}.frames.parquet")
    G = g["nb03/image"].shape[1] // a.grid_factor
    r_ref = float(g.attrs["r_ref"]) / a.grid_factor
    mu = normalize.mu_map(G, r_ref)
    disk = mu > np.sqrt(1 - 0.95 ** 2)
    core = mu > np.sqrt(1 - 0.9 ** 2)
    resp_path = config.ROOT / "outputs" / "phase2" / "response" / f"response_{a.store}.npz"
    pointing = {int(r.store_index): (float(r.reg_x0), float(r.reg_y0)) for r in fr.itertuples()}

    # 1. frame cache
    frames = np.full((len(fr), G, G), np.nan, np.float16)
    with ProcessPoolExecutor(CFG["run"]["workers"], initializer=_init,
                             initargs=(zpath, a.grid_factor, str(resp_path), pointing)) as ex:
        for i, img in ex.map(_frame, fr.store_index.astype(int).tolist(), chunksize=8):
            med = np.nanmedian(img[core])
            img = np.where(disk, img / med, np.nan)
            frames[i] = img.astype(np.float16)
    np.save(OUT / f"frames_{G}.npy", frames)
    print(f"frame cache {frames.shape} ({time.time() - t0:.0f} s)", flush=True)

    # 2. samples
    seqf = pd.read_parquet(SEQ / "frames.parquet")
    pos2store = pd.Series(fr.store_index.values, index=fr.frame_id).reindex(seqf.frame_id).values.astype(int)
    man = pd.read_parquet(config.out_dir(CFG) / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file")
    win = pd.read_parquet(SEQ / "windows.parquet")
    win = win[(win.context == CONTEXT) & win.horizon.isin(HORIZONS) & win.split.isin(["train", "val"])].copy()
    runs_train = sorted(seqf.loc[seqf.split == "train", "run"].unique())
    last_run = runs_train[-1]
    win["run"] = seqf.run.values[win["last"].values]
    win["set"] = np.where(win.split == "val", "val", np.where(win.run == last_run, "holdout", "train"))
    parts = []
    for (s, h), d in win.groupby(["set", "horizon"]):
        parts.append(d.iloc[::a.train_stride] if s == "train" else d)
    win = pd.concat(parts).sort_values(["set", "horizon", "last"]).reset_index(drop=True)

    meta_rows = []
    X = np.lib.format.open_memmap(OUT / f"X_{G}.npy", mode="w+", dtype=np.float16, shape=(len(win), CONTEXT, G, G))
    Y = np.lib.format.open_memmap(OUT / f"Y_{G}.npy", mode="w+", dtype=np.float16, shape=(len(win), G, G))
    coord_cache = {}
    for n, w in win.iterrows():
        pos = list(range(int(w["first"]), int(w["last"]) + 1))
        tgt = int(w.target)
        t_tgt = seqf.t.iloc[tgt]
        b0 = float(man.loc[seqf.frame_id.iloc[tgt], "HGLT_OBS"])
        dts = []
        for k, p in enumerate(pos):
            dt = (t_tgt - seqf.t.iloc[p]).total_seconds()
            key = (round(dt / 2) * 2, round(b0, 1))  # 2-s bins: < 0.001 px of rotation at 384²
            if key not in coord_cache:
                coord_cache[key] = baselines.derotation_coords(G, r_ref, key[1], key[0])
            X[n, k] = baselines.rotated_persistence(frames[pos2store[p]].astype(np.float32), r_ref, b0, dt,
                                                    coords=coord_cache[key]).astype(np.float16)
            dts.append(dt)
        Y[n] = frames[pos2store[tgt]]
        meta_rows.append({"sample": n, "set": w["set"], "horizon": int(w.horizon), "run": int(w.run),
                          "t_last": seqf.t.iloc[int(w["last"])], "t_target": t_tgt, "dt_target_s": dts[-1],
                          "dt_context_s": dts, "store_last": int(pos2store[int(w['last'])]),
                          "store_target": int(pos2store[tgt])})
        if len(coord_cache) > 3000:
            coord_cache.clear()
        if n % 500 == 0:
            print(f"{n}/{len(win)} samples ({time.time() - t0:.0f} s)", flush=True)
    X.flush()
    Y.flush()
    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(OUT / f"samples_{G}.parquet", index=False)
    np.save(OUT / f"mu_{G}.npy", mu.astype(np.float32))
    # trusted region (Phase 2, training split, after the response correction), resized to G
    tr = np.load(config.ROOT / "outputs" / "phase2" / "noise_maps" / f"noise_maps_{a.store}_train_g2_resp.npz")["trusted"]
    idx = (np.arange(G) * tr.shape[0] / G).astype(int)
    np.save(OUT / f"trusted_{G}.npy", tr[np.ix_(idx, idx)])
    info = {"grid": G, "r_ref": r_ref, "context": CONTEXT, "horizons": HORIZONS, "train_stride": a.train_stride,
            "holdout_run": int(last_run), "samples": meta.groupby(["set", "horizon"]).size().unstack().to_dict(),
            "seconds": round(time.time() - t0, 1), "store": a.store, "store_attrs": dict(g.attrs), **CFG["_meta"]}
    (OUT / "prepare_meta.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("grid", "holdout_run", "samples", "seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()
