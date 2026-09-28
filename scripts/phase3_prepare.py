"""Phase 3 data preparation: the frame cache and the sample index of one data set.

    python scripts/phase3_prepare.py

Settings: configs/phase3.toml [samples]. Writes outputs/datasets/<name>/phase3/cache/:
  frames_<G>.npy        every store frame once (float16, NaN = invalid): QC-masked, block-averaged to
                        G = 1536 / grid_factor, pointing-response corrected (PHASE2 §4.2), divided by its
                        own disk median (PHASE2 §4.3)
  samples_<G>.parquet   one row per window (context K, horizon H): set, horizon, run, ctx (K frame indices),
                        tgt (target frame index), dt_context_s (K elapsed seconds to the target), b0, times.
                        Sets: train = training runs except the last; holdout = the last training run
                        (early stopping and baseline tuning only); val = validation. The sealed test split
                        is not read here.
  mu_<G>.npy, trusted_<G>.npy (the Phase 2 trusted region, resized), prepare_meta.json (provenance)
Samples are assembled on the fly (suitdyn/ml/data.py); no sample array is stored.
"""
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, normalize, paths, progress, response  # noqa: E402
from suitdyn.ml import samples  # noqa: E402

CFG = config.load_dataset()
P3 = config.load_phase3()
_G = {}


def _init(zpath, f, resp_path, pointing):
    g = zarr.open_group(zpath, mode="r")
    _G.update(img=g["nb03/image"], mask=g["nb03/mask"], f=f, resp=response.load(resp_path), pointing=pointing)


def _frame(i):
    import warnings
    img = _G["img"][i].astype(np.float32)
    img[_G["mask"][i] != 0] = np.nan
    f = _G["f"]
    n = img.shape[0] // f
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img = np.nanmean(img.reshape(n, f, n, f), axis=(1, 3))
    x0, y0 = _G["pointing"][i]
    return i, img * response.factor(_G["resp"], n, x0, y0)


def main():
    t0 = time.time()
    S = P3["samples"]
    name = config.DATASET
    out = paths.phase3("cache")
    zpath = str(paths.stores(f"{name}.zarr"))
    g = zarr.open_group(zpath, mode="r")
    fr = pd.read_parquet(paths.stores(f"{name}.frames.parquet"))
    G = g["nb03/image"].shape[1] // S["grid_factor"]
    r_ref = float(g.attrs["r_ref"]) / S["grid_factor"]
    mu = normalize.mu_map(G, r_ref)
    disk = mu > np.sqrt(1 - 0.95 ** 2)
    core = mu > np.sqrt(1 - 0.9 ** 2)
    resp_path = paths.phase2("response", f"response_{name}.npz")
    pointing = {int(r.store_index): (float(r.reg_x0), float(r.reg_y0)) for r in fr.itertuples()}

    # 1. frame cache
    frames = np.full((len(fr), G, G), np.nan, np.float16)
    fid = fr.set_index("store_index").frame_id
    with ProcessPoolExecutor(CFG["run"]["workers"], initializer=_init,
                             initargs=(zpath, S["grid_factor"], str(resp_path), pointing)) as ex:
        for k, (i, img) in enumerate(ex.map(_frame, fr.store_index.astype(int).tolist(), chunksize=8)):
            progress.report("phase3_prepare: frame cache", item=fid.get(i), i=k, n=len(fr))
            med = np.nanmedian(img[core])
            frames[i] = np.where(disk, img / med, np.nan).astype(np.float16)
    atomic.save_npy(out / f"frames_{G}.npy", frames)
    print(f"frame cache {frames.shape} ({time.time() - t0:.0f} s)", flush=True)

    # 2. sample index
    seqf = pd.read_parquet(paths.sequences("frames.parquet"))
    man = pd.read_parquet(paths.archive("manifest.parquet"), columns=["file", "HGLT_OBS"]).set_index("file")
    win = pd.read_parquet(paths.sequences("windows.parquet"))
    K = int(S["context"])
    # train / val only: the sealed test split is read solely by the one-time test evaluation
    win = win[(win.context == K) & win.horizon.isin(S["horizons"]) & win.split.isin(["train", "val"])].copy()
    runs_train = sorted(seqf.loc[seqf.split == "train", "run"].unique())
    holdout_run = runs_train[-1]
    win["run"] = seqf.run.values[win["last"].values]
    win["set"] = np.where(win.split == "val", "val", np.where(win.run == holdout_run, "holdout", "train"))
    if S.get("train_stride", 1) > 1:
        win = pd.concat([d.iloc[::S["train_stride"]] if s_ == "train" else d
                         for (s_, h), d in win.groupby(["set", "horizon"])])
    win = win.sort_values(["set", "horizon", "last"]).reset_index(drop=True)
    idx = samples.build(win, seqf, fr, man.HGLT_OBS, win.set.values)
    # the embargo between splits (configs/datasets/<name>.toml [split] embargo_h) also separates the training
    # runs from the hold-out run that early stopping and every tuned baseline use
    emb = float(CFG["split"].get("embargo_h", 4.0))
    n0 = len(idx)
    idx = samples.embargo_before(idx, "holdout", "train", emb)
    print(f"embargo {emb} h before the hold-out run: {n0 - len(idx)} training samples dropped", flush=True)
    atomic.to_parquet(idx, out / f"samples_{G}.parquet")
    atomic.save_npy(out / f"mu_{G}.npy", mu.astype(np.float32))
    tr = np.load(paths.phase2("noise_maps", f"noise_maps_{name}_train_g2_resp.npz"))["trusted"]
    sel = (np.arange(G) * tr.shape[0] / G).astype(int)
    atomic.save_npy(out / f"trusted_{G}.npy", tr[np.ix_(sel, sel)])
    counts = idx.groupby(["set", "horizon"]).size().unstack().fillna(0).astype(int)
    info = {"grid": G, "r_ref": r_ref, "context": K, "horizons": S["horizons"], "holdout_run": int(holdout_run),
            "embargo_h": emb, "embargo_dropped_train": int(n0 - len(idx)),
            "samples": {str(h): counts[h].to_dict() for h in counts.columns}, "store": name,
            "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "frames_gb": round(frames.nbytes / 1e9, 3), "seconds": round(time.time() - t0, 1),
            "store_attrs": dict(g.attrs), **CFG["_meta"], **P3["_meta"]}
    atomic.write_json(out / "prepare_meta.json", info)
    print(counts.to_string())
    print(f"done in {time.time() - t0:.0f} s; frame cache {info['frames_gb']} GB")


if __name__ == "__main__":
    main()
