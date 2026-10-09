"""Answer frames for scoring the sealed blind forecasts (docs/PREREGISTRATION.md, Addendum B; test D3).

    python scripts/answer_frames.py --sealed outputs/sealed_forecast/<UTC time> [--workers 4]

For every sealed target time, the real NB03 full-disk 2048 frame nearest in time (within +-2 min) is processed like
the final_offset frame cache, with the FROZEN final_offset products:
  registration  the pipeline's method (scripts/registration_study.py adopt_nb03), re-run on every 2048 NB03 frame
                from 2026-09-27 03:00 to the last target: per-frame limb edges from the header guess
                (suitdyn.geometry.limb, [limb] settings, as scripts/process_frames.py), circle fits on the rays
                usable in >= 98 % of these frames, runs (gaps > 300 s) and jumps (> 1 px, phase correlation of
                consecutive frames), image motion with the fixed pattern removed (suitdyn.motion, keyframes every
                10 frames, solar rotation removed), each segment anchored to its circle fits, radius = robust
                segment mean scaled by DSUN_OBS
  store + cache suitdyn.store.process (QC mask, final_offset additive fixed pattern, exposure, 1536 grid, r_ref
                690); block-averaged to 384; final_offset pointing-response factor; divided by its own disk median
VALIDATION (always): the window starts inside final_offset's last observing stretch, so the same frames registered
here and by the pipeline are compared (reg_x0/y0/R), and frames built here are compared with the cache.
Writes outputs/sealed_forecast/answers_<sealed name>/. Reads raw files only; the pipeline's archive, manifest,
stores and state are not touched.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DS = "final_offset"
MATCH_S, F, G_STORE = 120.0, 4, 1536
RUN_GAP_S, JUMP_PX, KEY, FP_FRAMES = 300, 1.0, 10, 240
START = pd.Timestamp("2026-09-27 03:00")
STAMP = re.compile(r"(20\d\d-\d\d-\d\d)T(\d\d)\.(\d\d)\.(\d\d)\.(\d+)")


def robust_sigma(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def edges_one(path):
    """Header values and limb edge points of one 2048 frame (as scripts/process_frames.py full_frame)."""
    os.environ.setdefault("SUITDYN_DATASET", DS)
    from suitdyn import config, geometry, io
    L = config.load()["limb"]
    try:
        im, h = io.read(path)
        if im.shape != (2048, 2048):
            return None
        geo = geometry.limb(im, float(h["CRPIX1"]) - 1, float(h["CRPIX2"]) - 1, float(h["R_SUN"]), rays=L["rays"],
                            window=L["window"], smooth_px=L["smooth_px"], edge_margin_px=L["edge_margin_px"],
                            harmonics=L["harmonics"], clip_sigma=L["clip_sigma"], scale=1.0)
        if geo is None:
            return None
        e = geo["edges"]
        return {"CROTA2": float(h.get("CROTA2", 0.0)), "DSUN_OBS": float(h["DSUN_OBS"]), "HGLT_OBS": float(h["HGLT_OBS"]),
                "CMD_EXPT": float(h["CMD_EXPT"]), "BZERO": float(h.get("BZERO", 0.0)), "BSCALE": float(h.get("BSCALE", 1.0)),
                "edge_x": e["x"].astype(np.float32), "edge_y": e["y"].astype(np.float32), "edge_usable": e["usable"]}
    except Exception:
        return None


def pc_one(pair):
    """Phase-correlation motion of consecutive raw frames (jump detection, as process_frames.phase_motion)."""
    from suitdyn import io, motion
    a, b = (motion.highpass(io.read(p)[0]) for p in pair)
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    return motion.shift(a * w, b * w, upsample=20)[:2]


def fp_one(path):
    from suitdyn import io, motion
    return motion.highpass(io.read(path)[0])


def segment_positions(job):
    """Image-content positions within one jump-free segment (registration_study._segment_positions)."""
    from suitdyn import io, motion
    paths_, ts, rot_rate, fp = job
    n = len(paths_)
    prep = {}

    def P(i):
        if i not in prep:
            prep[i] = motion.prepared(io.read(paths_[i])[0], fp)
        return prep[i]

    keys = list(range(0, n, KEY))
    kpos = {0: np.zeros(2)}
    for a, b in zip(keys[:-1], keys[1:]):
        dx, dy, _ = motion.shift(P(a), P(b))
        kpos[b] = kpos[a] + np.array([dx, dy]) - rot_rate * (ts[b] - ts[a])
    pos = np.zeros((n, 2))
    for i in range(n):
        k = min(keys, key=lambda kk: abs(kk - i))
        if i == k:
            pos[i] = kpos[k]
            continue
        dx, dy, _ = motion.shift(P(k), P(i))
        pos[i] = kpos[k] + np.array([dx, dy]) - rot_rate * (ts[i] - ts[k])
    return pos


def register(nb, workers):
    """nb: time-sorted frames (t, file, path) -> reg_x0, reg_y0, reg_R and header values."""
    from concurrent.futures import ProcessPoolExecutor
    from suitdyn import config, geometry, pointing, solar
    clip = config.load()["limb"]["clip_sigma"]
    with ProcessPoolExecutor(workers) as ex:
        ed = list(ex.map(edges_one, nb.path.tolist(), chunksize=4))
    nb = nb[[e is not None for e in ed]].reset_index(drop=True)
    ed = [e for e in ed if e is not None]
    for k in ("CROTA2", "DSUN_OBS", "HGLT_OBS", "CMD_EXPT", "BZERO", "BSCALE"):
        nb[k] = [e[k] for e in ed]
    U = np.array([e["edge_usable"] for e in ed])
    rays = U.mean(0) >= 0.98
    cc = []
    for e in ed:
        m = rays & e["edge_usable"]
        f = geometry.fit_limb(e["edge_x"][m], e["edge_y"][m], 0, clip)
        cc.append((f["x0"], f["y0"], f["R"]))
    nb[["cc_x0", "cc_y0", "cc_R"]] = np.array(cc)
    # runs, jumps, segments
    with ProcessPoolExecutor(workers) as ex:
        pc = list(ex.map(pc_one, list(zip(nb.path[:-1], nb.path[1:])), chunksize=4))
    nb["pc_dx"], nb["pc_dy"] = [np.nan] + [p[0] for p in pc], [np.nan] + [p[1] for p in pc]
    dt = nb.t.diff().dt.total_seconds()
    nb["run"] = (dt.isna() | (dt > RUN_GAP_S)).cumsum()
    nb["jump_px"] = np.hypot(nb.pc_dx, nb.pc_dy).where(nb.run.eq(nb.run.shift()))
    nb["segment"] = (nb.run.ne(nb.run.shift()) | (nb.jump_px > JUMP_PX)).cumsum()
    nb["pointing_mode"] = pointing.mode(nb.cc_x0, nb.cc_y0, 2048)
    nb["segment"] = (nb.segment.ne(nb.segment.shift()) | nb.pointing_mode.ne(nb.pointing_mode.shift())).cumsum()
    # fixed pattern from these frames, image motion per segment, anchors
    pick = np.linspace(0, len(nb) - 1, min(FP_FRAMES, len(nb))).astype(int)
    with ProcessPoolExecutor(workers) as ex:
        st = np.array(list(ex.map(fp_one, nb.path[pick].tolist(), chunksize=4)))
    fp = np.median(st, 0)
    jobs, groups = [], []
    for _, g in nb.groupby("segment"):
        ts = (g.t - g.t.iloc[0]).dt.total_seconds().values
        rx, ry = solar.disk_centre_motion_px(float(g.cc_R.median()), float(g.HGLT_OBS.iloc[0]), 1.0, float(g.CROTA2.median()))
        jobs.append((g.path.tolist(), ts, np.array([rx, ry]), fp))
        groups.append(g.index)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(segment_positions, jobs))
    nb["img_x"], nb["img_y"] = np.nan, np.nan
    for gi, pos in zip(groups, res):
        nb.loc[gi, "img_x"], nb.loc[gi, "img_y"] = pos[:, 0], pos[:, 1]
    for ax in ("x", "y"):
        d = nb[f"cc_{ax}0"] - nb[f"img_{ax}"]
        off = d.groupby(nb.segment).transform(lambda s: s[np.abs(s - s.median()) <= 4 * max(robust_sigma(s), 0.3)].mean())
        nb[f"reg_{ax}0"] = nb[f"img_{ax}"] + off
    rn = nb.cc_R * nb.DSUN_OBS
    nb["reg_R"] = rn.groupby(nb.segment).transform(
        lambda s: s[np.abs(s - s.median()) < 4 * max(robust_sigma(s), 1e-6)].mean()) / nb.DSUN_OBS
    return nb


def build(path, row, ctx):
    """Raw frame -> a 384 frame-cache frame (store.process, then the scripts/phase3_prepare.py steps)."""
    import warnings
    from suitdyn import response, store
    cfg, pattern, pmode, resp, disk, core = ctx
    r = {"path": path, "reg_x0": row["reg_x0"], "reg_y0": row["reg_y0"], "reg_R": row["reg_R"], "CROTA2": row["CROTA2"],
         "CMD_EXPT": row["CMD_EXPT"], "clip_lo": row["BZERO"] + row["BSCALE"] * -32768, "clip_hi": row["BZERO"] + row["BSCALE"] * 32767}
    img, m = store.process(r, G_STORE, 690.0, cfg["qc"], int(cfg["limb"]["edge_margin_px"]), pattern, pmode)
    img = img.astype(np.float32)
    img[m != 0] = np.nan
    n = img.shape[0] // F
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img = np.nanmean(img.reshape(n, F, n, F), axis=(1, 3))
    img = img * response.factor(resp, n, row["reg_x0"], row["reg_y0"])
    return np.where(disk, img / np.nanmedian(img[core]), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = DS
    from suitdyn import atomic, config, normalize, paths, response
    cfg = config.load()
    sealed = Path(a.sealed)
    out = sealed.parent / f"answers_{sealed.name}"
    out.mkdir(exist_ok=True)
    t_start = time.time()

    targets = json.loads((sealed / "targets.json").read_text())
    tt = [t for v in targets["short"]["target_utc"].values() for t in v] + [t for v in targets["roll"]["target_utc"] for t in v]
    t_need = pd.to_datetime(pd.Series(tt)).dt.tz_localize(None)
    hi = t_need.max() + pd.Timedelta(minutes=30)
    files = []
    for dp, _, fn in os.walk(cfg["paths"]["raw_root"]):
        for f in fn:
            if f.endswith(".fits") and "NB03" in f and (m := STAMP.search(f)):
                t = pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}.{m.group(5)}")
                p = os.path.join(dp, f)
                if START <= t <= hi and 8_000_000 < os.path.getsize(p) < 12_000_000:   # 2048 full-disk only
                    files.append((t, f, p))
    nb = pd.DataFrame(files, columns=["t", "file", "path"]).sort_values("t").reset_index(drop=True)
    print(f"2048 full-disk NB03 frames {START} .. {hi}: {len(nb)}", flush=True)
    nb = register(nb, a.workers)
    atomic.to_parquet(nb.drop(columns=["path"]), out / "registration.parquet")
    print(f"registered {len(nb)} frames in {nb.segment.nunique()} segments ({time.time() - t_start:.0f} s)", flush=True)

    pattern = np.load(paths.calibration("nb03_pattern.npy"))
    pmode = json.loads(paths.calibration("nb03_pattern.json").read_text()).get("mode", "multiplicative")
    resp = response.load(str(paths.phase2("response", f"response_{DS}.npz")))
    G = G_STORE // F
    mu = normalize.mu_map(G, 690.0 / F)
    ctx = (cfg, pattern, pmode, resp, mu > np.sqrt(1 - 0.95 ** 2), mu > np.sqrt(1 - 0.9 ** 2))

    # VALIDATION: frames registered both here and by the pipeline
    preg = pd.read_parquet(paths.phase1("registration.parquet"))
    fr = pd.read_parquet(paths.stores(f"{DS}.frames.parquet"))
    cache = np.load(paths.phase3("cache", f"frames_{G}.npy"), mmap_mode="r")
    both = nb.merge(preg[["file", "reg_x0", "reg_y0", "reg_R"]], on="file", suffixes=("", "_pipe"))
    both = both.merge(fr[["frame_id", "store_index"]], left_on="file", right_on="frame_id")
    d = np.hypot(both.reg_x0 - both.reg_x0_pipe, both.reg_y0 - both.reg_y0_pipe)
    vrep = {"frames_compared": int(len(both)), "d_centre_px_median": float(d.median()), "d_centre_px_p95": float(d.quantile(0.95)),
            "d_centre_px_max": float(d.max()), "d_R_px_median": float((both.reg_R - both.reg_R_pipe).abs().median())}
    sub = both.iloc[np.linspace(0, len(both) - 1, min(12, len(both))).astype(int)]
    maes = []
    for r in sub.itertuples():
        img = build(nb.path[nb.file == r.file].iloc[0], r._asdict(), ctx)
        ref = cache[int(r.store_index)].astype(np.float32)
        ok = np.isfinite(img) & np.isfinite(ref)
        maes.append(float(np.abs(img - ref)[ok].mean()))
    vrep["mae_vs_cache_median"], vrep["mae_vs_cache_max"] = float(np.median(maes)), float(np.max(maes))
    atomic.write_json(out / "validate.json", vrep)
    print("validation:", vrep, flush=True)

    # the answer frames
    match = []
    for t in t_need:
        dd = (nb.t - t).abs().dt.total_seconds()
        j = int(dd.idxmin())
        match.append((str(t), j if dd[j] <= MATCH_S else None, float(dd[j])))
    need = sorted({j for _, j, _ in match if j is not None})
    ans = np.full((len(need), G, G), np.nan, np.float16)
    for k, j in enumerate(need):
        ans[k] = build(nb.path[j], nb.loc[j].to_dict(), ctx).astype(np.float16)
    atomic.save_npy(out / "answers_384.npy", ans)
    pos = {j: k for k, j in enumerate(need)}
    atomic.write_json(out / "answers.json", {
        "sealed": str(sealed), "match_window_s": MATCH_S,
        "registration": "pipeline method re-run (image motion with fixed pattern removed + segment circle anchor)",
        "frames": [{"file": nb.file[j], "utc": str(nb.t[j]), "reg_x0": float(nb.reg_x0[j]), "reg_y0": float(nb.reg_y0[j]),
                    "reg_R": float(nb.reg_R[j]), "segment": int(nb.segment[j])} for j in need],
        "targets": [{"target_utc": t, "answer": pos.get(j) if j is not None else None, "dt_s": dd} for t, j, dd in match],
        "products": {"pattern": str(paths.calibration("nb03_pattern.npy")), "pattern_mode": pmode,
                     "response": str(paths.phase2("response", f"response_{DS}.npz"))},
        "validation": vrep, "seconds": round(time.time() - t_start, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"{len(t_need)} targets, {sum(j is not None for _, j, _ in match)} matched within {MATCH_S:.0f} s, "
          f"{len(need)} answer frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
