"""Test G: the frozen final_offset models on an independent later period (docs/PREREGISTRATION.md, Addendum
G).

    python scripts/posthoc_new_period.py [--start "2026-09-28 00:00"] [--end "2026-10-06 23:59:59"] [--workers 4]
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
sys.path.insert(0, str(ROOT / "scripts"))
DS = "final_offset"
STAMP = re.compile(r"(20\d\d-\d\d-\d\d)T(\d\d)\.(\d\d)\.(\d\d)\.(\d+)")
PROG = re.compile(r"SUT_([A-Z0-9]+_\d+)_")
HORIZONS = (20, 40, 80, 160)
K = 5
_W = {}


def _init():
    os.environ["SUITDYN_DATASET"] = DS
    from suitdyn import config, normalize, paths, response
    cfg = config.load()
    pattern = np.load(paths.calibration("nb03_pattern.npy"))
    pmode = json.loads(paths.calibration("nb03_pattern.json").read_text()).get("mode", "multiplicative")
    resp = response.load(str(paths.phase2("response", f"response_{DS}.npz")))
    mu = normalize.mu_map(384, 690.0 / 4)
    _W["ctx"] = (cfg, pattern, pmode, resp, mu > np.sqrt(1 - 0.95 ** 2), mu > np.sqrt(1 - 0.9 ** 2))


def _build(job):
    import answer_frames as af
    i, path, row = job
    try:
        return i, af.build(path, row, _W["ctx"]).astype(np.float16), None
    except Exception as e:
        return i, None, f"{type(e).__name__}: {e}"[:200]


def prepare(a, af, cfg, out, cache, start, end):
    """Steps 1-4: registration, frame selection, the frame cache and the sample index."""
    from concurrent.futures import ProcessPoolExecutor
    from suitdyn import atomic, paths
    check_lo, check_hi = pd.Timestamp("2026-09-27 03:00"), pd.Timestamp("2026-09-27 14:59")

    # 1.
    reg_file = out / "registration.parquet"
    if reg_file.exists():
        nb = pd.read_parquet(reg_file)
        want, paths_ = set(nb.file), {}
        for dp, _, fn in os.walk(cfg["paths"]["raw_root"]):
            for f in fn:
                if f in want:
                    paths_[f] = os.path.join(dp, f)
        nb["path"] = nb.file.map(paths_)
    else:
        files = []
        for dp, _, fn in os.walk(cfg["paths"]["raw_root"]):
            for f in fn:
                if f.endswith(".fits") and "NB03" in f and (m := STAMP.search(f)):
                    t = pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}.{m.group(5)}")
                    p = os.path.join(dp, f)
                    if (check_lo <= t <= check_hi or start <= t <= end) and 8_000_000 < os.path.getsize(p) < 12_000_000:
                        files.append((t, f, p))
        nb = pd.DataFrame(files, columns=["t", "file", "path"]).sort_values("t").reset_index(drop=True)
        print(f"{len(nb)} frames to register ({(nb.t >= start).sum()} in the test period)", flush=True)
        nb = af.register(nb, a.workers)
        atomic.to_parquet(nb.drop(columns=["path"]), reg_file)
    preg = pd.read_parquet(paths.phase1("registration.parquet"))
    both = nb.merge(preg[["file", "reg_x0", "reg_y0", "reg_R"]], on="file", suffixes=("", "_pipe"))
    d = np.hypot(both.reg_x0 - both.reg_x0_pipe, both.reg_y0 - both.reg_y0_pipe)
    regcheck = {"frames_compared": int(len(both)), "d_centre_px_median": float(d.median()), "d_centre_px_p95": float(d.quantile(0.95))}
    print("registration check vs the pipeline (27 Sep):", regcheck, flush=True)

    # 2.
    nb = nb[(nb.t >= start) & (nb.t <= end)].sort_values("t").reset_index(drop=True)
    nb["program"] = nb.file.str.extract(PROG)[0]
    first = nb.program.ne(nb.program.shift())
    rs = lambda s: 1.4826 * np.median(np.abs(s - np.median(s))) if len(s) else np.nan
    lx = (nb.cc_x0 - nb.reg_x0).groupby(nb.segment).transform(lambda s: np.abs(s - s.median()) / max(rs(s), 0.3))
    ly = (nb.cc_y0 - nb.reg_y0).groupby(nb.segment).transform(lambda s: np.abs(s - s.median()) / max(rs(s), 0.3))
    outlier = (lx > 4) | (ly > 4)
    keep = ~first & ~outlier
    sel = {"frames": int(len(nb)), "dropped_programme_first": int(first.sum()), "dropped_limb_outlier": int((outlier & ~first).sum())}
    nb = nb[keep].reset_index(drop=True)
    print("frame selection:", sel, flush=True)

    # 3.
    fpath = cache / "frames_384.npy"
    if fpath.exists() and np.load(fpath, mmap_mode="r").shape[0] == len(nb):
        frames = np.array(np.load(fpath, mmap_mode="r"))
        todo = [i for i in range(len(nb)) if not np.isfinite(frames[i, ::7, ::7]).any()]
        print(f"  existing cache: {len(todo)} empty frames to build again", flush=True)
    else:
        frames = np.full((len(nb), 384, 384), np.nan, np.float16)
        todo = list(range(len(nb)))
    if todo:
        errors = {}
        for attempt in range(3):
            jobs = [(i, nb.path[i], nb.loc[i].drop(labels=["path"]).to_dict()) for i in todo]
            with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
                for k, (i, img, err) in enumerate(ex.map(_build, jobs, chunksize=8)):
                    if img is None:
                        errors[i] = err
                    else:
                        frames[i] = img
                        errors.pop(i, None)
                    if k % 500 == 0:
                        print(f"  frame cache {k}/{len(jobs)} (attempt {attempt + 1})", flush=True)
            todo = sorted(errors)
            print(f"  attempt {attempt + 1}: {len(todo)} frames failed" +
                  (f"; first error: {errors[todo[0]]}" if todo else ""), flush=True)
            if not todo:
                break
            time.sleep(30)
        sel["build_failed"] = len(todo)
        atomic.save_npy(fpath, frames)
        if len(todo) > 0.02 * len(nb):
            print(f"WARNING: {len(todo)} of {len(nb)} frames could not be built; the windows cover less of the period",
                  flush=True)
    src = paths.phase3("cache")
    for f in ("mu_384.npy", "trusted_384.npy"):
        (cache / f).write_bytes((src / f).read_bytes())

    # 4.
    from suitdyn import sequences
    from suitdyn.ml import samples as smp
    ok = np.isfinite(frames.reshape(len(nb), -1)[:, ::97]).any(1)
    seqf = pd.DataFrame({"frame_id": nb.file, "t": nb.t, "run": sequences.runs(nb.t, 300), "ok": ok})
    win = []
    for r, g in seqf.groupby("run"):
        pos = g.index.values
        for j in range(K - 1, len(pos)):
            for H in HORIZONS:
                if j + H < len(pos) and g.ok.iloc[j - K + 1:j + 1].all() and g.ok.iloc[j + H]:
                    win.append({"first": pos[j - K + 1], "last": pos[j], "target": pos[j + H], "horizon": H})
    win = pd.DataFrame(win)
    if a.max_windows:
        win = win.iloc[np.linspace(0, len(win) - 1, a.max_windows).astype(int)].reset_index(drop=True)
    store = pd.DataFrame({"frame_id": nb.file, "store_index": np.arange(len(nb)), "reg_x0": nb.reg_x0, "reg_y0": nb.reg_y0})
    b0 = pd.Series(nb.HGLT_OBS.values, index=nb.file)
    idx = smp.build(win, seqf, store, b0, ["val"] * len(win))
    atomic.to_parquet(idx, cache / "samples_384.parquet")
    atomic.write_json(cache / "prepare_meta.json", {"grid": 384, "r_ref": 690.0 / 4, "context": K, "horizons": list(HORIZONS),
                                                    "source": "posthoc_new_period.py (frozen final_offset products)"})
    print(f"{len(idx)} windows: {idx.horizon.value_counts().sort_index().to_dict()}", flush=True)
    del frames  # the Bank below loads the cache file itself
    return sel, regcheck, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-09-28 00:00:00")
    ap.add_argument("--end", default="2026-10-06 23:59:59")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-windows", type=int, default=0, help="code tests only")
    ap.add_argument("--score-only", action="store_true", help="re-score the existing cache; no registration")
    ap.add_argument("--prepare-only", action="store_true", help="build or repair the cache and windows, no scoring (CPU)")
    ap.add_argument("--runs-dir", default="runs", help="model folder under phase3/ (runs_v1_targetmask, runs, ...)")
    ap.add_argument("--mask", choices=["context", "target"], default="context",
                    help="the models' mask channel: context frames only, or the original target-aware mask (v1)")
    ap.add_argument("--tag", default="", help="suffix of the output files (summary_<tag>.csv, ...)")
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = DS
    os.environ["SUITDYN_RUNS_DIR"] = a.runs_dir
    import answer_frames as af
    from suitdyn import atomic, config, paths
    t0 = time.time()
    cfg = config.load()
    out = paths.phase3("newperiod")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    if a.score_only:  # re-score the existing cache (another model set or mask); nothing is registered again
        prev = next((json.loads((out / f).read_text()) for f in ("prepare.json", "meta.json") if (out / f).exists()), {})
        sel, regcheck = prev.get("selection"), prev.get("registration_check")
        idx = pd.read_parquet(cache / "samples_384.parquet")
    else:
        sel, regcheck, idx = prepare(a, af, cfg, out, cache, start, end)
        atomic.write_json(out / "prepare.json", {"selection": sel, "registration_check": regcheck, "windows": int(len(idx)),
                                                 "days": int(pd.to_datetime(idx.t_last).dt.floor("D").nunique()),
                                                 "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        if a.prepare_only:
            return

    # 5.
    import torch
    import phase3_evaluate as pe
    from suitdyn import progress
    from suitdyn.ml import data, geometry, thermal
    P3, dev = pe.P3, a.device
    E, bf16 = P3["eval"], P3["train"]["bf16"]
    bank = data.Bank(cache, dev)
    bgz = data.load_background(bank, paths.phase3("background", "static_bg_384.npz"))
    qmap = torch.from_numpy(np.interp(bank.mu.cpu().numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32)).to(dev)
    qmap = torch.where(torch.isfinite(bank.S), qmap, torch.full_like(qmap, np.nan))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).to(dev) for k in bgz.files if k.startswith("M_H")}
    runs = [r for r in pe.load_runs(bank, dev) if not r[4]]
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    ids_all = np.arange(len(bank.index))
    rows = []
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report("posthoc: new period", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], len(ids), bank.K)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                 "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}
        ens = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and dev.startswith("cuda")):
            for name, kind, inputs, m, *_ in runs:
                x = b["x_bg"] if inputs == "bg" else b["x_plain"]
                mi = data.input_mask(x)
                if a.mask == "target":  # the original v1 mask channel (Addendum H)
                    mi = mi & torch.isfinite(b["y"])
                preds[name] = x[:, -1] + m(torch.nan_to_num(x), mi.float(), bank.mu[None, None], b["h"]).float()[:, 0]
                ens.setdefault(f"{kind}_{inputs}-ens", []).append(preds[name])
        preds.update({k: torch.stack(v).mean(0) for k, v in ens.items() if len(v) > 1})
        y = b["y"][:, 0]
        valid = torch.isfinite(y) & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0) & inner
        lvl = pe.gauss_level(preds["B1"], E["plage_sigma_px"])
        regions = {"disk": valid, "trusted": valid & bank.trusted, "plage": valid & (preds["B1"] / lvl > E["plage_contrast"])}
        for rname, (lo, hi) in pe.RINGS.items():
            regions[rname] = valid & (rho >= lo) & (rho < hi)
        recs = [{"sample": int(i), "horizon": int(bank.horizon[i]), "run": int(bank.index.run.iloc[i]),
                 "t_last": bank.index.t_last.iloc[i], "minutes": float(bank.index.dt_target_s.iloc[i]) / 60} for i in ids]
        for reg, msk in regions.items():
            for name, p in preds.items():
                mae, _, _, cnt = pe.masked_metrics(p, y, msk)
                for j in range(len(ids)):
                    if float(cnt[j]) >= 200:
                        recs[j][f"{reg}|{name}"] = float(mae[j])
        rows += recs
    sfx = f"_{a.tag}" if a.tag else ""
    res = pd.DataFrame(rows)
    atomic.to_parquet(res, out / f"errors{sfx}.parquet")
    methods = [c.split("|", 1)[1] for c in res.columns if c.startswith("disk|")]
    base = [m for m in E["baselines"] if m in methods]

    def day_ci(df, col, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        blocks = [g[col].values for _, g in df.groupby(pd.to_datetime(df.t_last).dt.floor("D"))]
        if len(blocks) < 2:
            return np.nan, np.nan
        bs = [np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])) for _ in range(n)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    summ = []
    for reg in ["disk", "trusted", "plage", *pe.RINGS]:
        for h, d in res.groupby("horizon"):
            cols = [f"{reg}|{m}" for m in methods]
            if not all(c in d for c in cols):
                continue
            d = d.dropna(subset=cols)
            if not len(d):
                continue
            strongest = min(base, key=lambda m: d[f"{reg}|{m}"].median())
            for m in methods:
                dd = d.assign(s=1 - d[f"{reg}|{m}"] / d[f"{reg}|{strongest}"])
                s, lo, hi = pe.block_ci(dd, "s", E["bootstrap"])
                dlo, dhi = day_ci(dd, "s")
                summ.append({"region": reg, "horizon": int(h), "minutes": float(dd.minutes.median()), "method": m,
                             "strongest_baseline": strongest, "n": len(dd), "days": int(pd.to_datetime(dd.t_last).dt.floor("D").nunique()),
                             "skill_vs_strongest": s, "lo": lo, "hi": hi, "day_lo": dlo, "day_hi": dhi,
                             "G1": bool(lo > 0) if m.endswith("-ens") else None,
                             "G2": bool(dlo > 0) if m.endswith("-ens") and np.isfinite(dlo) else None})
    s = pd.DataFrame(summ)
    s.to_csv(out / f"summary{sfx}.csv", index=False)
    atomic.write_json(out / f"meta{sfx}.json", {"registration": "docs/PREREGISTRATION.md, Addenda G and I", "period": [str(start), str(end)],
                                          "runs_dir": a.runs_dir, "mask": a.mask, "score_only": a.score_only,
                                          "selection": sel, "registration_check": regcheck, "windows": int(len(idx)),
                                          "runs": [r[0] for r in runs], "seconds": round(time.time() - t0, 1)})
    show = s[s.region.isin(["disk", "plage"]) & s.method.str.endswith("-ens")]
    print("new period (28 Sep - 6 Oct), skill vs the strongest baseline (%):")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill_vs_strongest") * 100).round(2).to_string())
    print(show[show.region == "disk"][["method", "horizon", "lo", "hi", "day_lo", "day_hi", "G1", "G2"]].to_string(index=False))


if __name__ == "__main__":
    main()
