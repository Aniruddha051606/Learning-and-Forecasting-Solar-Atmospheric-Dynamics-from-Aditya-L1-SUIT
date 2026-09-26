"""Phase 2: noise floor, baselines B0/B1/B2 across horizons, under each normalisation variant.

    python scripts/phase2_baselines.py --store v0 [--split val] [--per-horizon 120] [--grid-factor 1]

For each horizon H (frames) windows are drawn from the chosen split (never the sealed test split),
spread evenly over time. For each window: F(t-1), F(t) and the truth F(t+H) are read from the store;
pixels with any QC bit (seam, edge, off-limb, spike, clipped, no source) are invalid. Forecasts:
  B0 = F(t);  B1 = F(t) rotated by solar differential rotation over the true elapsed time;
  B2 = B1 plus the residual flow between F(t-1) and F(t), extrapolated.
Every normalisation variant is applied to all three images before scoring, so the variants are
compared on identical windows. Scores are on the valid disk (r < 0.9 of r_ref) and reported as
relative MAE (MAE / median truth level), SSIM, gradient correlation and bright-region scores.
Confidence intervals: bootstrap over runs × hour blocks (windows in the same hour are not independent).

Also written, per variant: how much of the true change in bright-region excess brightness between t
and t+H survives normalisation ("signal retained"), and the correlation of the frame level with
pointing (instrument leakage).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import baselines, config, metrics, normalize  # noqa: E402

CFG = config.load_phase2()
SEQ = config.ROOT / "outputs" / "phase2" / "sequences"
STORES = config.ROOT / "outputs" / "phase2" / "stores"
OUT = config.ROOT / "outputs" / "phase2" / "baselines"
B0_DEG = None
_G = {}


def _load(i):
    img = _G["image"][i].astype(np.float32)
    m = _G["mask"][i]
    img[m != 0] = np.nan  # any QC bit, or 255 = no source
    f = _G["factor"]
    if f > 1:
        n = img.shape[0] // f
        img = np.nanmean(img[:n * f, :n * f].reshape(n, f, n, f), axis=(1, 3))
    return img


def _init(store_path, factor, r_ref, level):
    g = zarr.open_group(store_path, mode="r")
    _G.update(image=g["nb03/image"], mask=g["nb03/mask"], factor=factor, r_ref=r_ref, level=level)


def _window(job):
    w, t_prev, t_last, t_tgt, b0 = job
    r_ref = _G["r_ref"]
    prev, last, truth = _load(w["prev"]), _load(w["last"]), _load(w["target"])
    dt, dtp = (t_tgt - t_last).total_seconds(), (t_last - t_prev).total_seconds()
    fc = {"B0": baselines.persistence(last), "B1": baselines.rotated_persistence(last, r_ref, b0, dt)}
    try:
        fc["B2"], _ = baselines.optical_flow_extrapolation(prev, last, r_ref, b0, dtp, dt)
    except Exception:
        fc["B2"] = np.full_like(last, np.nan)
    grid = last.shape[0]
    mu = normalize.mu_map(grid, r_ref)
    disk = (mu > np.sqrt(1 - 0.9 ** 2)) & np.isfinite(truth)
    rows = []
    for vname, fn in normalize.VARIANTS.items():
        kw = {"mu": mu} if vname == "quiet_sun_contrast" else {}
        try:
            tn = fn(truth, disk & np.isfinite(truth), _G["level"], **kw)
            ln = fn(last, disk & np.isfinite(last), _G["level"], **kw)
        except Exception:
            continue
        # the forecast is normalised with the statistics of the frame it came from (last), never the truth
        scale = ln / last
        for bname, f in fc.items():
            fn_ = f * scale if vname != "robust_percentile" else fn(f, disk & np.isfinite(f), _G["level"], **kw)
            m = disk & np.isfinite(fn_) & np.isfinite(tn)
            if m.sum() < 1000:
                continue
            s = metrics.all_metrics(fn_, tn, m)
            s["rel_mae"] = s["mae"] / float(np.nanmedian(np.abs(tn[m])))
            rows.append({"variant": vname, "baseline": bname, "horizon": int(w["horizon"]), "dt_s": dt,
                         "last": int(w["last"]), "t": t_last, "valid_px": int(m.sum()), **s})
        # signal retained: true change of bright-region excess between t and t+H in this variant
        m = disk & np.isfinite(ln) & np.isfinite(tn)
        ex = [float(np.sum((a[m] - np.median(a[m]))[a[m] > 1.3 * np.median(a[m])])) for a in (ln, tn)]
        rows.append({"variant": vname, "baseline": "signal", "horizon": int(w["horizon"]), "dt_s": dt,
                     "last": int(w["last"]), "t": t_last, "excess_last": ex[0], "excess_truth": ex[1],
                     "level_last": float(np.nanmedian(ln[m])), "level_truth": float(np.nanmedian(tn[m]))})
    return rows


def block_bootstrap(df, col, n=2000, seed=0):
    """Median of `col` with a 95 % interval from resampling (run, hour) blocks."""
    rng = np.random.default_rng(seed)
    blocks = [g[col].values for _, g in df.groupby(["run", df.t.dt.floor("h")])]
    if len(blocks) < 2:
        v = np.median(df[col])
        return v, np.nan, np.nan
    meds = [np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))
            for _ in range(n)]
    return float(np.median(df[col])), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="v0")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--per-horizon", type=int, default=120)
    ap.add_argument("--grid-factor", type=int, default=1, help="block-average the grid by this factor")
    ap.add_argument("--context", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    zpath = str(STORES / f"{a.store}.zarr")
    g = zarr.open_group(zpath, mode="r")
    r_ref = float(g.attrs["r_ref"]) / a.grid_factor
    fr = pd.read_parquet(STORES / f"{a.store}.frames.parquet")
    win = pd.read_parquet(SEQ / "windows.parquet")
    win = win[(win.split == a.split) & (win.context == a.context)]
    # store index = row of frames.parquet; window positions refer to sequences/frames.parquet → map by frame_id
    seqf = pd.read_parquet(SEQ / "frames.parquet")
    pos2store = pd.Series(fr.store_index.values, index=fr.frame_id).reindex(seqf.frame_id).values
    man = pd.read_parquet(config.out_dir(CFG) / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file")
    level = float(np.nanmedian([np.nanmedian(g["nb03/image"][i][600:900, 600:900].astype(np.float32))
                                for i in fr.index[fr.split == "train"][::50]]))
    jobs = []
    for h, gw in win.groupby("horizon"):
        pick = gw.iloc[np.linspace(0, len(gw) - 1, min(a.per_horizon, len(gw))).astype(int)]
        for _, w in pick.iterrows():
            i_prev, i_last, i_tgt = w["last"] - 1, w["last"], w["target"]
            ids = [pos2store[i_prev], pos2store[i_last], pos2store[i_tgt]]
            if np.any(pd.isna(ids)):
                continue
            ww = {"prev": int(ids[0]), "last": int(ids[1]), "target": int(ids[2]), "horizon": int(h)}
            t = [seqf.t.iloc[j] for j in (i_prev, i_last, i_tgt)]
            b0 = float(man.loc[seqf.frame_id.iloc[i_last], "HGLT_OBS"])
            jobs.append((ww, t[0], t[1], t[2], b0))
    with ProcessPoolExecutor(CFG["run"]["workers"], initializer=_init,
                             initargs=(zpath, a.grid_factor, r_ref, level)) as ex:
        rows = [r for rs in ex.map(_window, jobs, chunksize=2) for r in rs]
    res = pd.DataFrame(rows)
    res["run"] = res["last"].map(fr.set_index("store_index").run)
    tag = f"{a.store}_{a.split}_g{a.grid_factor}"
    res.to_parquet(OUT / f"scores_{tag}.parquet", index=False)

    sc = res[res.baseline != "signal"]
    summary_rows = []
    for (v, b, h), d in sc.groupby(["variant", "baseline", "horizon"]):
        med, lo, hi = block_bootstrap(d, "rel_mae")
        summary_rows.append({"variant": v, "baseline": b, "horizon": h, "minutes": d.dt_s.median() / 60,
                             "n": len(d), "rel_mae": med, "rel_mae_lo": lo, "rel_mae_hi": hi,
                             "ssim": d.ssim.median(), "grad_corr": d.grad_corr.median(),
                             "bright_iou": d.bright_iou.median()})
    summ = pd.DataFrame(summary_rows)
    # skill of B1 and B2 against B0 on the same windows
    piv = sc.pivot_table(index=["variant", "horizon", "last"], columns="baseline", values="rel_mae").reset_index()
    for b in ("B1", "B2"):
        piv[f"skill_{b}_vs_B0"] = 1 - piv[b] / piv["B0"]
    piv["skill_B2_vs_B1"] = 1 - piv["B2"] / piv["B1"]
    skill = piv.groupby(["variant", "horizon"])[[c for c in piv.columns if c.startswith("skill")]].median()
    sig = res[res.baseline == "signal"].copy()
    sig["d_excess"] = sig.excess_truth - sig.excess_last
    sig["d_level"] = sig.level_truth / sig.level_last - 1
    # signal retained: per horizon, correlation over windows between a variant's change of bright-region
    # excess and the change in the global (un-rescaled) variant. Units differ between variants, so a
    # correlation (does the variant keep the same pattern of solar change?) is the comparable quantity.
    wide = sig.pivot_table(index=["horizon", "last"], columns="variant", values="d_excess")
    rows_r = []
    for h, d in wide.groupby(level=0):
        for v in wide.columns:
            ok = d[v].notna() & d["global"].notna()
            rows_r.append({"horizon": h, "variant": v, "n": int(ok.sum()),
                           "corr_d_excess_with_global": float(np.corrcoef(d[v][ok], d["global"][ok])[0, 1])
                           if ok.sum() > 3 else np.nan,
                           "d_level_rms": float(sig[(sig.variant == v) & (sig.horizon == h)].d_level.std())})
    retained = pd.DataFrame(rows_r)
    summ.to_csv(OUT / f"summary_{tag}.csv", index=False)
    skill.to_csv(OUT / f"skill_{tag}.csv")
    retained.to_csv(OUT / f"signal_{tag}.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for v, c in zip(normalize.VARIANTS, ("k", "C0", "C1", "C2")):
        for b, ls in (("B0", ":"), ("B1", "-"), ("B2", "--")):
            d = summ[(summ.variant == v) & (summ.baseline == b)].sort_values("minutes")
            if len(d):
                ax[0].plot(d.minutes, d.rel_mae * 100, ls, color=c, marker="o", ms=3,
                           label=f"{v} {b}" if b == "B1" or v == "global" else None)
                if b == "B1":
                    ax[0].fill_between(d.minutes, d.rel_mae_lo * 100, d.rel_mae_hi * 100, color=c, alpha=.15)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("horizon (min, measured)")
    ax[0].set_ylabel("relative MAE (%)")
    ax[0].set_title(f"{a.split}: B0 dotted, B1 solid (95 % block CI), B2 dashed")
    ax[0].legend(fontsize=7)
    for v, c in zip(normalize.VARIANTS, ("k", "C0", "C1", "C2")):
        s = skill.loc[v] if v in skill.index.get_level_values(0) else None
        if s is not None:
            mins = summ[(summ.variant == v) & (summ.baseline == "B1")].set_index("horizon").minutes
            ax[1].plot(mins.reindex(s.index), s["skill_B1_vs_B0"], "-o", color=c, ms=3, label=f"{v}: B1 vs B0")
            ax[1].plot(mins.reindex(s.index), s["skill_B2_vs_B1"], "--x", color=c, ms=3, label=f"{v}: B2 vs B1")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("horizon (min)")
    ax[1].set_ylabel("skill (1 − MAE / MAE_ref)")
    ax[1].legend(fontsize=7)
    for v, c in zip(normalize.VARIANTS, ("k", "C0", "C1", "C2")):
        d = retained[retained.variant == v]
        ax[2].plot(d.horizon, d.corr_d_excess_with_global, "-o", color=c, ms=3, label=v)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("horizon (frames)")
    ax[2].set_ylabel("corr. of bright-region excess change with global")
    ax[2].set_title("signal retained by each normalisation")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"baselines_{tag}.png", dpi=80)
    plt.close(fig)
    meta = {"store": a.store, "split": a.split, "grid_factor": a.grid_factor, "windows": len(jobs),
            "seconds": round(time.time() - t0, 1), "store_attrs": dict(g.attrs), **CFG["_meta"]}
    (OUT / f"meta_{tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    pd.set_option("display.width", 220)
    print(summ[summ.baseline == "B1"].round(4).to_string(index=False))
    print(skill.round(3).to_string())


if __name__ == "__main__":
    main()
