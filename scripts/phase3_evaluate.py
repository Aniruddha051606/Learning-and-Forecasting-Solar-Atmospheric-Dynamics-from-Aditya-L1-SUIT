"""Phase 3 evaluation on the validation split (the test split stays sealed).

    python scripts/phase3_evaluate.py

For every validation sample (same windows for every method):
  B1              the last context frame rotated to the target time (rotation-corrected persistence)
  B1-avg          the mean of the K context frames, each rotated to the target time
  B1-avg-LD       B1-avg with each rotated frame corrected for limb darkening: rotation moves a feature to
                  a different mu, which changes its brightness by q(mu_target)/q(mu_source); q is the quiet-
                  Sun centre-to-limb profile measured on training frames. Pure geometry.
  B1-avg-LD-blur  B1-avg-LD smoothed (NaN-aware Gaussian) by the width that minimises the error on the
                  early-stopping hold-out run, per horizon: the MAE-optimal forecast under uncertainty is
                  smooth, so a model can gain by blurring alone. (B1-avg-blur likewise, without LD.)
  LT              per-pixel least-squares linear trend over the rotated context, extrapolated
  <model>         each trained run (B1 + predicted residual), and the seed ensemble of each model type
A model is only credited with skill it shows over the STRONGEST of these baselines.
Errors are relative MAE (frames are normalised to their disk median, so MAE is already relative) in three
regions: trusted (Phase 2 trusted region), disk (all valid pixels, r < 0.95), plage (pixels where the B1
forecast exceeds 1.3: defined from the forecast, never from the truth).
Skill per window = 1 − MAE_method / MAE_reference; reported as the median with a 95 % interval from a
bootstrap over (run, hour) blocks. The corrected Phase 2 ceilings (PHASE2 §4.7, 384²) are checked.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import baselines, config  # noqa: E402
from suitdyn.ml import models  # noqa: E402

CFG = config.load_phase2()
CACHE = config.phase3_dir("cache")
RUNS = config.phase3_dir("runs")
OUT = config.phase3_dir("eval")
G = 384
# corrected ceilings (PHASE2 §4.7, 384²), by horizon in frames (28 min, 57 min, 1.9 h, 3.8 h)
CEILING = {20: 0.42, 40: 0.51, 80: 0.59, 160: 0.71}


def block_ci(df, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [g[col].values for _, g in df.groupby(["run", df.t_last.dt.floor("h")])]
    med = float(np.median(df[col]))
    if len(blocks) < 2:
        return med, np.nan, np.nan
    bs = [np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])) for _ in range(n)]
    return med, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


SIGMAS = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nan_blur(a, sigma):
    if sigma <= 0:
        return a
    w = np.isfinite(a).astype(np.float32)
    num = gaussian_filter(np.nan_to_num(a), sigma)
    den = gaussian_filter(w, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, num / den, np.nan)


def ld_profile(frames, store_idx, mu, n_bins=30):
    """Centre-to-limb profile q(mu) of the per-frame-normalised training frames: per frame the median of
    every mu annulus (plage covers a small fraction of an annulus, so the median is the quiet level), then
    the median over frames. No brightness window: the NB03 profile spans ~0.67-1.30 of the disk median
    (r < 0.95), and a fixed 0.7-1.3 quiet-Sun window (used before 2026-09-27 16:40) truncated both ends,
    giving a flattened, non-monotonic profile and a wrong limb-darkening baseline."""
    edges = np.linspace(np.sqrt(1 - 0.95 ** 2), 1.0, n_bins + 1)
    prof = []
    for i in store_idx:
        f = frames[i].astype(np.float32)
        ok = np.isfinite(f) & (mu > 0)
        b = np.digitize(mu[ok], edges)
        prof.append([np.median(f[ok][b == k]) if (b == k).sum() > 50 else np.nan for k in range(1, n_bins + 1)])
    q = np.nanmedian(np.array(prof), 0)
    c = 0.5 * (edges[1:] + edges[:-1])
    good = np.isfinite(q)
    return c[good], q[good]


class LD:
    """Limb-darkening correction factor for a frame rotated over dt: q(mu_target) / q(mu_source)."""

    def __init__(self, qc, qv, r_ref, mu):
        self.qc, self.qv, self.r_ref, self.mu = qc, qv, r_ref, mu
        self.q_tgt = np.interp(mu, qc, qv)
        self.cache = {}

    def factor(self, b0, dt):
        key = (round(dt / 2) * 2, round(b0, 1))
        if key not in self.cache:
            (rows, cols), ok = baselines.derotation_coords(G, self.r_ref, key[1], key[0])
            c = (G - 1) / 2
            rho2 = ((rows - c) ** 2 + (cols - c) ** 2) / self.r_ref ** 2
            mu_src = np.sqrt(np.clip(1 - rho2, 0, None))
            f = self.q_tgt / np.interp(mu_src, self.qc, self.qv)
            f[~ok] = np.nan
            if len(self.cache) > 4000:
                self.cache.clear()
            self.cache[key] = f.astype(np.float32)
        return self.cache[key]


def baseline_preds(x, rows, b0s, ld, sig_by_h):
    """x (n, K, G, G) rotated context; rows: sample metadata; returns dict of baseline forecasts."""
    n = len(x)
    out = {"B1": x[:, -1], "B1-avg": x.mean(1)}
    ldx = np.empty_like(x)
    for j in range(n):
        for k, dt in enumerate(rows[j]["dt_context_s"]):
            ldx[j, k] = x[j, k] * ld.factor(b0s[j], dt)
    out["B1-avg-LD"] = ldx.mean(1)
    if sig_by_h is not None:
        for name, src in (("B1-avg-blur", out["B1-avg"]), ("B1-avg-LD-blur", out["B1-avg-LD"])):
            out[name] = np.stack([nan_blur(src[j], sig_by_h[name][rows[j]["horizon"]]) for j in range(n)])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(CACHE / f"samples_{G}.parquet")
    val = meta[meta.set == "val"]
    X = np.load(CACHE / f"X_{G}.npy", mmap_mode="r")
    Y = np.load(CACHE / f"Y_{G}.npy", mmap_mode="r")
    mu_np = np.load(CACHE / f"mu_{G}.npy")
    prep = json.loads((CACHE / "prepare_meta.json").read_text())
    r_ref = float(prep["r_ref"])
    frames = np.load(CACHE / f"frames_{G}.npy", mmap_mode="r")
    store_fr = pd.read_parquet(config.ROOT / "outputs" / "phase2" / "stores" / f"{prep['store']}.frames.parquet")
    man = pd.read_parquet(config.out_dir(CFG) / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file")
    b0_of = store_fr.set_index("store_index").frame_id.map(man.HGLT_OBS).astype(float)
    train_store = store_fr.loc[store_fr.split == "train", "store_index"].values[::10]
    qc, qv = ld_profile(frames, train_store, mu_np)
    ld = LD(qc, qv, r_ref, mu_np)

    # blur widths per horizon, chosen on the hold-out run (never on validation)
    ho = meta[meta.set == "holdout"]
    errs = {name: {} for name in ("B1-avg-blur", "B1-avg-LD-blur")}
    for n0 in range(0, len(ho), 16):
        ids = ho.index.values[n0:n0 + 16]
        x = np.asarray(X[ids], dtype=np.float32)
        y = np.asarray(Y[ids], dtype=np.float32)
        rws = ho.loc[ids].to_dict("records")
        b0s = [b0_of[r["store_target"]] for r in rws]
        bp = baseline_preds(x, rws, b0s, ld, None)
        valid = np.isfinite(y) & np.isfinite(x).all(1)
        for name, src in (("B1-avg-blur", bp["B1-avg"]), ("B1-avg-LD-blur", bp["B1-avg-LD"])):
            for j, r in enumerate(rws):
                for sg in SIGMAS:
                    e = np.abs(nan_blur(src[j], sg) - y[j])[valid[j]]
                    errs[name].setdefault((r["horizon"], sg), []).append(float(np.nanmean(e)))
    sig_by_h = {name: {} for name in errs}
    for name, d in errs.items():
        for h in sorted({k[0] for k in d}):
            sig_by_h[name][h] = min(SIGMAS, key=lambda sg: np.median(d[(h, sg)]))
    print("blur widths chosen on hold-out (px at 384):", sig_by_h, flush=True)
    trusted = np.load(CACHE / f"trusted_{G}.npy").astype(bool)
    mu = torch.from_numpy(mu_np)[None, None].cuda()

    runs = []
    for d in sorted(RUNS.glob("*_s*")):
        if (d / "best.pt").exists() and (d / "run.json").exists():
            info = json.loads((d / "run.json").read_text())
            m = models.MODELS[info["model"]]().cuda()
            m.load_state_dict(torch.load(d / "best.pt", map_location="cuda"))
            m.eval()
            runs.append((d.name, info["model"], m))
    print("runs:", [r[0] for r in runs], flush=True)

    rows = []
    for n0 in range(0, len(val), 8):
        ids = val.index.values[n0:n0 + 8]
        x = np.asarray(X[ids], dtype=np.float32)
        y = np.asarray(Y[ids], dtype=np.float32)
        valid = np.isfinite(y) & np.isfinite(x).all(1)
        rws = val.loc[ids].to_dict("records")
        preds = baseline_preds(x, rws, [b0_of[r["store_target"]] for r in rws], ld, sig_by_h)
        # linear trend: times of the context frames relative to the target (negative seconds)
        for j, i in enumerate(ids):
            tt = -np.asarray(val.loc[i, "dt_context_s"], float)
            A = np.c_[np.ones_like(tt), tt]
            coef = np.linalg.lstsq(A, x[j].reshape(len(tt), -1), rcond=None)[0]
            preds.setdefault("LT", np.empty_like(x[:, -1]))[j] = coef[0].reshape(G, G)
        xt = torch.from_numpy(np.nan_to_num(x)).cuda()
        mt = torch.from_numpy(valid.astype(np.float32))[:, None].cuda()
        ht = torch.from_numpy(val.horizon.values[n0:n0 + 8]).cuda()
        ens = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, kind, m in runs:
                r = m(xt, mt, mu, ht).float()[:, 0].cpu().numpy()
                preds[name] = x[:, -1] + r
                ens.setdefault(f"{kind}-ens", []).append(preds[name])
        for k, v in ens.items():
            preds[k] = np.mean(v, 0)
        for j, i in enumerate(ids):
            regions = {"trusted": valid[j] & trusted, "disk": valid[j], "plage": valid[j] & (x[j, -1] > 1.3)}
            rec = {"sample": int(i), "horizon": int(val.loc[i, "horizon"]), "run": int(val.loc[i, "run"]),
                   "t_last": val.loc[i, "t_last"], "minutes": float(val.loc[i, "dt_target_s"]) / 60}
            for reg, msk in regions.items():
                if msk.sum() < 200:
                    continue
                for name, p in preds.items():
                    rec[f"{reg}|{name}"] = float(np.mean(np.abs(p[j][msk] - y[j][msk])))
            rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "val_errors.parquet", index=False)

    methods = sorted({c.split("|")[1] for c in res.columns if "|" in c})
    base_names = ["B1-avg", "B1-avg-LD", "B1-avg-blur", "B1-avg-LD-blur"]
    summ = []
    for reg in ("trusted", "disk", "plage"):
        for h, d in res.groupby("horizon"):
            for mth in methods:
                col = f"{reg}|{mth}"
                if col not in d or d[col].isna().all():
                    continue
                dd = d.dropna(subset=[col, f"{reg}|B1", f"{reg}|B1-avg"]).copy()
                dd["skill_B1"] = 1 - dd[col] / dd[f"{reg}|B1"]
                # strongest baseline = the one with the lowest median error in this region and horizon
                best = min(base_names, key=lambda b: d[f"{reg}|{b}"].median())
                dd["skill_avg"] = 1 - dd[col] / dd[f"{reg}|{best}"]
                s1, l1, h1 = block_ci(dd, "skill_B1")
                s2, l2, h2 = block_ci(dd, "skill_avg")
                summ.append({"region": reg, "horizon": h, "minutes": float(dd.minutes.median()), "method": mth,
                             "n": len(dd), "rel_mae": float(dd[col].median()),
                             "skill_vs_B1": s1, "skill_vs_B1_lo": l1, "skill_vs_B1_hi": h1,
                             "strongest_baseline": best,
                             "skill_vs_best_baseline": s2, "skill_vs_best_lo": l2, "skill_vs_best_hi": h2,
                             "above_ceiling": bool(s1 > CEILING.get(h, 1.0))})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "summary.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    for k, reg in enumerate(("trusted", "disk", "plage")):
        for mth in methods:
            d = summ[(summ.region == reg) & (summ.method == mth)].sort_values("minutes")
            if d.empty or (mth.endswith(tuple("s0 s1 s2".split())) and "ens" not in mth and len(runs) > 2):
                continue
            ax[k].errorbar(d.minutes, d.skill_vs_B1 * 100, yerr=[(d.skill_vs_B1 - d.skill_vs_B1_lo) * 100,
                                                                 (d.skill_vs_B1_hi - d.skill_vs_B1) * 100],
                           marker="o", ms=4, capsize=3, label=mth)
        ax[k].plot(list(np.array([28.5, 57.1, 114.1, 228.1])), [c * 100 for c in CEILING.values()], "k--",
                   lw=.8, label="corrected ceiling")
        ax[k].axhline(0, color="k", lw=.5)
        ax[k].set_xscale("log")
        ax[k].set_xlabel("horizon (min)")
        ax[k].set_ylabel("skill vs B1 (%)")
        ax[k].set_title(f"{reg}: median skill vs B1, 95 % block CI (validation)")
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "skill_vs_B1.png", dpi=80)
    plt.close(fig)
    meta_out = {"runs": [r[0] for r in runs], "val_samples": int(len(val)), "blur_sigma_px": sig_by_h,
                "ld_profile": {"mu": qc.round(4).tolist(), "q": qv.round(5).tolist()}, **CFG["_meta"]}
    (OUT / "eval_meta.json").write_text(json.dumps(meta_out, indent=1, default=str))
    pd.set_option("display.width", 250)
    show = summ[summ.method.isin(base_names + [m for m in methods if m.endswith("-ens")])]
    show = show[["region", "horizon", "minutes", "method", "rel_mae", "skill_vs_B1", "strongest_baseline",
                 "skill_vs_best_baseline", "skill_vs_best_lo", "skill_vs_best_hi", "above_ceiling"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
