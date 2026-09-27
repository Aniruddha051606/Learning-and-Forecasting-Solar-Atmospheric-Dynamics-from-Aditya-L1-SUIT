"""Phase 3 evaluation on the validation split (the test split stays sealed).

    python scripts/phase3_evaluate.py

For every validation sample (same windows for every method):
  B1        the last context frame rotated to the target time (rotation-corrected persistence)
  B1-avg    the mean of the K context frames, each rotated to the target time
  LT        per-pixel least-squares linear trend over the rotated context, extrapolated to the target time
  <model>   each trained run (B1 + predicted residual), and the seed ensemble of each model type
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config  # noqa: E402
from suitdyn.ml import models  # noqa: E402

CFG = config.load_phase2()
CACHE = config.ROOT / "outputs" / "phase3" / "cache"
RUNS = config.ROOT / "outputs" / "phase3" / "runs"
OUT = config.ROOT / "outputs" / "phase3" / "eval"
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(CACHE / f"samples_{G}.parquet")
    val = meta[meta.set == "val"]
    X = np.load(CACHE / f"X_{G}.npy", mmap_mode="r")
    Y = np.load(CACHE / f"Y_{G}.npy", mmap_mode="r")
    mu_np = np.load(CACHE / f"mu_{G}.npy")
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
        preds = {"B1": x[:, -1], "B1-avg": x.mean(1)}
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
    summ = []
    for reg in ("trusted", "disk", "plage"):
        for h, d in res.groupby("horizon"):
            for mth in methods:
                col = f"{reg}|{mth}"
                if col not in d or d[col].isna().all():
                    continue
                dd = d.dropna(subset=[col, f"{reg}|B1", f"{reg}|B1-avg"]).copy()
                dd["skill_B1"] = 1 - dd[col] / dd[f"{reg}|B1"]
                dd["skill_avg"] = 1 - dd[col] / dd[f"{reg}|B1-avg"]
                s1, l1, h1 = block_ci(dd, "skill_B1")
                s2, l2, h2 = block_ci(dd, "skill_avg")
                summ.append({"region": reg, "horizon": h, "minutes": float(dd.minutes.median()), "method": mth,
                             "n": len(dd), "rel_mae": float(dd[col].median()),
                             "skill_vs_B1": s1, "skill_vs_B1_lo": l1, "skill_vs_B1_hi": h1,
                             "skill_vs_avg": s2, "skill_vs_avg_lo": l2, "skill_vs_avg_hi": h2,
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
    meta_out = {"runs": [r[0] for r in runs], "val_samples": int(len(val)), **CFG["_meta"]}
    (OUT / "eval_meta.json").write_text(json.dumps(meta_out, indent=1, default=str))
    pd.set_option("display.width", 250)
    show = summ[summ.method.isin(["B1-avg", "LT"] + [m for m in methods if m.endswith("-ens")])]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
