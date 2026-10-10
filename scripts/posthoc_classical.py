"""Post-hoc test E: tuned classical forecasters (docs/PREREGISTRATION.md, Addendum E; exploratory).

    python scripts/posthoc_classical.py --dataset final_offset [--train-per-horizon 300] [--device cuda]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
MT_EDGES = [0.43 + (1.0 - 0.43) * i / 12 for i in range(1, 12)]     # inner edges: 12 bins of mu_target
DM_EDGES = [-0.04 + 0.08 * i / 9 for i in range(1, 9)]              # inner edges: 9 bins of mu_target - mu_source
SIGMAS = (0.5, 0.75, 1.0, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-per-horizon", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="validation windows (0 = all; code tests only)")
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = a.dataset
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import numpy as np
    import pandas as pd
    import torch
    import phase3_evaluate as pe
    from suitdyn import atomic, paths, progress
    from suitdyn.ml import data, geometry, thermal

    t0 = time.time()
    P3, dev = pe.P3, a.device
    E, bf16 = P3["eval"], P3["train"]["bf16"]
    out_dir = paths.phase3("posthoc")
    bank = data.Bank(paths.phase3("cache"), dev)
    bgz = data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    G, K, r_ref = bank.G, bank.K, bank.r_ref
    qmap = torch.from_numpy(np.interp(bank.mu.cpu().numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32)).to(dev)
    qmap = torch.where(torch.isfinite(bank.S), qmap, torch.full_like(qmap, np.nan))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).to(dev) for k in bgz.files if k.startswith("M_H")}
    q_mu = torch.tensor(bgz["q_mu"], dtype=torch.float32, device=dev)
    q_v = torch.tensor(bgz["q"], dtype=torch.float32, device=dev)
    runs = [r for r in pe.load_runs(bank, dev) if not r[4]]
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    horizons = sorted(set(int(h) for h in bank.horizon))
    idx = bank.index

    def q_of(mu):
        """The background file's quiet-Sun profile q(mu), linear interpolation, clamped at the ends."""
        m = mu.clamp(float(q_mu[0]), float(q_mu[-1]))
        i = torch.searchsorted(q_mu, m.contiguous()).clamp(1, len(q_mu) - 1)
        x0, x1, y0, y1 = q_mu[i - 1], q_mu[i], q_v[i - 1], q_v[i]
        return y0 + (y1 - y0) * (m - x0) / (x1 - x0)

    def frames(b):
        """(x_bg (B,K,G,G), x_ld (B,K,G,G)): background-aware derotated context, and the same with the solar
        part scaled by q(mu_target)/q(mu_source).
        """
        B = b["x_bg"].shape[0]
        S = b["S"][:, None]
        g = b["grid"].view(B, K, G, G, 2)
        c = (G - 1) / 2
        cols, rows = (g[..., 0] + 1) / 2 * (G - 1), (g[..., 1] + 1) / 2 * (G - 1)
        mu_s = torch.sqrt(torch.clamp(1 - ((cols - c) ** 2 + (rows - c) ** 2) / r_ref ** 2, min=0))
        ratio = q_of(bank.mu)[None, None] / q_of(mu_s).clamp(min=1e-3)
        return b["x_bg"], (b["x_bg"] - S) * ratio + S, mu_s[:, -1]

    def combine(x, w):
        return (x * w[:-1].view(1, -1, 1, 1)).sum(1) + w[-1]

    def sharpen(img, alpha, sig):
        if alpha == 0:
            return img
        sm = pe.gauss_level(img, sig)
        return torch.where(torch.isfinite(img), img + alpha * (img - sm), img)

    def baselines(b, n):
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], n, K)
        return {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}

    # 1.
    rng = np.random.default_rng(0)
    tr = bank.ids("train")
    pick = np.concatenate([rng.choice(tr[bank.horizon[tr] == h], min(a.train_per_horizon, int((bank.horizon[tr] == h).sum())),
                                      replace=False) for h in horizons])
    mt_e = torch.tensor(MT_EDGES, device=dev)
    dm_e = torch.tensor(DM_EDGES, device=dev)
    nb_mt, nb_dm = len(MT_EDGES) + 1, len(DM_EDGES) + 1

    def bins(mu_s_last):
        """(B, G, G) flat bin index of (mu_target, mu_target - mu_source)."""
        mt = torch.bucketize(bank.mu.expand_as(mu_s_last).contiguous(), mt_e)
        dm = torch.bucketize((bank.mu[None] - mu_s_last).contiguous(), dm_e)
        return mt * nb_dm + dm

    def run_pass(fn, label):
        for n0 in range(0, len(pick), 8):
            th.check()
            ids = pick[n0:n0 + 8]
            progress.report(f"posthoc: classical {label} {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(pick) + 7) // 8)
            fn(ids, bank.batch(ids, "plain"))

    # E1: least-squares weights per horizon (normal equations)
    NE = {h: [torch.zeros(K + 1, K + 1, dtype=torch.float64, device=dev), torch.zeros(K + 1, dtype=torch.float64, device=dev)]
          for h in horizons}

    def acc_w(ids, b):
        xr, _, _ = frames(b)
        y = b["y"][:, 0]
        for j, i in enumerate(ids):
            v = torch.isfinite(y[j]) & torch.isfinite(xr[j]).all(0) & inner
            X = torch.cat([xr[j][:, v].T, torch.ones(int(v.sum()), 1, device=dev)], 1).double()
            NE[int(bank.horizon[i])][0] += X.T @ X
            NE[int(bank.horizon[i])][1] += X.T @ y[j][v].double()
    run_pass(acc_w, "weights")
    W = {h: torch.linalg.solve(v[0], v[1]).float() for h, v in NE.items()}

    def e1(xr, ids):
        return torch.stack([combine(xr[j:j + 1], W[int(bank.horizon[i])])[0] for j, i in enumerate(ids)])

    # E3: mean clipped ratio target/forecast per (horizon, mu_t bin, dmu bin), for B1-avg-bgS and for the E1
    # output
    LS = {(k, h): torch.zeros(nb_mt * nb_dm, dtype=torch.float64, device=dev) for k in ("bgS", "e1") for h in horizons}
    LC = {key: torch.zeros_like(v) for key, v in LS.items()}

    def acc_lut(ids, b):
        xr, _, mus = frames(b)
        y = b["y"][:, 0]
        bi = bins(mus)
        for kind, f in (("bgS", xr.mean(1)), ("e1", e1(xr, ids))):
            for j, i in enumerate(ids):
                v = torch.isfinite(y[j]) & torch.isfinite(f[j]) & inner & (f[j] > 0.05)
                r = (y[j][v] / f[j][v]).clamp(0.5, 2.0).double()
                LS[(kind, int(bank.horizon[i]))].index_add_(0, bi[j][v], r)
                LC[(kind, int(bank.horizon[i]))].index_add_(0, bi[j][v], torch.ones_like(r))
    run_pass(acc_lut, "centre-to-limb table")
    LUT = {k: torch.where(LC[k] > 50, LS[k] / LC[k].clamp(min=1), torch.ones_like(LS[k])).float() for k in LS}

    def lut(f, mus, ids, kind):
        bi = bins(mus)
        return torch.stack([f[j] * LUT[(kind, int(bank.horizon[i]))][bi[j]] for j, i in enumerate(ids)])

    # E2: sharpening of the E1 + E3 forecaster, (alpha, sigma) by median training MAE
    grid_err = {(al, sg): [] for al in ALPHAS for sg in SIGMAS}

    def acc_grid(ids, b):
        xr, _, mus = frames(b)
        y = b["y"][:, 0]
        base = lut(e1(xr, ids), mus, ids, "e1")
        for j in range(len(ids)):
            v = torch.isfinite(y[j]) & torch.isfinite(base[j]) & inner
            if int(v.sum()) < 200:
                continue
            for sg in SIGMAS:
                sm = pe.gauss_level(base[j:j + 1], sg)[0]
                for al in ALPHAS:
                    grid_err[(al, sg)].append(float((base[j] + al * (base[j] - sm) - y[j]).abs()[v].mean()))
    run_pass(acc_grid, "sharpening")
    best = min(grid_err, key=lambda k: np.median(grid_err[k]))
    print(f"E1 weights per horizon (5 frames oldest->newest, offset): { {h: [round(float(x), 3) for x in W[h]] for h in horizons} }", flush=True)
    print(f"E3 table range (e1): { {h: (round(float(LUT[('e1', h)].min()), 3), round(float(LUT[('e1', h)].max()), 3)) for h in horizons} }", flush=True)
    print(f"sharpening chosen on training windows: alpha {best[0]}, sigma {best[1]} px", flush=True)

    # 2.
    ids_all = bank.ids("val")
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    rows = []
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report(f"posthoc: classical score {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        preds = baselines(b, len(ids))
        xr, xl, mus = frames(b)
        w1 = e1(xr, ids)
        preds["E1-wavg"] = w1
        preds["E3-lut"] = lut(xr.mean(1), mus, ids, "bgS")
        preds["E3-qratio"] = xl.mean(1)
        preds["classical"] = sharpen(lut(w1, mus, ids, "e1"), best[0], best[1])
        out, _ = pe.predict(runs, b, bank, bf16)
        preds.update(out)
        y = b["y"][:, 0]
        valid = torch.isfinite(y) & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0) & inner
        lvl = pe.gauss_level(preds["B1"], E["plage_sigma_px"])
        regions = {"disk": valid, "trusted": valid & bank.trusted, "plage": valid & (preds["B1"] / lvl > E["plage_contrast"])}
        for rname, (lo, hi) in pe.RINGS.items():
            regions[rname] = valid & (rho >= lo) & (rho < hi)
        recs = [{"sample": int(i), "horizon": int(bank.horizon[i]), "run": int(idx.run.iloc[i]), "t_last": idx.t_last.iloc[i],
                 "minutes": float(idx.dt_target_s.iloc[i]) / 60} for i in ids]
        for reg, msk in regions.items():
            for name, p in preds.items():
                mae, _, _, cnt = pe.masked_metrics(p, y, msk)
                for j in range(len(ids)):
                    if float(cnt[j]) >= 200:
                        recs[j][f"{reg}|{name}"] = float(mae[j])
        rows += recs
    res = pd.DataFrame(rows)
    atomic.to_parquet(res, out_dir / "classical_errors.parquet")
    methods = [c.split("|", 1)[1] for c in res.columns if c.startswith("disk|")]
    base = [m for m in E["baselines"] if m in methods]
    summ, vsc = [], []
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
                for ref, lst in ((strongest, summ), ("classical", vsc)):
                    dd = d.assign(s=1 - d[f"{reg}|{m}"] / d[f"{reg}|{ref}"])
                    s, lo, hi = pe.block_ci(dd, "s", E["bootstrap"])
                    lst.append({"region": reg, "horizon": int(h), "minutes": float(dd.minutes.median()), "method": m,
                                "reference": ref, "n": len(dd), "skill": s, "lo": lo, "hi": hi})
    pd.DataFrame(summ).to_csv(out_dir / "classical_summary.csv", index=False)
    v = pd.DataFrame(vsc)
    v.to_csv(out_dir / "classical_vs_classical.csv", index=False)
    atomic.write_json(out_dir / "classical_meta.json", {
        "dataset": a.dataset, "registration": "docs/PREREGISTRATION.md, Addendum E", "train_windows": int(len(pick)),
        "val_windows": int(len(ids_all)), "weights_e1": {str(h): W[h].tolist() for h in horizons},
        "e3_table_e1": {str(h): LUT[("e1", h)].view(nb_mt, nb_dm).tolist() for h in horizons},
        "e3_bins": {"mu_target_inner_edges": MT_EDGES, "dmu_inner_edges": DM_EDGES},
        "sharpen_alpha": best[0], "sharpen_sigma_px": best[1],
        "sharpen_train_median_mae": float(np.median(grid_err[best])), "no_sharpen_train_median_mae": float(np.median(grid_err[(0.0, SIGMAS[0])])),
        "runs": [r[0] for r in runs], "device": dev, "seconds": round(time.time() - t0, 1)})
    s = pd.DataFrame(summ)
    show = s[s.region.isin(["disk", "plage"]) & s.method.isin(["E1-wavg", "E3-lut", "E3-qratio", "classical", "unet_bg-ens", "convlstm_bg-ens"])]
    print(f"{a.dataset} validation: skill vs the strongest of the five baselines (%)")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill") * 100).round(2).to_string())
    show = v[v.region.isin(["disk", "plage"]) & v.method.isin(["unet_bg-ens", "convlstm_bg-ens"])]
    print("models vs the tuned classical forecaster (%)")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill") * 100).round(2).to_string())


if __name__ == "__main__":
    main()
