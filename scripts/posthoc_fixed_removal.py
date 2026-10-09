"""Post-hoc test: detector-fixed correction removal (docs/PREREGISTRATION.md, Addendum C; exploratory).

    python scripts/posthoc_fixed_removal.py --dataset final_offset [--per-horizon 400] [--device cuda]

For each final model and horizon, M(H) = the per-pixel mean of the model's correction (forecast minus its
background-aware B1) over TRAINING samples. On the VALIDATION split three forecasts per model are scored, with the
evaluation's own regions, metrics and run-hour block bootstrap (scripts/phase3_evaluate.py):
  <model>          the model as evaluated
  <model>-minusM   the model minus M(H): skill that depends on the input frames
  <model>-Monly    background-aware B1 plus M(H): what a static map gives without solar information
  <model>-avgM     the strongest baseline (B1-avg-bgS) plus M(H)        (amendment: the fair "map only")
  B1-avg-bgS-smooth<s>  the strongest baseline smoothed (sigma s px)   (amendment: what pure denoising gives)
and their seed ensembles. Writes outputs/datasets/<name>/phase3/posthoc/fixed_removal_{summary.csv,errors.parquet,
maps_<G>.npz,meta.json}. The test split is not read.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOOTH = (0.7, 1.0, 1.5)  # px, Gaussian sigma of the smoothed-baseline references


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--per-horizon", type=int, default=400, help="training samples per horizon for M(H)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="validation samples (0 = all; code tests only)")
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
    E, P3, dev = pe.E if hasattr(pe, "E") else pe.P3["eval"], pe.P3, a.device
    bf16 = P3["train"]["bf16"]
    out_dir = paths.phase3("posthoc")
    bank = data.Bank(paths.phase3("cache"), dev)
    bgz = data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    # the evaluation's five baselines exactly (scripts/phase3_evaluate.py), so the mask and "strongest" match it
    qmap = torch.from_numpy(np.interp(bank.mu.cpu().numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32)).to(dev)
    qmap = torch.where(torch.isfinite(bank.S), qmap, torch.full_like(qmap, np.nan))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).to(dev) for k in bgz.files if k.startswith("M_H")}
    runs = [r for r in pe.load_runs(bank, dev) if not r[4]]
    names = [r[0] for r in runs]
    kinds = {r[0]: f"{r[1]}_{r[2]}" for r in runs}
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    G, horizons = bank.G, sorted(set(int(h) for h in bank.horizon))

    # 1. M(H) per run from training samples
    rng = np.random.default_rng(0)
    tr = bank.ids("train")
    pick = np.concatenate([rng.choice(tr[bank.horizon[tr] == h], min(a.per_horizon, int((bank.horizon[tr] == h).sum())),
                                      replace=False) for h in horizons])
    S = {(n, h): torch.zeros(G, G, device=dev) for n in names for h in horizons}
    C = {(n, h): torch.zeros(G, G, device=dev) for n in names for h in horizons}
    for n0 in range(0, len(pick), 8):
        th.check()
        ids = pick[n0:n0 + 8]
        progress.report(f"posthoc: fixed maps {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(pick) + 7) // 8)
        b = bank.batch(ids, "plain")
        ok = torch.isfinite(b["x_bg"]).all(1) & torch.isfinite(b["y"][:, 0])
        _, corr = pe.predict(runs, b, bank, bf16)
        for n in names:
            c = torch.where(ok, corr[n], torch.zeros_like(corr[n]))
            for j, i in enumerate(ids):
                h = int(bank.horizon[i])
                S[(n, h)] += c[j]
                C[(n, h)] += ok[j].float()
    M = {k: torch.where(C[k] > 0, S[k] / C[k].clamp(min=1), torch.zeros_like(S[k])) for k in S}
    np.savez_compressed(out_dir / f"fixed_removal_maps_{G}.npz",
                        **{f"{n}_H{h}": M[(n, h)].cpu().numpy().astype(np.float32) for n in names for h in horizons})

    # 2. score on validation
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    ids_all = bank.ids("val")
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    idx, rows = bank.index, []
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report(f"posthoc: fixed removal {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], len(ids), bank.K)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                 "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}
        for sg in SMOOTH:  # what pure denoising of the strongest baseline gives (Addendum C amendment)
            sm = pe.gauss_level(preds["B1-avg-bgS"], sg)
            preds[f"B1-avg-bgS-smooth{sg}"] = torch.where(torch.isfinite(preds["B1-avg-bgS"]), sm, preds["B1-avg-bgS"])
        out, _ = pe.predict(runs, b, bank, bf16)
        hs = [int(bank.horizon[i]) for i in ids]
        ens = {}
        for n in names:
            Mb = torch.stack([M[(n, h)] for h in hs])
            for tag, p in ((n, out[n]), (f"{n}-minusM", out[n] - Mb), (f"{n}-Monly", b["x_bg"][:, -1] + Mb),
                           (f"{n}-avgM", preds["B1-avg-bgS"] + Mb)):
                preds[tag] = p
                ens.setdefault(tag.replace(n, kinds[n] + "-ens"), []).append(p)
        preds.update({k: torch.stack(v).mean(0) for k, v in ens.items()})
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
    atomic.to_parquet(res, out_dir / "fixed_removal_errors.parquet")
    methods = [c.split("|", 1)[1] for c in res.columns if c.startswith("disk|")]
    base = [m for m in E["baselines"] if m in methods]  # the smoothed references are scored, never "strongest"
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
                summ.append({"region": reg, "horizon": int(h), "minutes": float(dd.minutes.median()), "method": m,
                             "strongest_baseline": strongest, "n": len(dd), "skill_vs_strongest": s, "lo": lo, "hi": hi})
    s = pd.DataFrame(summ)
    s.to_csv(out_dir / "fixed_removal_summary.csv", index=False)
    atomic.write_json(out_dir / "fixed_removal_meta.json", {
        "dataset": a.dataset, "registration": "docs/PREREGISTRATION.md, Addendum C", "train_samples_for_M": int(len(pick)),
        "val_samples": int(len(ids_all)), "runs": names, "device": dev, "seconds": round(time.time() - t0, 1)})
    show = s[(s.region.isin(["disk", "plage"])) & (s.method.str.contains("-ens") | s.method.str.contains("smooth"))]
    print(f"{a.dataset} validation: skill vs the strongest baseline (%)")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill_vs_strongest") * 100).round(2).to_string())


if __name__ == "__main__":
    main()
