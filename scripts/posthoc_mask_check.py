"""Addendum H: does the target-aware mask channel drive the skill?

    python scripts/posthoc_mask_check.py --dataset final_offset
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
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
    qmap = torch.from_numpy(np.interp(bank.mu.cpu().numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32)).to(dev)
    qmap = torch.where(torch.isfinite(bank.S), qmap, torch.full_like(qmap, np.nan))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).to(dev) for k in bgz.files if k.startswith("M_H")}
    runs = [r for r in pe.load_runs(bank, dev) if not r[4]]
    kinds = {r[0]: f"{r[1]}_{r[2]}" for r in runs}
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    idx = bank.index
    ids_all = bank.ids("val")
    rows = []
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report(f"posthoc: mask check {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], len(ids), bank.K)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                 "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}
        ens = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and dev.startswith("cuda")):
            for name, kind, inputs, m, *_ in runs:
                x = b["x_bg"] if inputs == "bg" else b["x_plain"]
                masks = {"": torch.isfinite(b["y"]) & torch.isfinite(x).all(1, keepdim=True),  # standard (until 2026-10-09)
                         "-ctxmask": torch.isfinite(x).all(1, keepdim=True)}                    # context-only
                for sfx, mi in masks.items():
                    r = m(torch.nan_to_num(x), mi.float(), bank.mu[None, None], b["h"]).float()[:, 0]
                    preds[f"{name}{sfx}"] = x[:, -1] + r
                    ens.setdefault(f"{kinds[name]}-ens{sfx}", []).append(preds[f"{name}{sfx}"])
        preds.update({k: torch.stack(v).mean(0) for k, v in ens.items() if len(v) > 1})
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
    atomic.to_parquet(res, out_dir / "mask_check_errors.parquet")
    methods = [c.split("|", 1)[1] for c in res.columns if c.startswith("disk|")]
    base = [m for m in E["baselines"] if m in methods]
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
    s.to_csv(out_dir / "mask_check_summary.csv", index=False)
    atomic.write_json(out_dir / "mask_check_meta.json", {"dataset": a.dataset, "registration": "docs/PREREGISTRATION.md, Addendum H",
                                                         "val_windows": int(len(ids_all)), "seconds": round(time.time() - t0, 1)})
    show = s[s.region.isin(["disk", "plage"]) & s.method.str.contains("-ens")]
    print(f"{a.dataset} validation, skill vs the strongest baseline (%): standard vs context-only mask")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill_vs_strongest") * 100).round(2).to_string())


if __name__ == "__main__":
    main()
