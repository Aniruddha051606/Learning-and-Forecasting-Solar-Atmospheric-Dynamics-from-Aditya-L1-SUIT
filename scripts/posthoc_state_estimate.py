"""Post-hoc test F: state estimate or forecast?

    python scripts/posthoc_state_estimate.py --dataset final_offset [--device cuda]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CADENCE_S = 89.0
SHORT = (20, 40, 80)
TARGETS = (80, 160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
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
    runs = [r for r in pe.load_runs(bank, dev) if not r[4]]
    kinds = {r[0]: f"{r[1]}_{r[2]}" for r in runs}
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    idx = bank.index
    nan = torch.tensor(float("nan"), device=dev)

    def run_model(m, x, h):
        """Background-aware forecast of one model from inputs x (B, K, G, G) at horizon h (frames)."""
        mi = torch.isfinite(x).all(1, keepdim=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and dev.startswith("cuda")):
            r = m(torch.nan_to_num(x), mi.float(), bank.mu[None, None], torch.full((x.shape[0],), h, device=dev)).float()[:, 0]
        return torch.where(mi[:, 0], x[:, -1] + r, nan)

    val = bank.ids("val")
    ids_all = val[np.isin(bank.horizon[val], TARGETS)]
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    rows = []
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report(f"posthoc: state estimate {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(ids_all) + 7) // 8)
        B = len(ids)
        b = bank.batch(ids, "plain")
        S = b["S"]                                                           # (B, G, G)
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], B, K)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                 "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}
        Hs = bank.horizon[ids].astype(int)
        ctx = bank._frames(bank.ctx[ids].reshape(-1)).view(B, K, G, G)
        dt_ctx = torch.from_numpy(bank.dt[ids]).to(dev)                       # (B, K) seconds context -> target
        dt_tgt = torch.from_numpy(idx.dt_target_s.values[ids].astype(np.float32)).to(dev)
        b0 = torch.from_numpy(bank.b0[ids]).to(dev)
        ens = {}
        for name, kind, inputs, m, *_ in runs:
            own = torch.stack([run_model(m, b["x_bg"][j:j + 1], int(Hs[j]))[0] for j in range(B)])
            ens.setdefault(f"{kinds[name]}-ens@H", []).append(own)
            for h in SHORT:
                use = torch.from_numpy(Hs > h).to(dev)
                if not bool(use.any()):
                    continue
                shift = dt_tgt - h * CADENCE_S                                 # seconds from t_last + h to the target
                grid, ok = geometry.derotation_grid(G, r_ref, b0, dt_ctx - shift[:, None])
                x = geometry.warp(ctx, grid, ok)
                x = x - geometry.warp(S[:, None].expand(B, K, G, G), grid, ok) + S[:, None]
                f = run_model(m, x, h)
                g2, ok2 = geometry.derotation_grid(G, r_ref, b0, shift[:, None])
                rot = geometry.warp((f - S)[:, None], g2, ok2)[:, 0] + S
                rot = torch.where(use[:, None, None], rot, nan)
                ens.setdefault(f"{kinds[name]}-ens@{h}>H", []).append(rot)
        preds.update({k: torch.stack(v).mean(0) for k, v in ens.items()})
        y = b["y"][:, 0]
        recs = [{"sample": int(i), "horizon": int(bank.horizon[i]), "run": int(idx.run.iloc[i]), "t_last": idx.t_last.iloc[i],
                 "minutes": float(idx.dt_target_s.iloc[i]) / 60} for i in ids]
        for j in range(B):
            names = [k for k in preds if not torch.isnan(preds[k][j]).all()]
            valid = torch.isfinite(y[j]) & inner & torch.stack([torch.isfinite(preds[k][j]) for k in names]).all(0)
            lvl = pe.gauss_level(preds["B1"][j:j + 1], E["plage_sigma_px"])[0]
            regions = {"disk": valid, "plage": valid & (preds["B1"][j] / lvl > E["plage_contrast"])}
            for rname, (lo, hi) in pe.RINGS.items():
                regions[rname] = valid & (rho >= lo) & (rho < hi)
            for reg, msk in regions.items():
                if int(msk.sum()) < 200:
                    continue
                for k in names:
                    recs[j][f"{reg}|{k}"] = float((preds[k][j] - y[j]).abs()[msk].mean())
        rows += recs
    res = pd.DataFrame(rows)
    atomic.to_parquet(res, out_dir / "state_estimate_errors.parquet")
    base = [m for m in E["baselines"] if f"disk|{m}" in res]
    summ, paired = [], []
    for reg in ["disk", "plage", *pe.RINGS]:
        for H, d in res.groupby("horizon"):
            ms = [c.split("|", 1)[1] for c in d.columns if c.startswith(f"{reg}|") and d[c].notna().all()]
            d = d.dropna(subset=[f"{reg}|{m}" for m in ms])
            if not len(d):
                continue
            strongest = min([m for m in base if m in ms], key=lambda m: d[f"{reg}|{m}"].median())
            for m in ms:
                dd = d.assign(s=1 - d[f"{reg}|{m}"] / d[f"{reg}|{strongest}"])
                s, lo, hi = pe.block_ci(dd, "s", E["bootstrap"])
                summ.append({"region": reg, "horizon": int(H), "minutes": float(d.minutes.median()), "method": m,
                             "strongest_baseline": strongest, "n": len(d), "skill_vs_strongest": s, "lo": lo, "hi": hi})
            for kind in sorted({m.split("-ens")[0] for m in ms if "-ens" in m}):
                own = f"{kind}-ens@H"
                for m in ms:
                    if m.startswith(f"{kind}-ens@") and m != own:
                        dd = d.assign(s=1 - d[f"{reg}|{own}"] / d[f"{reg}|{m}"])
                        s, lo, hi = pe.block_ci(dd, "s", E["bootstrap"])
                        paired.append({"region": reg, "horizon": int(H), "model": kind, "rotated_from": m.split("@")[1].split(">")[0],
                                       "n": len(d), "own_over_rotated": s, "lo": lo, "hi": hi})
    s = pd.DataFrame(summ)
    p = pd.DataFrame(paired)
    s.to_csv(out_dir / "state_estimate_summary.csv", index=False)
    p.to_csv(out_dir / "state_estimate_paired.csv", index=False)
    atomic.write_json(out_dir / "state_estimate_meta.json", {"dataset": a.dataset, "registration": "docs/PREREGISTRATION.md, Addendum F",
                                                             "val_windows": int(len(ids_all)), "runs": [r[0] for r in runs],
                                                             "device": dev, "seconds": round(time.time() - t0, 1)})
    show = s[s.region.isin(["disk", "plage"]) & s.method.str.contains("-ens")]
    print(f"{a.dataset} validation, skill vs the strongest baseline (%):")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill_vs_strongest") * 100).round(2).to_string())
    print("own forecast over the rotated shorter-horizon forecast (%):")
    print((p[p.region.isin(["disk", "plage"])].pivot_table(index=["region", "model", "rotated_from"], columns="horizon",
                                                             values="own_over_rotated") * 100).round(2).to_string())


if __name__ == "__main__":
    main()
