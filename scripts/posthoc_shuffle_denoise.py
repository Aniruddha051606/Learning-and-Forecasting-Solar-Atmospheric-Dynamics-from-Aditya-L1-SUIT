"""Post-hoc tests D1 (correction transplant) and D2 (edge-preserving denoising) (docs/PREREGISTRATION.md,
Addendum D).

    python scripts/posthoc_shuffle_denoise.py --dataset final_offset [--train-per-horizon 100] [--device cuda]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILTERS = ("median3", "median5", "bilateral1", "bilateral2")
LOWPASS_PX = 2.0  # Gaussian sigma for the low-pass corrections (Addendum D amendment: removes the noise-cancelling part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-per-horizon", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="validation windows (0 = all; code tests only)")
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = a.dataset
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
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
    names = [r[0] for r in runs]
    kinds = {r[0]: f"{r[1]}_{r[2]}" for r in runs}
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    idx = bank.index

    def baselines(b, n):
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], n, bank.K)
        return {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}

    def patches(img, k):
        """(B, G, G) -> (B, k*k, G, G) neighbourhoods, NaN outside the image."""
        p = F.pad(img[:, None], (k // 2,) * 4, value=float("nan"))
        return F.unfold(p, k).view(img.shape[0], k * k, *img.shape[1:])

    def denoise(img, kind):
        fin = torch.isfinite(img)
        if kind.startswith("median"):
            out = torch.nanmedian(patches(img, int(kind[-1])), dim=1).values
        else:
            c = float(kind[-1])
            resid = img - pe.gauss_level(img, 1.0)
            sig = []
            for j in range(img.shape[0]):  # each image's robust noise level (MAD of the high-pass residual)
                r = resid[j][fin[j]]
                sig.append(1.4826 * torch.median(torch.abs(r - torch.median(r))) if r.numel() else torch.tensor(1.0, device=img.device))
            sig_n = torch.stack(sig).clamp(min=1e-6).view(-1, 1, 1, 1)
            P = patches(img, 5)
            yy, xx = torch.meshgrid(torch.arange(-2, 3, device=img.device), torch.arange(-2, 3, device=img.device), indexing="ij")
            ws = torch.exp(-(xx ** 2 + yy ** 2).float() / 2.0).view(1, 25, 1, 1)
            wr = torch.exp(-((P - img[:, None]) ** 2) / (2 * (c * sig_n) ** 2))
            w = torch.where(torch.isfinite(P), ws * wr, torch.zeros_like(wr))
            out = (torch.nan_to_num(P) * w).sum(1) / w.sum(1).clamp(min=1e-12)
        return torch.where(fin, out, img)

    def scores(preds, y, n_ids):
        """Disk MAE per window for each prediction, on the evaluation's validity mask."""
        valid = torch.isfinite(y) & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0) & inner
        return {k: pe.masked_metrics(p, y, valid) for k, p in preds.items()}, valid

    # D2 step 1: choose the filter on training windows (median skill over B1-avg-bgS, disk)
    rng = np.random.default_rng(0)
    tr = bank.ids("train")
    horizons = sorted(set(int(h) for h in bank.horizon))
    pick = np.concatenate([rng.choice(tr[bank.horizon[tr] == h], min(a.train_per_horizon, int((bank.horizon[tr] == h).sum())),
                                      replace=False) for h in horizons])
    tskill = {f: [] for f in FILTERS}
    for n0 in range(0, len(pick), 8):
        th.check()
        ids = pick[n0:n0 + 8]
        progress.report(f"posthoc: denoise choice {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(pick) + 7) // 8)
        b = bank.batch(ids, "plain")
        pr = baselines(b, len(ids))
        pr.update({f"dn-{f}": denoise(pr["B1-avg-bgS"], f) for f in FILTERS})
        m, _ = scores(pr, b["y"][:, 0], len(ids))
        for f in FILTERS:
            ok = m["B1-avg-bgS"][3] >= 200
            tskill[f] += (1 - m[f"dn-{f}"][0][ok] / m["B1-avg-bgS"][0][ok]).tolist()
    tmed = {f: float(np.median(v)) for f, v in tskill.items()}
    best = max(tmed, key=tmed.get)
    print(f"training-window median skill of the filters over B1-avg-bgS: {tmed}; chosen: {best}", flush=True)

    # D1 + D2 on validation
    ids_all = bank.ids("val")
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    tl = pd.to_datetime(idx.t_last).values
    rows, seps = [], []
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report(f"posthoc: transplant+denoise {a.dataset}", item=f"batch {n0 // 8 + 1}", i=n0 // 8,
                        n=(len(ids_all) + 7) // 8)
        other = []
        for i in ids:
            cand = ids_all[(bank.horizon[ids_all] == bank.horizon[i]) & (ids_all != i)]
            dt = np.abs((tl[cand] - tl[i]).astype("timedelta64[s]").astype(float))
            far = cand[dt > np.median(dt)] if len(cand) > 1 else cand
            o = int(rng.choice(far))
            other.append(o)
            seps.append(abs(float((tl[o] - tl[i]).astype("timedelta64[s]").astype(float))) / 3600)
        b = bank.batch(ids, "plain")
        bo = bank.batch(np.array(other), "plain")
        preds = baselines(b, len(ids))
        for f in FILTERS:
            preds[f"B1-avg-bgS-{f}"] = denoise(preds["B1-avg-bgS"], f)
        out, _ = pe.predict(runs, b, bank, bf16)
        _, corr_o = pe.predict(runs, bo, bank, bf16)
        ens = {}
        for n in names:
            last = b["x_bg"][:, -1]
            lp_own, lp_other = pe.gauss_level(out[n] - last, LOWPASS_PX), pe.gauss_level(corr_o[n], LOWPASS_PX)
            avg, avg_o = preds["B1-avg-bgS"], bo["x_bg"].mean(1)
            alp_own = pe.gauss_level(out[n] - avg, LOWPASS_PX)
            alp_other = pe.gauss_level(bo["x_bg"][:, -1] + corr_o[n] - avg_o, LOWPASS_PX)
            for tag, p in ((n, out[n]), (f"{n}-transplant", last + corr_o[n]),
                           (f"{n}-lowpass", last + lp_own), (f"{n}-transplant-lowpass", last + lp_other),
                           (f"{n}-avglp", avg + alp_own), (f"{n}-transplant-avglp", avg + alp_other)):
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
    atomic.to_parquet(res, out_dir / "shuffle_denoise_errors.parquet")
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
    s.to_csv(out_dir / "shuffle_denoise_summary.csv", index=False)
    atomic.write_json(out_dir / "shuffle_denoise_meta.json", {
        "dataset": a.dataset, "registration": "docs/PREREGISTRATION.md, Addendum D", "val_windows": int(len(ids_all)),
        "train_windows_for_filter_choice": int(len(pick)), "filter_train_median_skill": tmed, "chosen_filter": best,
        "transplant_separation_h": {"median": float(np.median(seps)), "p10": float(np.percentile(seps, 10)),
                                    "p90": float(np.percentile(seps, 90))},
        "runs": names, "device": dev, "seconds": round(time.time() - t0, 1)})
    show = s[s.region.isin(["disk", "plage"]) & (s.method.str.contains("-ens") | s.method.str.startswith("B1-avg-bgS-"))]
    print(f"{a.dataset} validation: skill vs the strongest baseline (%); transplant separation median {np.median(seps):.1f} h")
    print((show.pivot_table(index=["region", "method"], columns="horizon", values="skill_vs_strongest") * 100).round(2).to_string())


if __name__ == "__main__":
    main()
