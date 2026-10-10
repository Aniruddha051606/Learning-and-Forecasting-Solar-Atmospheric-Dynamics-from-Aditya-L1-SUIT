"""Phase 3 evaluation: validation (default) or the one-time test evaluation, with negative controls.

    python scripts/phase3_evaluate.py [--device cuda] [--max-samples N]
    python scripts/phase3_evaluate.py --split test --reason "final run" [--allow-new-models]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, paths, progress, sequences  # noqa: E402
from suitdyn.ml import data, evalguard, geometry, models, samples, thermal  # noqa: E402

CFG = config.load_dataset()
P3 = config.load_phase3()
RINGS = {"ring_inner": (0.0, 0.5), "ring_mid": (0.5, 0.75), "ring_outer": (0.75, 0.9)}


def block_ci(df, col, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [g[col].values for _, g in df.groupby(["run", pd.to_datetime(df.t_last).dt.floor("h")])]
    med = float(np.median(df[col]))
    if len(blocks) < 2:
        return med, np.nan, np.nan
    bs = [np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])) for _ in range(n)]
    return med, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def gauss_level(img, sigma):
    """NaN-aware Gaussian smoothing of (B, G, G) on the device."""
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    w = torch.isfinite(img).float()[:, None]
    v = torch.nan_to_num(img)[:, None]
    for kk in (k, k.transpose(-1, -2)):
        pad = (r, r, 0, 0) if kk.shape[-1] > 1 else (0, 0, r, r)
        v, w = F.conv2d(F.pad(v, pad), kk), F.conv2d(F.pad(w, pad), kk)
    return (v / w.clamp(min=1e-6))[:, 0]


def masked_metrics(p, y, m):
    """Per window (B,): MAE, RMSE, gradient correlation and pixel count inside mask m (B, G, G)."""
    w = m.float()
    n = w.sum((1, 2))
    e = torch.nan_to_num(p - y)
    mae = (e.abs() * w).sum((1, 2)) / n.clamp(min=1)
    rmse = torch.sqrt((e * e * w).sum((1, 2)) / n.clamp(min=1))
    pn, yn = torch.nan_to_num(p), torch.nan_to_num(y)
    mx = (m[:, :, 1:] & m[:, :, :-1]).float()
    my = (m[:, 1:, :] & m[:, :-1, :]).float()
    gxp, gxy = pn[:, :, 1:] - pn[:, :, :-1], yn[:, :, 1:] - yn[:, :, :-1]
    gyp, gyy = pn[:, 1:, :] - pn[:, :-1, :], yn[:, 1:, :] - yn[:, :-1, :]
    num = (gxp * gxy * mx).sum((1, 2)) + (gyp * gyy * my).sum((1, 2))
    den = torch.sqrt(((gxp ** 2 * mx).sum((1, 2)) + (gyp ** 2 * my).sum((1, 2)))
                     * ((gxy ** 2 * mx).sum((1, 2)) + (gyy ** 2 * my).sum((1, 2)))).clamp(min=1e-12)
    return mae, rmse, num / den, n


def load_runs(bank, dev):
    runs = []
    for d in sorted(paths.runs().glob("*")):
        if not (d / "run.json").exists():
            continue
        info = json.loads((d / "run.json").read_text())
        kw = {"unet": {"k": bank.K, "base": info["model_cfg"]["unet_base"]},
              "convlstm": {"k": bank.K, "hidden": info["model_cfg"]["convlstm_hidden"]}}[info["model"]]
        m = models.MODELS[info["model"]](**kw).to(dev)
        m.load_state_dict(torch.load(d / "best.pt", map_location=dev))
        m.eval()
        runs.append((info["name"], info["model"], info["inputs"], m, info.get("smoke_test", False),
                     info.get("checkpoint_sha256") or hashlib.sha256((d / "best.pt").read_bytes()).hexdigest()))
    return runs


def predict(runs, b, bank, bf16):
    """Forecast and correction (forecast minus the run's own B1) of every run, plus seed ensembles."""
    out, corr, ens = {}, {}, {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for name, kind, inputs, m, *_ in runs:
            x = b["x_bg"] if inputs == "bg" else b["x_plain"]
            xi, mi = torch.nan_to_num(x), data.input_mask(x).float()  # inputs only (Addenda H, I)
            r = m(xi, mi, bank.mu[None, None], b["h"]).float()[:, 0]
            out[name], corr[name] = x[:, -1] + r, r
            ens.setdefault(f"{kind}_{inputs}-ens", []).append(out[name])
    for k, v in ens.items():
        if len(v) > 1:
            out[k] = torch.stack(v).mean(0)
    return out, corr


def test_samples(bank, reason, allow_new, runs, out_dir):
    """The one-time test evaluation: model-set check first, then the unseal (hash checked, read logged)."""
    fp = evalguard.models_fingerprint([(r[0], r[5]) for r in runs],
                                      {"phase3": P3["_meta"]["phase3_config_sha256"],
                                       "dataset": CFG["_meta"]["dataset_config_sha256"]})
    rec = out_dir / "test_reads.json"
    changed = evalguard.authorize(rec, fp, allow_new)
    S = P3["samples"]
    seqf = pd.read_parquet(paths.sequences("frames.parquet"))
    win = pd.read_parquet(paths.sequences("windows.parquet"))
    win = win[(win.context == int(S["context"])) & win.horizon.isin(S["horizons"])]
    tw = sequences.windows(win, seqf, "test", paths.sequences("test_seal.json"), unseal=True, reason=reason)
    fr = pd.read_parquet(paths.stores(f"{config.DATASET}.frames.parquet"))
    man = pd.read_parquet(paths.archive("manifest.parquet"), columns=["file", "HGLT_OBS"]).set_index("file")
    idx = samples.build(tw, seqf, fr, man.HGLT_OBS, ["test"] * len(tw))
    evalguard.record(rec, fp, [(r[0], r[5]) for r in runs], reason, changed)
    return bank.add_samples(idx), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--reason", default="", help="required for --split test; logged with the unseal")
    ap.add_argument("--allow-new-models", action="store_true", help="test only: evaluate a changed model set (recorded)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="evaluate a subset (smoke tests only)")
    a = ap.parse_args()
    if a.split == "test" and not a.reason.strip():
        sys.exit("--split test needs --reason (it is logged with the unseal)")
    t0 = time.time()
    E, C = P3["eval"], P3["controls"]
    dev = a.device
    out_dir = paths.evals() if a.split == "val" else paths.phase3("eval_test_smoke" if paths.smoke() else "eval_test")
    bank = data.Bank(paths.phase3("cache"), dev)
    bgz = data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    qmap = torch.from_numpy(np.interp(bank.mu.cpu().numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32)).to(dev)
    qmap = torch.where(torch.isfinite(bank.S), qmap, torch.full_like(qmap, np.nan))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).to(dev) for k in bgz.files if k.startswith("M_H")}
    runs = load_runs(bank, dev)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    th = thermal.controller(P3["thermal"], dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    fp = None
    if a.split == "test":
        ids_all, fp = test_samples(bank, a.reason, a.allow_new_models, runs, out_dir)
    else:
        ids_all = bank.ids("val")
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    idx = bank.index
    rows, methods = [], None
    G = bank.G
    acc = {}  # corotation: (run, horizon) -> [sum, sum of squares, count] of the correction field
    for n0 in range(0, len(ids_all), 8):
        th.check()
        ids = ids_all[n0:n0 + 8]
        progress.report(f"evaluate: {a.split}", item=idx.target_frame.iloc[ids[0]], i=n0 // 8, n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        B = len(ids)
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], B, bank.K)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                 "B1-avg-bgS": b["x_bg"].mean(1), "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}
        mp, corr = predict(runs, b, bank, P3["train"]["bf16"])
        preds.update(mp)
        methods = methods or list(preds)
        y = b["y"][:, 0]
        valid = torch.isfinite(y) & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0) & inner
        lvl = gauss_level(preds["B1"], E["plage_sigma_px"])
        regions = {"disk": valid, "trusted": valid & bank.trusted, "plage": valid & (preds["B1"] / lvl > E["plage_contrast"])}
        for rname, (lo, hi) in RINGS.items():
            regions[rname] = valid & (rho >= lo) & (rho < hi)
        recs = [{"sample": int(i), "horizon": int(bank.horizon[i]), "run": int(idx.run.iloc[i]), "t_last": idx.t_last.iloc[i],
                 "minutes": float(idx.dt_target_s.iloc[i]) / 60} for i in ids]
        for reg, msk in regions.items():
            for name, p in preds.items():
                mae, rmse, gc, n = masked_metrics(p, y, msk)
                for j in range(B):
                    if float(n[j]) >= 200:
                        recs[j][f"{reg}|{name}"] = float(mae[j])
                        recs[j][f"{reg}|{name}|rmse"] = float(rmse[j])
                        recs[j][f"{reg}|{name}|gc"] = float(gc[j])
        rows += recs
        if a.split == "val":
            for name, c in corr.items():
                for j in range(B):
                    h = int(b["h"][j])
                    s = acc.setdefault((name, h), [torch.zeros(G, G, device=dev), torch.zeros(G, G, device=dev),
                                                   torch.zeros(G, G, device=dev)])
                    v = valid[j]
                    s[0] += torch.where(v, c[j], 0)
                    s[1] += torch.where(v, c[j] ** 2, 0)
                    s[2] += v.float()
    res = pd.DataFrame(rows)
    atomic.to_parquet(res, out_dir / "errors.parquet")

    base = [m for m in E["baselines"] if m in methods]
    summ = []
    for reg in ["disk", "trusted", "plage", *RINGS]:
        for h, d in res.groupby("horizon"):
            cols = [f"{reg}|{m}" for m in methods]
            if not all(c in d for c in cols):
                continue
            d = d.dropna(subset=cols)
            if not len(d):
                continue
            strongest = min(base, key=lambda m: d[f"{reg}|{m}"].median())
            for m in methods:
                dd = d.assign(s_avg=1 - d[f"{reg}|{m}"] / d[f"{reg}|B1-avg"],
                              s_best=1 - d[f"{reg}|{m}"] / d[f"{reg}|{strongest}"],
                              r_avg=1 - d[f"{reg}|{m}|rmse"] / d[f"{reg}|B1-avg|rmse"],
                              r_best=1 - d[f"{reg}|{m}|rmse"] / d[f"{reg}|{strongest}|rmse"])
                s1, l1, h1 = block_ci(dd, "s_avg", E["bootstrap"])
                s2, l2, h2 = block_ci(dd, "s_best", E["bootstrap"])
                summ.append({"region": reg, "horizon": h, "minutes": float(dd.minutes.median()), "method": m, "n": len(dd),
                             "rel_mae": float(dd[f"{reg}|{m}"].median()), "skill_vs_B1avg": s1, "lo": l1, "hi": h1,
                             "strongest_baseline": strongest, "skill_vs_strongest": s2, "lo_strongest": l2,
                             "hi_strongest": h2, "rmse_skill_vs_B1avg": float(dd.r_avg.median()),
                             "rmse_skill_vs_strongest": float(dd.r_best.median()),
                             "gradient_corr": float(dd[f"{reg}|{m}|gc"].median()), "is_baseline": m in base})
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "summary.csv", index=False)

    csum, crot = [], []
    if a.split == "val":
        # corotation: how much of each model's correction is one fixed map on the grid
        for (name, h), (s, s2, n) in acc.items():
            ok = n >= 0.5 * n.max()
            mean = s / n.clamp(min=1)
            fixed = float((mean[ok] ** 2).sum() / (s2[ok] / n[ok]).sum().clamp(min=1e-12))
            mh = clim.get(h)
            cm = np.nan
            if mh is not None:
                ok2 = ok & torch.isfinite(mh)
                a1, a2 = mean[ok2].cpu().numpy(), mh[ok2].cpu().numpy()
                cm = float(np.corrcoef(a1, a2)[0, 1]) if len(a1) > 100 and a1.std() > 0 and a2.std() > 0 else np.nan
            crot.append({"model_run": name, "horizon": h, "fixed_share": fixed, "corr_with_M": cm})
        # shuffle and frozen
        ctrl = []
        rng = np.random.default_rng(0)
        val = ids_all
        cval = val[rng.choice(len(val), min(C["samples"], len(val)), replace=False)] if len(val) else val
        for control in [c for c in C["enabled"] if c in ("shuffle", "frozen")]:
            for n0 in range(0, len(cval), 8):
                th.check()
                ids = cval[n0:n0 + 8]
                progress.report(f"evaluate: control {control}", item=f"batch {n0 // 8}", i=n0 // 8, n=(len(cval) + 7) // 8)
                if control == "shuffle":
                    other = np.array([rng.choice(val[(bank.horizon[val] == bank.horizon[i]) & (val != i)]) for i in ids])
                    b = bank.batch(ids, "plain", ctx_override=bank.ctx[other])
                else:
                    b = bank.batch(ids, "plain")
                    noise_src = bank.batch(rng.choice(val, len(ids)), "plain")["x_plain"]
                    eta = (noise_src[:, -1] - noise_src[:, -2]) / np.sqrt(2)
                for run in runs:
                    name, inputs = run[0], run[2]
                    x = b["x_bg"] if inputs == "bg" else b["x_plain"]
                    y = b["y"][:, 0] if control == "shuffle" else x[:, -1] + eta
                    p = predict([run], {**b, "y": y[:, None]}, bank, P3["train"]["bf16"])[0][name]
                    v = torch.isfinite(y) & torch.isfinite(p) & torch.isfinite(x).all(1) & inner
                    for j, i in enumerate(ids):
                        if int(v[j].sum()) < 200:
                            continue
                        e_m, e_b = float((p[j] - y[j]).abs()[v[j]].mean()), float((x[j, -1] - y[j]).abs()[v[j]].mean())
                        ctrl.append({"control": control, "model_run": name, "horizon": int(bank.horizon[i]),
                                     "run": int(idx.run.iloc[i]), "t_last": idx.t_last.iloc[i], "skill_vs_own_B1": 1 - e_m / e_b})
        cdf = pd.DataFrame(ctrl)
        if len(cdf):
            for (control, model_run), d in cdf.groupby(["control", "model_run"]):
                s, lo, hi = block_ci(d, "skill_vs_own_B1", E["bootstrap"])
                csum.append({"control": control, "model_run": model_run, "n": len(d), "skill_vs_own_B1": s, "lo": lo, "hi": hi})
        pd.DataFrame(csum).to_csv(out_dir / "controls.csv", index=False)
        pd.DataFrame(crot).to_csv(out_dir / "corotation.csv", index=False)
    meta = {"split": a.split, "samples": int(len(ids_all)), "runs": [r[0] for r in runs],
            "smoke_runs": [r[0] for r in runs if r[4]], "baselines": base, "subset": bool(a.max_samples),
            "models_fingerprint": fp, "reason": a.reason or None, "seconds": round(time.time() - t0, 1),
            **CFG["_meta"], **P3["_meta"]}
    atomic.write_json(out_dir / "eval_meta.json", meta)
    pd.set_option("display.width", 220)
    show = summ[summ.region == "disk"].pivot(index="method", columns="horizon", values="skill_vs_B1avg") * 100
    print(f"{a.split}, disk: skill vs B1-avg (%)\n" + show.round(1).to_string())
    for tab in (csum, crot):
        if len(tab):
            print(pd.DataFrame(tab).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
