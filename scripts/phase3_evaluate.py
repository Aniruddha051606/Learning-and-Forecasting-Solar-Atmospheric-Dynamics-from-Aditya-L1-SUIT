"""Phase 3 evaluation on the validation split (the test split stays sealed), with negative controls.

    python scripts/phase3_evaluate.py [--device cuda] [--max-samples N]

Settings: configs/phase3.toml [eval], [controls]. Every validation sample, the same windows for every method:
  B1              last context frame derotated to the target time
  B1-avg          mean of the K derotated context frames
  B1-avg-LDadd    B1-avg + mean_k (q - rot_k q): the limb-darkening profile q(mu) kept fixed (additive)
  B1-avg-bgS      mean_k [rot_k(F_k - S) + S]: the static background S kept fixed (phase3_background.py)
  B1-avg-clim     B1-avg + M(H): the mean train + hold-out residual map of that horizon
  <run>           every finished training run (B1 of its inputs + predicted residual) and, per model and
                  inputs, the seed ensemble (<model>_<inputs>-ens)
Regions (all inside r < [eval] disk_rho_max = 0.9; the limb ring is unreliable for any derotation): disk (all valid pixels), trusted (Phase 2 trusted region), plage (B1 above plage_contrast
times its own Gaussian-smoothed level; defined from the forecast and relative to the local level, so the
offset-mode vignetting does not bias it). Skill per window = 1 - MAE / MAE_reference; the median with a 95 %
interval from a bootstrap over (run, hour) blocks. Every model is also scored against the STRONGEST
baseline (lowest median MAE among [eval] baselines, per region and horizon).
Negative controls (a model that shows skill here has learned an artefact):
  shuffle  the context comes from another validation sample of the same horizon; skill over its own B1
           shows how much of the correction does not depend on the solar content
  frozen   the target is the model's own B1 plus real frame-to-frame noise (a Sun that does not evolve);
           the right answer is to change nothing, so skill over B1 must be <= 0
Writes outputs/datasets/<name>/phase3/eval/: val_errors.parquet, summary.csv, controls.csv, eval_meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, paths, progress  # noqa: E402
from suitdyn.ml import data, geometry, models, thermal  # noqa: E402

CFG = config.load_dataset()
P3 = config.load_phase3()


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
        runs.append((info["name"], info["model"], info["inputs"], m, info.get("smoke_test", False)))
    return runs


def predict(runs, b, bank, bf16):
    out, ens = {}, {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for name, kind, inputs, m, _ in runs:
            x = b["x_bg"] if inputs == "bg" else b["x_plain"]
            xi, mi = torch.nan_to_num(x), (torch.isfinite(b["y"]) & torch.isfinite(x).all(1, keepdim=True)).float()
            p = x[:, -1] + m(xi, mi, bank.mu[None, None], b["h"]).float()[:, 0]
            out[name] = p
            ens.setdefault(f"{kind}_{inputs}-ens", []).append(p)
    for k, v in ens.items():
        if len(v) > 1:
            out[k] = torch.stack(v).mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="evaluate a subset (smoke tests only)")
    a = ap.parse_args()
    t0 = time.time()
    E, C = P3["eval"], P3["controls"]
    dev = a.device
    out_dir = paths.evals()
    bank = data.Bank(paths.phase3("cache"), dev)
    bgz = np.load(paths.phase3("background", f"static_bg_{bank.G}.npz"))
    bank.set_background(bgz["S"])
    qmap = torch.from_numpy(np.interp(bank.mu.cpu().numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32)).to(dev)
    qmap = torch.where(torch.isfinite(bank.S), qmap, torch.full_like(qmap, np.nan))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).to(dev) for k in bgz.files if k.startswith("M_H")}
    runs = load_runs(bank, dev)
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    th = thermal.Thermal(P3["thermal"]["max_c"], P3["thermal"]["resume_c"], P3["thermal"]["target_c"],
                         sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    val = bank.ids("val")
    if a.max_samples:
        val = val[np.linspace(0, len(val) - 1, min(a.max_samples, len(val))).astype(int)]
    idx = bank.index
    rows = []
    for n0 in range(0, len(val), 8):
        th.check()
        ids = val[n0:n0 + 8]
        progress.report("evaluate: validation", item=idx.target_frame.iloc[ids[0]], i=n0 // 8, n=(len(val) + 7) // 8)
        b = bank.batch(ids, "plain")
        B = len(ids)
        A = b["x_plain"].mean(1)
        qrot = geometry.warp_static(qmap, b["grid"], b["ok"], B, bank.K)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-LDadd": A + (qmap - qrot).mean(1),
                 "B1-avg-bgS": b["x_bg"].mean(1),
                 "B1-avg-clim": A + torch.stack([clim[int(h)] for h in b["h"]])}
        preds.update(predict(runs, b, bank, P3["train"]["bf16"]))
        y = b["y"][:, 0]
        valid = torch.isfinite(y) & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0) & inner
        lvl = gauss_level(preds["B1"], E["plage_sigma_px"])
        plage = valid & (preds["B1"] / lvl > E["plage_contrast"])
        regions = {"disk": valid, "trusted": valid & bank.trusted, "plage": plage}
        for j, i in enumerate(ids):
            rec = {"sample": int(i), "horizon": int(bank.horizon[i]), "run": int(idx.run.iloc[i]),
                   "t_last": idx.t_last.iloc[i], "minutes": float(idx.dt_target_s.iloc[i]) / 60}
            for reg, msk in regions.items():
                mk = msk[j]
                if int(mk.sum()) < 200:
                    continue
                for name, p in preds.items():
                    rec[f"{reg}|{name}"] = float((p[j] - y[j]).abs()[mk].mean())
            rows.append(rec)
    res = pd.DataFrame(rows)
    atomic.to_parquet(res, out_dir / "val_errors.parquet")

    methods = [c.split("|", 1)[1] for c in res.columns if c.startswith("disk|")]
    base = [m for m in E["baselines"] if m in methods]
    summ = []
    for reg in ("disk", "trusted", "plage"):
        for h, d in res.groupby("horizon"):
            cols = [f"{reg}|{m}" for m in methods]
            if not all(c in d for c in cols):
                continue
            d = d.dropna(subset=cols)
            strongest = min(base, key=lambda m: d[f"{reg}|{m}"].median())
            for m in methods:
                dd = d.assign(s_avg=1 - d[f"{reg}|{m}"] / d[f"{reg}|B1-avg"], s_best=1 - d[f"{reg}|{m}"] / d[f"{reg}|{strongest}"])
                s1, l1, h1 = block_ci(dd, "s_avg", E["bootstrap"])
                s2, l2, h2 = block_ci(dd, "s_best", E["bootstrap"])
                summ.append({"region": reg, "horizon": h, "minutes": float(dd.minutes.median()), "method": m, "n": len(dd),
                             "rel_mae": float(dd[f"{reg}|{m}"].median()), "skill_vs_B1avg": s1, "lo": l1, "hi": h1,
                             "strongest_baseline": strongest, "skill_vs_strongest": s2, "lo_strongest": l2,
                             "hi_strongest": h2, "is_baseline": m in base})
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "summary.csv", index=False)

    # negative controls
    ctrl = []
    rng = np.random.default_rng(0)
    cval = val[rng.choice(len(val), min(C["samples"], len(val)), replace=False)] if len(val) else val
    for control in C["enabled"]:
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
            for name, kind, inputs, m, _ in runs:
                x = b["x_bg"] if inputs == "bg" else b["x_plain"]
                y = b["y"][:, 0] if control == "shuffle" else x[:, -1] + eta
                p = predict([(name, kind, inputs, m, False)], {**b, "y": y[:, None]}, bank, P3["train"]["bf16"])[name]
                v = torch.isfinite(y) & torch.isfinite(p) & torch.isfinite(x).all(1)
                for j, i in enumerate(ids):
                    if int(v[j].sum()) < 200:
                        continue
                    e_m, e_b = float((p[j] - y[j]).abs()[v[j]].mean()), float((x[j, -1] - y[j]).abs()[v[j]].mean())
                    ctrl.append({"control": control, "model_run": name, "horizon": int(bank.horizon[i]),
                                 "run": int(idx.run.iloc[i]), "t_last": idx.t_last.iloc[i], "skill_vs_own_B1": 1 - e_m / e_b})
    cdf = pd.DataFrame(ctrl)
    csum = []
    if len(cdf):
        for (control, model_run), d in cdf.groupby(["control", "model_run"]):
            s, lo, hi = block_ci(d, "skill_vs_own_B1", E["bootstrap"])
            csum.append({"control": control, "model_run": model_run, "n": len(d), "skill_vs_own_B1": s, "lo": lo, "hi": hi})
    pd.DataFrame(csum).to_csv(out_dir / "controls.csv", index=False)
    meta = {"val_samples": int(len(val)), "runs": [r[0] for r in runs], "smoke_runs": [r[0] for r in runs if r[4]],
            "baselines": base, "subset": bool(a.max_samples), "seconds": round(time.time() - t0, 1),
            **CFG["_meta"], **P3["_meta"]}
    atomic.write_json(out_dir / "eval_meta.json", meta)
    pd.set_option("display.width", 220)
    show = summ[summ.region == "disk"].pivot(index="method", columns="horizon", values="skill_vs_B1avg") * 100
    print("disk: skill vs B1-avg (%)\n" + show.round(1).to_string())
    if len(csum):
        print(pd.DataFrame(csum).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
