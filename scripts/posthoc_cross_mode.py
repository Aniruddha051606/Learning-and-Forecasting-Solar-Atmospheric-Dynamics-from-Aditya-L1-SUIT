"""Post-hoc test: cross-mode transfer. Models trained on one pointing mode, scored on the other mode's data.

    python scripts/posthoc_cross_mode.py --train final_offset --on final_centred [--device cpu] [--max-samples N]

If a model learned the Sun, it should keep (some of) its skill on data taken with the other pointing; if it
learned the detector background of its own pointing, its skill should drop or turn negative there. In the --on
data set's validation windows, with its own static background for the background-aware inputs, this scores:
the --train data set's models and seed ensembles (tagged '@<train>'), the --on data set's own models (if
trained), and the baselines, all in one pass with one mask (scripts/phase3_evaluate.py regions and metrics).
Writes outputs/datasets/<on>/phase3/posthoc/cross_from_<train>_errors.parquet and _summary.csv.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="data set whose models are transferred")
    ap.add_argument("--on", required=True, help="data set whose validation windows they are scored on")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow-smoke-runs", action="store_true", help=argparse.SUPPRESS)  # code tests only
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = a.on
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from posthoc_flow_baseline import low_priority
    low_priority()
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(a.threads)
    import phase3_evaluate as pe
    from suitdyn import atomic, paths, progress
    from suitdyn.ml import data, models

    t0 = time.time()
    E, dev = pe.P3["eval"], a.device
    out_dir = paths.phase3("posthoc")
    bank = data.Bank(paths.phase3("cache"), dev)
    data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    own = pe.load_runs(bank, dev)
    foreign = []
    for d in sorted(paths.runs(name=a.train, make=False).glob("*")):
        if not (d / "run.json").exists():
            continue
        info = json.loads((d / "run.json").read_text())
        if info.get("smoke_test") and not a.allow_smoke_runs:
            continue
        kw = {"unet": {"k": bank.K, "base": info["model_cfg"]["unet_base"]},
              "convlstm": {"k": bank.K, "hidden": info["model_cfg"]["convlstm_hidden"]}}[info["model"]]
        m = models.MODELS[info["model"]](**kw).to(dev)
        m.load_state_dict(torch.load(d / "best.pt", map_location=dev))
        m.eval()
        foreign.append((f"{info['name']}@{a.train}", f"{info['model']}@{a.train}", info["inputs"], m, False,
                        hashlib.sha256((d / "best.pt").read_bytes()).hexdigest()))
    if not foreign:
        sys.exit(f"no finished runs in {a.train}")
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    ids_all = bank.ids("val")
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    idx, rows = bank.index, []
    for n0 in range(0, len(ids_all), 8):
        ids = ids_all[n0:n0 + 8]
        progress.report(f"posthoc: cross {a.train} -> {a.on}", item=f"batch {n0 // 8 + 1}", i=n0 // 8,
                        n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        A = b["x_plain"].mean(1)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": A, "B1-avg-bgS": b["x_bg"].mean(1)}
        preds.update(pe.predict(own + foreign, b, bank, pe.P3["train"]["bf16"])[0])
        y = b["y"][:, 0]
        valid = torch.isfinite(y) & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0) & inner
        lvl = pe.gauss_level(preds["B1"], E["plage_sigma_px"])
        regions = {"disk": valid, "plage": valid & (preds["B1"] / lvl > E["plage_contrast"])}
        for rname, (lo, hi) in pe.RINGS.items():
            regions[rname] = valid & (rho >= lo) & (rho < hi)
        recs = [{"sample": int(i), "horizon": int(bank.horizon[i]), "run": int(idx.run.iloc[i]), "t_last": idx.t_last.iloc[i],
                 "minutes": float(idx.dt_target_s.iloc[i]) / 60} for i in ids]
        for reg, msk in regions.items():
            for name, p in preds.items():
                mae, rmse, gc, n = pe.masked_metrics(p, y, msk)
                for j in range(len(ids)):
                    if float(n[j]) >= 200:
                        recs[j][f"{reg}|{name}"] = float(mae[j])
        rows += recs
    res = pd.DataFrame(rows)
    tag = f"cross_from_{a.train}"
    atomic.to_parquet(res, out_dir / f"{tag}_errors.parquet")
    methods = [c.split("|", 1)[1] for c in res.columns if c.startswith("disk|")]
    base = [m for m in E["baselines"] if m in methods]
    summ = []
    for reg in ["disk", "plage", *pe.RINGS]:
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
                             "trained_on": a.train if f"@{a.train}" in m else (a.on if m not in base else "-"),
                             "strongest_baseline": strongest, "n": len(dd), "skill_vs_strongest": s, "lo": lo, "hi": hi})
    s = pd.DataFrame(summ)
    s.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    atomic.write_json(out_dir / f"{tag}_meta.json", {"train": a.train, "on": a.on, "samples": int(len(ids_all)),
                                                    "foreign_runs": [r[0] for r in foreign], "own_runs": [r[0] for r in own],
                                                    "device": dev, "seconds": round(time.time() - t0, 1)})
    show = s[s.region == "disk"].pivot(index="method", columns="horizon", values="skill_vs_strongest") * 100
    print(f"models of {a.train} on {a.on} val, disk: skill vs the strongest baseline (%)\n" + show.round(2).to_string())


if __name__ == "__main__":
    main()
