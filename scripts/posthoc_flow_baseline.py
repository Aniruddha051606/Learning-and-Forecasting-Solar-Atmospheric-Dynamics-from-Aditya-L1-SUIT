"""Post-hoc test: an optical-flow advection baseline, scored with the models in the same windows and masks.

    python scripts/posthoc_flow_baseline.py --dataset final_offset [--device cpu] [--max-samples N] [--no-models]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOW = {"pyr_scale": 0.5, "levels": 3, "winsize": 31, "iterations": 3, "poly_n": 7, "poly_sigma": 1.5}
FLOW_SIGMA_PX, MAX_SHIFT_PX = 8.0, 12.0


def low_priority():
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL: training keeps priority
    except Exception:
        pass


def to_u8(img, ok):
    """Disk-median-normalised 8-bit image for the flow estimate (invalid pixels set to the median)."""
    import numpy as np
    med = float(np.median(img[ok])) if ok.any() else 1.0
    v = np.where(ok, img, med) / max(med, 1e-6)
    return np.clip(v / 3.0 * 255, 0, 255).astype(np.uint8)


def advect(base, prev, last, t_span_s, t_lead_s):
    """Warp base (G, G) by the smoothed, capped flow prev->last scaled from t_span_s to t_lead_s."""
    import cv2
    import numpy as np
    ok = np.isfinite(prev) & np.isfinite(last)
    flow = cv2.calcOpticalFlowFarneback(to_u8(prev, ok), to_u8(last, ok), None, flags=0, **FLOW)
    flow = cv2.GaussianBlur(flow, (0, 0), FLOW_SIGMA_PX)
    d = flow * (t_lead_s / max(t_span_s, 1.0))
    mag = np.hypot(d[..., 0], d[..., 1])
    d *= np.minimum(1.0, MAX_SHIFT_PX / np.maximum(mag, 1e-6))[..., None]
    G = base.shape[0]
    yy, xx = np.mgrid[0:G, 0:G].astype(np.float32)
    return cv2.remap(base.astype(np.float32), xx - d[..., 0], yy - d[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan")), float(np.median(mag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = a.dataset
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    low_priority()
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(a.threads)
    import phase3_evaluate as pe
    from suitdyn import atomic, paths, progress
    from suitdyn.ml import data

    t0 = time.time()
    E, dev = pe.P3["eval"], a.device
    out_dir = paths.phase3("posthoc")
    bank = data.Bank(paths.phase3("cache"), dev)
    data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    runs = [] if a.no_models else pe.load_runs(bank, dev)
    rho = torch.sqrt(torch.clamp(1 - bank.mu ** 2, min=0))
    inner = bank.mu > float(np.sqrt(1 - E["disk_rho_max"] ** 2))
    ids_all = bank.ids("val")
    if a.max_samples:
        ids_all = ids_all[np.linspace(0, len(ids_all) - 1, min(a.max_samples, len(ids_all))).astype(int)]
    idx, rows, shifts = bank.index, [], []
    for n0 in range(0, len(ids_all), 8):
        ids = ids_all[n0:n0 + 8]
        progress.report("posthoc: flow baseline", item=f"batch {n0 // 8 + 1}", i=n0 // 8, n=(len(ids_all) + 7) // 8)
        b = bank.batch(ids, "plain")
        bgS = b["x_bg"].mean(1)
        xb = b["x_bg"].cpu().numpy()
        base = bgS.cpu().numpy()
        of = np.empty_like(base)
        for j, i in enumerate(ids):
            dts = bank.dt[i]
            of[j], s = advect(base[j], xb[j, 0], xb[j, -1], float(dts[0] - dts[-1]), float(np.mean(dts)))
            shifts.append(s)
        preds = {"B1": b["x_plain"][:, -1], "B1-avg": b["x_plain"].mean(1), "B1-avg-bgS": bgS,
                 "OF-adv": torch.from_numpy(of).to(dev)}
        if runs:
            preds.update(pe.predict(runs, b, bank, pe.P3["train"]["bf16"])[0])
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
                mae, rmse, gc, n = pe.masked_metrics(p, y, msk)
                for j in range(len(ids)):
                    if float(n[j]) >= 200:
                        recs[j][f"{reg}|{name}"] = float(mae[j])
                        recs[j][f"{reg}|{name}|rmse"] = float(rmse[j])
                        recs[j][f"{reg}|{name}|gc"] = float(gc[j])
        rows += recs
    res = pd.DataFrame(rows)
    atomic.to_parquet(res, out_dir / "flow_errors.parquet")
    methods = [c.split("|")[1] for c in res.columns if c.startswith("disk|") and c.count("|") == 1]
    summ = []
    for reg in ["disk", "trusted", "plage", *pe.RINGS]:
        for h, d in res.groupby("horizon"):
            cols = [f"{reg}|{m}" for m in methods]
            if not all(c in d for c in cols):
                continue
            d = d.dropna(subset=cols)
            if not len(d):
                continue
            for m in methods:
                for ref in ("B1-avg-bgS", "OF-adv"):
                    dd = d.assign(s=1 - d[f"{reg}|{m}"] / d[f"{reg}|{ref}"])
                    s, lo, hi = pe.block_ci(dd, "s", E["bootstrap"])
                    summ.append({"region": reg, "horizon": int(h), "minutes": float(dd.minutes.median()), "method": m,
                                 "reference": ref, "n": len(dd), "skill": s, "lo": lo, "hi": hi})
    pd.DataFrame(summ).to_csv(out_dir / "flow_summary.csv", index=False)
    meta = {"dataset": a.dataset, "samples": int(len(ids_all)), "runs": [r[0] for r in runs], "device": dev,
            "flow": FLOW, "flow_sigma_px": FLOW_SIGMA_PX, "max_shift_px": MAX_SHIFT_PX,
            "median_shift_px": float(np.median(shifts)) if shifts else None, "seconds": round(time.time() - t0, 1)}
    atomic.write_json(out_dir / "flow_meta.json", meta)
    s = pd.DataFrame(summ)
    show = s[(s.region == "disk") & (s.reference == "B1-avg-bgS")].pivot(index="method", columns="horizon", values="skill") * 100
    print(f"{a.dataset} val, disk: skill vs B1-avg-bgS (%)\n" + show.round(2).to_string())
    print(f"median applied shift {meta['median_shift_px']:.2f} px; {meta['seconds']:.0f} s")


if __name__ == "__main__":
    main()
