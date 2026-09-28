"""Phase 3: the static background S of one data set (suitdyn/ml/background.py).

    python scripts/phase3_background.py [--device cuda]

1. M(H) from the TRAIN pairs; for each lambda in configs/phase3.toml [background] lambdas, S is solved and
   scored on the HOLD-OUT run (median MAE of B1-avg-bgS vs B1-avg per horizon); the lambda with the largest
   mean gain is chosen. Validation is never used.
2. S is solved again from train + hold-out pairs with that lambda and saved.
Writes outputs/datasets/<name>/phase3/background/: static_bg_<G>.npz (S, the M(H) maps of train + hold-out,
the limb-darkening profile), background_meta.json (lambda scan, fit, geometry, provenance), background.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, paths, progress  # noqa: E402
from suitdyn.ml import background as bgm  # noqa: E402
from suitdyn.ml import data, thermal  # noqa: E402

CFG = config.load_dataset()
P3 = config.load_phase3()


def ld_profile(bank, ids, n_bins=30):
    """Centre-to-limb profile q(mu): per frame the median of each mu annulus (r < 0.95), median over frames
    (no brightness window: PHASE3 §3.2)."""
    mu = bank.mu.cpu().numpy()
    edges = np.linspace(np.sqrt(1 - 0.95 ** 2), 1.0, n_bins + 1)
    b = np.digitize(mu, edges)
    prof = []
    for i in np.unique(bank.tgt[ids])[::10]:
        f = bank.frames[int(i)].float().cpu().numpy()
        prof.append([np.nanmedian(f[(b == k) & np.isfinite(f)]) if ((b == k) & np.isfinite(f)).sum() > 50 else np.nan
                     for k in range(1, n_bins + 1)])
    q = np.nanmedian(np.array(prof), 0)
    c = 0.5 * (edges[1:] + edges[:-1])
    return c[np.isfinite(q)], q[np.isfinite(q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    B = P3["background"]
    out = paths.phase3("background")
    bank = data.Bank(paths.phase3("cache"), a.device)
    G, r_ref = bank.G, bank.r_ref
    pos = bgm.disk_index(G, r_ref, B["rho_max"])
    train, ho = bank.ids("train"), bank.ids("holdout")
    TH = P3["thermal"]
    th = thermal.Thermal(TH["max_c"], TH["resume_c"], TH["target_c"]) if a.device.startswith("cuda") else None
    tick = th.check if th else None

    rep = lambda stage: (lambda i, n: progress.report(stage, item=f"sample {i}", i=i, n=n))  # noqa: E731
    M_tr, geo_tr = bgm.mean_residuals(bank, train, B["residual_clip"], report=rep("background: M(H) train"), tick=tick)
    ops_tr = {h: bgm.horizon_operator(G, r_ref, dts, b0, pos) for h, (dts, b0) in geo_tr.items()}
    scan, x0 = {}, None
    for k, lam in enumerate(B["lambdas"]):
        progress.report("background: lambda scan", item=f"lambda = {lam}", i=k, n=len(B["lambdas"]), every_s=0)
        S, x0, info = bgm.solve(M_tr, ops_tr, pos, lam, x0)
        sc = bgm.score(bank, ho, S, rho_max=B["score_rho_max"], tick=tick)
        gain = {h: 1 - v["B1-avg-bgS"] / v["B1-avg"] for h, v in sc.items()}
        scan[lam] = {"holdout_gain_vs_B1avg": gain, "mean_gain": float(np.mean(list(gain.values()))), **info}
        print(f"lambda {lam}: hold-out gain {({h: round(100 * g, 2) for h, g in sorted(gain.items())})} %", flush=True)
    lam = max(scan, key=lambda v: scan[v]["mean_gain"])
    edge = lam in (B["lambdas"][0], B["lambdas"][-1])

    both = np.concatenate([train, ho])

    # Pointing groups. The detector background (vignetting, seams) is fixed on the CCD and moves across the
    # registered grid when the pointing drifts, so over many days one S smears. Samples are grouped by their
    # target's pointing; each group gets its own S, pulled toward the common S (shrinkage) so small groups stay
    # stable. Grouping is kept only if it beats the single S on the hold-out run (groups fitted on train only).
    def group_maps(ids_all, groups, prior):
        maps = []
        for gi in range(groups.max() + 1):
            M_g, geo_g = bgm.mean_residuals(bank, ids_all[groups == gi], B["residual_clip"], tick=tick)
            ops_g = {h: bgm.horizon_operator(G, r_ref, dts, b0, pos) for h, (dts, b0) in geo_g.items()}
            maps.append(bgm.solve(M_g, ops_g, pos, lam, prior=prior, shrink=B["pointing_shrink"])[0])
        return np.stack(maps)

    pxy = bank.index[["px", "py"]].to_numpy(float) if "px" in bank.index else None
    grouping = {"used": False}
    if pxy is not None:
        grouping.update(px_range=float(np.ptp(pxy[both, 0])), py_range=float(np.ptp(pxy[both, 1])),
                        bin_px=B["pointing_bin_px"])
        g_tr, cen_tr = bgm.pointing_groups(pxy[train, 0], pxy[train, 1], B["pointing_bin_px"], B["pointing_min_pairs"])
        grouping["train_groups"] = int(len(cen_tr))
        if len(cen_tr) > 1:
            progress.report("background: pointing groups", item=f"{len(cen_tr)} groups", i=0, n=1, every_s=0)
            S_tr = bgm.solve(M_tr, ops_tr, pos, lam, x0)[0]
            maps_tr = group_maps(train, g_tr, S_tr)
            one = bgm.score(bank, ho, S_tr, rho_max=B["score_rho_max"], tick=tick)
            grp = bgm.score(bank, ho, (maps_tr, cen_tr), rho_max=B["score_rho_max"], tick=tick)
            g1 = float(np.mean([1 - v["B1-avg-bgS"] / v["B1-avg"] for v in one.values()]))
            g2 = float(np.mean([1 - v["B1-avg-bgS"] / v["B1-avg"] for v in grp.values()]))
            grouping.update(holdout_mean_gain_single=g1, holdout_mean_gain_groups=g2, used=bool(g2 > g1))
            print(f"pointing groups: {len(cen_tr)}; hold-out mean gain single S {100 * g1:.2f} %, grouped "
                  f"{100 * g2:.2f} % -> {'grouped' if g2 > g1 else 'single'}", flush=True)

    M, geo = bgm.mean_residuals(bank, both, B["residual_clip"], report=rep("background: M(H) train+holdout"), tick=tick)
    ops = {h: bgm.horizon_operator(G, r_ref, dts, b0, pos) for h, (dts, b0) in geo.items()}
    S, _, info = bgm.solve(M, ops, pos, lam, x0)
    extra = {}
    if grouping["used"]:
        g_all, cen_all = bgm.pointing_groups(pxy[both, 0], pxy[both, 1], B["pointing_bin_px"], B["pointing_min_pairs"])
        extra = {"S_groups": group_maps(both, g_all, S).astype(np.float32), "centres": cen_all}
        grouping.update(final_groups=int(len(cen_all)), centres=cen_all.round(1).tolist())
    qc, qv = ld_profile(bank, both)
    atomic.savez(out / f"static_bg_{G}.npz", S=S, q_mu=qc, q=qv, **extra,
                 **{f"M_H{h}": m.astype(np.float32) for h, m in M.items()})
    meta = {"grid": G, "r_ref": r_ref, "lambda": lam, "lambda_at_scan_edge": edge, "pointing_grouping": grouping,
            "scan": {str(k): v for k, v in scan.items()}, "final_fit": info,
            "pairs": {"train": int(len(train)), "holdout": int(len(ho))},
            "geometry": {str(h): {"median_context_dt_s": g[0].tolist(), "b0_deg": g[1]} for h, g in geo.items()},
            "seconds": round(time.time() - t0, 1), **CFG["_meta"], **P3["_meta"]}
    atomic.write_json(out / "background_meta.json", meta)

    hmax = max(M)
    Rbar, good = ops[hmax]
    v = np.full(G * G, np.nan)
    s_d = np.nan_to_num(S[pos >= 0])
    pred = s_d - Rbar @ s_d
    pred[~good] = np.nan
    v[(pos >= 0).ravel()] = pred
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for k, (title, img, lim) in enumerate([("static background S", S, None), (f"M(H={hmax}), train + hold-out", M[hmax], 0.02),
                                           (f"S - Rbar S (what S predicts for M, H={hmax})", v.reshape(G, G), 0.02)]):
        kw = {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim} if lim else {"cmap": "gray"}
        im = ax[k].imshow(img, origin="lower", **kw)
        ax[k].set_title(title, fontsize=9)
        ax[k].axis("off")
        plt.colorbar(im, ax=ax[k], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "background.png", dpi=80)
    plt.close(fig)
    print(json.dumps({"lambda": lam, "at_scan_edge": edge, "final_fit": info, "seconds": meta["seconds"]}, indent=1))
    if edge:
        print("WARNING: the chosen lambda is at the edge of the scanned range; widen [background] lambdas", flush=True)


if __name__ == "__main__":
    main()
