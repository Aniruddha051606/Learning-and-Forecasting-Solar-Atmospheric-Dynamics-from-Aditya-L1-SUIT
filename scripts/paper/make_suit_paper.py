"""Paper results package: every table and figure of the SUIT-DYN paper from the stored outputs.

    python scripts/paper/make_suit_paper.py [--example-device cpu]
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "results"
DS = {"final_offset": "offset", "final_centred": "centred"}
ENS = {"unet_bg-ens": "U-Net", "convlstm_bg-ens": "ConvLSTM"}
COL = {"unet_bg-ens": "#3a6aa6", "convlstm_bg-ens": "#c25d27"}
MARK = {"final_offset": ("o", "-"), "final_centred": ("s", "--")}
REGIONS = ["disk", "trusted", "plage", "ring_inner", "ring_mid", "ring_outer"]


def d3(ds, *p):
    return ROOT / "outputs" / "datasets" / ds / "phase3" / Path(*p)


def pct(x):
    return np.nan if x is None or pd.isna(x) else round(100 * float(x), 2)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def tables():
    T = {}
    rep = json.loads((ROOT / "outputs/tests/report.json").read_text())
    # T1 data and splits
    rows = []
    for ds, mode in DS.items():
        import tomllib
        cfg = tomllib.loads((ROOT / "configs/datasets" / f"{ds}.toml").read_text(encoding="utf-8"))
        meta = json.loads(d3(ds, "cache", "prepare_meta.json").read_text())
        test = pd.read_csv(d3(ds, "eval_test", "summary.csv"))
        for split in ("train", "val", "test"):
            lo, hi = cfg["split"][split]
            for h in (20, 40, 80, 160):
                n = (meta["samples"][str(h)].get(split, 0) + (meta["samples"][str(h)].get("holdout", 0) if split == "train" else 0)
                     if split != "test" else int(test[(test.region == "disk") & (test.horizon == h)].n.iloc[0]))
                rows.append({"dataset": ds, "pointing": mode, "split": split, "start_utc": lo, "end_utc": hi, "horizon_frames": h, "windows": n})
    T["T1_data_splits"] = pd.DataFrame(rows)
    # T2 sealed test (pre-registered)
    rows = []
    for ds in DS:
        for r in rep["datasets"][ds]["test"]:
            rows.append({"dataset": ds, "model": ENS.get(r["method"], r["method"]), "horizon_frames": r["horizon"], "minutes": round(r["minutes"], 1),
                         "strongest_baseline": r["strongest"], "skill_pct": pct(r["skill"]), "ci_lo_pct": pct(r["lo"]), "ci_hi_pct": pct(r["hi"]),
                         "P1": r["P1"], "day_ci_lo_pct": pct(r.get("lo_day")), "day_ci_hi_pct": pct(r.get("hi_day")), "P2": r.get("P2"),
                         "windows": r["n"], "gradient_corr": r.get("gradient_corr")})
    T["T2_sealed_test"] = pd.DataFrame(rows)
    # T3 per seed and T4 regions (sealed test)
    s3, s4 = [], []
    for ds in DS:
        s = pd.read_csv(d3(ds, "eval_test", "summary.csv"))
        for r in s[(s.region == "disk") & ~s.is_baseline].itertuples():
            s3.append({"dataset": ds, "run": r.method, "horizon_frames": r.horizon, "skill_pct": pct(r.skill_vs_strongest),
                       "ci_lo_pct": pct(r.lo_strongest), "ci_hi_pct": pct(r.hi_strongest)})
        for r in s[s.method.isin(ENS) & s.region.isin(REGIONS)].itertuples():
            s4.append({"dataset": ds, "model": ENS[r.method], "region": r.region, "horizon_frames": r.horizon,
                       "skill_pct": pct(r.skill_vs_strongest), "ci_lo_pct": pct(r.lo_strongest), "ci_hi_pct": pct(r.hi_strongest)})
    T["T3_sealed_test_per_seed"], T["T4_sealed_test_regions"] = pd.DataFrame(s3), pd.DataFrame(s4)
    # T5 controls
    rows = []
    for ds in DS:
        c = rep["datasets"][ds]["controls"]
        for k in ("frozen", "shuffle"):
            for r in c[k]:
                rows.append({"dataset": ds, "control": k, "run": r["run"], "horizon_frames": "all", "value_pct": pct(r["skill"]),
                             "ci_lo_pct": pct(r["lo"]), "ci_hi_pct": pct(r["hi"])})
        for r in c["corotation"]:
            rows.append({"dataset": ds, "control": "corotation_fixed_share", "run": r["run"], "horizon_frames": r["horizon"],
                         "value_pct": pct(r["fixed_share"]), "ci_lo_pct": np.nan, "ci_hi_pct": np.nan})
    T["T5_controls"] = pd.DataFrame(rows)
    # T6 diagnostics on validation (Addenda C, D, E)
    rows = []
    keep = {"fixed_removal_summary.csv": ["-ens", "-ens-minusM", "-ens-avgM"],
            "shuffle_denoise_summary.csv": ["-ens-avglp", "-ens-transplant-avglp", "-ens-transplant", "B1-avg-bgS-bilateral1", "B1-avg-bgS-median3"],
            "classical_summary.csv": ["E1-wavg", "E3-lut", "E3-qratio", "classical"]}
    for ds in DS:
        for f, pats in keep.items():
            p = d3(ds, "posthoc", f)
            if not p.exists():
                continue
            s = pd.read_csv(p)
            sk, lo, hi = ("skill_vs_strongest", "lo", "hi") if "skill_vs_strongest" in s else ("skill", "lo", "hi")
            for r in s[s.region.isin(["disk", "plage"])].itertuples():
                if any((r.method.endswith(q) if q.startswith("-") else r.method == q) for q in pats):
                    rows.append({"dataset": ds, "test": f.replace("_summary.csv", ""), "region": r.region, "method": r.method,
                                 "horizon_frames": r.horizon, "skill_vs_strongest_pct": pct(getattr(r, sk)),
                                 "ci_lo_pct": pct(getattr(r, lo)), "ci_hi_pct": pct(getattr(r, hi))})
        p = d3(ds, "posthoc", "classical_vs_classical.csv")
        if p.exists():
            s = pd.read_csv(p)
            for r in s[s.region.isin(["disk", "plage"]) & s.method.isin(ENS)].itertuples():
                rows.append({"dataset": ds, "test": "models_vs_tuned_classical", "region": r.region, "method": r.method,
                             "horizon_frames": r.horizon, "skill_vs_strongest_pct": pct(r.skill), "ci_lo_pct": pct(r.lo), "ci_hi_pct": pct(r.hi)})
    T["T6_diagnostics_validation"] = pd.DataFrame(rows)
    # T7 flow and cross-pointing (validation)
    rows = []
    for ds in DS:
        for r in rep["datasets"][ds].get("flow") or []:
            rows.append({"dataset": ds, "test": "optical_flow", "method": r["method"], "reference": r.get("reference"),
                         "horizon_frames": r["horizon"], "skill_pct": pct(r["skill"]), "ci_lo_pct": pct(r["lo"]), "ci_hi_pct": pct(r["hi"])})
        for r in rep["datasets"][ds].get("cross") or []:
            rows.append({"dataset": ds, "test": "cross_pointing", "method": r["method"], "reference": f"trained_on {r.get('trained_on')}",
                         "horizon_frames": r["horizon"], "skill_pct": pct(r["skill"]), "ci_lo_pct": pct(r["lo"]), "ci_hi_pct": pct(r["hi"])})
    T["T7_flow_cross_validation"] = pd.DataFrame(rows)
    # T8 sealed blind forecasts
    sc = sorted(glob.glob(str(ROOT / "outputs/sealed_forecast/score_*")))
    if sc:
        s = pd.read_csv(Path(sc[-1]) / "summary.csv")
        s = s[s.method.isin(ENS)].assign(model=lambda d: d.method.map(ENS), median_skill_pct=lambda d: (100 * d.median_skill).round(2))
        T["T8_sealed_blind"] = s[["kind", "key", "lead_h", "model", "strongest", "n_targets", "n_in_archive", "median_skill_pct"]]
        bd = Path(sc[-1]) / "roll_blur_diagnostic.csv"
        if bd.exists():
            T["T8b_blind_roll_vs_blurred"] = pd.read_csv(bd)
    # T2b the first (v1) read of the sealed test, kept for comparison
    rows = []
    for ds in DS:
        p = d3(ds, "eval_test_v1_targetmask", "summary.csv")
        if p.exists():
            s = pd.read_csv(p)
            for r in s[(s.region == "disk") & s.method.isin(ENS)].itertuples():
                rows.append({"dataset": ds, "model": ENS[r.method], "horizon_frames": r.horizon, "minutes": round(r.minutes, 1),
                             "skill_pct": pct(r.skill_vs_strongest), "ci_lo_pct": pct(r.lo_strongest), "ci_hi_pct": pct(r.hi_strongest)})
    T["T2b_sealed_test_v1_first_read"] = pd.DataFrame(rows)
    # T10 the leak: v1 with its original mask, v1 with the input-only mask, v2 (validation, new period, test)
    sets = [("val", "v1 target-aware mask", ("eval_v1_targetmask", "summary.csv"), "", "_strongest"),
            ("val", "v1 input-only mask", ("posthoc_v1_targetmask", "mask_check_summary.csv"), "-ctxmask", ""),
            ("val", "v2", ("eval", "summary.csv"), "", "_strongest"),
            ("new period", "v1 target-aware mask", ("newperiod", "summary_v1_targetmask.csv"), "", ""),
            ("new period", "v1 input-only mask", ("newperiod", "summary_v1_ctxmask.csv"), "", ""),
            ("new period", "v2", ("newperiod", "summary_v2.csv"), "", ""),
            ("new period, 28-29 Sep only", "v2", ("newperiod", "summary_v2_partialcache.csv"), "", ""),
            ("test", "v1 target-aware mask", ("eval_test_v1_targetmask", "summary.csv"), "", "_strongest"),
            ("test", "v2", ("eval_test", "summary.csv"), "", "_strongest")]
    rows = []
    for ds in DS:
        swapped = d3(ds, "runs_v1_targetmask").exists() and not d3(ds, "runs_v2").exists()
        for split, version, parts, sfx, ci in sets:
            p = d3(ds, *parts)
            if not p.exists() or (version == "v2" and parts[0] in ("eval", "eval_test") and not swapped):
                continue
            s = pd.read_csv(p)
            for m, name in ENS.items():
                for r in s[(s.region == "disk") & (s.method == m + sfx)].itertuples():
                    rows.append({"dataset": ds, "split": split, "models": version, "model": name, "horizon_frames": r.horizon,
                                 "minutes": round(r.minutes, 1), "skill_pct": pct(r.skill_vs_strongest),
                                 "ci_lo_pct": pct(getattr(r, "lo" + ci)), "ci_hi_pct": pct(getattr(r, "hi" + ci)),
                                 "windows": getattr(r, "n", np.nan), "days": getattr(r, "days", np.nan)})
    T["T10_leak_v1_v2"] = pd.DataFrame(rows)
    # T11 the independent later period (v2, all 9 days): criteria G1 (run-hour blocks) and G2 (day blocks)
    p = d3("final_offset", "newperiod", "summary_v2.csv")
    if p.exists():
        s = pd.read_csv(p)
        s = s[(s.region == "disk") & s.method.isin(ENS)]
        T["T11_new_period_v2"] = pd.DataFrame([{
            "model": ENS[r.method], "horizon_frames": r.horizon, "minutes": round(r.minutes, 1),
            "strongest_baseline": r.strongest_baseline, "skill_pct": pct(r.skill_vs_strongest), "ci_lo_pct": pct(r.lo),
            "ci_hi_pct": pct(r.hi), "G1": r.G1, "day_ci_lo_pct": pct(r.day_lo), "day_ci_hi_pct": pct(r.day_hi), "G2": r.G2,
            "windows": r.n, "days": r.days} for r in s.itertuples()])
    # T12 size of the leak: pixels the original mask channel switched off
    rows = []
    for ds in DS:
        p = d3(ds, "posthoc", "mask_difference.json")
        if p.exists():
            for split, v in json.loads(p.read_text())["splits"].items():
                rows.append({"dataset": ds, "split": split, **v})
    T["T12_mask_difference"] = pd.DataFrame(rows)
    # T9 training (v2; the v1 runs for comparison)
    rows = []
    for ds in DS:
        for version, sub in (("v2", "runs"), ("v1", "runs_v1_targetmask")):
            for d in sorted(d3(ds, sub).glob("*")):
                if not (d / "run.json").exists():
                    continue
                if (json.loads((d / "run.json").read_text()).get("input_mask") == "context") != (version == "v2"):
                    continue
                lg = pd.read_parquet(d / "log.parquet")
                h = lg.dropna(subset=["holdout_l1"])
                b = h.loc[h.holdout_l1.idxmin()]
                rows.append({"dataset": ds, "version": version, "run": d.name, "epochs": int(lg.epoch.max()) + 1,
                             "best_epoch": int(b.epoch),
                             "best_holdout_skill_vs_B1_pct": pct(b.holdout_skill_vs_B1),
                             "train_hours": round(float(lg.seconds.max()) / 3600, 1), "gpu_peak_C": float(lg.gpu_temp_peak.max())})
    T["T9_training"] = pd.DataFrame(rows)
    return T, rep


def figures(T, rep, example_device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False})
    # F1 skill vs horizon, sealed test (filled) and validation (open)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    t2 = T["T2_sealed_test"]
    for k, ds in enumerate(DS):
        val = pd.read_csv(d3(ds, "eval", "summary.csv"))
        for m, name in ENS.items():
            d = t2[(t2.dataset == ds) & (t2.model == name)].sort_values("minutes")
            ax[k].errorbar(d.minutes, d.skill_pct, yerr=[d.skill_pct - d.ci_lo_pct, d.ci_hi_pct - d.skill_pct], color=COL[m],
                           marker=MARK[ds][0], ls="-", capsize=2, label=f"{name}, sealed test")
            v = val[(val.region == "disk") & (val.method == m)].sort_values("minutes")
            ax[k].plot(v.minutes, 100 * v.skill_vs_strongest, color=COL[m], marker=MARK[ds][0], mfc="none", ls=":", label=f"{name}, validation")
            if ds == "final_offset" and "T11_new_period_v2" in T:
                g = T["T11_new_period_v2"]
                g = g[g.model == name].sort_values("minutes")
                ax[k].errorbar(g.minutes * 1.04, g.skill_pct, yerr=[g.skill_pct - g.ci_lo_pct, g.ci_hi_pct - g.skill_pct],
                               color=COL[m], marker="^", ms=4, ls="--", lw=0.9, capsize=2, label=f"{name}, 28 Sep - 6 Oct")
        ax[k].axhline(0, color="0.3", lw=0.8)
        ax[k].set_xscale("log")
        ax[k].set_xticks([30, 60, 120, 240])
        ax[k].set_xticklabels(["30 min", "1 h", "2 h", "4 h"])
        ax[k].minorticks_off()
        ax[k].set_title(f"{DS[ds]} pointing")
    ax[0].set_ylabel("MAE skill over the strongest\nphysical baseline (%)")
    ax[1].legend(fontsize=7, loc="upper left")
    save(fig, "F1_skill_vs_horizon")
    plt.close(fig)
    # F2 regions (sealed test)
    t4 = T["T4_sealed_test_regions"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    lab = {"disk": "disk", "trusted": "trusted", "plage": "plage", "ring_inner": "r<0.5", "ring_mid": "0.5-0.75", "ring_outer": "0.75-0.9"}
    for k, ds in enumerate(DS):
        for j, (m, name) in enumerate(ENS.items()):
            for hh, alpha in ((20, 0.45), (160, 1.0)):
                d = t4[(t4.dataset == ds) & (t4.model == name) & (t4.horizon_frames == hh)].set_index("region").reindex(REGIONS)
                x = np.arange(len(REGIONS)) + (j - 0.5) * 0.38 + (0.09 if hh == 160 else -0.09)
                ax[k].bar(x, d.skill_pct, width=0.17, color=COL[m], alpha=alpha, label=f"{name}, {'~30 min' if hh == 20 else '~4 h'}")
                ax[k].errorbar(x, d.skill_pct, yerr=[d.skill_pct - d.ci_lo_pct, d.ci_hi_pct - d.skill_pct], fmt="none", ecolor="0.25", lw=0.6)
        ax[k].set_xticks(range(len(REGIONS)))
        ax[k].set_xticklabels([lab[r] for r in REGIONS], rotation=30, fontsize=7.5)
        ax[k].axhline(0, color="0.3", lw=0.8)
        ax[k].set_title(f"{DS[ds]} pointing, sealed test")
    ax[0].set_ylabel("skill over the strongest baseline (%)")
    ax[1].legend(fontsize=6.5)
    save(fig, "F2_regions")
    plt.close(fig)
    # F3 controls: fixed share vs horizon (C2) and frozen / shuffle
    t5 = T["T5_controls"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ds in DS:
        c = t5[(t5.dataset == ds) & (t5.control == "corotation_fixed_share")].copy()
        c["h"] = c.horizon_frames.astype(int)
        for m in ENS:
            d = c[c.run.str.startswith(m.split("_")[0])].groupby("h").value_pct.agg(["min", "max", "mean"])
            ax[0].plot(d.index, d["mean"], color=COL[m], marker=MARK[ds][0], ls=MARK[ds][1], label=f"{ENS[m]}, {DS[ds]}")
            ax[0].fill_between(d.index, d["min"], d["max"], color=COL[m], alpha=0.12)
    ax[0].axhline(20, color="#a33", lw=1, ls="--")
    ax[0].text(22, 21, "C2 limit 20 %", color="#a33", fontsize=7)
    ax[0].set_xscale("log")
    ax[0].set_xticks([20, 40, 80, 160])
    ax[0].set_xticklabels(["30 min", "1 h", "2 h", "4 h"])
    ax[0].minorticks_off()
    ax[0].set_ylabel("share of the correction that\nis one fixed map (%)")
    ax[0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    sub = t5[t5.control.isin(["frozen", "shuffle"])]
    for k, (ctl, g) in enumerate(sub.groupby(["dataset", "control"])):
        x = k + np.linspace(-0.3, 0.3, len(g))
        ax[1].scatter(x, g.value_pct, s=12, color=[COL["unet_bg-ens"] if r.startswith("unet") else COL["convlstm_bg-ens"] for r in g.run])
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([f"{DS[d]}\n{c}" for d, c in sub.groupby(["dataset", "control"]).groups], fontsize=7)
    ax[1].axhline(0, color="0.3", lw=0.8)
    ax[1].set_ylabel("skill over its own B1 (%)")
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], color=COL[m], marker="o", ls="none", label=f"{ENS[m]} seeds") for m in ENS],
                 fontsize=6.5, loc="center right")
    save(fig, "F3_controls")
    plt.close(fig)
    # F4 fixed correction maps (validation-time M(H), training mean) and the static background
    p = d3("final_offset", "posthoc", "fixed_removal_maps_384.npz")
    if p.exists():
        z = np.load(p)
        bg = np.load(d3("final_offset", "background", "static_bg_384.npz"))
        fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
        for k, (m, title) in enumerate((("unet_bg", "U-Net mean correction, 4 h"), ("convlstm_bg", "ConvLSTM mean correction, 4 h"))):
            M = np.mean([z[f"{m}_s{s}_H160"] for s in range(3)], 0)
            v = np.nanpercentile(np.abs(M[M != 0]), 99)
            im = ax[k].imshow(np.where(M == 0, np.nan, M)[::-1], cmap="RdBu_r", vmin=-v, vmax=v)
            ax[k].set_title(title, fontsize=8)
            plt.colorbar(im, ax=ax[k], fraction=0.046)
        S = bg["S_groups"].mean(0) if "S_groups" in bg.files else bg["S"]
        im = ax[2].imshow(S[::-1], cmap="RdBu_r", vmin=-np.nanpercentile(np.abs(S), 99), vmax=np.nanpercentile(np.abs(S), 99))
        ax[2].set_title("static detector background S", fontsize=8)
        plt.colorbar(im, ax=ax[2], fraction=0.046)
        for a in ax:
            a.axis("off")
        save(fig, "F4_fixed_maps_offset")
        plt.close(fig)
    # F5 diagnostics (validation, disk): what explains the advantage
    t6 = T["T6_diagnostics_validation"]
    if len(t6):
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        series = [("unet_bg-ens", "fixed_removal", "U-Net full", "#3a6aa6", "-"),
                  ("unet_bg-ens-minusM", "fixed_removal", "U-Net minus fixed map", "#3a6aa6", "--"),
                  ("unet_bg-ens-transplant-avglp", "shuffle_denoise", "large-scale correction from another time", "0.45", ":"),
                  ("B1-avg-bgS-bilateral1", "shuffle_denoise", "best edge-preserving denoiser", "0.65", "-."),
                  ("classical", "classical", "tuned classical forecaster", "#2f7d4f", "-"),
                  ("unet_bg-ens", "models_vs_tuned_classical", "U-Net over the tuned classical", "#7b3fa0", "-")]
        for k, ds in enumerate(DS):
            for m, test, lab_, c, ls in series:
                d = t6[(t6.dataset == ds) & (t6.region == "disk") & (t6.method == m) & (t6.test == test)].sort_values("horizon_frames")
                if len(d):
                    ax[k].plot(d.horizon_frames, d.skill_vs_strongest_pct, color=c, ls=ls, marker=".", label=lab_)
            ax[k].axhline(0, color="0.3", lw=0.8)
            ax[k].set_xscale("log")
            ax[k].set_xticks([20, 40, 80, 160])
            ax[k].set_xticklabels(["30 min", "1 h", "2 h", "4 h"])
            ax[k].minorticks_off()
            ax[k].set_title(f"{DS[ds]} pointing, validation")
        ax[0].set_ylabel("skill (%) over the strongest baseline\n(purple: over the tuned classical)")
        h_, l_ = ax[0].get_legend_handles_labels()
        fig.legend(h_, l_, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
        save(fig, "F5_diagnostics")
        plt.close(fig)
    # F6 sealed blind forecasts: skill vs lead
    if "T8_sealed_blind" in T:
        t8 = T["T8_sealed_blind"]
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        for m, name in ENS.items():
            d = t8[(t8.model == name) & (t8.kind == "roll")].sort_values("lead_h")
            ax.plot(d.lead_h, d.median_skill_pct, color=COL[m], marker="o", ms=3, label=f"{name} chain vs persistence")
            dd = t8[(t8.model == name) & (t8.kind == "short")]
            ax.scatter(dd.lead_h, dd.median_skill_pct, color=COL[m], marker="*", s=40, zorder=3)
        if "T8b_blind_roll_vs_blurred" in T:
            b = T["T8b_blind_roll_vs_blurred"]
            ax.plot(b.lead_h, b["ConvLSTM_vs_best_smoothed_%"], color=COL["convlstm_bg-ens"], ls="--", lw=1, label="ConvLSTM chain vs best-blurred persistence")
            ax.plot(b.lead_h, b["smoothed_persistence_skill_%"], color="0.5", ls=":", lw=1, label="best-blurred persistence vs persistence")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xlabel("lead time after the last real frame (h)")
        ax.set_ylabel("skill (%)")
        ax.legend(fontsize=6.3)
        save(fig, "F6_sealed_blind")
        plt.close(fig)
    # F8 training curves
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for k, ds in enumerate(DS):
        for d in sorted(d3(ds, "runs").glob("*")):
            if (d / "log.parquet").exists() and json.loads((d / "run.json").read_text()).get("input_mask") == "context":
                lg = pd.read_parquet(d / "log.parquet").dropna(subset=["holdout_l1"])
                m = "unet_bg-ens" if d.name.startswith("unet") else "convlstm_bg-ens"
                ax[k].plot(lg.epoch, 100 * lg.holdout_skill_vs_B1, color=COL[m], lw=0.9, alpha=0.8)
        ax[k].set_title(f"{DS[ds]}: hold-out skill over B1 during training", fontsize=8)
        ax[k].set_xlabel("epoch (a quarter of the training set each)")
    ax[0].set_ylabel("skill over B1 (%)")
    save(fig, "F8_training")
    plt.close(fig)
    # F9 the leak and the retraining (offset pointing): v1 with its original mask, v1 with the input-only mask,
    # v2
    t10 = T["T10_leak_v1_v2"]
    if len(t10):
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
        style = {"v1 target-aware mask": ("--", "s", 0.55), "v1 input-only mask": (":", "x", 0.55), "v2": ("-", "o", 1.0)}
        for k, split in enumerate(("val", "new period")):
            for m, name in ENS.items():
                for version, (ls, mk, al) in style.items():
                    d = t10[(t10.dataset == "final_offset") & (t10.split == split) & (t10.models == version)
                            & (t10.model == name)].sort_values("minutes")
                    if len(d):
                        ax[k].plot(d.minutes, d.skill_pct, color=COL[m], ls=ls, marker=mk, ms=3.5, alpha=al, label=f"{name}, {version}")
            ax[k].axhline(0, color="0.3", lw=0.8)
            ax[k].set_xscale("log")
            ax[k].set_xticks([30, 60, 120, 240])
            ax[k].set_xticklabels(["30 min", "1 h", "2 h", "4 h"])
            ax[k].minorticks_off()
            ax[k].set_title("offset, validation" if split == "val" else "offset, 28 Sep - 6 Oct", fontsize=8)
        ax[0].set_ylabel("skill over the strongest\nphysical baseline (%)")
        h_, l_ = ax[0].get_legend_handles_labels()
        fig.legend(h_, l_, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
        save(fig, "F9_leak_v1_v2")
        plt.close(fig)
    if example_device:
        example(example_device, plt)


def example(device, plt):
    """F7: one validation window (offset, 4 h): last frame, target, strongest baseline, U-Net ensemble,
    errors.
    """
    os.environ["SUITDYN_DATASET"] = "final_offset"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase3_evaluate as pe
    from suitdyn import paths
    from suitdyn.ml import data
    bank = data.Bank(paths.phase3("cache"), device)
    data.load_background(bank, paths.phase3("background", f"static_bg_{bank.G}.npz"))
    err = pd.read_parquet(paths.evals("errors.parquet"))
    e = err[err.horizon == 160].dropna(subset=["disk|unet_bg-ens", "disk|B1-avg-bgS"])
    s = 1 - e["disk|unet_bg-ens"] / e["disk|B1-avg-bgS"]
    i = int(e["sample"].iloc[(s - s.median()).abs().argmin()])
    runs = [r for r in pe.load_runs(bank, device) if r[1] == "unet"]
    b = bank.batch(np.array([i]), "plain")
    out, _ = pe.predict(runs, b, bank, False)
    f = out["unet_bg-ens"][0].cpu().numpy()
    base = b["x_bg"].mean(1)[0].cpu().numpy()
    y = b["y"][0, 0].cpu().numpy()
    last = b["x_plain"][0, -1].cpu().numpy()
    v = np.nanpercentile(y, [1, 99.5])
    fig, ax = plt.subplots(1, 5, figsize=(7.4, 1.9))
    ev = np.nanpercentile(np.abs(base - y), 98)
    for a, img, t, kw in ((ax[0], last, "last context frame\n(derotated)", dict(cmap="inferno", vmin=v[0], vmax=v[1])),
                          (ax[1], y, "target (+3.9 h)", dict(cmap="inferno", vmin=v[0], vmax=v[1])),
                          (ax[2], f, "U-Net ensemble", dict(cmap="inferno", vmin=v[0], vmax=v[1])),
                          (ax[3], np.abs(base - y), "|error|, strongest\nbaseline", dict(cmap="magma", vmin=0, vmax=ev)),
                          (ax[4], np.abs(f - y), "|error|, U-Net", dict(cmap="magma", vmin=0, vmax=ev))):
        a.imshow(img[::-1], **kw)
        a.set_title(t, fontsize=7)
        a.axis("off")
    fig.suptitle(f"final_offset validation window {i} ({pd.Timestamp(bank.index.t_target.iloc[i]):%d %b %H:%M} UT), "
                 f"disk skill {100 * float(s.median()):.1f} % (the median window)", fontsize=7.5, y=1.04)
    save(fig, "F7_example_offset_4h")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example-device", default="cpu", help="device for the one example figure ('' to skip)")
    a = ap.parse_args()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    T, rep = tables()
    md = [f"# SUIT-DYN paper tables (generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/paper/make_suit_paper.py)", ""]
    for name, df in T.items():
        df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
        try:
            body = df.to_markdown(index=False)
        except ImportError:  # tabulate not installed
            body = "```\n" + df.to_string(index=False) + "\n```"
        md += [f"## {name}", "", body, ""]
    (OUT / "TABLES.md").write_text("\n".join(md), encoding="utf-8")
    figures(T, rep, a.example_device)
    print(f"wrote {len(T)} tables and {len(list((OUT / 'figures').glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
