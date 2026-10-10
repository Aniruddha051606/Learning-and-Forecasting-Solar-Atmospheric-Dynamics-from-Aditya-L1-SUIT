"""Numbers and tables of the manuscript, generated from the stored results (no number is typed by hand).

    python scripts/paper/make_tex.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "paper" / "results" / "tables"
MS = ROOT / "paper" / "manuscript"
HL = {20: "A", 40: "B", 80: "C", 160: "D"}
DSL = {"final_offset": "Off", "final_centred": "Cen"}
ML = {"U-Net": "Unet", "ConvLSTM": "Clstm", "unet_bg-ens": "Unet", "convlstm_bg-ens": "Clstm"}
M = {}


def put(name, value, fmt="{:.1f}"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        M[name] = r"\tbd{" + name + "}"
    elif isinstance(value, (int, np.integer)):
        M[name] = f"{int(value):,}".replace(",", r"\,")
    elif isinstance(value, (float, np.floating)):
        s = fmt.format(value) if fmt else str(value)
        if s.lstrip("-").replace("0", "").replace(".", "") == "":  # no "-0.0"
            s = s.lstrip("-")
        M[name] = s.replace("-", "$-$")
    else:
        M[name] = str(value)


def signed(v):
    """+1.2 / -0.4 with a plain hyphen: the callers turn '-' into $-$ once (doing it here as well gave
    '$$-$$').
    """
    return ("+" if v >= 0 else "-") + f"{abs(v):.1f}"


def rng(series):
    s = pd.Series(series).dropna()
    return (s.min(), s.max()) if len(s) else (np.nan, np.nan)


def opt_csv(name):
    """A results table that may not exist yet (or is empty): an empty frame then."""
    try:
        return pd.read_csv(RES / f"{name}.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def d3(ds, *p):
    return ROOT / "outputs" / "datasets" / ds / "phase3" / Path(*p)


def main():
    (MS / "tables").mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(RES / "T1_data_splits.csv")
    t2 = pd.read_csv(RES / "T2_sealed_test.csv")
    t3 = pd.read_csv(RES / "T3_sealed_test_per_seed.csv")
    t4 = pd.read_csv(RES / "T4_sealed_test_regions.csv")
    t5 = pd.read_csv(RES / "T5_controls.csv")
    t6 = pd.read_csv(RES / "T6_diagnostics_validation.csv")
    t9 = pd.read_csv(RES / "T9_training.csv")
    t9 = t9[t9.version == "v2"] if "version" in t9 else t9
    t2b = opt_csv("T2b_sealed_test_v1_first_read")
    t10 = opt_csv("T10_leak_v1_v2")
    t11 = opt_csv("T11_new_period_v2")
    t12 = opt_csv("T12_mask_difference")

    # data
    for ds, L in DSL.items():
        for split in ("train", "val", "test"):
            d = t1[(t1.dataset == ds) & (t1.split == split)]
            put(f"win{L}{split.capitalize()}", int(d.windows.sum()))
            put(f"start{L}{split.capitalize()}", str(d.start_utc.iloc[0])[:16])
            put(f"end{L}{split.capitalize()}", str(d.end_utc.iloc[0])[:16])
        seq = pd.read_parquet(ROOT / "outputs" / "datasets" / ds / "sequences" / "frames.parquet")
        put(f"frames{L}", len(seq))
        reg = pd.read_parquet(ROOT / "outputs" / "datasets" / ds / "phase1" / "registration.parquet")
        reg = reg[reg.file.isin(seq.frame_id)]
        put(f"regSd{L}", float(np.nanmedian(np.r_[reg.reg_x0_anchor_sd, reg.reg_y0_anchor_sd])), "{:.2f}")
        put(f"limbReg{L}", float(np.nanmedian(np.hypot(reg.limb_minus_reg_x, reg.limb_minus_reg_y))), "{:.2f}")
        put(f"radius{L}", float(reg.reg_R.median()), "{:.0f}")

    # sealed test
    for ds, L in DSL.items():
        d = t2[t2.dataset == ds]
        lo, hi = rng(d.skill_pct)
        put(f"test{L}Min", lo)
        put(f"test{L}Max", hi)
        for r in d.itertuples():
            put(f"test{L}{ML[r.model]}{HL[r.horizon_frames]}", r.skill_pct)
            put(f"test{L}{ML[r.model]}{HL[r.horizon_frames]}Lo", r.ci_lo_pct)
            put(f"test{L}{ML[r.model]}{HL[r.horizon_frames]}Hi", r.ci_hi_pct)
        put(f"min{L}A", d[d.horizon_frames == 20].minutes.median(), "{:.0f}")
        put(f"min{L}D", d[d.horizon_frames == 160].minutes.median(), "{:.0f}")
    put("nPone", int(t2.P1.sum()))
    put("nPtwo", int((t2.P2 == True).sum()))  # noqa: E712
    put("nTests", len(t2))
    seeds = t3.copy()
    put("seedMin", seeds.skill_pct.min())
    put("seedMax", seeds.skill_pct.max())
    put("seedLoMin", seeds.ci_lo_pct.min())
    for ds, L in DSL.items():
        p = t4[(t4.dataset == ds) & (t4.region == "plage")]
        put(f"plage{L}Min", p.skill_pct.min())
        put(f"plage{L}Max", p.skill_pct.max())
        o = t4[(t4.dataset == ds) & (t4.region == "ring_outer")]
        put(f"outer{L}Max", o.skill_pct.max())

    # controls
    f = t5[t5.control == "frozen"]
    put("frozenMin", f.value_pct.min())
    put("frozenMax", f.value_pct.max())
    s = t5[t5.control == "shuffle"]
    put("shuffleMin", s.value_pct.min())
    put("shuffleMax", s.value_pct.max())
    c = t5[t5.control == "corotation_fixed_share"].copy()
    c["h"] = c.horizon_frames.astype(int)
    put("fixedShortMin", c[c.h < 160].value_pct.min())
    put("fixedShortMax", c[c.h < 160].value_pct.max())
    put("fixedLongMin", c[c.h == 160].value_pct.min())
    put("fixedLongMax", c[c.h == 160].value_pct.max())

    # diagnostics (validation)
    def diag(test, method, region="disk"):
        return t6[(t6.test == test) & (t6.method == method) & (t6.region == region)]
    for ds, L in DSL.items():
        x = t6[t6.dataset == ds]
        full = x[(x.test == "fixed_removal") & x.method.isin(["unet_bg-ens", "convlstm_bg-ens"]) & (x.region == "disk")]
        mm = x[(x.test == "fixed_removal") & x.method.str.endswith("-minusM") & (x.region == "disk")]
        put(f"minusM{L}Min", mm.skill_vs_strongest_pct.min())
        put(f"minusM{L}Max", mm.skill_vs_strongest_pct.max())
        put(f"full{L}Min", full.skill_vs_strongest_pct.min())
        put(f"full{L}Max", full.skill_vs_strongest_pct.max())
        avg = x[(x.test == "fixed_removal") & x.method.str.endswith("-avgM") & (x.region == "disk")]
        put(f"avgM{L}Min", avg.skill_vs_strongest_pct.min())
        put(f"avgM{L}Max", avg.skill_vs_strongest_pct.max())
        tp = x[(x.test == "shuffle_denoise") & x.method.str.endswith("-transplant-avglp") & (x.region == "disk")]
        put(f"tpLp{L}Min", tp.skill_vs_strongest_pct.min())
        put(f"tpLp{L}Max", tp.skill_vs_strongest_pct.max())
        tw = x[(x.test == "shuffle_denoise") & x.method.str.endswith("-transplant") & (x.region == "disk")]
        put(f"tpWhole{L}Min", tw.skill_vs_strongest_pct.min())
        put(f"tpWhole{L}Max", tw.skill_vs_strongest_pct.max())
        own = x[(x.test == "shuffle_denoise") & x.method.str.endswith("-ens-avglp") & (x.region == "disk")]
        put(f"ownLp{L}Max", own.skill_vs_strongest_pct.max())
        dn = x[(x.test == "shuffle_denoise") & (x.method == "B1-avg-bgS-bilateral1") & (x.region == "disk")]
        put(f"denoise{L}Min", dn.skill_vs_strongest_pct.min())
        put(f"denoise{L}Max", dn.skill_vs_strongest_pct.max())
        cl = x[(x.test == "classical") & (x.method == "classical") & (x.region == "disk")]
        put(f"classical{L}Min", cl.skill_vs_strongest_pct.min())
        put(f"classical{L}Max", cl.skill_vs_strongest_pct.max())
        vc = x[(x.test == "models_vs_tuned_classical") & (x.region == "disk")]
        put(f"vsClassical{L}Min", vc.skill_vs_strongest_pct.min())
        put(f"vsClassical{L}Max", vc.skill_vs_strongest_pct.max())
        put(f"vsClassical{L}LoMin", vc.ci_lo_pct.min())
        q = x[(x.test == "classical") & (x.method == "E3-qratio") & (x.region == "disk")]
        put(f"qratio{L}Min", q.skill_vs_strongest_pct.min())
        pl = x[(x.test == "fixed_removal") & (x.method == "unet_bg-ens-minusM") & (x.region == "plage")]
        pl = pl[pl.horizon_frames <= 40]
        put(f"plageMinusM{L}Min", pl.skill_vs_strongest_pct.min())
        put(f"plageMinusM{L}Max", pl.skill_vs_strongest_pct.max())
    p = d3("final_offset", "posthoc", "classical_meta.json")
    if p.exists():
        meta = json.loads(p.read_text())
        put("sharpAlpha", float(meta["sharpen_alpha"]))
        w = np.array([v[:5] for v in meta["weights_e1"].values()])
        put("wMin", float(w.min()), "{:.2f}")
        put("wMax", float(w.max()), "{:.2f}")
        tab = np.array([v for v in meta["e3_table_e1"].values()])
        put("lutMin", float(tab.min()), "{:.2f}")
        put("lutMax", float(tab.max()), "{:.2f}")
    else:  # the v2 analyses have not run yet
        for k in ("sharpAlpha", "wMin", "wMax", "lutMin", "lutMax"):
            put(k, None)
    for ds, L, f in (("final_offset", "Off", "{:.0f}"), ("final_centred", "Cen", "{:.1f}")):
        p = d3(ds, "posthoc", "shuffle_denoise_meta.json")
        sd = json.loads(p.read_text())["transplant_separation_h"] if p.exists() else {}
        put(f"tpSep{L}Med", sd.get("median"), f)
        put(f"tpSep{L}Lo", sd.get("p10"), f)
        put(f"tpSep{L}Hi", sd.get("p90"), f)

    # training
    for kind, L in (("unet", "Unet"), ("convlstm", "Clstm")):
        d = t9[t9.run.str.startswith(kind)]
        put(f"best{L}Min", int(d.best_epoch.min()))
        put(f"best{L}Max", int(d.best_epoch.max()))
        put(f"epochs{L}Max", int(d.epochs.max()))
        put(f"hours{L}Min", float(d.train_hours.min()))
        put(f"hours{L}Max", float(d.train_hours.max()))
        put(f"hold{L}Min", float(d.best_holdout_skill_vs_B1_pct.min()))
        put(f"hold{L}Max", float(d.best_holdout_skill_vs_B1_pct.max()))
    put("gpuPeak", float(t9.gpu_peak_C.max()), "{:.0f}")

    # sealed blind forecasts
    p8 = RES / "T8_sealed_blind.csv"
    if p8.exists():
        b = pd.read_csv(p8)
        for model, L in (("U-Net", "Unet"), ("ConvLSTM", "Clstm")):
            for key, K in (("h80", "C"), ("h160", "D")):
                put(f"blind{L}{K}", float(b[(b.kind == "short") & (b.key == key) & (b.model == model)].median_skill_pct.iloc[0]))
        r = b[(b.kind == "roll") & (b.model == "ConvLSTM") & (b.lead_h > 15)]
        put("rollClstmMin", r.median_skill_pct.min())
        put("rollClstmMax", r.median_skill_pct.max())
        bb = pd.read_csv(RES / "T8b_blind_roll_vs_blurred.csv")
        bb = bb[bb.lead_h > 15]
        put("rollBlurMin", bb["ConvLSTM_vs_best_smoothed_%"].min())
        put("rollBlurMax", bb["ConvLSTM_vs_best_smoothed_%"].max())
        put("blurOnlyMin", bb["smoothed_persistence_skill_%"].min())
        put("blurOnlyMax", bb["smoothed_persistence_skill_%"].max())
        put("nBlind", int(b[(b.model == "U-Net")].n_targets.sum()))

    # Addendum F (state estimate or forecast)
    for ds, L in DSL.items():
        p = d3(ds, "posthoc", "state_estimate_paired.csv")
        if p.exists():
            q = pd.read_csv(p)
            q = q[(q.region == "disk") & (q.horizon == 160)]
            for src, S in (("20", "A"), ("40", "B"), ("80", "C")):
                z = q[q.rotated_from.astype(str) == src]
                put(f"own{L}From{S}Min", z.own_over_rotated.min() * 100)
                put(f"own{L}From{S}Max", z.own_over_rotated.max() * 100)
                put(f"own{L}From{S}LoMin", z.lo.min() * 100)
        else:
            for S in "ABC":
                put(f"own{L}From{S}Min", None)
                put(f"own{L}From{S}Max", None)
                put(f"own{L}From{S}LoMin", None)

    # Addendum G (independent later period): v2 on all 9 days (Addendum I); the 2-day first scoring separately
    q = d3("final_offset", "newperiod", "summary_v2_partialcache.csv")
    if q.exists():
        g = pd.read_csv(q)
        e = g[(g.region == "disk") & g.method.isin(["unet_bg-ens", "convlstm_bg-ens"])]
        put("newPartMin", e.skill_vs_strongest.min() * 100)
        put("newPartMax", e.skill_vs_strongest.max() * 100)
        put("newPartWindows", int(e.n.sum() // 2))
    p = d3("final_offset", "newperiod", "summary_v2.csv")
    if p.exists():
        g = pd.read_csv(p)
        e = g[(g.region == "disk") & g.method.isin(["unet_bg-ens", "convlstm_bg-ens"])]
        for r in e.itertuples():
            k = f"new{ML[r.method]}{HL[r.horizon]}"
            put(k, r.skill_vs_strongest * 100)
            put(k + "Lo", r.lo * 100)
            put(k + "Hi", r.hi * 100)
        put("newMin", e.skill_vs_strongest.min() * 100)
        put("newMax", e.skill_vs_strongest.max() * 100)
        put("nGone", int(e.G1.astype(bool).sum()))
        put("nGtwo", int(e.G2.fillna(False).astype(bool).sum()))
        put("newWindows", int(e.n.sum() // 2))
        put("newDays", int(e.days.max()))
    else:
        for k in ("newMin", "newMax", "nGone", "nGtwo", "newWindows", "newDays"):
            put(k, None)

    # the leak (Addenda H and I): v1 with its original mask, v1 with the input-only mask, v2
    SPL = {"val": "Val", "new period": "New", "test": "Test"}
    VER = {"v1 target-aware mask": "Old", "v1 input-only mask": "Ctx", "v2": "Two"}
    if len(t10):
        for r in t10[t10.split.isin(SPL)].itertuples():
            k = f"lk{SPL[r.split]}{VER[r.models]}{DSL[r.dataset]}{ML[r.model]}{HL[r.horizon_frames]}"
            put(k, r.skill_pct)
            put(k + "Lo", r.ci_lo_pct)
            put(k + "Hi", r.ci_hi_pct)
        for (split, ver, ds), d in t10[t10.split.isin(SPL)].groupby(["split", "models", "dataset"]):
            put(f"lk{SPL[split]}{VER[ver]}{DSL[ds]}Min", d.skill_pct.min())
            put(f"lk{SPL[split]}{VER[ver]}{DSL[ds]}Max", d.skill_pct.max())
            put(f"lk{SPL[split]}{VER[ver]}{DSL[ds]}LoMin", d.ci_lo_pct.min())
        # how much the leak was worth: v1 with its own mask minus v2, per model and horizon (validation)
        for ds, L in DSL.items():
            a = t10[(t10.dataset == ds) & (t10.split == "val") & (t10.models == "v1 target-aware mask")]
            b = t10[(t10.dataset == ds) & (t10.split == "val") & (t10.models == "v2")]
            dd = a.merge(b, on=["model", "horizon_frames"], suffixes=("_a", "_b"))
            if len(dd):
                diff = dd.skill_pct_a - dd.skill_pct_b
                put(f"leakWorth{L}Min", diff.min())
                put(f"leakWorth{L}Max", diff.max())
    if len(t2b):
        for ds, L in DSL.items():
            d = t2b[t2b.dataset == ds]
            put(f"testOld{L}Min", d.skill_pct.min())
            put(f"testOld{L}Max", d.skill_pct.max())
    if len(t12):
        for ds, L in DSL.items():
            d = t12[t12.dataset == ds]
            put(f"maskDisk{L}Min", d.disk_pixels_removed_pct.min(), "{:.2f}")
            put(f"maskDisk{L}Max", d.disk_pixels_removed_pct.max(), "{:.2f}")
            put(f"maskWin{L}Min", d.windows_affected_pct.min(), "{:.1f}")
    else:
        for ds, L in DSL.items():
            for k in ("maskDisk{}Min", "maskDisk{}Max", "maskWin{}Min"):
                put(k.format(L), None)
    reads = d3("final_offset", "eval_test", "test_reads.json")
    if reads.exists():
        put("nTestReadsOff", len(json.loads(reads.read_text())["reads"]))

    # settings read from the code and configs
    import sys
    import tomllib
    sys.path.insert(0, str(ROOT))
    from suitdyn.ml import models
    p3 = tomllib.loads((ROOT / "configs" / "phase3.toml").read_text(encoding="utf-8"))
    run = json.loads(next((ROOT / "outputs/datasets/final_offset/phase3/runs").glob("unet_*/run.json")).read_text())
    cfgm = run["model_cfg"]
    nu = sum(p.numel() for p in models.MODELS["unet"](k=5, base=cfgm["unet_base"]).parameters())
    nc = sum(p.numel() for p in models.MODELS["convlstm"](k=5, hidden=cfgm["convlstm_hidden"]).parameters())
    put("paramsUnet", int(nu))
    put("paramsClstm", int(nc))
    ev = p3["eval"]
    put("plageContrast", float(ev["plage_contrast"]), "{:.2f}")
    put("plageSigma", float(ev["plage_sigma_px"]), "{:g}")
    put("diskRho", float(ev["disk_rho_max"]), "{:g}")
    put("nBoot", int(ev["bootstrap"]))
    tr = p3["train"]
    for k, name in (("epochs", "nEpochs"), ("patience", "nPatience")):
        if k in tr:
            put(name, int(tr[k]))
    if "epoch_fraction" in tr:
        put("epochFraction", float(tr["epoch_fraction"]), "{:g}")
    th = p3.get("thermal", {})
    for k, name in (("target_c", "thermTarget"), ("max_c", "thermMax")):
        if k in th:
            put(name, float(th[k]), "{:.0f}")

    # figures for the manuscript
    import shutil
    (MS / "figures").mkdir(exist_ok=True)
    for f in (ROOT / "paper" / "results" / "figures").glob("*.pdf"):
        shutil.copy2(f, MS / "figures" / f.name)

    # numbers.tex
    lines = ["% generated by scripts/paper/make_tex.py; do not edit", r"\providecommand{\tbd}[1]{\textbf{[#1]}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (MS / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: sealed test
    rows = []
    for ds, title in (("final_offset", "Offset"), ("final_centred", "Centred")):
        for model in ("U-Net", "ConvLSTM"):
            d = t2[(t2.dataset == ds) & (t2.model == model)].sort_values("horizon_frames")
            cells = [f"{signed(r.skill_pct)} [{r.ci_lo_pct:.1f}, {r.ci_hi_pct:.1f}]".replace("-", "$-$") for r in d.itertuples()]
            rows.append(f"{title} & {model} & " + " & ".join(cells) + r" \\")
    mins = [f"{t2[(t2.dataset == 'final_offset') & (t2.horizon_frames == h)].minutes.median():.0f}" for h in (20, 40, 80, 160)]
    tex = [r"\begin{tabular}{llcccc}", r"\hline",
           r"Pointing & Model & \multicolumn{4}{c}{Skill (\%) at a lead time of} \\",
           " & & " + " & ".join(f"{m} min" for m in mins) + r" \\", r"\hline"] + rows + [r"\hline", r"\end{tabular}"]
    (MS / "tables" / "tab_sealed_test.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # Table: the leak (offset pointing; Addenda H and I)
    if len(t10):
        rows, prev = [], None
        for split, sname in (("val", "Validation"), ("new period", "28 Sep--6 Oct"), ("test", "Sealed test")):
            for ver, vname in (("v1 target-aware mask", "v1, target-aware mask"), ("v1 input-only mask", "v1, input-only mask"),
                               ("v2", "v2 (retrained)")):
                for model in ("U-Net", "ConvLSTM"):
                    d = t10[(t10.dataset == "final_offset") & (t10.split == split) & (t10.models == ver)
                            & (t10.model == model)].sort_values("horizon_frames")
                    if not len(d):
                        continue
                    cells = [f"{signed(r.skill_pct)} [{r.ci_lo_pct:.1f}, {r.ci_hi_pct:.1f}]".replace("-", "$-$") for r in d.itertuples()]
                    rows.append(f"{sname if split != prev else ''} & {vname} & {model} & " + " & ".join(cells) + r" \\")
                    prev = split
        tex = [r"\begin{tabular}{lllcccc}", r"\hline",
               r"Data & Models & Model & \multicolumn{4}{c}{Skill (\%) at a lead time of} \\",
               r" & & & 29 min & 58 min & 1.9 h & 3.8 h \\", r"\hline"] + rows + [r"\hline", r"\end{tabular}"]
        (MS / "tables" / "tab_leak.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    # Table: the independent later period (v2)
    if len(t11):
        rows = []
        for model in ("U-Net", "ConvLSTM"):
            for r in t11[t11.model == model].sort_values("horizon_frames").itertuples():
                num = (f"{signed(r.skill_pct)} & [{r.ci_lo_pct:.1f}, {r.ci_hi_pct:.1f}] & {'yes' if r.G1 else 'no'} & "
                       f"[{r.day_ci_lo_pct:.1f}, {r.day_ci_hi_pct:.1f}]").replace("-", "$-$")
                g2 = "yes" if r.G2 is True or str(r.G2) == "True" else "no"
                rows.append(f"{model} & {r.minutes:.0f} min & {num} & {g2} & {int(r.windows)}" + r" \\")
        tex = [r"\begin{tabular}{llrcccccr}", r"\hline",
               r"Model & Lead & Skill (\%) & 95\% interval & G1 & Day-block interval & G2 & Windows \\", r"\hline"] + rows + \
              [r"\hline", r"\end{tabular}"]
        (MS / "tables" / "tab_new_period.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # Table: data sets and splits
    rows = []
    for ds, title in (("final_offset", "Offset"), ("final_centred", "Centred")):
        for split, name in (("train", "Training"), ("val", "Validation"), ("test", "Sealed test")):
            d = t1[(t1.dataset == ds) & (t1.split == split)]
            a, b = pd.Timestamp(d.start_utc.iloc[0]), pd.Timestamp(d.end_utc.iloc[0])
            rows.append(f"{title} & {name} & {a:%Y %b.} {a.day} {a:%H:%M} & {b:%Y %b.} {b.day} {b:%H:%M} & "
                        + f"{int(d.windows.sum()):,}".replace(",", r"\,") + r" \\")
    tex = [r"\begin{tabular}{lllll}", r"\hline", r"Pointing & Split & Start (UT) & End (UT) & Windows \\", r"\hline"] + rows + \
          [r"\hline", r"\end{tabular}"]
    (MS / "tables" / "tab_data.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote {len(M)} numbers ({sum('tbd' in v for v in M.values())} pending) and the tables to {MS}")


if __name__ == "__main__":
    main()
