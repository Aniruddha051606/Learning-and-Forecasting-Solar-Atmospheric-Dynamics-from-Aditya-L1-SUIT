"""Check every \\pending[id]{...} statement of the manuscript against the v2 results, and resolve the ones
that hold.

    python scripts/paper/verify_claims.py [--dry-run]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "paper" / "results" / "tables"
MS = ROOT / "paper" / "manuscript"
DS = ("final_offset", "final_centred")


def tab(name):
    try:
        return pd.read_csv(RES / f"{name}.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def d3(ds, *p):
    return ROOT / "outputs" / "datasets" / ds / "phase3" / Path(*p)


def diag(t6, test, method_end, region="disk", exact=False):
    if not len(t6):
        return t6
    m = t6.method == method_end if exact else t6.method.str.endswith(method_end)
    return t6[(t6.test == test) & m & (t6.region == region)]


def checks():
    t2, t3, t4, t5, t6, t11 = (tab(n) for n in ("T2_sealed_test", "T3_sealed_test_per_seed", "T4_sealed_test_regions",
                                                  "T5_controls", "T6_diagnostics_validation", "T11_new_period_v2"))
    rp = ROOT / "outputs" / "tests" / "report.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    fpair = {ds: pd.read_csv(d3(ds, "posthoc", "state_estimate_paired.csv")) for ds in DS
             if d3(ds, "posthoc", "state_estimate_paired.csv").exists()}
    C = {}

    def need(*frames):
        return all(len(f) for f in frames)

    # the later period (G): every interval above zero
    if need(t11):
        lo = t11.ci_lo_pct
        C["newall"] = (bool((lo > 0).all()), f"lowest 95% bound over 8 cases {lo.min():+.2f} %; G1 true in {int(t11.G1.sum())} of {len(t11)}")
        if need(t2):
            off = t2[t2.dataset == "final_offset"].set_index(["model", "horizon_frames"]).skill_pct
            new = t11.set_index(["model", "horizon_frames"]).skill_pct
            d = (new - off).dropna()
            C["newclose"] = (bool(len(d) and d.abs().max() <= 1.0),
                             f"later period minus offset test, per model and lead: {', '.join(f'{v:+.1f}' for v in d)} points (pass: all within 1.0)")
    # the strongest baseline on the full disk
    if need(t2):
        names = set(t2.strongest_baseline) | (set(t11.strongest_baseline) if need(t11) else set())
        C["strongestbg"] = (names == {"B1-avg-bgS"}, f"strongest baselines found: {sorted(names)}")
        p2fail = t2[t2.P2.astype(str) != "True"]
        exc_ok = len(p2fail) > 0 and set(zip(p2fail.dataset, p2fail.horizon_frames)) <= {("final_centred", 160)}
        p3 = [r.get("P3") for r in rep.get("solar", [])]
        C["ptwoexc"] = (bool(exc_ok and p3 and all(p3)),
                        f"P2 fails for {sorted(set(zip(p2fail.dataset, p2fail.model, p2fail.horizon_frames)))}; "
                        f"P3 true in {sum(bool(x) for x in p3)} of {len(p3)}")
    if need(t3):
        C["seedspos"] = (bool(t3.skill_pct.min() > 0), f"lowest seed skill {t3.skill_pct.min():+.2f} %")
    if need(t4):
        x = t4.pivot_table(index=["dataset", "model", "horizon_frames"], columns="region", values="skill_pct")
        frac = float((x["plage"] < x["disk"]).mean()) if {"plage", "disk"} <= set(x.columns) else np.nan
        C["plagesmall"] = (bool(frac >= 0.75), f"plage below the full disk in {100 * frac:.0f} % of the cases (pass: at least 75 %)")
    ctl = {ds: rep.get("datasets", {}).get(ds, {}).get("controls", {}) for ds in DS}
    if all(ctl.values()):
        C["cone"] = (all(c.get("C1") is True for c in ctl.values()), f"C1: {[c.get('C1') for c in ctl.values()]}")
    if need(t5):
        c = t5[t5.control == "corotation_fixed_share"].copy()
        c["h"] = c.horizon_frames.astype(int)
        short, long_ = c[c.h < 160].value_pct.max(), c[c.h == 160].value_pct.max()
        C["ctwo"] = (bool(short <= 20 < long_), f"largest fixed share up to 2 h {short:.1f} %, at 4 h {long_:.1f} % (limit 20 %)")
    # diagnostics (validation)
    if need(t6):
        full = t6[(t6.test == "fixed_removal") & t6.method.isin(["unet_bg-ens", "convlstm_bg-ens"]) & (t6.region == "disk")]
        mm = diag(t6, "fixed_removal", "-minusM")
        if need(full, mm):
            ratio = {ds: mm[mm.dataset == ds].skill_vs_strongest_pct.mean() / full[full.dataset == ds].skill_vs_strongest_pct.mean()
                     for ds in DS}
            C["minusmmost"] = (all(v >= 0.5 for v in ratio.values()),
                               "minus-map skill / full skill: " + ", ".join(f"{k.split('_')[1]} {v:.2f}" for k, v in ratio.items()) + " (pass: at least 0.5)")
            C["minusmlo"] = (bool((mm.ci_lo_pct > 0).all()), f"lowest bound with the map removed {mm.ci_lo_pct.min():+.2f} %")
        avg = diag(t6, "fixed_removal", "-avgM")
        if need(avg):
            # adding the networks' fixed map to the strongest baseline must not help (it already carries that
            # correction)
            C["fixedrepeat"] = (bool(avg.skill_vs_strongest_pct.max() <= 0.3),
                                f"strongest baseline plus the fixed map: {avg.skill_vs_strongest_pct.min():+.2f} to {avg.skill_vs_strongest_pct.max():+.2f} % (pass: no gain above 0.3)")
        tp, tw = diag(t6, "shuffle_denoise", "-transplant-avglp"), diag(t6, "shuffle_denoise", "-transplant")
        if need(tp):
            C["tplpworse"] = (bool(tp.skill_vs_strongest_pct.max() < 0), f"large-scale transplant: up to {tp.skill_vs_strongest_pct.max():+.2f} %")
        if need(tp, tw):
            C["tpwholeworse"] = (bool(tw.skill_vs_strongest_pct.mean() < tp.skill_vs_strongest_pct.mean() - 1),
                                 f"mean whole transplant {tw.skill_vs_strongest_pct.mean():+.2f} % vs large-scale {tp.skill_vs_strongest_pct.mean():+.2f} %")
        bl, md = diag(t6, "shuffle_denoise", "B1-avg-bgS-bilateral1", exact=True), diag(t6, "shuffle_denoise", "B1-avg-bgS-median3", exact=True)
        if need(bl, md):
            j = bl.merge(md, on=["dataset", "horizon_frames"], suffixes=("_b", "_m"))
            C["medianworse"] = (bool((j.skill_vs_strongest_pct_m < j.skill_vs_strongest_pct_b).all()),
                                f"median filter minus edge-preserving: up to {(j.skill_vs_strongest_pct_m - j.skill_vs_strongest_pct_b).max():+.2f} points")
        cl = diag(t6, "classical", "classical", exact=True)
        if need(cl):
            C["classicalnobetter"] = (bool(cl.skill_vs_strongest_pct.max() <= 0.3),
                                      f"tuned classical forecaster: up to {cl.skill_vs_strongest_pct.max():+.2f} % (pass: at most 0.3)")
        vc = t6[(t6.test == "models_vs_tuned_classical") & (t6.region == "disk")]
        if need(vc):
            C["vsclassicallo"] = (bool((vc.ci_lo_pct > 0).all()), f"networks over the tuned classical: lowest bound {vc.ci_lo_pct.min():+.2f} %")
        own = diag(t6, "shuffle_denoise", "-ens-avglp")
        if need(own, full):  # like for like: the large-scale part of a model's correction over its full correction
            o = own.assign(m=own.method.str.replace("-avglp", "", regex=False))
            f = full.assign(m=full.method)
            j = o.merge(f, on=["dataset", "m", "horizon_frames"], suffixes=("_lp", "_full"))
            r = (j.skill_vs_strongest_pct_lp / j.skill_vs_strongest_pct_full).replace([np.inf, -np.inf], np.nan).dropna()
            C["finescale"] = (bool(len(r) and r.median() < 0.5),
                              f"large-scale part / full skill per model and lead: median {r.median():.2f}, range "
                              f"{r.min():.2f} to {r.max():.2f} (pass: median below 0.5)")
        if "notexplained" not in C:
            parts = [C.get(k) for k in ("minusmlo", "tplpworse", "classicalnobetter", "vsclassicallo")]
            dn = diag(t6, "shuffle_denoise", "B1-avg-bgS-bilateral1", exact=True)
            dn_ok = bool(len(dn) and dn.skill_vs_strongest_pct.max() < 0.5)
            if all(parts) and len(dn):
                C["notexplained"] = (all(p[0] for p in parts) and dn_ok,
                                     "needs minusmlo, tplpworse, classicalnobetter, vsclassicallo and a best denoiser below +0.5 % "
                                     f"(denoiser up to {dn.skill_vs_strongest_pct.max():+.2f} %)")
    for ds in DS:
        p = d3(ds, "posthoc", "classical_meta.json")
        if p.exists() and "nosharp" not in C:
            a = [json.loads(d3(x, "posthoc", "classical_meta.json").read_text()).get("sharpen_alpha")
                 for x in DS if d3(x, "posthoc", "classical_meta.json").exists()]
            C["nosharp"] = (all(v == 0 for v in a), f"chosen sharpening strength: {a}")
    # state estimate or forecast (Addendum F)
    if len(fpair) == 2:
        rows = {ds: q[(q.region == "disk") & (q.horizon == 160)] for ds, q in fpair.items()}
        a20 = {ds: r[r.rotated_from.astype(str) == "20"] for ds, r in rows.items()}
        ok = all(len(r) and (r.own_over_rotated > 0).all() and (r.lo > 0).all() for r in a20.values())
        C["evolution"] = (bool(ok), "own 4-h forecast over the 30-min forecast carried forward: " + "; ".join(
            f"{ds.split('_')[1]} {100 * r.own_over_rotated.min():+.1f} to {100 * r.own_over_rotated.max():+.1f} % (lowest bound {100 * r.lo.min():+.1f} %)"
            for ds, r in a20.items()))
        off = rows["final_offset"]
        C["fnumbers"] = (bool(len(off) and (off.own_over_rotated > 0).all() and (off.lo > 0).all() and C["evolution"][0]),
                         f"offset, from 30 min / 1 h / 2 h: lowest {100 * off.own_over_rotated.min():+.1f} %, lowest bound {100 * off.lo.min():+.1f} %")
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = (MS / "main.tex").read_text(encoding="utf-8")
    C = checks()
    found = []
    i = src.index(r"\begin{document}")  # claims live in the body only (the preamble defines the macro)
    while (j := src.find(r"\pending[", i)) != -1:
        k = src.find("]{", j)
        cid = src[j + 9:k]
        if k < 0 or not cid.isalpha():  # not a tagged claim
            i = j + 9
            continue
        m, depth = k + 2, 1
        while depth:
            depth += {"{": 1, "}": -1}.get(src[m], 0)
            m += 1
        found.append((j, m, cid, src[k + 2:m - 1]))
        i = m
    md = [f"# Claims check ({time.strftime('%Y-%m-%d %H:%M')}, scripts/paper/verify_claims.py)", "",
          "PASS claims are turned into normal text in main.tex; FAIL and UNKNOWN stay blue and need rewriting by hand.", "",
          "| # | id | result | claim | numbers |", "|---|---|---|---|---|"]
    out, last, n = [], 0, {"PASS": 0, "FAIL": 0, "UNKNOWN": 0}
    for k, (j, m, cid, text) in enumerate(found):
        ok, detail = C.get(cid, (None, "no result yet (the analysis has not run)"))
        res = "PASS" if ok is True else "FAIL" if ok is False else "UNKNOWN"
        n[res] += 1
        md.append(f"| {k} | {cid} | **{res}** | {' '.join(text.split())[:120]} | {detail} |")
        out.append(src[last:j] + (text if res == "PASS" else src[j:m]))
        last = m
    out.append(src[last:])
    md += ["", f"{n['PASS']} PASS, {n['FAIL']} FAIL, {n['UNKNOWN']} UNKNOWN"]
    (MS / "CLAIMS_CHECK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if not a.dry_run and n["PASS"]:
        shutil.copy2(MS / "main.tex", MS / "main.before_check.tex")
        (MS / "main.tex").write_text("".join(out), encoding="utf-8")
    print(f"{n['PASS']} PASS, {n['FAIL']} FAIL, {n['UNKNOWN']} UNKNOWN -> {MS / 'CLAIMS_CHECK.md'}"
          + ("" if a.dry_run else " (PASS claims unwrapped in main.tex)"))


if __name__ == "__main__":
    main()
