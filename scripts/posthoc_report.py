"""Post-hoc report: every test result against the pre-registered criteria (docs/PREREGISTRATION.md).

    python scripts/posthoc_report.py [--datasets final_offset,final_centred]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from suitdyn import atomic, paths  # noqa: E402

REGION = "disk"
ENSEMBLES = ("unet_bg-ens", "convlstm_bg-ens")
FROZEN_MAX = 0.01
COROT_MAX = 0.20
BOOT = 2000


def read(p):
    try:
        return pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    except Exception:
        return None


def day_ci(err, method, strongest, horizon, n=BOOT, seed=0):
    """Median skill with a bootstrap over whole observing days (coarser than the evaluation's run-hour
    blocks).
    """
    a, b = f"{REGION}|{method}", f"{REGION}|{strongest}"
    d = err[(err.horizon == horizon)].dropna(subset=[a, b])
    if not len(d):
        return np.nan, np.nan, np.nan
    s = (1 - d[a] / d[b]).values
    days = pd.to_datetime(d.t_last).dt.floor("D").values
    blocks = [s[days == u] for u in np.unique(days)]
    med = float(np.median(s))
    if len(blocks) < 2:
        return med, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = [np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])) for _ in range(n)]
    return med, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 5)


def skill_rows(summ, err=None):
    out = []
    if summ is None:
        return out
    d = summ[(summ.region == REGION) & summ.method.isin(ENSEMBLES)]
    for _, r in d.sort_values(["method", "horizon"]).iterrows():
        row = {"method": r.method, "horizon": int(r.horizon), "minutes": f(r.minutes), "strongest": r.strongest_baseline,
               "skill": f(r.skill_vs_strongest), "lo": f(r.lo_strongest), "hi": f(r.hi_strongest), "n": int(r.n),
               "gradient_corr": f(r.gradient_corr)}
        row["P1"] = bool(np.isfinite(r.lo_strongest) and r.lo_strongest > 0)
        if err is not None:
            s, lo, hi = day_ci(err, r.method, r.strongest_baseline, r.horizon)
            row.update(lo_day=f(lo), hi_day=f(hi), P2=bool(lo > 0) if np.isfinite(lo) else None)
        out.append(row)
    return out


def dataset_block(ds):
    p3 = paths.phase3(name=ds, make=False)
    ev, et, ph = p3 / "eval", p3 / "eval_test", p3 / "posthoc"
    val, test = read(ev / "summary.csv"), read(et / "summary.csv")
    ctrl, corot = read(ev / "controls.csv"), read(ev / "corotation.csv")
    block = {"status": {"val": val is not None, "test": test is not None,
                        "flow": (ph / "flow_summary.csv").exists(), "cross": bool(list(ph.glob("cross_from_*_summary.csv")))
                        if ph.exists() else False},
             "val": skill_rows(val), "test": skill_rows(test, read(et / "errors.parquet") if test is not None else None)}
    c = {"frozen": [], "shuffle": [], "corotation": [], "C1": None, "C2": None}
    if ctrl is not None and len(ctrl):
        for _, r in ctrl.iterrows():
            c[r.control].append({"run": r.model_run, "skill": f(r.skill_vs_own_B1), "lo": f(r.lo), "hi": f(r.hi)})
        fr = ctrl[ctrl.control == "frozen"]
        c["C1"] = bool(len(fr) and (fr.hi <= FROZEN_MAX).all())
    if corot is not None and len(corot):
        c["corotation"] = [{"run": r.model_run, "horizon": int(r.horizon), "fixed_share": f(r.fixed_share),
                            "corr_with_M": f(r.corr_with_M)} for _, r in corot.iterrows()]
        c["C2"] = bool((corot.fixed_share <= COROT_MAX).all())
        c["max_fixed_share"] = f(corot.fixed_share.max())
    block["controls"] = c
    fl = read(ph / "flow_summary.csv")
    block["flow"] = [] if fl is None else [
        {"method": r.method, "reference": r.reference, "horizon": int(r.horizon), "minutes": f(r.minutes),
         "skill": f(r.skill), "lo": f(r.lo), "hi": f(r.hi)}
        for _, r in fl[(fl.region == REGION) & (fl.method.isin(("OF-adv",) + ENSEMBLES))].iterrows()]
    block["cross"] = []
    for cs in sorted(ph.glob("cross_from_*_summary.csv")) if ph.exists() else []:
        x = read(cs)
        for _, r in x[(x.region == REGION) & x.method.str.contains("-ens")].iterrows():
            block["cross"].append({"method": r.method, "trained_on": r.trained_on, "horizon": int(r.horizon),
                                   "minutes": f(r.minutes), "skill": f(r.skill_vs_strongest), "lo": f(r.lo), "hi": f(r.hi)})
    return block


def headline(rep):
    lines = []
    for ds, b in rep["datasets"].items():
        if b["test"]:
            k = {m: sum(r["P1"] for r in b["test"] if r["method"] == m) for m in ENSEMBLES}
            lines.append(f"{ds}: test skill over the strongest baseline at " +
                         ", ".join(f"{m} {k[m]}/4 horizons" for m in ENSEMBLES))
        elif b["val"]:
            lines.append(f"{ds}: validation only so far (sealed test pending)")
        else:
            lines.append(f"{ds}: no evaluation yet")
    if rep["solar"]:
        n = sum(r["P3"] for r in rep["solar"])
        lines.append(f"skill in BOTH pointing modes (solar origin): {n}/{len(rep['solar'])} (model, horizon) pairs")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="final_offset,final_centred")
    a = ap.parse_args()
    dss = [d.strip() for d in a.datasets.split(",") if d.strip()]
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "region": REGION, "ensembles": list(ENSEMBLES),
           "criteria": {"P1": "test: CI lower bound of skill vs strongest baseline > 0 (run-hour blocks)",
                        "P2": "same with observing-day blocks", "P3": "P1 in both pointing modes",
                        "C1": f"frozen control: upper CI <= {FROZEN_MAX}", "C2": f"corotation: fixed_share <= {COROT_MAX}"},
           "datasets": {ds: dataset_block(ds) for ds in dss}, "solar": []}
    if len(dss) == 2 and all(rep["datasets"][d]["test"] for d in dss):
        t = {d: {(r["method"], r["horizon"]): r for r in rep["datasets"][d]["test"]} for d in dss}
        for key in sorted(set(t[dss[0]]) & set(t[dss[1]])):
            r0, r1 = t[dss[0]][key], t[dss[1]][key]
            rep["solar"].append({"method": key[0], "horizon": key[1], "minutes": r1["minutes"],
                                 dss[0]: r0["skill"], dss[1]: r1["skill"], "P3": r0["P1"] and r1["P1"]})
    rep["headline"] = headline(rep)
    out = paths.OUT / "tests"
    out.mkdir(parents=True, exist_ok=True)
    atomic.write_json(out / "report.json", rep)
    md = [f"# SUIT-DYN test report ({rep['generated']})", "", *[f"- {h}" for h in rep["headline"]], ""]
    for ds, b in rep["datasets"].items():
        md += [f"## {ds}", "", f"status: {b['status']}", ""]
        for split in ("test", "val"):
            if b[split]:
                md += [f"### {split}: skill vs the strongest baseline, {REGION}", "",
                       "| model | horizon | min | strongest | skill | 95% CI | P1 |" + (" day-block CI | P2 |" if split == "test" else ""),
                       "|---|---|---|---|---|---|---|" + ("---|---|" if split == "test" else "")]
                for r in b[split]:
                    md.append(f"| {r['method']} | {r['horizon']} | {r['minutes']:.0f} | {r['strongest']} | {r['skill']:+.4f} | "
                              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {'yes' if r['P1'] else 'no'} |" +
                              (f" [{r['lo_day']}, {r['hi_day']}] | {'yes' if r.get('P2') else 'no'} |" if split == "test" else ""))
                md.append("")
        c = b["controls"]
        md += [f"controls: frozen C1 = {c['C1']}, corotation C2 = {c['C2']} (max fixed share {c.get('max_fixed_share')})", ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(rep["headline"]))
    print(f"wrote {out / 'report.json'} and REPORT.md")


if __name__ == "__main__":
    main()
