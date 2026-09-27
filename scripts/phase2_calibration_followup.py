"""Phase 2 calibration follow-up: three checks the first study could not settle (docs/PHASE2.md §2).

    python scripts/phase2_calibration_followup.py

Uses the residual cache written by phase2_calibration.py (outputs/phase1/cache), restricted to a
detector band seen on the disk in both pointing modes (rows 352-1248, cols 600-1700).

C1  Which spatial scales are a detector pattern? Pattern estimates from independent subsets (early
    days vs last day; offset vs centred pointing) are compared after high-passing at several scales.
    A detector pattern agrees between subsets at every scale where it is estimated well; solar
    leakage (active regions that did not move far enough) does not.
C2  Multiplicative or additive, with a lever arm: per pixel, frames where plage covers it (local
    level > 1.3x its median) vs quiet frames (below median). With k = level ratio plage/quiet:
    multiplicative -> |relative residual| same in both, |absolute| scales by k; additive -> the reverse.
    The noise inflation of the (fewer) plage frames is measured on pixels with no pattern and divided
    out. The first study compared slopes of two noisy estimates, which cannot separate the cases.
C3  Rotation test on the longest OFFSET-mode run (the first used a run containing the pointing slew):
    plain phase correlation of consecutive frames, uncorrected / additive / multiplicative /
    high-passed pattern; median x-motion should equal the predicted rotation, ~0.17 px per frame.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, flat, io, motion, solar  # noqa: E402

CFG = config.load()
OUT = config.out_dir(CFG)
CAL = OUT / "calibration"
CACHE = OUT / "cache"
SHAPE = (2048, 2048)
ROWS, COLS = slice(352, 1248), slice(600, 1700)
STATE_CHANGE = pd.Timestamp("2026-09-24 15:00")


def nan_highpass(p, sigma):
    """p minus its NaN-aware Gaussian smoothing."""
    w = np.isfinite(p).astype(np.float32)
    num = gaussian_filter(np.nan_to_num(p), sigma)
    den = gaussian_filter(w, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, p - num / den, np.nan)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 else np.nan


def band_pass(rel_mm, sm_mm, groups, off, rows=32):
    H, W = ROWS.stop - ROWS.start, COLS.stop - COLS.start
    P = {k: np.full((H, W), np.nan, np.float32) for k in groups}
    keys = ("rel_p", "rel_q", "abs_p", "abs_q", "s_p", "s_q")
    L = {k: np.full((H, W), np.nan, np.float32) for k in keys}
    npl = np.zeros((H, W), np.int32)
    for y0 in range(0, H, rows):
        sl = slice(ROWS.start + y0, ROWS.start + y0 + rows)
        r = np.asarray(rel_mm[:, sl, COLS], dtype=np.float32)
        s = np.asarray(sm_mm[:, sl, COLS], dtype=np.float32)
        for k, m in groups.items():
            P[k][y0:y0 + rows] = flat.nanmedian0(r[m])
        ro, so = r[off], s[off]
        med = flat.nanmedian0(so)
        pl, qu = so > 1.3 * med, so <= med
        npl[y0:y0 + rows] = pl.sum(0)
        ab = ro * so
        for tag, sel in (("p", pl), ("q", qu)):
            L[f"rel_{tag}"][y0:y0 + rows] = flat.nanmedian0(np.where(sel, ro, np.nan))
            L[f"abs_{tag}"][y0:y0 + rows] = flat.nanmedian0(np.where(sel, ab, np.nan))
            L[f"s_{tag}"][y0:y0 + rows] = flat.nanmedian0(np.where(sel, so, np.nan))
    return P, L, npl


def rotation_test(patterns, n_pairs=150):
    reg = pd.read_parquet(OUT / "registration.parquet")
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path", "HGLT_OBS"]).set_index("file")
    nb = reg[(reg.frame == "full_binned") & (reg.pointing_mode == "offset")].sort_values("t").reset_index(drop=True)
    run = nb.run.value_counts().idxmax()
    g = nb[nb.run == run].reset_index(drop=True)
    zero = np.zeros(motion.highpass(np.zeros(SHAPE)).shape)
    rows = []
    cache = {}
    for i in range(1, min(n_pairs + 1, len(g))):
        if (g.jump_px.iloc[i] or 0) > 1:
            continue
        rot = solar.disk_centre_motion_px(float(g.cc_R.iloc[i]), float(man.loc[g.file[i], "HGLT_OBS"]),
                                          (g.t[i] - g.t[i - 1]).total_seconds(), float(g.CROTA2.iloc[i]))[0]
        exp = rot + (g.reg_x0.iloc[i] - g.reg_x0.iloc[i - 1])
        for name, (pat, mode) in patterns.items():
            ims = []
            for j in (i - 1, i):
                if (name, j) not in cache:
                    im = io.read(man.loc[g.file[j], "path"])[0]
                    cache[(name, j)] = motion.prepared(im if pat is None else flat.correct(im, pat, mode), zero)
                ims.append(cache[(name, j)])
            dx, dy, _ = motion.shift(ims[0], ims[1])
            rows.append({"i": i, "method": name, "dx": dx, "dy": dy, "pred_rot": rot, "expected_dx": exp})
        cache = {k: v for k, v in cache.items() if k[1] >= i}
    d = pd.DataFrame(rows)
    out = {"run_start": str(g.t.iloc[0]), "pairs": int(d.i.nunique()), "predicted_rotation_median": float(d.pred_rot.median())}
    for name, dd in d.groupby("method"):
        out[name] = {"median_dx": float(dd.dx.median()), "median_dx_minus_pointing": float((dd.dx - (dd.expected_dx - dd.pred_rot)).median()),
                     "corr_with_expected": float(np.corrcoef(dd.dx, dd.expected_dx)[0, 1]),
                     "rms_dx_minus_expected": float(np.sqrt(np.mean((dd.dx - dd.expected_dx) ** 2)))}
    return out, d


def main():
    t0 = time.time()
    sel_files = json.loads((CACHE / "nb03_selection.json").read_text())
    reg = pd.read_parquet(OUT / "registration.parquet").set_index("file")
    sel = reg.loc[sel_files].reset_index()
    n = len(sel)
    rel = np.memmap(CACHE / "nb03_rel.f16", dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    sm = np.memmap(CACHE / "nb03_sm.f16", dtype=np.float16, mode="r", shape=(n,) + SHAPE)
    off = (sel.pointing_mode == "offset").values
    days = sel.t.dt.floor("D")
    last_day = days[off].max()
    k = np.cumsum(off) - 1
    groups = {"offset": off, "centred": ~off, "early_days": off & (days < last_day).values,
              "last_day": off & (days == last_day).values, "A": off & (k % 2 == 0), "B": off & (k % 2 == 1)}
    P, L, npl = band_pass(rel, sm, groups, off)
    np.savez_compressed(CAL / "followup_band.npz", **{f"P_{k}": v for k, v in P.items()}, **L, n_plage=npl,
                        rows=[ROWS.start, ROWS.stop], cols=[COLS.start, COLS.stop])
    print(f"band pass {time.time() - t0:.0f} s", flush=True)

    # C1: agreement by scale
    scales = {}
    for sig in (None, 32, 16, 8, 4, 2):
        f = (lambda p: p) if sig is None else (lambda p, s=sig: nan_highpass(p, s))
        scales["full" if sig is None else f"highpass_{sig}px"] = {
            "early_vs_last_day": corr(f(P["early_days"]), f(P["last_day"])),
            "offset_vs_centred": corr(f(P["offset"]), f(P["centred"])),
            "half_A_vs_B": corr(f(P["A"]), f(P["B"])),
            "rms_offset": float(np.nanstd(f(P["offset"])))}

    # C2: multiplicative vs additive with a plage lever arm
    sd = np.nanstd(P["offset"])
    ok = npl >= 10
    strong = ok & (np.abs(P["offset"]) > 2 * sd)
    weak = ok & (np.abs(P["offset"]) < 0.3 * sd)
    k_ratio = float(np.nanmedian((L["s_p"] / L["s_q"])[strong]))

    def amp(key, m):
        return float(np.nanmedian(np.abs(L[f"{key}_p"][m])) / np.nanmedian(np.abs(L[f"{key}_q"][m])))

    a_rel, a_abs = amp("rel", strong), amp("abs", strong)
    b_rel, b_abs = amp("rel", weak), amp("abs", weak)
    c2 = {"pixels_strong": int(strong.sum()), "pixels_control": int(weak.sum()), "level_ratio_plage_over_quiet": k_ratio,
          "amp_ratio_relative": a_rel, "amp_ratio_absolute": a_abs,
          "noise_ratio_relative_control": b_rel, "noise_ratio_absolute_control": b_abs,
          "relative_corrected": a_rel / b_rel, "absolute_corrected": a_abs / b_abs,
          "expected_multiplicative": {"relative_corrected": 1.0, "absolute_corrected": k_ratio},
          "expected_additive": {"relative_corrected": 1 / k_ratio, "absolute_corrected": 1.0}}
    # Decision on the raw amplitude ratios. The "noise control" above divides by near-zero medians of
    # noise-dominated pixels and is kept only as a diagnostic (docs/PHASE2.md §2).
    d_mult = abs(np.log(a_rel)) + abs(np.log(a_abs / k_ratio))
    d_add = abs(np.log(a_rel * k_ratio)) + abs(np.log(a_abs))
    c2["log_distance_to_multiplicative"] = float(d_mult)
    c2["log_distance_to_additive"] = float(d_add)
    c2["closer_to"] = "multiplicative" if d_mult < d_add else "additive"

    # C3: rotation test on an offset-only run with each correction
    pat_rel = np.load(CAL / "nb03_pattern_relative.npy")
    level = np.load(CAL / "nb03_level.npy")
    pat_add = pat_rel * level
    hp_px = float(CFG["calibration"]["pattern_highpass_px"])
    pat_rel_hp = nan_highpass(pat_rel, hp_px)
    mode = c2["closer_to"]
    adopted = nan_highpass(pat_add, hp_px) if mode == "additive" else pat_rel_hp
    adopted = adopted.astype(np.float32)
    c3, rot = rotation_test({"uncorrected": (None, None), "additive": (pat_add, "additive"),
                             "multiplicative": (pat_rel, "multiplicative"),
                             "adopted": (adopted, mode)})
    rot.to_csv(CAL / "followup_rotation_test.csv", index=False)
    apath = CAL / "nb03_pattern_adopted.npy"
    np.save(apath, adopted)
    import hashlib
    prov = {"file": apath.name, "sha256": hashlib.sha256(apath.read_bytes()).hexdigest(), "mode": mode,
            "highpass_px": hp_px, "units": "counts (Level-1 NB03 binned, 300 ms)" if mode == "additive" else "relative",
            "made_from": "nb03_pattern_relative.npy x nb03_level.npy (phase2_calibration.py), NaN-aware high-pass",
            "rms": float(np.nanstd(adopted)), "valid_pixels": int(np.isfinite(adopted).sum()),
            "why_mode": c2, "why_scale": scales, "rotation_test": c3, **CFG["_meta"]}
    (CAL / "nb03_pattern_adopted.json").write_text(json.dumps(prov, indent=1, default=float))

    summary = {"band": {"rows": [ROWS.start, ROWS.stop], "cols": [COLS.start, COLS.stop]},
               "c1_agreement_by_scale": scales, "c2_mult_vs_add": c2, "c3_rotation_test_offset_run": c3,
               "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    (CAL / "followup_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    labels = list(scales)
    for key, c in (("early_vs_last_day", "C0"), ("offset_vs_centred", "C1"), ("half_A_vs_B", "C2")):
        ax[0].plot(labels, [scales[s][key] for s in labels], "-o", color=c, label=key)
    ax[0].set_ylabel("correlation of independent pattern estimates")
    ax[0].set_title("C1: which scales are a detector pattern?")
    ax[0].legend()
    ax[0].tick_params(axis="x", labelrotation=30)
    ax[1].bar(["rel (corr.)", "abs (corr.)"], [c2["relative_corrected"], c2["absolute_corrected"]], color="0.5")
    ax[1].axhline(1, color="k", lw=.5)
    ax[1].axhline(k_ratio, color="r", ls="--", label=f"plage/quiet level ratio {k_ratio:.2f}")
    ax[1].axhline(1 / k_ratio, color="b", ls="--", label=f"1/ratio {1 / k_ratio:.2f}")
    ax[1].set_title(f"C2: plage vs quiet amplitude ratios -> {c2['closer_to']}\nmult: rel=1, abs=k; add: rel=1/k, abs=1")
    ax[1].legend(fontsize=8)
    for name, c in zip(("uncorrected", "additive", "multiplicative", "adopted"), ("0.5", "C0", "C1", "C2")):
        dd = rot[rot.method == name]
        ax[2].plot(dd.expected_dx, dd.dx, "o", ms=3, color=c, label=f"{name}: median {dd.dx.median():.2f}")
    ax[2].plot([-4, 4], [-4, 4], "k-", lw=.5)
    ax[2].set_xlabel("expected x-motion (rotation + pointing change), px")
    ax[2].set_ylabel("measured, plain phase correlation, px")
    ax[2].set_title(f"C3: offset run, predicted rotation {c3['predicted_rotation_median']:.2f} px/frame")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CAL / "followup.png", dpi=75)
    plt.close(fig)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
