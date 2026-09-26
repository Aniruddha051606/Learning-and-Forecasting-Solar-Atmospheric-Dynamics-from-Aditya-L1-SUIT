"""Registration study and adopted per-frame transforms, plus the Phase 1 frame-level QC decision.

    python scripts/registration_study.py

What Phase 1 found (docs/PHASE1.md §Registration), which this script implements and re-checks:
  * The pointing oscillates smoothly (period ~1.5-2 h, ±6 px x, ±10 px y in 2048 frames) and
    jumps by 5-9 px a few times a day. Header CRPIX follows the oscillation.
  * Phase correlation of consecutive frames cannot measure sub-pixel motion: it locks on a pattern
    fixed on the CCD and returns ~0. It is reliable only for shifts of more than ~1 px, so it is used
    to find the jumps and nothing else.
  * A limb fit with distortion harmonics is 3-4x noisier than a plain circle on real frames (the
    harmonics trade off against the centre over a 65-75 % limb). The circle is used, always on the
    same set of rays so that the distortion bias is the same in every frame.

Adopted NB03 centre = circle-fit centre, outliers removed, smoothed by a local quadratic within each
jump-free segment. Radius = robust mean per segment, scaled by the Sun-spacecraft distance.
Unbinned burst frames: per-frame circle fit on their own common ray set (checked against the NB03
pointing, see summary).

Validation: register frame pairs 15-90 min apart; the measured image motion minus predicted solar
rotation must be ~0. Done for the adopted method and for header-only and unsmoothed alternatives.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from suitdyn import config, geometry, io, motion, register, solar  # noqa: E402

CFG = config.load()
OUT = config.out_dir(CFG)
FIG = config.out_dir(CFG, "registration")
R = CFG["register"]
RUN_GAP_S = 300
JUMP_PX = 1.0
KEY = 10          # keyframe spacing (frames, ~15 min)
FP_FRAMES = 240   # frames used to estimate the fixed pattern


def robust_sigma(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def common_rays(df, min_frac=0.98):
    """Rays usable in at least min_frac of the frames; frames missing some of them use the rest."""
    U = np.array(df.edge_usable.tolist())
    return U.mean(0) >= min_frac


def circle_fits(df, rays):
    X, Y, U = (np.array(df[c].tolist()) for c in ("edge_x", "edge_y", "edge_usable"))
    out = []
    for i in range(len(df)):
        m = rays & U[i]
        f = geometry.fit_limb(X[i][m], Y[i][m], 0, CFG["limb"]["clip_sigma"])
        out.append((f["x0"], f["y0"], f["R"], f["rms"], int(m.sum())))
    return pd.DataFrame(out, columns=["cc_x0", "cc_y0", "cc_R", "cc_rms", "cc_rays"], index=df.index)


def local_quadratic(t, v, half_window_s, clip=4.0):
    """Robust local quadratic smoother: at each time, fit a quadratic to the points within
    ±half_window_s, drop points beyond clip robust sigma, refit, evaluate at that time."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    out = np.empty_like(v)
    for i, ti in enumerate(t):
        m = np.abs(t - ti) <= half_window_s
        tt, vv = t[m] - ti, v[m]
        deg = 2 if m.sum() >= 7 else (1 if m.sum() >= 3 else 0)
        for _ in range(2):
            p = np.polyfit(tt, vv, deg)
            r = vv - np.polyval(p, tt)
            s = robust_sigma(r) if len(r) > 4 else np.inf
            keep = np.abs(r) <= clip * max(s, 0.3)
            if keep.all() or keep.sum() <= deg + 1:
                break
            tt, vv = tt[keep], vv[keep]
        out[i] = np.polyval(p, 0.0)
    return out


def pointing_mode(x0, y0, size):
    """'centred' when the disk centre is within 200 (binned) px of the detector centre, else 'offset'.
    The spacecraft pointing changed on 2026-09-23 ~05:00 UT from centred to offset (docs/PHASE2.md)."""
    c = (size - 1) / 2
    return np.where(np.hypot(x0 - c, y0 - c) < 200 * size / 2048, "centred", "offset")


def _segment_positions(job):
    """Image-content positions of every frame of one jump-free segment, relative to its first frame,
    rotation removed. Keyframes every KEY frames are linked to each other; every frame is correlated
    with its nearest keyframe (never chained frame to frame)."""
    files, paths, ts, rot_rate, fp = job
    n = len(files)
    prep = {}

    def P(i):
        if i not in prep:
            prep[i] = motion.prepared(io.read(paths[i])[0], fp)
        return prep[i]

    keys = list(range(0, n, KEY))
    kpos = {0: np.zeros(2)}
    for a, b in zip(keys[:-1], keys[1:]):
        dx, dy, _ = motion.shift(P(a), P(b))
        kpos[b] = kpos[a] + np.array([dx, dy]) - rot_rate * (ts[b] - ts[a])
    pos = np.zeros((n, 2))
    err = np.zeros(n)
    for i in range(n):
        k = min(keys, key=lambda kk: abs(kk - i))
        if i == k:
            pos[i] = kpos[k]
            continue
        dx, dy, e = motion.shift(P(k), P(i))
        pos[i] = kpos[k] + np.array([dx, dy]) - rot_rate * (ts[i] - ts[k])
        err[i] = e
    return files, pos, err


def adopt_nb03(nb, fp):
    nb = nb.sort_values("t").copy()
    dt = nb.t.diff().dt.total_seconds()
    nb["run"] = (dt.isna() | (dt > RUN_GAP_S)).cumsum()
    nb["jump_px"] = np.hypot(nb.pc_dx, nb.pc_dy).where(nb.run.eq(nb.run.shift()))
    nb["segment"] = (nb.run.ne(nb.run.shift()) | (nb.jump_px > JUMP_PX)).cumsum()
    nb["pointing_mode"] = pointing_mode(nb.cc_x0, nb.cc_y0, 2048)
    nb["segment"] = (nb.segment.ne(nb.segment.shift()) | nb.pointing_mode.ne(nb.pointing_mode.shift())).cumsum()
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path", "HGLT_OBS"]).set_index("file")
    jobs = []
    for _, g in nb.groupby("segment"):
        ts = (g.t - g.t.iloc[0]).dt.total_seconds().values
        rx, ry = solar.disk_centre_motion_px(float(g.cc_R.median()), float(man.loc[g.file.iloc[0], "HGLT_OBS"]), 1.0,
                                             float(g.CROTA2.median()))
        jobs.append((g.file.tolist(), man.loc[g.file, "path"].tolist(), ts, np.array([rx, ry]), fp))
    with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
        res = list(ex.map(_segment_positions, jobs))
    pos = pd.DataFrame([(f, p[0], p[1], e) for files, P_, E in res for f, p, e in zip(files, P_, E)],
                       columns=["file", "img_x", "img_y", "img_err"]).set_index("file")
    nb = nb.join(pos, on="file")
    for ax in ("x", "y"):
        d = nb[f"cc_{ax}0"] - nb[f"img_{ax}"]
        off = d.groupby(nb.segment).transform(
            lambda s: s[np.abs(s - s.median()) <= 4 * max(robust_sigma(s), 0.3)].mean())
        nb[f"reg_{ax}0"] = nb[f"img_{ax}"] + off
        nb[f"reg_{ax}0_anchor_sd"] = d.groupby(nb.segment).transform(lambda s: robust_sigma(s) / np.sqrt(len(s)))
        nb[f"limb_minus_reg_{ax}"] = nb[f"cc_{ax}0"] - nb[f"reg_{ax}0"]
        # the earlier (rejected) method, kept for the comparison
        ts = (nb.t - nb.t.iloc[0]).dt.total_seconds().values
        sm = np.full(len(nb), np.nan)
        for _, idx in nb.groupby("segment").groups.items():
            ii = nb.index.get_indexer(idx)
            sm[ii] = local_quadratic(ts[ii], nb[f"cc_{ax}0"].values[ii], R["smooth_half_window_s"])
        nb[f"circle_smoothed_{ax}0"] = sm
    rn = nb.cc_R * nb.DSUN_OBS
    nb["reg_R"] = rn.groupby(nb.segment).transform(
        lambda s: s[np.abs(s - s.median()) < 4 * max(robust_sigma(s), 1e-6)].mean()) / nb.DSUN_OBS
    nb["reg_method"] = "image_motion(fixed_pattern_removed)+segment_circle_anchor"
    return nb


def adopt_full(full, nb):
    """Burst frames: own circle fit (their own common rays). The NB03 pointing interpolated to the
    frame time (x_4096 = 2·x_2048 + 0.5) is recorded next to it to measure the per-filter image
    offsets from the filter wedges (Tripathi et al. 2025, Table 4)."""
    full = full.sort_values("t").copy()
    t_nb = nb.t.astype("int64").values
    t_f = full.t.astype("int64").values
    for ax in ("x", "y"):
        full[f"nb_{ax}0"] = 2 * np.interp(t_f, t_nb, nb[f"reg_{ax}0"].values) + 0.5
        full[f"filter_offset_{ax}"] = full[f"cc_{ax}0"] - full[f"nb_{ax}0"]
        full[f"reg_{ax}0"] = full[f"cc_{ax}0"]
        full[f"limb_minus_reg_{ax}"] = 0.0
    full["nb_dt_s"] = np.abs(t_f[:, None] - t_nb[None, :]).min(1) / 1e9
    full["reg_R"] = full.cc_R
    full["reg_method"] = "circle_common_rays_per_frame"
    full["pointing_mode"] = pointing_mode(full.cc_x0, full.cc_y0, 4096)
    return full


def register_frame(row, path, x0, y0, Rr):
    im, _ = io.read(path)
    A, b = register.transform(x0, y0, Rr, row.CROTA2, R["grid"], R["r_ref"])
    return register.apply(im, A, b, R["grid"])


def validate(nb, fp, lags=(1, 10, 40), n_per_lag=20, seed=0):
    """Independent check: pairs of frames in DIFFERENT jump-free segments of the same run, whose absolute
    positions come from independent limb anchors. The image motion between them (fixed pattern removed,
    detector px) minus predicted rotation must equal the difference of their adopted centres. Lag 1 =
    the jump pair itself. The same pairs score the alternatives: per-frame circle, smoothed circle
    (the method first adopted and then rejected), and the header."""
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path", "HGLT_OBS"]).set_index("file")
    rng = np.random.default_rng(seed)
    rows = []
    for lag in lags:
        cands = [i for i in range(len(nb) - lag) if nb.run.iloc[i] == nb.run.iloc[i + lag]
                 and nb.segment.iloc[i] != nb.segment.iloc[i + lag]
                 and nb.pointing_mode.iloc[i] == nb.pointing_mode.iloc[i + lag]
                 and np.hypot(nb.reg_x0.iloc[i + lag] - nb.reg_x0.iloc[i],
                              nb.reg_y0.iloc[i + lag] - nb.reg_y0.iloc[i]) < 60]
        for i in rng.choice(cands, min(n_per_lag, len(cands)), replace=False):
            a, b = nb.iloc[i], nb.iloc[i + lag]
            dts = (b.t - a.t).total_seconds()
            rx, ry = solar.disk_centre_motion_px(float(a.cc_R), float(man.loc[a.file, "HGLT_OBS"]), dts, float(a.CROTA2))
            dx, dy, err = motion.shift(motion.prepared(io.read(man.loc[a.file, "path"])[0], fp),
                                       motion.prepared(io.read(man.loc[b.file, "path"])[0], fp))
            mx, my = dx - rx, dy - ry
            rec = {"lag": lag, "dt_min": dts / 60, "measured_dx": mx, "measured_dy": my, "corr_err": err,
                   "file_a": a.file, "file_b": b.file}
            for name, (xa, ya, xb, yb) in {
                "adopted": (a.reg_x0, a.reg_y0, b.reg_x0, b.reg_y0),
                "circle_unsmoothed": (a.cc_x0, a.cc_y0, b.cc_x0, b.cc_y0),
                "circle_smoothed": (a.circle_smoothed_x0, a.circle_smoothed_y0, b.circle_smoothed_x0, b.circle_smoothed_y0),
                "header": (a.CRPIX1, a.CRPIX2, b.CRPIX1, b.CRPIX2),
            }.items():
                rec[f"{name}_ex"], rec[f"{name}_ey"] = mx - (xb - xa), my - (yb - ya)
            rows.append(rec)
    return pd.DataFrame(rows)


def qc_decision(fr):
    """Frame-level QC. Reject = unusable; flags = usable but recorded. z-scores are robust, computed on
    this archive's own distributions."""
    reasons, usable = [], []
    for _, r in fr.iterrows():
        rs, ok = [], True
        if not np.isfinite(r.get("cc_R", np.nan)):
            rs.append("limb_fit_failed")
            ok = False
        elif r.disk_on_ccd < 0.9:
            rs.append("disk_off_ccd")
            ok = False
        for col, name in (("z_spike", "spike_rate"), ("z_bright", "brightness_jump"), ("z_limb", "limb_outlier")):
            if abs(r.get(col, 0) or 0) > 8:
                rs.append(name)
        if (r.get("jump_px", 0) or 0) > JUMP_PX:
            rs.append("pointing_jump")
        if r.get("mode_change", False):
            rs.append("pointing_mode_change")
        reasons.append(";".join(rs))
        usable.append(ok)
    return reasons, np.array(usable)


METHODS = (("header", "g"), ("circle_unsmoothed", "0.5"), ("circle_smoothed", "b"), ("adopted", "r"))


def figures(nb, val, fp_info):
    fig, ax = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    for k, ax_ in enumerate(("x", "y")):
        ax[k].plot(nb.t, nb[f"cc_{ax_}0"], ".", ms=1.5, color="0.6", label="circle fit (per frame)")
        ax[k].plot(nb.t, (nb.CRPIX1 if ax_ == "x" else nb.CRPIX2) - 1, ".", ms=1, color="g", alpha=.5, label="header")
        ax[k].plot(nb.t, nb[f"reg_{ax_}0"], "r.", ms=1.2, label="adopted (image motion + segment anchor)")
        ax[k].set_ylabel(f"disk centre {ax_} (2048 px)")
        ax[k].legend(fontsize=7, ncol=3)
    ax[2].plot(nb.t, nb.cc_R, ".", ms=1.5, color="0.6", label="circle R per frame")
    ax[2].plot(nb.t, nb.reg_R, "r.", ms=1, label="adopted R (segment mean)")
    ax[2].set_ylabel("disk radius (px)")
    ax[2].legend(fontsize=7)
    ax[0].set_title("NB03 pointing: jitter (~1.4 px/frame), ~1.5-2 h oscillation, jumps")
    fig.tight_layout()
    fig.savefig(FIG / "pointing_adopted.png", dpi=80)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, comp in enumerate(("ex", "ey")):
        for name, c in METHODS:
            ax[k].plot(val.dt_min, val[f"{name}_{comp}"], "o", color=c, ms=4, label=name, alpha=.8)
        ax[k].axhline(0, color="k", lw=.5)
        ax[k].set_xscale("symlog", linthresh=2)
        ax[k].set_xlabel("pair separation (min), pairs straddle a pointing jump")
        ax[k].set_ylabel(f"measured - predicted motion, {comp[1]} (px)")
        ax[k].legend(fontsize=8)
    fig.suptitle("Registration validation against image motion (independent segment anchors)")
    fig.tight_layout()
    fig.savefig(FIG / "validation.png", dpi=80)
    plt.close(fig)

    fp, split_r, _ = fp_info
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    v = np.percentile(np.abs(fp), 99)
    ax.imshow(fp, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"NB03 fixed detector pattern (high-passed), box {motion.BOX}\nrms {fp.std():.0f} counts, "
                 f"split-half r = {split_r:.3f}")
    fig.tight_layout()
    fig.savefig(FIG / "fixed_pattern.png", dpi=80)
    plt.close(fig)


def before_after(nb):
    """A pair ~60 min apart spanning a large pointing excursion: raw difference vs registered and
    rotation-shifted difference."""
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path"]).set_index("file").path
    best, pair = -1, None
    for _, g in nb.groupby("segment"):
        if len(g) > 45:
            d = np.hypot(g.reg_x0.diff(40), g.reg_y0.diff(40))
            j = d.idxmax()
            if d[j] > best:
                best, pair = d[j], (nb.index.get_loc(j) - 40, nb.index.get_loc(j))
    a, b = nb.iloc[pair[0]], nb.iloc[pair[1]]
    raw = [io.read(man[r.file])[0] for r in (a, b)]
    reg = [register_frame(r, man[r.file], r.reg_x0, r.reg_y0, r.reg_R) for r in (a, b)]
    from scipy.ndimage import shift as ndshift
    rot_dx, _ = solar.disk_centre_motion_px(R["r_ref"], 7.0, (b.t - a.t).total_seconds(), 0.0)
    reg0_rot = ndshift(np.nan_to_num(reg[0]), (0, rot_dx), order=1)
    lim = 500
    c = int((R["grid"] - 1) / 2)
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow((raw[1] - raw[0])[450:950, 1100:1600], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0].set_title(f"raw difference, {(b.t - a.t).total_seconds() / 60:.0f} min apart, pointing moved {best:.1f} px")
    ax[1].imshow((reg[1] - reg[0])[c - 250:c + 250, c - 100:c + 400], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1].set_title("registered difference (pointing removed; rotation remains)")
    ax[2].imshow((np.nan_to_num(reg[1]) - reg0_rot)[c - 250:c + 250, c - 100:c + 400], origin="lower", cmap="RdBu_r",
                 vmin=-lim, vmax=lim)
    ax[2].set_title(f"registered, first frame shifted by predicted disk-centre rotation ({rot_dx:.1f} px)")
    fig.tight_layout()
    fig.savefig(FIG / "before_after.png", dpi=70)
    plt.close(fig)


def main():
    fr = pd.read_parquet(OUT / "frames_full.parquet")
    fr = fr[fr.limb_R.notna()].copy()
    parts = []
    for kind, g in fr.groupby("frame"):
        rays = common_rays(g)
        parts.append(g.join(circle_fits(g, rays)).assign(common_rays=int(rays.sum())))
    fr = pd.concat(parts)
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path"]).set_index("file").path
    nbf = fr[fr.frame == "full_binned"].sort_values("t")
    pick = nbf.file.iloc[np.linspace(0, len(nbf) - 1, min(FP_FRAMES, len(nbf))).astype(int)]
    fp_info = motion.fixed_pattern([io.read(man[f])[0] for f in pick])
    fp = fp_info[0]
    np.save(OUT / "fixed_pattern_nb03_box.npy", fp)

    nb = adopt_nb03(nbf, fp)
    full = adopt_full(fr[fr.frame == "full"], nb)

    reg = pd.concat([nb, full], ignore_index=True)
    reg["z_spike"] = reg.groupby("frame").n_spike.transform(lambda v: (v - v.median()) / max(robust_sigma(v), 1))
    dev = np.hypot(reg.limb_minus_reg_x, reg.limb_minus_reg_y)
    isnb = reg.frame == "full_binned"
    reg["z_limb"] = np.where(isnb, (dev - dev[isnb].median()) / max(robust_sigma(dev[isnb]), .1), 0.0)
    roll = nb.set_index("t").disk_median.rolling("30min", center=True).median()
    d = nb.disk_median.values - roll.values
    reg["z_bright"] = reg.file.map(pd.Series(d / max(robust_sigma(d), 1), index=nb.file)).fillna(0)
    isnb = reg.frame == "full_binned"
    reg["mode_change"] = False
    nbi = reg[isnb].sort_values("t")
    # frames within 10 min of a change of pointing mode (the slew)
    ch = nbi.t[nbi.pointing_mode.ne(nbi.pointing_mode.shift()) & nbi.pointing_mode.shift().notna()]
    for tc in ch:
        reg.loc[isnb & ((reg.t - tc).abs() < pd.Timedelta("10min")), "mode_change"] = True
    reg["qc_reasons"], reg["qc_usable"] = qc_decision(reg)

    val = validate(nb, fp)
    figures(nb, val, fp_info)
    before_after(nb)

    keep = ["file", "t", "frame", "FTR_NAME", "OBS_MODE", "pointing_mode", "run", "segment", "reg_method", "reg_x0", "reg_y0", "reg_R",
            "reg_x0_anchor_sd", "reg_y0_anchor_sd", "CROTA2", "img_x", "img_y", "img_err", "cc_x0", "cc_y0", "cc_R",
            "cc_rms", "cc_rays", "common_rays", "circle_smoothed_x0", "circle_smoothed_y0", "CRPIX1", "CRPIX2", "R_SUN",
            "jump_px", "limb_minus_reg_x", "limb_minus_reg_y", "nb_x0", "nb_y0", "nb_dt_s", "filter_offset_x",
            "filter_offset_y", "z_spike", "z_limb", "z_bright", "qc_usable", "qc_reasons"]
    reg = reg.reindex(columns=keep)
    reg["grid"], reg["r_ref"] = R["grid"], R["r_ref"]
    reg.to_parquet(OUT / "registration.parquet", index=False)
    val.to_csv(OUT / "registration_validation.csv", index=False)

    def vstat(name, sub=val):
        e = np.hypot(sub[f"{name}_ex"], sub[f"{name}_ey"])
        return {"rms_x_px": float(np.sqrt(np.mean(sub[f"{name}_ex"] ** 2))),
                "rms_y_px": float(np.sqrt(np.mean(sub[f"{name}_ey"] ** 2))),
                "median_abs_px": float(np.median(e)), "p90_abs_px": float(np.quantile(e, .9))}

    diffs = nb.groupby("segment")[["cc_x0", "cc_y0"]].diff()
    fo = full[full.nb_dt_s < 600].groupby("FTR_NAME")[["filter_offset_x", "filter_offset_y"]]
    summary = {
        "fixed_pattern": {"rms_counts": float(fp.std()), "split_half_r": fp_info[1],
                          "highpass_frame_rms_counts": fp_info[2], "frames": int(len(pick))},
        "pointing_modes": {m: {"frames": int(len(g)), "start": str(g.t.min()), "end": str(g.t.max()),
                               "median_centre": [float(g.reg_x0.median()), float(g.reg_y0.median())]}
                           for m, g in nb.groupby("pointing_mode")},
        "nb03": {"frames": int(len(nb)), "runs": int(nb.run.nunique()), "segments": int(nb.segment.nunique()),
                 "pointing_jumps": int((nb.jump_px > JUMP_PX).sum()), "common_rays": int(nb.common_rays.iloc[0]),
                 "circle_frame_to_frame_px": [float(diffs.cc_x0.std() / np.sqrt(2)), float(diffs.cc_y0.std() / np.sqrt(2))],
                 "circle_minus_adopted_sigma_px": [robust_sigma(nb.limb_minus_reg_x), robust_sigma(nb.limb_minus_reg_y)],
                 "anchor_sd_px_median": [float(nb.reg_x0_anchor_sd.median()), float(nb.reg_y0_anchor_sd.median())],
                 "header_minus_adopted_median_px": [float(np.median(nb.CRPIX1 - 1 - nb.reg_x0)),
                                                    float(np.median(nb.CRPIX2 - 1 - nb.reg_y0))],
                 "header_minus_adopted_sigma_px": [robust_sigma(nb.CRPIX1 - 1 - nb.reg_x0),
                                                   robust_sigma(nb.CRPIX2 - 1 - nb.reg_y0)],
                 "adopted_range_px": {"x": [float(nb.reg_x0.min()), float(nb.reg_x0.max())],
                                      "y": [float(nb.reg_y0.min()), float(nb.reg_y0.max())]},
                 "radius_px": {"median": float(nb.reg_R.median()), "segment_spread": robust_sigma(nb.reg_R)}},
        "validation": {"pairs": int(len(val)), **{n: vstat(n) for n, _ in METHODS},
                       "by_lag_adopted": {int(l): vstat("adopted", g) for l, g in val.groupby("lag")}},
        "full_frames": {"circle_rms_px_median": full.groupby("FTR_NAME").cc_rms.median().round(2).to_dict(),
                        "offset_vs_nb03_median_px": fo.median().round(2).to_dict("index"),
                        "offset_vs_nb03_sigma_px": fo.agg(robust_sigma).round(2).to_dict("index"),
                        "radius_px_median": full.groupby("FTR_NAME").cc_R.median().round(1).to_dict()},
        "qc": {"usable": int(reg.qc_usable.sum()), "rejected": int((~reg.qc_usable).sum()),
               "flag_counts": reg.qc_reasons.str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict()},
        **CFG["_meta"],
    }
    (OUT / "registration_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("fixed_pattern", "nb03", "validation", "qc")}, indent=1, default=float))


if __name__ == "__main__":
    main()
