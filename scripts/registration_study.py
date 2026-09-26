"""Registration study and adopted per-frame transforms, plus the Phase 1 frame-level QC decision.

    python scripts/registration_study.py

Question: how should each full-disk frame be placed on the common grid? Candidates:
  H  header CRPIX / R_SUN
  L  per-frame limb fit (circle + distortion harmonics)
  C  chained NB03 pointing: frame-to-frame image motion (phase correlation) minus the predicted
     solar-rotation motion, summed within each continuous run, anchored to the run-mean limb fit.
The script measures each against the others, adopts one, validates it on registered frame pairs,
and writes registration.parquet (one transform per frame, with its provenance) and figures.
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
from suitdyn import config, io, register, solar  # noqa: E402
from scripts.process_frames import phase_motion  # noqa: E402

CFG = config.load()
OUT = config.out_dir(CFG)
FIG = config.out_dir(CFG, "registration")
RUN_GAP_S = 300


def robust_sigma(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def chain_nb03(nb):
    """Pointing chain per continuous run: cumulative (measured motion − predicted rotation)."""
    nb = nb.sort_values("t").copy()
    dt = nb.t.diff().dt.total_seconds()
    nb["run"] = (dt.isna() | (dt > RUN_GAP_S)).cumsum()
    hg = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file").HGLT_OBS
    b0 = hg.reindex(nb.file).astype(float).values
    rx, ry = solar.disk_centre_motion_px(nb.limb_R.values, b0, dt.fillna(0).values, nb.CROTA2.values)
    nb["rot_dx"], nb["rot_dy"] = rx, ry
    first = nb.run.ne(nb.run.shift())
    for ax in ("x", "y"):
        step = (nb[f"pc_d{ax}"] - nb[f"rot_d{ax}"]).where(~first, 0.0).fillna(0.0)
        nb[f"chain_{ax}"] = step.groupby(nb.run).cumsum()
    return nb


def adopt_nb03(nb):
    """Adopted centre = chain + per-run offset (robust mean of limb − chain); radius = per-run robust
    mean of the limb radius, scaled by the Sun-spacecraft distance."""
    out = []
    for run, g in nb.groupby("run"):
        g = g.copy()
        for ax in ("x", "y"):
            d = g[f"limb_{ax}0"] - g[f"chain_{ax}"]
            med = np.median(d)
            keep = np.abs(d - med) < 5 * max(robust_sigma(d), 0.5)
            g[f"reg_{ax}0"] = g[f"chain_{ax}"] + d[keep].mean()
            g[f"limb_minus_reg_{ax}"] = g[f"limb_{ax}0"] - g[f"reg_{ax}0"]
            g[f"reg_{ax}0_sd"] = d[keep].std() / np.sqrt(keep.sum())
        rn = g.limb_R * g.DSUN_OBS
        g["reg_R"] = np.median(rn) / g.DSUN_OBS
        g["reg_method"] = "chain+limb_anchor" if len(g) >= 20 else "limb_only"
        if len(g) < 20:  # too short to anchor a chain: use the limb fit itself
            g["reg_x0"], g["reg_y0"], g["reg_R"] = g.limb_x0, g.limb_y0, g.limb_R
        out.append(g)
    return pd.concat(out)


def adopt_full(full, nbreg):
    """Unbinned burst frames: NB03 binned pointing interpolated to the frame time, doubled (2x2 binning,
    x_4096 = 2·x_2048 + 0.5), plus a constant per-filter offset and radius (filter wedge tilts shift the
    image, Tripathi et al. 2025 Table 4), estimated as the median over bursts of the limb fit."""
    full = full.sort_values("t").copy()
    t_nb = nbreg.t.astype("int64").values
    t_f = full.t.astype("int64").values
    for ax in ("x", "y"):
        full[f"nb_{ax}0"] = 2 * np.interp(t_f, t_nb, nbreg[f"reg_{ax}0"].values) + 0.5
    near = np.abs(t_f[:, None] - t_nb[None, :]).min(1) / 1e9
    full["nb_dt_s"] = near
    rows = []
    for f, g in full.groupby("FTR_NAME"):
        g = g.copy()
        ok = g.nb_dt_s < 600
        for ax in ("x", "y"):
            off = (g[f"limb_{ax}0"] - g[f"nb_{ax}0"])[ok]
            g[f"filter_offset_{ax}"] = np.median(off) if ok.any() else np.nan
            g[f"reg_{ax}0"] = np.where(ok, g[f"nb_{ax}0"] + g[f"filter_offset_{ax}"], g[f"limb_{ax}0"])
            g[f"limb_minus_reg_{ax}"] = g[f"limb_{ax}0"] - g[f"reg_{ax}0"]
        g["reg_R"] = np.median(g.limb_R * g.DSUN_OBS) / g.DSUN_OBS
        g["reg_method"] = np.where(ok, "nb03_chain+filter_offset", "limb_only")
        rows.append(g)
    return pd.concat(rows)


def qc_decision(fr):
    """Frame-level QC. Reject = unusable; flags = usable but recorded. Thresholds are robust z-scores
    on this archive's own distributions, so they adapt to the data rather than being guessed."""
    reasons, reject = [], []
    for _, r in fr.iterrows():
        rs, rj = [], False
        if not np.isfinite(r.get("limb_R", np.nan)):
            rs.append("limb_fit_failed")
            rj = True
        elif r.disk_on_ccd < 0.9:
            rs.append("disk_off_ccd")
            rj = True
        for col, name in (("z_spike", "spike_rate"), ("z_bright", "brightness_jump"), ("z_limb", "limb_outlier")):
            if abs(r.get(col, 0) or 0) > 8:
                rs.append(name)
        if (r.get("pc_err", 0) or 0) > 0.9:
            rs.append("poor_motion_estimate")
        if (r.get("jump_px", 0) or 0) > 1.0:
            rs.append("pointing_jump")
        reasons.append(";".join(rs))
        reject.append(rj)
    return reasons, ~np.array(reject)


def validate_pairs(nb, n=60, seed=0):
    """Residual motion between consecutive frames after registration. If registration is right, the only
    motion left is solar rotation (≈0.16 px per 87 s near disk centre)."""
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path"]).set_index("file").path
    rng = np.random.default_rng(seed)
    cand = nb[(nb.t.diff().dt.total_seconds() < RUN_GAP_S) & nb.run.eq(nb.run.shift())].index
    jumps = nb.index[(nb.jump_px > 1.0)]
    pick = list(rng.choice(cand, min(n, len(cand)), replace=False)) + [j for j in jumps if j in cand][:20]
    grid, r_ref = CFG["register"]["grid"], CFG["register"]["r_ref"]
    rows = []
    for i in pick:
        cur, prev = nb.loc[i], nb.loc[nb.index[nb.index.get_loc(i) - 1]]
        ims = []
        for row, use in ((prev, "reg"), (cur, "reg")):
            im, _ = io.read(man[row.file])
            A, b = register.transform(row.reg_x0, row.reg_y0, row.reg_R, row.CROTA2, grid, r_ref)
            ims.append(np.nan_to_num(register.apply(im, A, b, grid)))
        c = (grid - 1) / 2
        box = (int(c - 260), int(c + 260), int(c - 150), int(c + 370))  # disk interior, clear of the seam band
        dx, dy, err = phase_motion(ims[0], ims[1], box)
        rows.append({"file": cur.file, "t": cur.t, "dt_s": (cur.t - prev.t).total_seconds(), "jump_pair": i in jumps,
                     "resid_dx": dx, "resid_dy": dy, "raw_dx": cur.pc_dx, "raw_dy": cur.pc_dy,
                     "expected_rot_dx_grid": cur.rot_dx * r_ref / cur.reg_R, "err": err})
    return pd.DataFrame(rows)


def before_after(nb):
    """Difference images of a frame pair across the largest pointing jump, raw and registered."""
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path"]).set_index("file").path
    i = nb.jump_px.idxmax()
    pos = nb.index.get_loc(i)
    prev, cur = nb.iloc[pos - 1], nb.iloc[pos]
    grid, r_ref = CFG["register"]["grid"], CFG["register"]["r_ref"]
    raw, reg = [], []
    for row in (prev, cur):
        im, _ = io.read(man[row.file])
        raw.append(im)
        A, b = register.transform(row.reg_x0, row.reg_y0, row.reg_R, row.CROTA2, grid, r_ref)
        reg.append(register.apply(im, A, b, grid))
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    lim = 400
    ax[0, 0].imshow(raw[1], origin="lower", cmap="inferno", vmin=0, vmax=np.percentile(raw[1], 99.5))
    ax[0, 0].set_title(f"raw {cur.t:%m-%d %H:%M:%S}")
    ax[0, 1].imshow(raw[1] - raw[0], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0, 1].set_title(f"raw difference (Δt {(cur.t - prev.t).total_seconds():.0f} s, jump {cur.jump_px:.1f} px)")
    ax[0, 2].imshow((raw[1] - raw[0])[400:900, 1100:1600], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[0, 2].set_title("raw difference, zoom")
    ax[1, 0].imshow(reg[1], origin="lower", cmap="inferno", vmin=0, vmax=np.nanpercentile(reg[1], 99.5))
    c = (grid - 1) / 2
    t = np.linspace(0, 2 * np.pi, 400)
    ax[1, 0].plot(c + r_ref * np.cos(t), c + r_ref * np.sin(t), "c-", lw=.6)
    ax[1, 0].set_title("registered (cyan: r_ref circle), solar north up")
    d = reg[1] - reg[0]
    ax[1, 1].imshow(d, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[1, 1].set_title("registered difference")
    ax[1, 2].imshow(d[int(c) - 250:int(c) + 250, int(c) - 100:int(c) + 400], origin="lower", cmap="RdBu_r", vmin=-lim,
                    vmax=lim)
    ax[1, 2].set_title("registered difference, zoom")
    fig.tight_layout()
    fig.savefig(FIG / "before_after_jump.png", dpi=70)
    plt.close(fig)


def main():
    fr = pd.read_parquet(OUT / "frames_full.parquet")
    fr = fr[fr.limb_R.notna()].copy()
    nb = chain_nb03(fr[fr.frame == "full_binned"])
    nb["jump_px"] = np.hypot(nb.pc_dx - nb.rot_dx, nb.pc_dy - nb.rot_dy)
    nb = adopt_nb03(nb)
    full = adopt_full(fr[fr.frame == "full"], nb)

    # sign check of the rotation model: median measured x-motion per 87 s should match the prediction
    steady = nb[(nb.jump_px < 0.5) & nb.pc_dx.notna()]
    sign = {"median_pc_dx_per_frame": float(steady.pc_dx.median()), "median_pred_rot_dx": float(steady.rot_dx.median()),
            "median_pc_dy_per_frame": float(steady.pc_dy.median()), "median_pred_rot_dy": float(steady.rot_dy.median())}

    reg = pd.concat([nb, full], ignore_index=True)
    reg["z_spike"] = reg.groupby("frame").n_spike.transform(lambda v: (v - v.median()) / max(robust_sigma(v), 1))
    reg["z_limb"] = np.hypot(reg.limb_minus_reg_x, reg.limb_minus_reg_y)
    reg["z_limb"] = reg.groupby("frame").z_limb.transform(lambda v: (v - v.median()) / max(robust_sigma(v), 0.1))
    roll = nb.set_index("t").disk_median.rolling("30min", center=True).median()
    dev = nb.disk_median.values - roll.values
    nb_z = pd.Series(dev / max(robust_sigma(dev), 1), index=nb.file)
    reg["z_bright"] = reg.file.map(nb_z).fillna(0)
    reg["qc_reasons"], reg["qc_usable"] = qc_decision(reg)

    val = validate_pairs(nb)
    before_after(nb)

    keep = ["file", "t", "frame", "FTR_NAME", "OBS_MODE", "run", "reg_method", "reg_x0", "reg_y0", "reg_R", "reg_x0_sd",
            "reg_y0_sd", "CROTA2", "limb_x0", "limb_y0", "limb_R", "circ_x0", "circ_y0", "CRPIX1", "CRPIX2", "R_SUN",
            "chain_x", "chain_y", "rot_dx", "rot_dy", "jump_px", "limb_minus_reg_x", "limb_minus_reg_y",
            "filter_offset_x", "filter_offset_y", "z_spike", "z_limb", "z_bright", "qc_usable", "qc_reasons"]
    reg = reg.reindex(columns=keep)
    reg["grid"], reg["r_ref"] = CFG["register"]["grid"], CFG["register"]["r_ref"]
    reg.to_parquet(OUT / "registration.parquet", index=False)
    val.to_csv(OUT / "registration_validation.csv", index=False)

    nbv = reg[reg.frame == "full_binned"]
    summary = {
        "rotation_sign_check": sign,
        "nb03": {
            "frames": int(len(nbv)), "runs": int(nb.run.nunique()),
            "limb_minus_adopted_sigma_px": [robust_sigma(nbv.limb_minus_reg_x), robust_sigma(nbv.limb_minus_reg_y)],
            "circle_minus_adopted_sigma_px": [robust_sigma(nb.circ_x0 - nb.reg_x0), robust_sigma(nb.circ_y0 - nb.reg_y0)],
            "header_minus_adopted_median_px": [float(np.median(nb.CRPIX1 - 1 - nb.reg_x0)),
                                               float(np.median(nb.CRPIX2 - 1 - nb.reg_y0))],
            "header_minus_adopted_sigma_px": [robust_sigma(nb.CRPIX1 - 1 - nb.reg_x0), robust_sigma(nb.CRPIX2 - 1 - nb.reg_y0)],
            "adopted_range_px": {"x": [float(nb.reg_x0.min()), float(nb.reg_x0.max())],
                                 "y": [float(nb.reg_y0.min()), float(nb.reg_y0.max())]},
            "pointing_jumps_gt1px": int((nb.jump_px > 1).sum()),
            "anchor_sd_px_median": [float(nb.reg_x0_sd.median()), float(nb.reg_y0_sd.median())],
        },
        "validation_pairs": {
            "n": int(len(val)),
            "resid_motion_median_px": [float(val.resid_dx.median()), float(val.resid_dy.median())],
            "resid_motion_p95_abs_px": [float(val.resid_dx.abs().quantile(.95)), float(val.resid_dy.abs().quantile(.95))],
            "raw_motion_p95_abs_px": [float(val.raw_dx.abs().quantile(.95)), float(val.raw_dy.abs().quantile(.95))],
            "expected_rotation_dx_median_px": float(val.expected_rot_dx_grid.median()),
            "jump_pairs": int(val.jump_pair.sum()),
            "jump_pairs_resid_max_px": float(np.hypot(val[val.jump_pair].resid_dx, val[val.jump_pair].resid_dy).max())
            if val.jump_pair.any() else None,
        },
        "full_frames_filter_offsets_px": full.groupby("FTR_NAME")[["filter_offset_x", "filter_offset_y"]].first()
        .round(2).to_dict("index"),
        "full_frames_limb_minus_adopted_sigma_px": {
            f: [robust_sigma(g.limb_minus_reg_x), robust_sigma(g.limb_minus_reg_y)] for f, g in full.groupby("FTR_NAME")},
        "qc": {"usable": int(reg.qc_usable.sum()), "rejected": int((~reg.qc_usable).sum()),
               "flag_counts": reg.qc_reasons.str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict()},
        **CFG["_meta"],
    }
    (OUT / "registration_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
