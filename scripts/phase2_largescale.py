"""Phase 2 fix: self-calibrate the NB03 large-scale detector response (vignetting, seam, bands).

    python scripts/phase2_largescale.py [--train-frames 140] [--centred-frames 100] [--knot 32]

Fit (suitdyn/largescale.py) on pattern-corrected, 2x2-averaged frames: training-split offset-mode frames
plus centred-mode frames (never validation or test). Quiet-disk pixels only: per frame, pixels within
0.85-1.2 of the frame's own mu-annulus median, away from the seam band and the vignetted CCD edge.

Validation against measurements the fit never saw:
  V1  the gradient of the fitted response must predict the per-pixel pointing sensitivity measured from
      one-frame differences on the VALIDATION split (phase2_noise_maps.py): slope = −100 ∂r/∂x_det;
  V2  the fitted seam step must match the step measured directly across x = 1023|1024 (Phase 1 seam
      profiles, offset-mode frames);
  V3  (downstream) a store corrected with it (build_store.py --response-map) must lower the validation
      pointing sensitivity and baseline errors below the first-order correction (phase2_response.py).
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import map_coordinates, median_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, flat, io, largescale, register, response  # noqa: E402

CFG = config.load_phase2()
P1 = config.out_dir(CFG)
CAL = P1 / "calibration"
OUT = config.ROOT / "outputs" / "phase2" / "largescale"
PATTERN = CAL / "nb03_pattern_additive_hp8.npy"  # identical to nb03_pattern_adopted.npy (checked by hash)


def _sample(args):
    path, x0, y0, R, clip_lo, clip_hi, k, seed, n_samp = args
    im, _ = io.read(path)
    bad = (im <= clip_lo) | (im >= clip_hi)
    im = flat.correct(im, np.load(PATTERN), "additive")
    im[bad] = np.nan
    h = im.reshape(1024, 2, 1024, 2)
    im2 = np.nanmean(h, axis=(1, 3))
    im2 = median_filter(np.nan_to_num(im2, nan=0.0), 3)
    yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float64)
    xf, yf = 2 * xx + 0.5, 2 * yy + 0.5  # detector (binned 2048) coordinates of the 2x2 block centres
    rr = np.hypot(xf - x0, yf - y0) / R
    mu = np.sqrt(np.clip(1 - rr ** 2, 0, None))
    edge = 40
    ok = (rr < 0.989) & (mu > 0.15) & (np.abs(xf - 1023.5) > 6) & (im2 > 0)
    ok &= (xf > edge) & (xf < 2048 - edge) & (yf > edge) & (yf < 2048 - edge)
    bins = np.linspace(0.15, 1.0, 21)
    ib = np.digitize(mu, bins)
    med = np.zeros(len(bins) + 1)
    for b in np.unique(ib[ok]):
        med[b] = np.median(im2[ok & (ib == b)])
    ratio = im2 / np.where(med[ib] > 0, med[ib], np.nan)
    ok &= (ratio > 0.85) & (ratio < 1.2)
    idx = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, min(n_samp, len(idx)), replace=False)
    return xf.ravel()[idx], yf.ravel()[idx], mu.ravel()[idx], np.full(len(idx), k), np.log(im2.ravel()[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frames", type=int, default=140)
    ap.add_argument("--centred-frames", type=int, default=100)
    ap.add_argument("--knot", type=int, default=32)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    adopted = CAL / "nb03_pattern_adopted.npy"
    same = adopted.exists() and hashlib.sha256(adopted.read_bytes()).hexdigest() == \
        hashlib.sha256(PATTERN.read_bytes()).hexdigest()

    reg = pd.read_parquet(P1 / "registration.parquet")
    man = pd.read_parquet(P1 / "manifest.parquet", columns=["file", "path", "clip_lo", "clip_hi"])
    seq = pd.read_parquet(config.ROOT / "outputs" / "phase2" / "sequences" / "frames.parquet")
    train_ids = set(seq.loc[seq.split == "train", "frame_id"])
    nb = reg[(reg.frame == "full_binned") & reg.qc_usable].merge(man, on="file").sort_values("t")
    bad = nb.qc_reasons.fillna("").str.contains("pointing_mode_change|limb_outlier|spike_rate")
    nb = nb[~bad]
    tr = nb[nb.file.isin(train_ids)]
    ce = nb[nb.pointing_mode == "centred"]
    pick = pd.concat([tr.iloc[np.linspace(0, len(tr) - 1, min(a.train_frames, len(tr))).astype(int)],
                      ce.iloc[np.linspace(0, len(ce) - 1, min(a.centred_frames, len(ce))).astype(int)]])
    pick = pick.reset_index(drop=True)
    jobs = [(r.path, r.reg_x0, r.reg_y0, r.reg_R, r.clip_lo, r.clip_hi, k, k, a.samples) for k, r in pick.iterrows()]
    with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
        parts = list(ex.map(_sample, jobs, chunksize=2))
    X, Y, MU, F, L = (np.concatenate([p[i] for p in parts]) for i in range(5))
    print(f"sampled {len(X)} pixels from {len(pick)} frames ({time.time() - t0:.0f} s)", flush=True)

    basis = largescale.Basis(size=2048, knot=a.knot)
    coef, res = largescale.fit(basis, X, Y, MU, F, L, len(pick), smooth=a.smooth)
    print(f"fit done ({time.time() - t0:.0f} s), residual rms {np.std(res):.4f}", flush=True)
    r_map = basis.evaluate_r(coef).astype(np.float32)
    coverage = np.zeros((2048, 2048), bool)
    for cx, cy, R in pick[["reg_x0", "reg_y0", "reg_R"]].drop_duplicates().values[::5]:
        yy, xx = np.ogrid[0:2048, 0:2048]
        coverage |= np.hypot(xx - cx, yy - cy) < 0.99 * R
    r_map[~coverage] = np.nan
    np.save(OUT / "nb03_logresponse.npy", r_map)

    # V1: predicted vs measured pointing sensitivity on the validation split (registered 768 grid)
    nm = np.load(config.ROOT / "outputs" / "phase2" / "noise_maps" / "noise_maps_v0_val_g2.npz")
    sx_m, sy_m = response.nan_smooth(nm["slope_x"], 6), response.nan_smooth(nm["slope_y"], 6)
    gy, gx = np.gradient(np.nan_to_num(r_map))
    fr = pd.read_parquet(config.ROOT / "outputs" / "phase2" / "stores" / "v0.frames.parquet")
    v = fr[fr.split == "val"]
    x0, y0, R, rho = v.reg_x0.median(), v.reg_y0.median(), v.reg_R.median(), v.CROTA2.median()
    grid, r_ref = sx_m.shape[0], CFG["register"]["r_ref"] * sx_m.shape[0] / CFG["register"]["grid"]
    A, b = register.transform(x0, y0, R, rho, grid, r_ref)
    vv, uu = np.indices((grid, grid), dtype=np.float64)
    dx = A[0, 0] * uu + A[0, 1] * vv + b[0]
    dy = A[1, 0] * uu + A[1, 1] * vv + b[1]
    sx_p = -100 * map_coordinates(gx, [dy, dx], order=1)
    sy_p = -100 * map_coordinates(gy, [dy, dx], order=1)
    cov_reg = map_coordinates(coverage.astype(float), [dy, dx], order=0) > 0.5
    far_seam = np.abs(dx - 1023.5) > 12
    m = cov_reg & far_seam & np.isfinite(sx_m) & np.isfinite(sy_m)
    v1 = {"pixels": int(m.sum()),
          "corr_x": float(np.corrcoef(sx_p[m], sx_m[m])[0, 1]), "corr_y": float(np.corrcoef(sy_p[m], sy_m[m])[0, 1]),
          "slope_x_measured_on_predicted": float(np.polyfit(sx_p[m], sx_m[m], 1)[0]),
          "slope_y_measured_on_predicted": float(np.polyfit(sy_p[m], sy_m[m], 1)[0]),
          "rms_measured": [float(np.std(sx_m[m])), float(np.std(sy_m[m]))],
          "rms_measured_minus_predicted": [float(np.std((sx_m - sx_p)[m])), float(np.std((sy_m - sy_p)[m]))]}

    # V2: seam step, fitted vs measured
    sp = pd.read_parquet(P1 / "seam_profiles.parquet")
    offs = set(reg.loc[(reg.frame == "full_binned") & (reg.pointing_mode == "offset"), "file"])
    meas = sp[(sp.axis == "vertical") & sp.file.isin(offs)].groupby("pos").step.median()
    fit_step = {}
    for pos in meas.index:
        yv = int(pos)
        left = np.nanmean(r_map[max(yv - 32, 0):yv + 32, 1013:1019])
        right = np.nanmean(r_map[max(yv - 32, 0):yv + 32, 1024:1030])
        fit_step[pos] = float(np.exp(right - left) - 1)
    s = pd.DataFrame({"measured": meas, "fitted": pd.Series(fit_step)}).dropna()
    v2 = {"rows": s.index.tolist(), "measured": s.measured.round(4).tolist(), "fitted": s.fitted.round(4).tolist(),
          "corr": float(np.corrcoef(s.measured, s.fitted)[0, 1]) if len(s) > 2 else None,
          "rms_difference": float(np.sqrt(np.mean((s.measured - s.fitted) ** 2))) if len(s) else None}

    q = basis.evaluate_q(coef, np.linspace(0.15, 1, 18))
    prov = {"file": "nb03_logresponse.npy", "sha256": hashlib.sha256((OUT / "nb03_logresponse.npy").read_bytes()).hexdigest(),
            "pattern_used": PATTERN.name, "pattern_identical_to_adopted": same,
            "frames": {"train_offset": int((pick.pointing_mode == "offset").sum()),
                       "centred": int((pick.pointing_mode == "centred").sum())},
            "samples": int(len(X)), "knot_px": a.knot, "smooth": a.smooth, "residual_rms": float(np.std(res)),
            "response_range": [float(np.nanpercentile(r_map, 1)), float(np.nanpercentile(r_map, 99))],
            "limb_darkening_q": {"mu": np.linspace(0.15, 1, 18).round(3).tolist(), "q": np.round(q, 4).tolist()},
            "v1_pointing_sensitivity": v1, "v2_seam_step": v2, "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    (OUT / "nb03_logresponse.json").write_text(json.dumps(prov, indent=1, default=float))

    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    im0 = ax[0, 0].imshow(np.exp(r_map), origin="lower", cmap="viridis")
    ax[0, 0].set_title("fitted large-scale response R(x) (detector)")
    plt.colorbar(im0, ax=ax[0, 0], fraction=.046)
    ax[0, 1].plot(np.linspace(0.15, 1, 18), np.exp(q), "o-")
    ax[0, 1].set_xlabel("mu")
    ax[0, 1].set_title("fitted quiet-Sun limb darkening exp(q)")
    ax[0, 2].plot(s.index, s.measured * 100, "k-o", ms=3, label="measured across 1023|1024")
    ax[0, 2].plot(s.index, s.fitted * 100, "r-s", ms=3, label="fitted response")
    ax[0, 2].set_xlabel("row")
    ax[0, 2].set_ylabel("seam step (%)")
    ax[0, 2].legend()
    ax[0, 2].set_title(f"V2 seam step: r = {v2['corr']:.2f}" if v2["corr"] is not None else "V2")
    lim = np.nanpercentile(np.abs(sx_m[m]), 99)
    for j, (pred, meas_, lab) in enumerate(((sx_p, sx_m, "x"), (sy_p, sy_m, "y"))):
        ax[1, j].imshow(np.where(m, meas_, np.nan), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1, j].contour(np.where(m, pred, 0), levels=np.linspace(-lim, lim, 9), colors="k", linewidths=.5)
        ax[1, j].set_title(f"V1 {lab}: measured (colour) vs predicted (contours), r = {v1['corr_' + lab]:.2f}")
    ax[1, 2].plot(sx_p[m][::10], sx_m[m][::10], ".", ms=1, label="x")
    ax[1, 2].plot(sy_p[m][::10], sy_m[m][::10], ".", ms=1, label="y")
    ax[1, 2].plot([-lim, lim], [-lim, lim], "k-", lw=.5)
    ax[1, 2].set_xlabel("predicted slope (%/px)")
    ax[1, 2].set_ylabel("measured slope, validation (%/px)")
    ax[1, 2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "largescale.png", dpi=70)
    plt.close(fig)
    print(json.dumps({k: prov[k] for k in ("frames", "samples", "residual_rms", "response_range",
                                          "pattern_identical_to_adopted", "v1_pointing_sensitivity",
                                          "v2_seam_step")}, indent=1, default=float))


if __name__ == "__main__":
    main()
