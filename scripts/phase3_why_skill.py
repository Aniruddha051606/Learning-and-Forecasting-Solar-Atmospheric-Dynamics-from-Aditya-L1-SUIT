"""Phase 3: why does the models' skill over the context mean (B1-avg) GROW with horizon?

    python scripts/phase3_why_skill.py [--runs unet_s0,convlstm_s0] [--threads 4]

Explanations that are not learned solar evolution, each turned into a baseline fitted WITHOUT the
validation split, and then compared with what the models actually do:

  H-rot   Mg II k features do not rotate at the Snodgrass & Ulrich magnetic rate used by B1. A rate error
          displaces features by an amount that grows linearly with horizon, so a model that learns the
          correction gains more at long horizons.
          Fit: the linearised residual r = y - F ~ dw * z, z = -dt * (dpos/dlon . grad F), F = B1-avg,
          by least squares per 10-degree latitude band and as dw = a + b sin^2(lat), with a translation per
          pair as nuisance (pointing). Pairs: train + hold-out. Interval: bootstrap over (run, hour) blocks.
          Baseline: B1-avg-rot = context mean derotated at SU90 + (a, b).
  H-damp  Regression to the mean: bright structure fades and dark structure fills in over hours, so the
          MAE-optimal forecast shrinks contrast toward a local mean, more at longer horizons.
          Baseline: B1-avg-damp = L + beta(H) * (B1-avg - L), L = B1-avg Gaussian-blurred by sigma_L;
          (sigma_L, beta) per horizon chosen on the hold-out run. B1-avg-rot-damp combines both.
  H-reg   Registration drift between context and target: the per-pair translation of the same fit, by
          horizon (the pointing oscillation is +-10 px at 1536 = +-2.5 px here).
  H-inst  Derotation moves the WHOLE image, including anything fixed on the registered grid (the large-scale
          instrument pattern that Phase 2 left uncorrected, the seam, CCD-edge vignetting, the limb-darkening
          background). That error does not depend on solar content and grows with the rotation shift, i.e.
          with horizon; a model told the horizon can learn to undo it.
          Baseline: B1-avg-clim = B1-avg + M(H), M(H) = mean residual map (target - B1-avg) over the train +
          hold-out pairs of that horizon (solar evolution averages out; a static pattern does not).
          Diagnostics: mean correction map of each model vs M(H) (correlation, and the share of each
          model correction explained by M(H), per sample).
          Root-cause baseline (no learning, no validation data): rotate only the solar part,
            B1-avg-bg = mean_k [ rot_k(x_k - Bg) + Bg ] = B1-avg + mean_k (Bg - rot_k(Bg)),
          Bg = per-pixel median of the training-split frames (limb darkening + static instrument pattern;
          solar structure moves through and is suppressed by the median). B1-avg-LDadd does the same with
          only the limb-darkening profile q(mu) as the static background (separates LD from instrument).
          Multiplicative forms (physically right for LD, I = S q(mu)): B1-avg-LD = mean_k x_k,rot q(mu_tgt)/
          q(mu_src); B1-avg-bgmult = Bg * mean_k rot_k(x_k / Bg).
          B1-avg-bgS: the same with the static background S solved from derotation residuals of the train +
          hold-out pairs (scripts/phase3_background.py), when that file exists.
Model diagnostics on validation (CPU inference, so a training job can keep the GPU):
  * scale decomposition: RMS of the error field below and above a 4-px Gaussian scale (384 grid);
  * effective damping: beta_model = sum (P-L)(A-L) / sum (A-L)^2, P = model forecast, A = B1-avg;
  * effective rate offset: the linearised fit applied to P - A (what motion the model adds).
Validation is used for reporting only; the test split stays sealed. Writes outputs/phase3/why/.
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase3_evaluate as ev  # noqa: E402  (shared constants and helpers)
from suitdyn import baselines, config, progress  # noqa: E402
from suitdyn.ml import models  # noqa: E402

CFG, G, CACHE, RUNS = ev.CFG, ev.G, ev.CACHE, ev.RUNS
OUT = config.phase3_dir("why")
BANDS = np.arange(-60, 61, 10)          # latitude band edges, degrees
NB = len(BANDS) - 1
RHO_MAX = 0.9                           # fit region: away from the limb (foreshortening, LD)
R_CLIP = 0.25                           # fit region: drop |residual| above this (brightenings, spikes)
SIG_L = (2.0, 4.0, 8.0, 16.0, 32.0)
BETAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)
SCALE_PX = 4.0
DEG_DAY = 180 / np.pi * 86400           # rad/s -> deg/day


class Geo:
    """Per-B0 geometry on the target grid: d(col,row)/d(longitude) in px/rad, latitude band, sin^2 lat."""

    def __init__(self, r_ref):
        self.r_ref, self.cache = r_ref, {}
        c = (G - 1) / 2
        v, u = np.indices((G, G))
        self.rho = np.hypot(u - c, v - c) / r_ref

    def get(self, b0):
        k = round(float(b0), 1)
        if k not in self.cache:
            lat, lon = baselines.heliographic(G, self.r_ref, k)
            b = np.deg2rad(k)
            vc = self.r_ref * np.cos(lat) * np.cos(lon)
            vr = self.r_ref * np.cos(lat) * np.sin(lon) * np.sin(b)
            latd = np.rad2deg(lat)
            band = np.digitize(np.nan_to_num(latd, nan=999.0), BANDS) - 1
            band[(band < 0) | (band >= NB) | ~np.isfinite(latd)] = -1
            self.cache[k] = (vc, vr, band, np.sin(lat) ** 2)
        return self.cache[k]


def pair_system(F, r, mask, dt, geo_b):
    """Normal equations of r ~ rate columns + translation (t_col, t_row) for one pair.

    Returns the rate block with the translation eliminated (Schur complement) for the per-band model and
    for the (a, b sin^2) model, plus per-pair single-rate and translation estimates."""
    vc, vr, band, s2 = geo_b
    gr, gc = np.gradient(F)
    z = -dt * (vc * gc + vr * gr)
    ok = mask & np.isfinite(z) & np.isfinite(r) & (band >= 0)
    if ok.sum() < 2000:
        return None
    zb, bb, rr, ss = z[ok], band[ok], r[ok], s2[ok]
    T = np.stack([-gc[ok], -gr[ok]], 1)
    Att = T.T @ T
    ct = T.T @ rr
    Ainv = np.linalg.inv(Att)
    # per band
    Aaa = np.diag(np.bincount(bb, zb * zb, NB))
    Aat = np.stack([np.bincount(bb, zb * T[:, 0], NB), np.bincount(bb, zb * T[:, 1], NB)], 1)
    ca = np.bincount(bb, zb * rr, NB)
    S_band, u_band = Aaa - Aat @ Ainv @ Aat.T, ca - Aat @ Ainv @ ct
    # a + b sin^2
    Z2 = np.stack([zb, zb * ss], 1)
    A2, At2, c2 = Z2.T @ Z2, Z2.T @ T, Z2.T @ rr
    S_ab, u_ab = A2 - At2 @ Ainv @ At2.T, c2 - At2 @ Ainv @ ct
    # single rate + translation, and translation only
    M = np.c_[zb, T]
    th = np.linalg.solve(M.T @ M, M.T @ rr)
    t0 = Ainv @ ct
    return {"S_band": S_band, "u_band": u_band, "S_ab": S_ab, "u_ab": u_ab, "S_abnt": A2, "u_abnt": c2,
            "rate": th[0] * DEG_DAY,
            "t_col": th[1], "t_row": th[2], "t_col_only": t0[0], "t_row_only": t0[1], "npix": int(ok.sum())}


def solve(S, u):
    return np.linalg.lstsq(S, u, rcond=None)[0]


def boot_solve(recs, key, n=400, seed=0):
    """Bootstrap over (run, hour) blocks of the pooled normal equations."""
    rng = np.random.default_rng(seed)
    blk = {}
    for r in recs:
        b = blk.setdefault(r["block"], [0, 0])
        b[0] = b[0] + r[f"S_{key}"]
        b[1] = b[1] + r[f"u_{key}"]
    blocks = list(blk.values())
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        out.append(solve(sum(blocks[i][0] for i in pick), sum(blocks[i][1] for i in pick)))
    return np.array(out)


def fit_rates(recs):
    res = {}
    S_b, u_b = sum(r["S_band"] for r in recs), sum(r["u_band"] for r in recs)
    S_a, u_a = sum(r["S_ab"] for r in recs), sum(r["u_ab"] for r in recs)
    band, ab = solve(S_b, u_b) * DEG_DAY, solve(S_a, u_a) * DEG_DAY
    bb, ba = boot_solve(recs, "band") * DEG_DAY, boot_solve(recs, "ab") * DEG_DAY
    res["band_deg_day"] = band.tolist()
    res["band_lo"], res["band_hi"] = np.percentile(bb, 2.5, 0).tolist(), np.percentile(bb, 97.5, 0).tolist()
    res["ab_deg_day"] = ab.tolist()
    res["ab_lo"], res["ab_hi"] = np.percentile(ba, 2.5, 0).tolist(), np.percentile(ba, 97.5, 0).tolist()
    # the same (a, b) model without the per-pair translation: shows how far pointing and rate trade off
    res["ab_no_translation_deg_day"] = (solve(sum(r["S_abnt"] for r in recs), sum(r["u_abnt"] for r in recs))
                                        * DEG_DAY).tolist()
    res["band_has_data"] = (np.diag(S_b) > 0).astype(int).tolist()
    res["pairs"] = len(recs)
    return res


def context_store(row, t_store):
    """Store indices of the context frames of a sample, from their times."""
    ts = pd.Timestamp(row["t_target"]) - pd.to_timedelta(np.asarray(row["dt_context_s"], float), unit="s")
    pos = np.searchsorted(t_store.index.values, ts.values)
    out = []
    for p, t in zip(pos, ts.values):
        cand = [q for q in (p - 1, p) if 0 <= q < len(t_store)]
        q = min(cand, key=lambda q: abs(t_store.index.values[q] - t))
        assert abs(t_store.index.values[q] - t) < np.timedelta64(500, "ms"), "context frame not found"
        out.append(int(t_store.values[q]))
    return out


def rotated_mean(frames, stores, dts, r_ref, b0, rate_offset, cache, divisor=None):
    """Mean of the context frames rotated to the target time; with `divisor`, each frame is divided by it
    before rotating and the mean is multiplied by it afterwards (a static multiplicative background)."""
    acc = []
    for s, dt in zip(stores, dts):
        key = (round(dt / 2) * 2, round(b0, 1))
        if key not in cache:
            if len(cache) > 3000:
                cache.clear()
            cache[key] = baselines.derotation_coords(G, r_ref, key[1], key[0], rate_offset=rate_offset)
        f = frames[s].astype(np.float32)
        acc.append(baselines.rotated_persistence(f if divisor is None else f / divisor, r_ref, b0, dt,
                                                 coords=cache[key]))
    return np.mean(acc, 0) if divisor is None else np.mean(acc, 0) * divisor


def static_shift_error(Bimg, dts, r_ref, b0, rate_offset, cache):
    """mean_k (Bimg - rot_k(Bimg)): what derotating the context adds to a background fixed on the grid."""
    acc = []
    for dt in dts:
        key = (round(dt / 2) * 2, round(b0, 1))
        if key not in cache:
            if len(cache) > 3000:
                cache.clear()
            cache[key] = baselines.derotation_coords(G, r_ref, key[1], key[0], rate_offset=rate_offset)
        acc.append(Bimg - baselines.rotated_persistence(Bimg, r_ref, b0, dt, coords=cache[key]))
    return np.mean(acc, 0)


def damp(A, sig, beta):
    L = ev.nan_blur(A, sig)
    return L + beta * (A - L), L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="unet_s0,convlstm_s0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth sample (smoke tests only)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_start = time.time()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(CACHE / f"samples_{G}.parquet").iloc[::a.stride]
    X = np.load(CACHE / f"X_{G}.npy", mmap_mode="r")
    Y = np.load(CACHE / f"Y_{G}.npy", mmap_mode="r")
    Xbg = np.load(CACHE / f"X_{G}_bg.npy", mmap_mode="r") if (CACHE / f"X_{G}_bg.npy").exists() else None
    frames = np.load(CACHE / f"frames_{G}.npy", mmap_mode="r")
    mu_np = np.load(CACHE / f"mu_{G}.npy")
    trusted = np.load(CACHE / f"trusted_{G}.npy").astype(bool)
    prep = json.loads((CACHE / "prepare_meta.json").read_text())
    r_ref = float(prep["r_ref"])
    store_fr = pd.read_parquet(config.ROOT / "outputs" / "phase2" / "stores" / f"{prep['store']}.frames.parquet")
    man = pd.read_parquet(config.out_dir(ev.CFG) / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file")
    b0_of = store_fr.set_index("store_index").frame_id.map(man.HGLT_OBS).astype(float)
    t_store = store_fr.set_index("t").store_index.sort_index()
    geo = Geo(r_ref)
    # static background: per-pixel median of the training-split frames (every 2nd frame; memory)
    tr_store = store_fr.loc[store_fr.split == "train", "store_index"].values
    Bg = np.nanmedian(np.asarray(frames[np.sort(tr_store)[::2]], dtype=np.float32), 0)
    qc, qv = ev.ld_profile(frames, tr_store[::10], mu_np)
    qmap = np.where(np.isfinite(Bg), np.interp(mu_np, qc, qv), np.nan).astype(np.float32)
    np.savez_compressed(out_dir / "static_background.npz", Bg=Bg, qmap=qmap, q_mu=qc, q=qv)
    s_path = config.phase3_dir("background", f"static_bg_{G}.npz")
    S_bg = np.load(s_path)["S"].astype(np.float32) if s_path.exists() else None
    print("background S:", s_path if S_bg is not None else "none (run scripts/phase3_background.py)", flush=True)
    ld = ev.LD(qc, qv, r_ref, mu_np)
    print("limb-darkening profile q(mu):", dict(zip(np.round(qc[::5], 2), np.round(qv[::5], 3))), flush=True)
    fitreg = geo.rho < RHO_MAX
    meta["block"] = list(zip(meta.run, pd.to_datetime(meta.t_last).dt.floor("h")))

    # ---- A. rate / translation fit on train + hold-out pairs ------------------------------------------
    recs = []
    msum, mcnt = {}, {}
    fit = meta[meta.set.isin(["train", "holdout"])]
    for n, (i, row) in enumerate(fit.iterrows()):
        x = np.asarray(X[i], dtype=np.float32)
        y = np.asarray(Y[i], dtype=np.float32)
        F = x.mean(0)
        r = y - F
        mask = np.isfinite(r) & fitreg & (np.abs(r) < R_CLIP)
        ok = np.isfinite(r) & (np.abs(r) < R_CLIP)
        h = int(row.horizon)
        msum[h] = msum.get(h, 0) + np.where(ok, r, 0.0)
        mcnt[h] = mcnt.get(h, 0) + ok
        ps = pair_system(F, r, mask, float(np.mean(row.dt_context_s)), geo.get(b0_of[row.store_target]))
        if ps is not None:
            ps.update(sample=int(i), set=row.set, horizon=int(row.horizon), block=row.block,
                      minutes=float(row.dt_target_s) / 60)
            recs.append(ps)
        if n % 500 == 0:
            print(f"fit pairs {n}/{len(fit)} ({time.time() - t_start:.0f} s)", flush=True)
    npair = fit.horizon.value_counts()
    # a pixel's mean needs it valid in at least half the pairs of that horizon; elsewhere M(H) = 0
    clim = {h: np.where(mcnt[h] >= 0.5 * npair[h], msum[h] / np.maximum(mcnt[h], 1), 0.0).astype(np.float32)
            for h in msum}
    np.savez_compressed(out_dir / "mean_residual_maps.npz", **{f"H{h}": m for h, m in clim.items()})
    rate = {"all": fit_rates(recs)}
    for h in sorted({r["horizon"] for r in recs}):
        rate[f"H{h}"] = fit_rates([r for r in recs if r["horizon"] == h])
    ab = tuple(rate["all"]["ab_deg_day"])
    print("rate offset (deg/day) a + b sin^2:", np.round(ab, 3), "CI a", np.round(rate["all"]["ab_lo"], 3),
          np.round(rate["all"]["ab_hi"], 3), flush=True)
    per_pair = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith(("S_", "u_"))} for r in recs])
    per_pair["block"] = per_pair.block.astype(str)
    per_pair.to_parquet(out_dir / "fit_pairs.parquet", index=False)

    # ---- B. damping (sigma_L, beta) per horizon on the hold-out run ------------------------------------
    ho = meta[meta.set == "holdout"]
    errs = {}
    for i, row in ho.iterrows():
        x = np.asarray(X[i], dtype=np.float32)
        y = np.asarray(Y[i], dtype=np.float32)
        A = x.mean(0)
        valid = np.isfinite(y) & np.isfinite(x).all(0)
        for sg in SIG_L:
            L = ev.nan_blur(A, sg)
            for be in BETAS:
                e = np.abs(L + be * (A - L) - y)[valid]
                errs.setdefault((int(row.horizon), sg, be), []).append(float(e.mean()))
    damp_choice = {}
    for h in sorted(ho.horizon.unique()):
        best = min(((sg, be) for sg in SIG_L for be in BETAS), key=lambda k: np.median(errs[(int(h), *k)]))
        damp_choice[int(h)] = {"sigma_L": best[0], "beta": best[1],
                               "holdout_gain_vs_B1avg": 1 - np.median(errs[(int(h), *best)]) /
                               np.median(errs[(int(h), best[0], 1.0)])}
    print("damping chosen on hold-out:", damp_choice, flush=True)

    # ---- C. validation: baselines, models, diagnostics --------------------------------------------------
    runs = []
    for name in [s for s in a.runs.split(",") if s]:
        d = RUNS / name
        info = json.loads((d / "run.json").read_text())
        m = models.MODELS[info["model"]]()
        m.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
        m.eval()
        runs.append((name, m))
    mu_t = torch.from_numpy(mu_np)[None, None]
    val = meta[meta.set == "val"]
    cache0, cache1 = {}, {}
    vsum, vcnt = {}, {}
    rows, diag = [], []
    fid = store_fr.set_index("store_index").frame_id
    for n, (i, row) in enumerate(val.iterrows()):
        progress.report("why_skill: validation", item=fid.get(int(row.store_target)), i=n, n=len(val),
                        horizon=int(row.horizon))
        x = np.asarray(X[i], dtype=np.float32)
        y = np.asarray(Y[i], dtype=np.float32)
        valid = np.isfinite(y) & np.isfinite(x).all(0)
        h = int(row.horizon)
        b0 = float(b0_of[row.store_target])
        stores = context_store(row, t_store)
        if n < 12:  # the reconstruction at the SU90 rate must reproduce the cached B1-avg
            A0 = rotated_mean(frames, stores, row.dt_context_s, r_ref, b0, None, cache0)
            dev = np.nanmax(np.abs(A0 - x.mean(0))[valid])
            assert dev < 0.02, f"context reconstruction differs by {dev}"
        A = x.mean(0)
        Arot = rotated_mean(frames, stores, row.dt_context_s, r_ref, b0, ab, cache1)
        Ebg = static_shift_error(Bg, row.dt_context_s, r_ref, b0, None, cache0)
        Ebg_rot = static_shift_error(Bg, row.dt_context_s, r_ref, b0, ab, cache1)
        Eq = static_shift_error(qmap, row.dt_context_s, r_ref, b0, None, cache0)
        Ald = np.mean([x[k] * ld.factor(b0, dt) for k, dt in enumerate(row.dt_context_s)], 0)
        Abgm = rotated_mean(frames, stores, row.dt_context_s, r_ref, b0, None, cache0, divisor=Bg)
        Es = static_shift_error(S_bg, row.dt_context_s, r_ref, b0, None, cache0) if S_bg is not None else 0 * A
        dc = damp_choice[h]
        Adamp, L = damp(A, dc["sigma_L"], dc["beta"])
        Arotdamp, _ = damp(Arot, dc["sigma_L"], dc["beta"])
        preds = {"B1": x[-1], "B1-avg": A, "B1-avg-rot": Arot, "B1-avg-damp": Adamp, "B1-avg-rot-damp": Arotdamp,
                 "B1-avg-clim": A + clim[h], "B1-avg-rot-clim": Arot + clim[h],
                 "B1-avg-bg": A + Ebg, "B1-avg-rot-bg": Arot + Ebg_rot, "B1-avg-LDadd": A + Eq,
                 "B1-avg-LD": Ald, "B1-avg-bgmult": Abgm}
        if S_bg is not None:
            preds["B1-avg-bgS"] = A + Es
        with torch.no_grad():
            for name, m in runs:
                # runs trained on background-aware inputs (name ends in _bg) get those inputs
                xin = np.asarray(Xbg[i], dtype=np.float32) if name.endswith("_bg") else x
                vin = np.isfinite(y) & np.isfinite(xin).all(0)
                xt = torch.from_numpy(np.nan_to_num(xin))[None]
                mt = torch.from_numpy(vin.astype(np.float32))[None, None]
                preds[name] = xin[-1] + m(xt, mt, mu_t, torch.tensor([h]))[0, 0].numpy()
        vr = valid & np.isfinite(Arot) & np.isfinite(Ebg_rot) & np.isfinite(Eq) & np.isfinite(Ald) & np.isfinite(Abgm)             & np.isfinite(Es)
        regions = {"trusted": vr & trusted, "disk": vr, "plage": vr & (x[-1] > 1.3)}
        rec = {"sample": int(i), "horizon": h, "run": int(row.run), "t_last": row.t_last,
               "minutes": float(row.dt_target_s) / 60}
        for reg, msk in regions.items():
            if msk.sum() < 200:
                continue
            for name, p in preds.items():
                rec[f"{reg}|{name}"] = float(np.mean(np.abs(p[msk] - y[msk])))
        rows.append(rec)
        # diagnostics on the disk region
        dg = {"sample": int(i), "horizon": h, "minutes": rec["minutes"]}
        for name, p in preds.items():
            e = np.where(vr, p - y, np.nan)
            eL = ev.nan_blur(e, SCALE_PX)
            dg[f"rms_large|{name}"] = float(np.sqrt(np.nanmean(eL[vr] ** 2)))
            dg[f"rms_small|{name}"] = float(np.sqrt(np.nanmean((e - eL)[vr] ** 2)))
        dA = (A - L)[vr]
        for name, _ in runs:
            dg[f"beta|{name}"] = float(np.sum((preds[name] - L)[vr] * dA) / np.sum(dA * dA))
        gb = geo.get(b0)
        fm = vr & fitreg
        for name in ["truth"] + [r[0] for r in runs]:
            tgt = y if name == "truth" else preds[name]
            r_ = tgt - A
            ps = pair_system(A, r_, fm & (np.abs(r_) < R_CLIP), float(np.mean(row.dt_context_s)), gb)
            if ps is not None:
                dg[f"rate|{name}"], dg[f"t_col|{name}"], dg[f"t_row|{name}"] = ps["rate"], ps["t_col"], ps["t_row"]
        for nm, E in (("static-bg", Ebg), ("static-LD", Eq), ("static-S", Es)):
            vs = vsum.setdefault((nm, h), np.zeros((G, G)))
            vs += np.where(vr, np.nan_to_num(E), 0.0)
            vcnt[(nm, h)] = vcnt.get((nm, h), 0) + vr
        Mv = clim[h][vr]
        for name in ["truth"] + [r[0] for r in runs]:
            corr = ((y if name == "truth" else preds[name]) - A)
            c = corr[vr]
            g = float(np.sum(c * Mv) / np.sum(Mv * Mv))
            dg[f"clim_coef|{name}"] = g
            dg[f"clim_share|{name}"] = float(1 - np.sum((c - g * Mv) ** 2) / np.sum(c * c))
            vs = vsum.setdefault((name, h), np.zeros((G, G)))
            vs += np.where(vr, np.nan_to_num(corr), 0.0)
            vcnt[(name, h)] = vcnt.get((name, h), 0) + vr
        diag.append(dg)
        if n % 100 == 0:
            print(f"val {n}/{len(val)} ({time.time() - t_start:.0f} s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(out_dir / "val_errors.parquet", index=False)
    dg = pd.DataFrame(diag)
    dg.to_parquet(out_dir / "val_diagnostics.parquet", index=False)

    # ---- summaries ---------------------------------------------------------------------------------------
    methods = [c.split("|")[1] for c in res.columns if c.startswith("disk|")]
    base = ["B1-avg", "B1-avg-rot", "B1-avg-damp", "B1-avg-rot-damp", "B1-avg-clim", "B1-avg-rot-clim",
            "B1-avg-bg", "B1-avg-rot-bg", "B1-avg-LDadd", "B1-avg-LD", "B1-avg-bgmult", "B1-avg-bgS"]
    base = [b for b in base if f"disk|{b}" in res.columns]
    summ = []
    for reg in ("trusted", "disk", "plage"):
        for h, d in res.groupby("horizon"):
            strongest = min(base, key=lambda b: d[f"{reg}|{b}"].median())
            for mth in methods:
                col = f"{reg}|{mth}"
                dd = d.dropna(subset=[col, f"{reg}|B1-avg", f"{reg}|{strongest}"]).copy()
                dd["s_avg"] = 1 - dd[col] / dd[f"{reg}|B1-avg"]
                dd["s_best"] = 1 - dd[col] / dd[f"{reg}|{strongest}"]
                s1, l1, h1 = ev.block_ci(dd, "s_avg")
                s2, l2, h2 = ev.block_ci(dd, "s_best")
                summ.append({"region": reg, "horizon": h, "minutes": float(dd.minutes.median()), "method": mth,
                             "n": len(dd), "rel_mae": float(dd[col].median()),
                             "skill_vs_B1avg": s1, "lo": l1, "hi": h1, "strongest_baseline": strongest,
                             "skill_vs_strongest": s2, "lo_strongest": l2, "hi_strongest": h2})
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "summary.csv", index=False)
    dsum = dg.groupby("horizon").median(numeric_only=True).drop(columns=["sample"])
    dsum.to_csv(out_dir / "diagnostics_by_horizon.csv")
    trans = per_pair.assign(t_abs=np.hypot(per_pair.t_col, per_pair.t_row),
                            t_abs_only=np.hypot(per_pair.t_col_only, per_pair.t_row_only)) \
        .groupby("horizon")[["rate", "t_col", "t_row", "t_abs", "t_abs_only"]].median()
    trans.to_csv(out_dir / "fit_pairs_by_horizon.csv")

    # ---- figures -----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
    cen = 0.5 * (BANDS[1:] + BANDS[:-1])
    for key, st in [("all", "k-o")] + [(f"H{h}", "--") for h in sorted(meta.horizon.unique())]:
        rr = rate[key]
        yb = np.array(rr["band_deg_day"])
        if key == "all":
            ax[0].errorbar(cen, yb, yerr=[yb - rr["band_lo"], np.array(rr["band_hi"]) - yb], fmt=st, capsize=3,
                           label="all horizons (95 % block CI)")
            lat = np.linspace(-60, 60, 121)
            ax[0].plot(lat, ab[0] + ab[1] * np.sin(np.deg2rad(lat)) ** 2, "r-", lw=1, label="a + b sin²φ fit")
        else:
            ax[0].plot(cen, yb, st, lw=.8, label=key)
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_xlabel("latitude (deg)")
    ax[0].set_ylabel("rate offset vs SU90 (deg/day)")
    ax[0].set_title("Fitted rotation-rate offset (train + hold-out)")
    ax[0].legend(fontsize=7)
    for k, reg in enumerate(("disk", "plage")):
        for mth in methods:
            if mth == "B1":
                continue
            d = summ[(summ.region == reg) & (summ.method == mth)].sort_values("minutes")
            ax[k + 1].errorbar(d.minutes, d.skill_vs_B1avg * 100, yerr=[(d.skill_vs_B1avg - d.lo) * 100,
                                                                        (d.hi - d.skill_vs_B1avg) * 100],
                               marker="o", ms=4, capsize=3, label=mth)
        ax[k + 1].axhline(0, color="k", lw=.5)
        ax[k + 1].set_xscale("log")
        ax[k + 1].set_xlabel("horizon (min)")
        ax[k + 1].set_ylabel("skill vs B1-avg (%)")
        ax[k + 1].set_title(f"{reg}: skill vs B1-avg, 95 % block CI (validation)")
        ax[k + 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "why_skill.png", dpi=80)
    plt.close(fig)

    vmean = {k: np.where(vcnt[k] >= 0.5 * vcnt[k].max(), vsum[k] / np.maximum(vcnt[k], 1), np.nan) for k in vsum}
    np.savez_compressed(out_dir / "val_mean_corrections.npz", **{f"{k[0]}_H{k[1]}": v for k, v in vmean.items()})
    names = ["truth"] + [r[0] for r in runs] + ["static-bg", "static-LD"] + (["static-S"] if S_bg is not None else [])
    hs = sorted(clim)
    map_corr = {}
    fig, ax = plt.subplots(len(hs), 1 + len(names), figsize=(4.2 * (1 + len(names)), 4 * len(hs)))
    for i, h in enumerate(hs):
        disk = np.isfinite(vmean[("truth", h)])
        for j, (lab, img) in enumerate([("train+hold-out mean residual M(H)", clim[h])] +
                                       [(f"val mean {'residual' if n == 'truth' else 'correction ' + n}",
                                         vmean[(n, h)]) for n in names]):
            img = np.nan_to_num(img)
            ax[i, j].imshow(np.where(disk, img, np.nan), origin="lower", cmap="RdBu_r", vmin=-0.02, vmax=0.02)
            ax[i, j].set_title(f"H{h}: {lab}", fontsize=8)
            ax[i, j].axis("off")
            if j > 0:
                map_corr[f"H{h}|{names[j - 1]}"] = float(np.corrcoef(clim[h][disk], img[disk])[0, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "mean_maps.png", dpi=70)
    plt.close(fig)
    print("correlation of val mean maps with M(H):", {k: round(v, 3) for k, v in map_corr.items()}, flush=True)
    out = {"map_corr_with_M": map_corr, "rate_fit": rate, "rate_offset_ab_deg_day": list(ab), "damping": damp_choice,
           "fit_region": {"rho_max": RHO_MAX, "residual_clip": R_CLIP}, "scale_px": SCALE_PX,
           "runs": [r[0] for r in runs], "val_samples": int(len(val)),
           "seconds": round(time.time() - t_start, 1), **CFG["_meta"]}
    (out_dir / "why_meta.json").write_text(json.dumps(out, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(summ[summ.region.isin(["disk", "plage"])][["region", "horizon", "method", "rel_mae", "skill_vs_B1avg",
                                                     "lo", "hi", "strongest_baseline", "skill_vs_strongest",
                                                     "lo_strongest", "hi_strongest"]].round(4).to_string(index=False))
    print(dsum.round(4).T.to_string())
    print(trans.round(4).to_string())


if __name__ == "__main__":
    main()
