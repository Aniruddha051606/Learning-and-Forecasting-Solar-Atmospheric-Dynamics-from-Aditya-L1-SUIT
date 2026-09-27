"""Phase 3: the static (non-rotating) background S on the sample grid, estimated from derotation residuals.

    python scripts/phase3_background.py [--lambdas 0.1,0.3,1,3,10,30]

B1 derotates the whole frame, including what is fixed on the registered grid: limb darkening, the
seams, the large-scale detector pattern (PHASE3 §3.1). Write each frame as F = solar + S. The mean
residual of the context-mean baseline over many pairs of horizon H is then
    M(H) = mean(target - B1-avg) ~ S - Rbar_H S,
where Rbar_H is the mean over the context frames of the derotation operator. Solar evolution
averages out over many pairs; S does not. S is solved from all horizons jointly by least squares with
a gradient penalty lambda. This is a Kuhn-Lin-Loranz-type estimate, with solar rotation (not
pointing) as the known image shift. S is determined only up to functions of latitude alone, which
derotation leaves unchanged anyway. The model is additive; PHASE3 §3.2 found additive LD better than
multiplicative (plage does not scale with the quiet-Sun limb darkening).

No validation data. lambda is chosen on the early-stop hold-out run with S fitted on train pairs
only (score: median hold-out MAE of B1-avg-bgS = B1-avg + mean_k (S - rot_k S), per horizon). The
adopted S is then refit on train + hold-out with that lambda.
Writes outputs/phase3/background/: static_bg_<G>.npz (S, mask, fit), background_meta.json,
background.png.
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
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import lsqr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase3_evaluate as ev  # noqa: E402
import phase3_why_skill as wy  # noqa: E402
from suitdyn import baselines, config, progress  # noqa: E402

CFG, G, CACHE = ev.CFG, ev.G, ev.CACHE
OUT = config.phase3_dir("background")
R_CLIP = 0.25       # residuals above this are dropped from M(H) (brightenings, spikes)
RHO_MAX = 0.95      # the sample disk


def mean_residual(meta, X, Y, rows):
    """M(H) per horizon over the given samples (clipped mean, valid where >= half of the pairs)."""
    acc = {}
    for i in rows:
        h = int(meta.loc[i, "horizon"])
        x = np.asarray(X[i], dtype=np.float32)
        r = np.asarray(Y[i], dtype=np.float32) - x.mean(0)
        ok = np.isfinite(r) & (np.abs(r) < R_CLIP)
        s, c, n = acc.get(h, (0.0, 0, 0))
        acc[h] = (s + np.where(ok, r, 0.0), c + ok, n + 1)
    return {h: np.where(c >= 0.5 * n, s / np.maximum(c, 1), np.nan) for h, (s, c, n) in acc.items()}


def rot_matrix(r_ref, b0, dt, pos):
    """Sparse bilinear derotation operator on the disk pixels (rows: target pixel, cols: source pixel);
    rows whose source is off the disk are all-zero and flagged invalid."""
    (rows, cols), ok = baselines.derotation_coords(G, r_ref, b0, dt)
    tgt = np.flatnonzero(pos.ravel() >= 0)
    rr, cc = rows.ravel()[tgt], cols.ravel()[tgt]
    r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
    fr, fc = rr - r0, cc - c0
    good = ok.ravel()[tgt] & (r0 >= 0) & (c0 >= 0) & (r0 + 1 < G) & (c0 + 1 < G)
    I, J, V = [], [], []
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        src = np.full(len(tgt), -1)
        src[good] = pos[r0[good] + dr, c0[good] + dc]
        good &= (src >= 0) | (w < 1e-9)
        I.append(np.arange(len(tgt)))
        J.append(np.maximum(src, 0))
        V.append(np.where(src >= 0, w, 0.0))
    n = len(tgt)
    R = sparse.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, n))
    return R, good


def horizon_operator(meta, rows, h, r_ref, pos):
    """Rbar_H for the median context times and B0 of the pairs of horizon h."""
    d = meta.loc[rows]
    d = d[d.horizon == h]
    dts = np.median(np.stack(d.dt_context_s.values), 0)
    b0 = float(np.median(d.b0))
    mats = [rot_matrix(r_ref, b0, float(dt), pos) for dt in dts]
    Rbar = sum(m[0] for m in mats) / len(mats)
    good = np.logical_and.reduce([m[1] for m in mats])
    return Rbar, good, dts.tolist(), b0


def gradient_matrix(pos):
    n = int((pos >= 0).sum())
    I, J, V, k = [], [], [], 0
    for a, b in ((pos[:, :-1], pos[:, 1:]), (pos[:-1, :], pos[1:, :])):
        m = (a >= 0) & (b >= 0)
        cnt = int(m.sum())
        I += [np.arange(k, k + cnt)] * 2
        J += [a[m], b[m]]
        V += [np.ones(cnt), -np.ones(cnt)]
        k += cnt
    return sparse.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(k, n))


def solve_S(Ms, ops, Dg, lam, pos, x0=None):
    blocks, rhs = [], []
    for h, M in Ms.items():
        Rbar, good = ops[h][0], ops[h][1]
        m = M.ravel()[pos.ravel() >= 0]
        use = good & np.isfinite(m)
        D = (sparse.identity(Rbar.shape[0], format="csr") - Rbar)[use]
        blocks.append(D)
        rhs.append(m[use])
    A = sparse.vstack(blocks + [np.sqrt(lam) * Dg]).tocsr()
    b = np.concatenate(rhs + [np.zeros(Dg.shape[0])])
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=4000, x0=x0)
    S = np.full((G, G), np.nan, np.float32)
    S[pos >= 0] = sol[0]
    fit_rms = float(np.sqrt(np.mean((A[:len(b) - Dg.shape[0]] @ sol[0] - b[:len(b) - Dg.shape[0]]) ** 2)))
    return S, sol[0], {"iterations": int(sol[2]), "fit_rms": fit_rms, "istop": int(sol[1])}


def score(meta, X, Y, rows, S, r_ref, cache):
    """Median per-horizon MAE of B1-avg and B1-avg-bgS on the given samples."""
    out = {}
    for i in rows:
        row = meta.loc[i]
        x = np.asarray(X[i], dtype=np.float32)
        y = np.asarray(Y[i], dtype=np.float32)
        A = x.mean(0)
        E = wy.static_shift_error(S, row.dt_context_s, r_ref, float(row.b0), None, cache)
        v = np.isfinite(y) & np.isfinite(x).all(0) & np.isfinite(E)
        d = out.setdefault(int(row.horizon), {"B1-avg": [], "B1-avg-bgS": []})
        d["B1-avg"].append(float(np.abs(A - y)[v].mean()))
        d["B1-avg-bgS"].append(float(np.abs(A + E - y)[v].mean()))
    return {h: {k: float(np.median(v)) for k, v in d.items()} for h, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="0.1,0.3,1,3,10,30")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(CACHE / f"samples_{G}.parquet")
    X = np.load(CACHE / f"X_{G}.npy", mmap_mode="r")
    Y = np.load(CACHE / f"Y_{G}.npy", mmap_mode="r")
    mu = np.load(CACHE / f"mu_{G}.npy")
    prep = json.loads((CACHE / "prepare_meta.json").read_text())
    r_ref = float(prep["r_ref"])
    store_fr = pd.read_parquet(config.ROOT / "outputs" / "phase2" / "stores" / f"{prep['store']}.frames.parquet")
    man = pd.read_parquet(config.out_dir(CFG) / "manifest.parquet", columns=["file", "HGLT_OBS"]).set_index("file")
    b0_of = store_fr.set_index("store_index").frame_id.map(man.HGLT_OBS).astype(float)
    meta["b0"] = b0_of.reindex(meta.store_target.values).values

    rho = np.sqrt(np.clip(1 - mu ** 2, 0, None))
    disk = (mu > 0) & (rho < RHO_MAX)
    pos = np.full((G, G), -1)
    pos[disk] = np.arange(int(disk.sum()))
    Dg = gradient_matrix(pos)
    train = meta.index[meta.set == "train"].values
    ho = meta.index[meta.set == "holdout"].values
    hs = sorted(meta.horizon.unique())

    # 1. lambda on the hold-out run, S from train pairs only
    M_tr = mean_residual(meta, X, Y, train)
    ops_tr = {h: horizon_operator(meta, train, h, r_ref, pos) for h in hs}
    print(f"train M(H) and operators ready ({time.time() - t0:.0f} s)", flush=True)
    scores, x0, cache = {}, None, {}
    lams = [float(v) for v in a.lambdas.split(",")]
    for k, lam in enumerate(lams):
        progress.report("phase3_background: lambda scan", item=f"lambda = {lam}", i=k, n=len(lams), every_s=0)
        S, x0, info = solve_S(M_tr, ops_tr, Dg, lam, pos, x0)
        sc = score(meta, X, Y, ho, S, r_ref, cache)
        gain = {h: 1 - sc[h]["B1-avg-bgS"] / sc[h]["B1-avg"] for h in sc}
        scores[lam] = {"holdout_gain_vs_B1avg": gain, "mean_gain": float(np.mean(list(gain.values()))), **info}
        print(f"lambda {lam}: hold-out gain vs B1-avg {({h: round(g * 100, 2) for h, g in gain.items()})} "
              f"iters {info['iterations']} ({time.time() - t0:.0f} s)", flush=True)
    lam = max(scores, key=lambda k: scores[k]["mean_gain"])

    # 2. adopted S: train + hold-out, chosen lambda
    both = np.concatenate([train, ho])
    M_all = mean_residual(meta, X, Y, both)
    ops = {h: horizon_operator(meta, both, h, r_ref, pos) for h in hs}
    S, _, info = solve_S(M_all, ops, Dg, lam, pos, x0)
    # interpretation only: how much of S is the limb-darkening profile (least squares S ~ c0 + c1 q(mu))
    qc, qv = ev.ld_profile(np.load(CACHE / f"frames_{G}.npy", mmap_mode="r"),
                           store_fr.loc[store_fr.split == "train", "store_index"].values[::10], mu)
    q = np.interp(mu, qc, qv)
    m = np.isfinite(S)
    coef = np.linalg.lstsq(np.c_[np.ones(m.sum()), q[m]], S[m], rcond=None)[0]
    inst = np.where(m, S - coef[0] - coef[1] * q, np.nan).astype(np.float32)
    r2 = 1 - np.nanvar(inst) / np.nanvar(S)
    np.savez_compressed(OUT / f"static_bg_{G}.npz", S=S, instrument_part=inst, disk=disk,
                        **{f"M_H{h}": M_all[h].astype(np.float32) for h in hs})
    meta_out = {"grid": G, "r_ref": r_ref, "lambda_chosen": lam, "lambda_scores": scores, "final_fit": info,
                "pairs": {"train": int(len(train)), "holdout": int(len(ho))},
                "operators": {int(h): {"median_context_dt_s": ops[h][2], "b0_deg": ops[h][3]} for h in hs},
                "ld_fit": {"c0": float(coef[0]), "c1": float(coef[1]), "r2_of_S": float(r2)},
                "model": "additive; F = solar + S; S up to functions of latitude", "r_clip": R_CLIP,
                "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    (OUT / "background_meta.json").write_text(json.dumps(meta_out, indent=1, default=str))

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for k, (title, img, lim) in enumerate([("static background S", S, None),
                                           (f"instrument part: S - ({coef[0]:.2f} + {coef[1]:.2f} q(mu))", inst, 0.03),
                                           ("M(H=160), train + hold-out", M_all[160], 0.02),
                                           ("S - Rbar_160 S (what S predicts for M)", None, 0.02)]):
        if img is None:
            v = np.full(G * G, np.nan)
            vv = np.nan_to_num(S[pos >= 0]) - ops[160][0] @ np.nan_to_num(S[pos >= 0])
            vv[~ops[160][1]] = np.nan
            v[(pos >= 0).ravel()] = vv
            img = v.reshape(G, G)
        kw = {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim} if lim else {"cmap": "gray"}
        im = ax[k].imshow(img, origin="lower", **kw)
        ax[k].set_title(title, fontsize=9)
        ax[k].axis("off")
        plt.colorbar(im, ax=ax[k], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "background.png", dpi=80)
    plt.close(fig)
    print(json.dumps({k: meta_out[k] for k in ("lambda_chosen", "final_fit", "ld_fit", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
