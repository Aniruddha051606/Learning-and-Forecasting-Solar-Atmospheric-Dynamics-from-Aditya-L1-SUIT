"""Phase 2 fix: is the one-frame noise floor instrumental or partly solar (3-min chromospheric oscillation)?

    python scripts/phase2_floor_origin.py --store v0

Structure function of the rotation-corrected difference, D(tau) = median over frame pairs of the relative
MAE between B1(F(t)) and F(t+tau) on the disk, for every pair of frames in the same run with
tau up to ~10 min. Instrument noise, registration residuals and slow solar evolution make D rise
monotonically with tau. A strong oscillation with period P makes D dip near tau = P (back in
phase) after a maximum near P/2. Mg II k shows ~3-min oscillations, so the test is a dip near 180 s.

The 87-s cadence samples tau = 87, 174, 261 s; the short 21-s-cadence stretch of 23 Sep (training
split; used here only for this physical test, never for model evaluation) samples it finely.
Response correction and per-frame median normalisation are applied as in the baselines.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import baselines, config, normalize, response  # noqa: E402

CFG = config.load_phase2()
STORES = config.ROOT / "outputs" / "phase2" / "stores"
OUT = config.ROOT / "outputs" / "phase2" / "floor_origin"


def loader(g, resp, pointing, f):
    def load(i):
        img = g["nb03/image"][i].astype(np.float32)
        img[g["nb03/mask"][i] != 0] = np.nan
        n = img.shape[0] // f
        img = np.nanmean(img[:n * f, :n * f].reshape(n, f, n, f), axis=(1, 3))
        x0, y0 = pointing[i]
        return img * response.factor(resp, img.shape[0], x0, y0)
    return load


def structure(fr, load, r_ref, grid, max_tau_s, max_pairs_per_lag=400, rng=None):
    mu = normalize.mu_map(grid, r_ref)
    disk = mu > np.sqrt(1 - 0.9 ** 2)
    quiet_cache = {}
    rows = []
    pairs = []
    for run, g in fr.groupby("run"):
        g = g.sort_values("t")
        t = g.t.values
        idx = g.store_index.values
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                tau = (t[b] - t[a]) / np.timedelta64(1, "s")
                if tau > max_tau_s:
                    break
                pairs.append((idx[a], idx[b], tau))
    pairs = pd.DataFrame(pairs, columns=["a", "b", "tau"])
    pairs["lag_bin"] = np.round(pairs.tau / 10) * 10
    sel = pairs.groupby("lag_bin", group_keys=False).apply(
        lambda d: d.sample(min(len(d), max_pairs_per_lag), random_state=0))
    for r in sel.itertuples():
        for k in (r.a, r.b):
            if k not in quiet_cache:
                im = load(k)
                quiet_cache[k] = im / np.nanmedian(im[disk])
        pred = baselines.rotated_persistence(quiet_cache[r.a], r_ref, 7.0, r.tau)
        tr = quiet_cache[r.b]
        m = disk & np.isfinite(pred) & np.isfinite(tr)
        e = np.abs(pred[m] - tr[m])
        bright = tr[m] > 1.3
        rows.append({"tau": r.tau, "d_all": float(e.mean()), "d_quiet": float(e[~bright].mean()),
                     "d_bright": float(e[bright].mean()) if bright.any() else np.nan})
        if len(quiet_cache) > 400:
            quiet_cache.clear()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="v0")
    ap.add_argument("--grid-factor", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(STORES / f"{a.store}.zarr"), mode="r")
    fr = pd.read_parquet(STORES / f"{a.store}.frames.parquet")
    resp = response.load(config.ROOT / "outputs" / "phase2" / "response" / f"response_{a.store}.npz")
    pointing = {int(r.store_index): (float(r.reg_x0), float(r.reg_y0)) for r in fr.itertuples()}
    load = loader(g, resp, pointing, a.grid_factor)
    grid = g["nb03/image"].shape[1] // a.grid_factor
    r_ref = float(g.attrs["r_ref"]) / a.grid_factor

    val = fr[fr.split == "val"]
    d87 = structure(val, load, r_ref, grid, max_tau_s=600)
    # the fast-cadence stretch: consecutive training frames closer than 40 s
    tr = fr[fr.split == "train"].sort_values("t").reset_index(drop=True)
    fast = tr.t.diff().dt.total_seconds() < 40
    fast = fast | fast.shift(-1, fill_value=False)
    seg = tr[fast].copy()
    seg["run"] = (seg.t.diff().dt.total_seconds() > 60).cumsum()
    d21 = structure(seg, load, r_ref, grid, max_tau_s=420) if len(seg) > 4 else pd.DataFrame()

    def curve(d):
        if d.empty:
            return pd.DataFrame()
        d = d.copy()
        d["lag"] = np.round(d.tau / 21.75) * 21.75 if d.tau.min() < 60 else np.round(d.tau / 87) * 87
        return d.groupby("lag").agg(n=("tau", "size"), tau=("tau", "median"), d_all=("d_all", "median"),
                                    d_quiet=("d_quiet", "median"), d_bright=("d_bright", "median")).reset_index()

    c87, c21 = curve(d87), curve(d21)
    c87.to_csv(OUT / "structure_87s_val.csv", index=False)
    if not c21.empty:
        c21.to_csv(OUT / "structure_21s_train.csv", index=False)

    def dip(c, col):
        """Is there a local minimum near 150-210 s after a maximum at shorter lag?"""
        if c.empty:
            return None
        near = c[(c.tau > 140) & (c.tau < 220)]
        before = c[(c.tau > 60) & (c.tau <= 140)]
        if near.empty or before.empty:
            return None
        return {"max_60_140s": float(before[col].max()), "min_140_220s": float(near[col].min()),
                "dip": bool(near[col].min() < before[col].max())}

    summary = {"fast_cadence_frames": int(len(seg)), "fast_cadence_span": [str(seg.t.min()), str(seg.t.max())]
               if len(seg) else None,
               "val_87s": c87.round(5).to_dict("records"), "train_21s": c21.round(5).to_dict("records"),
               "dip_test_87s": {c: dip(c87, c) for c in ("d_all", "d_quiet", "d_bright")},
               "dip_test_21s": {c: dip(c21, c) for c in ("d_all", "d_quiet", "d_bright")}, **CFG["_meta"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c, lab, mk in ((c87, "87 s cadence (val)", "o-"), (c21, "21 s cadence (train stretch)", "s--")):
        if c.empty:
            continue
        for col, colr in (("d_quiet", "C0"), ("d_bright", "C3")):
            ax.plot(c.tau, c[col] * 100, mk, color=colr, ms=4, label=f"{lab}: {col[2:]}")
    ax.axvline(180, color="k", ls=":", lw=.8)
    ax.set_xlabel("time separation τ (s)")
    ax.set_ylabel("median |B1(F(t)) − F(t+τ)| (% of disk level)")
    ax.set_title("Structure function of the rotation-corrected difference (dotted: 180 s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "structure_function.png", dpi=80)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("fast_cadence_frames", "fast_cadence_span", "dip_test_87s",
                                              "dip_test_21s")}, indent=1, default=float))
    print(c87.round(4).to_string(index=False))
    print(c21.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
