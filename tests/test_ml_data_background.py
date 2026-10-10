"""The on-the-fly sample bank and the static-background solver, on a small synthetic Sun."""
import json

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from suitdyn import baselines, normalize
from suitdyn.ml import background as bgm
from suitdyn.ml import data

G, R_REF, B0, CAD = 64, 28.75, 7.0, 87.0 * 40  # 40 "frames" per step so rotation moves ~1.3 px at this grid


def _synthetic_cache(tmp_path, S_true, n_runs=6, run_len=16, seed=0):
    """Frames = a solar field rotating with the Sun + a static background S_true fixed on the grid."""
    rng = np.random.default_rng(seed)
    c = (G - 1) / 2
    v, u = np.indices((G, G))
    disk = np.hypot(u - c, v - c) / R_REF < 0.95
    frames, run_of = [], []
    for r in range(n_runs):
        base = gaussian_filter(rng.normal(size=(G, G)), 1.5) * 0.3 + 1.0
        for k in range(run_len):
            # the base field rotated forward by k steps (sources behind the limb filled with the mean level)
            sol = baselines.rotated_persistence(base, R_REF, B0, k * CAD)
            sol = np.where(np.isfinite(sol), sol, 1.0) + rng.normal(0, 0.003, (G, G))
            frames.append(np.where(disk, sol + S_true, np.nan))
            run_of.append(r)
    frames = np.array(frames, np.float16)
    K, rows = 3, []
    for h in (2, 4):
        for last in range(len(frames) - h):
            first, r = last - K + 1, run_of[last]
            if first < 0 or run_of[first] != r or run_of[last + h] != r:
                continue
            ctx = list(range(first, last + 1))
            rows.append({"set": "train" if r < n_runs - 1 else "holdout", "horizon": h, "run": r, "ctx": ctx,
                         "tgt": last + h, "dt_context_s": [(last + h - j) * CAD for j in ctx], "b0": B0,
                         "t_last": pd.Timestamp("2026-09-20") + pd.Timedelta(hours=r * 12, seconds=last * CAD),
                         "t_target": pd.Timestamp("2026-09-20"), "dt_target_s": h * CAD, "target_frame": f"f{last + h}"})
    np.save(tmp_path / f"frames_{G}.npy", frames)
    pd.DataFrame(rows).to_parquet(tmp_path / f"samples_{G}.parquet")
    np.save(tmp_path / f"mu_{G}.npy", normalize.mu_map(G, R_REF).astype(np.float32))
    np.save(tmp_path / f"trusted_{G}.npy", disk)
    (tmp_path / "prepare_meta.json").write_text(json.dumps({"grid": G, "r_ref": R_REF}))
    return disk


def test_bank_batches_and_background_identity(tmp_path):
    _synthetic_cache(tmp_path, np.zeros((G, G)))
    bank = data.Bank(tmp_path, "cpu")
    b = bank.batch(bank.ids("train")[:4], "plain")
    assert b["x"].shape == (4, bank.K, G, G) and b["y"].shape == (4, 1, G, G)
    bank.set_background(np.zeros((G, G), np.float32))
    b = bank.batch(bank.ids("train")[:4], "bg")
    m = torch.isfinite(b["x_bg"]) & torch.isfinite(b["x_plain"])
    assert torch.allclose(b["x_bg"][m], b["x_plain"][m], atol=1e-5)  # S = 0: both derotations agree


def test_background_solver_recovers_a_planted_static_pattern(tmp_path):
    c = (G - 1) / 2
    v, u = np.indices((G, G))
    x = (u - c) / R_REF
    S_true = (0.6 * x - 0.3 * x ** 2).astype(np.float32)  # an east-west ramp like the offset-mode vignetting
    _synthetic_cache(tmp_path, S_true)
    bank = data.Bank(tmp_path, "cpu")
    train, ho = bank.ids("train"), bank.ids("holdout")
    M, geo = bgm.mean_residuals(bank, train, clip=1.0)
    pos = bgm.disk_index(G, R_REF, 0.95)
    ops = {h: bgm.horizon_operator(G, R_REF, dts, b0, pos) for h, (dts, b0) in geo.items()}
    # lambda chosen on the hold-out run inside rho < 0.9, as the pipeline does (the outer ring is poorly
    # constrained: all target pixels whose rotation source leaves the disk lie there)
    scans = {lam: bgm.score(bank, ho, bgm.solve(M, ops, pos, lam=lam)[0], rho_max=0.9) for lam in (0.01, 0.1, 1.0)}
    lam = min(scans, key=lambda k: np.mean([v["B1-avg-bgS"] for v in scans[k].values()]))
    sc, best = scans[lam], bgm.score(bank, ho, S_true, rho_max=0.9)
    for h, v_ in sc.items():
        # the static-background error is mostly removed, and nearly as well as with the true pattern
        assert v_["B1-avg-bgS"] < 0.5 * v_["B1-avg"], (h, v_)
        assert v_["B1-avg-bgS"] < 1.25 * best[h]["B1-avg-bgS"], (h, v_, best[h])
