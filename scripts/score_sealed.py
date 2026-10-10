"""Score the sealed blind forecasts against the answer frames (docs/PREREGISTRATION.md, Addendum B; test D3).

    python scripts/score_sealed.py --sealed outputs/sealed_forecast/<UTC time>
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS = "final_offset"
K = 5
CADENCE_S = 89.0
IN_ARCHIVE_UNTIL = pd.Timestamp("2026-09-27 18:58:59")
METHODS = ("unet_bg-ens", "convlstm_bg-ens", "unet_bg_s0", "unet_bg_s1", "unet_bg_s2",
           "convlstm_bg_s0", "convlstm_bg_s1", "convlstm_bg_s2")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ts(x):
    """A timestamp as naive UTC (the sealed files use ISO 'Z' strings, the answer index naive UTC)."""
    t = pd.Timestamp(x)
    return t.tz_convert(None) if t.tzinfo is not None else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True)
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = DS
    sys.path.insert(0, str(ROOT))
    import torch
    from suitdyn import paths
    from suitdyn.ml import geometry

    sealed = Path(a.sealed)
    # 1.
    seal = json.loads((sealed / "SEAL.json").read_text())
    files = {p.relative_to(sealed).as_posix(): sha(p) for p in sorted(sealed.rglob("*")) if p.is_file() and p.name != "SEAL.json"}
    digest = hashlib.sha256("".join(f"{k} {v}\n" for k, v in files.items()).encode()).hexdigest()
    if files != seal["files"] or digest != seal["digest_sha256"]:
        bad = sorted(set(files.items()) ^ set(seal["files"].items()))
        sys.exit(f"SEAL BROKEN: the sealed files differ from SEAL.json ({bad[:4]}); nothing scored")
    print(f"seal OK: {len(files)} files, digest {digest} (sealed {seal['sealed_utc']})", flush=True)

    inp = json.loads((sealed / "inputs.json").read_text())
    tgt = json.loads((sealed / "targets.json").read_text())
    ans_dir = sealed.parent / f"answers_{sealed.name}"
    answers = np.load(ans_dir / "answers_384.npy").astype(np.float32)
    aj = json.loads((ans_dir / "answers.json").read_text())
    amap = {ts(t["target_utc"]): t for t in aj["targets"]}
    a_utc = [ts(f["utc"]) for f in aj["frames"]]

    G, r_ref, b0 = int(inp["grid"]), float(inp["r_ref"]), float(inp["b0_deg"])
    cache = np.load(paths.phase3("cache", f"frames_{G}.npy"), mmap_mode="r")
    sidx = np.array(inp["store_index"])
    real = torch.from_numpy(cache[sidx].astype(np.float32))                       # (R, G, G)
    t_real = np.array([ts(t).value / 1e9 for t in inp["origin_utc"]])
    R = len(sidx)
    bgz = np.load(paths.phase3("background", f"static_bg_{G}.npz"))
    S = torch.from_numpy(bgz["S_groups"][inp["background_group"]] if inp["background_group"] is not None else bgz["S"]).float()
    mu = torch.from_numpy(np.load(paths.phase3("cache", f"mu_{G}.npy"))).float()
    qmap = torch.from_numpy(np.interp(mu.numpy(), bgz["q_mu"], bgz["q"]).astype(np.float32))
    qmap = torch.where(torch.isfinite(S), qmap, torch.full_like(qmap, float("nan")))
    clim = {int(k[3:]): torch.from_numpy(bgz[k]).float() for k in bgz.files if k.startswith("M_H")}
    inner = mu > float(np.sqrt(1 - 0.9 ** 2))

    def derot(frames, dt_s):
        """frames (B, Kc, G, G) moved by solar rotation over dt_s (B, Kc) seconds."""
        grid, ok = geometry.derotation_grid(G, r_ref, torch.full((frames.shape[0],), b0), torch.as_tensor(dt_s, dtype=torch.float32))
        return geometry.warp(frames, grid, ok), grid, ok

    def baselines(o, t_ans, h):
        """The physical baselines from the real context ending at origin index o (into the R origin frames)."""
        ctx = real[o - K + 1:o + 1][None]
        dt = (t_ans - t_real[o - K + 1:o + 1])[None]
        xp, grid, ok = derot(ctx, dt)
        A = xp.mean(1)
        out = {"B1": xp[:, -1], "B1-avg": A,
               "B1-avg-LDadd": A + (qmap - geometry.warp_static(qmap, grid, ok, 1, K)).mean(1),
               "B1-avg-bgS": (xp - geometry.warp_static(S, grid, ok, 1, K) + S).mean(1)}
        if h in clim:
            out["B1-avg-clim"] = A + clim[h]
        return {k: v[0] for k, v in out.items()}

    rows = []

    def score(kind, key, t_target, o, h, forecasts):
        m = amap.get(ts(t_target))
        if m is None or m["answer"] is None:
            rows.append({"kind": kind, "key": key, "target_utc": t_target, "scored": False})
            return
        y = torch.from_numpy(answers[m["answer"]])
        t_ans = a_utc[m["answer"]].value / 1e9
        t_t = ts(t_target).value / 1e9
        preds = baselines(o, t_ans, h)
        for name, f in forecasts.items():
            preds[name] = derot(torch.from_numpy(f.astype(np.float32))[None, None], np.array([[t_ans - t_t]]))[0][0, 0]
        valid = torch.isfinite(y) & inner & torch.stack([torch.isfinite(p) for p in preds.values()]).all(0)
        n = int(valid.sum())
        rec = {"kind": kind, "key": key, "target_utc": t_target, "answer_utc": str(a_utc[m["answer"]]), "dt_s": m["dt_s"],
               "scored": n >= 200, "pixels": n, "in_archive_at_sealing": a_utc[m["answer"]] <= IN_ARCHIVE_UNTIL,
               "lead_h": (t_ans - t_real[o]) / 3600}
        for name, p in preds.items():
            rec[name] = float((p - y).abs()[valid].mean()) if n else np.nan
        rows.append(rec)

    # B-short: origins = the last 5 real frames (indices R-5 ..
    hs = [int(h) for h in inp["horizons"]]
    short = {m: np.load(sealed / f"short_{m}.npy") for m in METHODS}
    for k in range(K):
        for hi, h in enumerate(hs):
            score("short", f"h{h}", tgt["short"]["target_utc"][str(h)][k], R - K + k, h, {m: short[m][k, hi] for m in METHODS})
    # B-roll: jump j, chain frame k; lineage origin = real frame R-5+k
    roll = {m: np.load(sealed / f"roll_{m}.npy") for m in METHODS}
    for j, tl in enumerate(tgt["roll"]["target_utc"]):
        for k in range(K):
            score("roll", f"jump{j + 1:02d}", tl[k], R - K + k, 160 if j == 0 else -1, {m: roll[m][j, k] for m in METHODS})

    df = pd.DataFrame(rows)
    out = sealed.parent / f"score_{sealed.name}"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "per_target.csv", index=False)
    summ = []
    sc = df[df.scored.fillna(False).astype(bool)]
    for (kind, key), d in sc.groupby(["kind", "key"]):
        base = [b for b in ("B1", "B1-avg", "B1-avg-LDadd", "B1-avg-bgS", "B1-avg-clim") if b in d and d[b].notna().all()]
        strongest = min(base, key=lambda b: d[b].median())
        for m in METHODS + tuple(base):
            s = 1 - d[m] / d[strongest]
            summ.append({"kind": kind, "key": key, "lead_h": float(d.lead_h.median()), "method": m, "strongest": strongest,
                         "n_targets": len(d), "n_in_archive": int(d.in_archive_at_sealing.sum()),
                         "median_skill": float(s.median()), "min_skill": float(s.min()), "max_skill": float(s.max()),
                         "skill_unseen_only": float(s[~d.in_archive_at_sealing.astype(bool)].median())
                         if (~d.in_archive_at_sealing.astype(bool)).any() else np.nan})
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    lines = [f"# Sealed blind forecasts, scored {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Seal verified: digest {digest} (sealed {seal['sealed_utc']}). Answer frames: {ans_dir.name} "
             f"(validation {aj.get('validation', {})}).", "",
             f"Targets scored: {int(sc.shape[0])} of {len(df)} (the rest had no full-disk frame within 120 s).", "",
             "Median skill over the strongest baseline (same origin frames), disk r < 0.9, % (n targets):", ""]
    for kind in ("short", "roll"):
        t = s[(s.kind == kind) & s.method.isin(["unet_bg-ens", "convlstm_bg-ens"])]
        if len(t):
            p = t.pivot_table(index="key", columns="method", values="median_skill") * 100
            n = t.groupby("key").n_targets.first()
            lead = t.groupby("key").lead_h.first()
            lines += [f"## {kind}", "", "| | lead (h) | n | ConvLSTM ens | UNet ens | strongest |", "|---|---|---|---|---|---|"]
            st = t.groupby("key").strongest.first()
            lines += [f"| {k} | {lead[k]:.1f} | {n[k]} | {p.loc[k].get('convlstm_bg-ens', np.nan):+.1f} | "
                      f"{p.loc[k].get('unet_bg-ens', np.nan):+.1f} | {st[k]} |" for k in p.index] + [""]
    (out / "SCORE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
