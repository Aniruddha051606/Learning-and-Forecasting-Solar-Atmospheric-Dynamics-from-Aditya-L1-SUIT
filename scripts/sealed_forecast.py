"""Sealed blind forecasts beyond the data (docs/PREREGISTRATION.md, Addendum B).

    python scripts/sealed_forecast.py [--jumps 18] [--device cuda] [--out DIR]

The 6 frozen final_offset models get ONLY frames we already have: the last continuous observing stretch of
final_offset (its frame cache ends 2026-09-27 14:58:26 UT, so nothing later can be read). They forecast
  B-short  from each of the last 5 frames, at the trained horizons 20/40/80/160 frames (target = origin + h x 89 s);
  B-roll   chained jumps of the trained horizon 160 (3.96 h each): jump 1 from the last 4 x jumps + 5 real frames,
           every later jump from 5-frame contexts made only of the previous jump's forecasts. Each seed rolls its
           own forecasts; an architecture's ensemble is the mean of its seeds' rollouts.
Writes outputs/sealed_forecast/<UTC time>/: forecast arrays (float16, the 384 frame-cache grid and units, NaN = no
forecast), target times, checksums of the models, cache, background and this script, previews, and SEAL.json (the
SHA-256 of every file and one digest over them). It changes nothing in the pipeline's outputs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = "final_offset"
CADENCE_S = 89.0  # median frame spacing of the training samples (88.6-89.1 s by horizon)
HORIZONS = (20, 40, 80, 160)
K = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumps", type=int, default=18)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", help="output folder (default outputs/sealed_forecast/<UTC time>)")
    a = ap.parse_args()
    os.environ["SUITDYN_DATASET"] = DATASET
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    import torch
    from PIL import Image
    from suitdyn import config, paths
    from suitdyn.ml import data, geometry, models

    t_start = datetime.now(timezone.utc)
    out = Path(a.out) if a.out else ROOT / "outputs" / "sealed_forecast" / t_start.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    bf16 = dev.type == "cuda" and tomllib.loads((ROOT / "configs/phase3.toml").read_text(encoding="utf-8"))["train"]["bf16"]

    # the frozen models, background and frame cache of final_offset
    bank = data.Bank(paths.phase3("cache"), dev)
    bg_file = paths.phase3("background", f"static_bg_{bank.G}.npz")
    data.load_background(bank, bg_file)
    G, r_ref, mu = bank.G, bank.r_ref, bank.mu[None, None]
    runs = []
    for d in sorted(paths.runs().glob("*")):
        if not (d / "run.json").exists():
            continue
        info = json.loads((d / "run.json").read_text())
        kw = {"unet": {"k": K, "base": info["model_cfg"]["unet_base"]},
              "convlstm": {"k": K, "hidden": info["model_cfg"]["convlstm_hidden"]}}[info["model"]]
        m = models.MODELS[info["model"]](**kw).to(dev)
        m.load_state_dict(torch.load(d / "best.pt", map_location=dev))
        m.eval()
        runs.append({"name": info["name"], "kind": info["model"], "inputs": info["inputs"], "m": m, "sha256": sha(d / "best.pt")})
    assert len(runs) == 6 and all(r["inputs"] == "bg" for r in runs), [r["name"] for r in runs]

    # the last continuous observing stretch (frame spacing <= 300 s), its newest R frames
    seqf = pd.read_parquet(paths.sequences("frames.parquet")).sort_values("t").reset_index(drop=True)
    store = pd.read_parquet(paths.stores(f"{DATASET}.frames.parquet")).set_index("frame_id")
    gaps = np.flatnonzero((seqf.t.diff().dt.total_seconds() > 300).values)
    stretch = seqf.iloc[gaps[-1] if len(gaps) else 0:]
    R = 4 * a.jumps + K
    if len(stretch) < max(R, 2 * K - 1):
        sys.exit(f"the last stretch has {len(stretch)} frames; {R} are needed for {a.jumps} jumps")
    real = stretch.iloc[-R:]
    sidx = store.store_index.reindex(real.frame_id).to_numpy()
    assert sidx.max() < len(bank.frames) and int(sidx.max()) == len(store) - 1, "origin is not the newest cached frame"
    t_real = (real.t - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()  # UTC seconds (t is naive UTC)
    man = pd.read_parquet(paths.archive("manifest.parquet"), columns=["file", "HGLT_OBS"]).set_index("file")
    b0 = float(man.HGLT_OBS.reindex([real.frame_id.iloc[-1]]).iloc[0])
    px, py = float(store.reg_x0.loc[real.frame_id.iloc[-1]]), float(store.reg_y0.loc[real.frame_id.iloc[-1]])
    if bank.S_groups is not None:
        g = int(np.argmin(np.hypot(bank.centres[:, 0] - px, bank.centres[:, 1] - py)))
        S = bank.S_groups[g]
    else:
        g, S = None, bank.S
    frames_real = bank.frames[torch.as_tensor(sidx)].to(dev).float()  # (R, G, G)
    nan = torch.tensor(float("nan"), device=dev)

    def forecast(m, ctx, t_ctx, t_tgt, h):
        """ctx (B, K, G, G) frames, t_ctx (B, K) and t_tgt (B,) UTC seconds: the model's forecast at t_tgt."""
        B = ctx.shape[0]
        dt = torch.as_tensor(t_tgt[:, None] - t_ctx, dtype=torch.float32, device=dev)
        grid, ok = geometry.derotation_grid(G, r_ref, torch.full((B,), b0, device=dev), dt)
        x = geometry.warp(ctx, grid, ok) - geometry.warp_static(S, grid, ok, B, K) + S  # background-aware inputs
        mi = torch.isfinite(x).all(1, keepdim=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            r = m(torch.nan_to_num(x), mi.float(), mu, torch.full((B,), h, device=dev)).float()[:, 0]
        return torch.where(mi[:, 0], x[:, -1] + r, nan)

    def run_windows(m, frames, times, h):
        """Forecast from every 5-frame window of `frames` (L, G, G): (L-4, G, G) and their target times."""
        L = frames.shape[0]
        starts = np.arange(L - K + 1)
        t_tgt = times[starts + K - 1] + h * CADENCE_S
        outs = []
        for i in range(0, len(starts), a.batch):
            s = starts[i:i + a.batch]
            ctx = torch.stack([frames[j:j + K] for j in s])
            outs.append(forecast(m, ctx, np.stack([times[j:j + K] for j in s]), t_tgt[i:i + a.batch], h))
        return torch.cat(outs), t_tgt

    short, roll, short_t, roll_t = {}, {}, None, None
    for r in runs:
        # B-short: origins = the last 5 real frames
        fs = []
        for h in HORIZONS:
            f, tt = run_windows(r["m"], frames_real[-(2 * K - 1):], t_real[-(2 * K - 1):], h)
            fs.append(f.cpu().numpy())
            short_t = (short_t or {}) | {h: tt.tolist()}
        short[r["name"]] = np.stack(fs, 1)  # (5 origins, 4 horizons, G, G)
        # B-roll: chained jumps of the trained horizon 160
        cur, cur_t, keep, keep_t = frames_real, t_real, [], []
        for j in range(a.jumps):
            cur, cur_t = run_windows(r["m"], cur, cur_t, 160)
            keep.append(cur[-K:].cpu().numpy())
            keep_t.append(cur_t[-K:].tolist())
        roll[r["name"]], roll_t = np.stack(keep), keep_t  # (jumps, 5, G, G)
        print(f"{r['name']}: short {short[r['name']].shape}, roll {roll[r['name']].shape}", flush=True)
    for kind in ("unet", "convlstm"):
        names = [r["name"] for r in runs if r["kind"] == kind]
        short[f"{kind}_bg-ens"] = np.mean([short[n] for n in names], 0)
        roll[f"{kind}_bg-ens"] = np.mean([roll[n] for n in names], 0)

    iso = lambda s: datetime.fromtimestamp(float(s), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    for name in short:
        np.save(out / f"short_{name}.npy", short[name].astype(np.float16))
        np.save(out / f"roll_{name}.npy", roll[name].astype(np.float16))
    (out / "targets.json").write_text(json.dumps({
        "short": {"axes": "origin (5, oldest first) x horizon (20, 40, 80, 160 frames)", "origin_utc": [iso(t) for t in t_real[-K:]],
                  "target_utc": {str(h): [iso(t) for t in v] for h, v in short_t.items()}},
        "roll": {"axes": "jump (1..N) x chain frame (5, oldest first)", "jump_hours": 160 * CADENCE_S / 3600,
                 "target_utc": [[iso(t) for t in v] for v in roll_t]}}, indent=1))
    meta = json.loads((paths.phase3("cache") / "prepare_meta.json").read_text())
    (out / "inputs.json").write_text(json.dumps({
        "registration": "docs/PREREGISTRATION.md, Addendum B", "dataset": DATASET, "created_utc": iso(t_start.timestamp()),
        "newest_real_frame_utc": iso(t_real[-1]), "newest_real_frame": real.frame_id.iloc[-1],
        "origin_frames": real.frame_id.tolist(), "origin_utc": [iso(t) for t in t_real], "store_index": sidx.tolist(),
        "cadence_s": CADENCE_S, "horizons": list(HORIZONS), "jumps": a.jumps, "b0_deg": b0, "pointing_px": [px, py],
        "background_group": g, "background_sha256": sha(bg_file), "frame_cache_sha256": meta.get("frames_sha256"),
        "grid": G, "r_ref": r_ref, "units": "frame-cache units (as the models see them); NaN = no forecast",
        "models": {r["name"]: r["sha256"] for r in runs}, "ensembles": "mean of the 3 seeds of each architecture",
        "script_sha256": sha(__file__), "git": config.git_state(), "torch": torch.__version__, "device": str(dev), "bf16": bf16},
        indent=1, default=str))

    def png(arr, path):
        v = arr[np.isfinite(arr)]
        lo, hi = np.percentile(v, [2, 99.8]) if v.size else (0, 1)
        x = np.clip((np.nan_to_num(arr, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
        Image.fromarray((255 * np.arcsinh(6 * x) / np.arcsinh(6)).astype(np.uint8)[::-1]).save(path)
    prev = out / "previews"
    prev.mkdir()
    for kind in ("unet", "convlstm"):
        e = f"{kind}_bg-ens"
        png(short[e][-1, -1], prev / f"{e}_short_h160.png")
        for j in sorted({1, a.jumps // 3, 2 * a.jumps // 3, a.jumps} - {0}):
            png(roll[e][j - 1, -1], prev / f"{e}_roll_jump{j:02d}.png")
    png(frames_real[-1].cpu().numpy(), prev / "newest_real_frame.png")

    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    digest = hashlib.sha256("".join(f"{k} {v}\n" for k, v in files.items()).encode()).hexdigest()
    sealed = datetime.now(timezone.utc)
    (out / "SEAL.json").write_text(json.dumps({
        "sealed_utc": iso(sealed.timestamp()), "digest_sha256": digest, "files": files,
        "statement": f"Forecasts made by the frozen final_offset models from frames up to {iso(t_real[-1])} only; "
                     "sealed before any real image of the target times was on our disk (Addendum B)."}, indent=1))
    print(f"sealed {len(files)} files in {out}\ndigest {digest}\nsealed at {iso(sealed.timestamp())} "
          f"({time.time() - t_start.timestamp():.0f} s)")


if __name__ == "__main__":
    main()
