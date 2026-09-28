"""How fast can this laptop train without overheating? A before/after test for cooling changes.

    python scripts/thermal_benchmark.py [--minutes 5] [--note "cooling pad, max fan"]

Runs the real training workload (UNetSmall forward + backward on 384² batches of 8, bf16) under the same
duty-cycle thermal controller the pipeline uses (configs/phase3.toml [thermal]) and reports what it can
sustain: mean duty, mean and peak GPU temperature, batches per minute, and the implied time per 1,000
training batches. Every result is appended to outputs/logs/thermal_benchmark.jsonl with the note, so the
effect of each change (cooling pad, raised rear, fan mode, AC power, clean vents, GPU clock cap) can be
compared. The first minute is a warm-up and is not counted. Needs the GPU; do not run next to training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, paths, progress  # noqa: E402
from suitdyn.ml import models, thermal  # noqa: E402

P3 = config.load_phase3()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    TH, G, K, B = P3["thermal"], 384, P3["samples"]["context"], P3["train"]["batch"]
    model = models.UNetSmall(k=K, base=P3["model"]["unet_base"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.rand(B, K, G, G, device="cuda")
    m = torch.ones(B, 1, G, G, device="cuda")
    mu = torch.rand(1, 1, G, G, device="cuda")
    h = torch.full((B,), 40, device="cuda")
    y = torch.rand(B, 1, G, G, device="cuda")
    th = thermal.Thermal(TH["max_c"], TH["resume_c"], TH["target_c"], sync=torch.cuda.synchronize)
    t_start, t_end = time.time(), time.time() + 60 * (a.minutes + 1)
    counted, warm = 0, True
    while time.time() < t_end:
        th.check()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model(x, m, mu, h).float()
        loss = (r - y).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        float(loss)
        if warm and time.time() - t_start > 60:
            warm, t_count = False, time.time()
            th.reset()
        elif not warm:
            counted += 1
        progress.report("thermal benchmark", item=f"{'warm-up' if warm else 'measuring'}; GPU {thermal.gpu_temp():.0f} C; "
                        f"duty {th.d:.2f}", i=int(time.time() - t_start), n=int(t_end - t_start))
    stats = th.reset()
    minutes = (time.time() - t_count) / 60
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": a.note, "minutes_measured": round(minutes, 2),
           "batches_per_min": round(counted / minutes, 1), "minutes_per_1000_batches": round(1000 * minutes / max(counted, 1), 1),
           **stats, "gpu": torch.cuda.get_device_name(0), "thermal_cfg": TH, **config.load()["_meta"]}
    log = paths.OUT / "logs" / "thermal_benchmark.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("note", "batches_per_min", "minutes_per_1000_batches", "duty_mean",
                                         "gpu_temp_mean", "gpu_temp_peak", "thermal_pause_s")}, indent=1))
    prev = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()[:-1] if ln.strip()]
    if prev:
        print(f"previous: {prev[-1]['batches_per_min']} batches/min ({prev[-1].get('note') or 'no note'}); "
              f"now {rec['batches_per_min']} ({np.round(rec['batches_per_min'] / max(prev[-1]['batches_per_min'], 1e-9), 2)}x)")


if __name__ == "__main__":
    main()
