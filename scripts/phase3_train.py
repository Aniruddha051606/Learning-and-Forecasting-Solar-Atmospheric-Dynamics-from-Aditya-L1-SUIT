"""Phase 3: train one forecaster of the B1 residual.

    python scripts/phase3_train.py --model unet --seed 0 [--epochs 40] [--batch 8] [--lr 3e-4]

Train on the 'train' samples, early-stop on the 'holdout' run (masked L1 of the residual), never look at
'val' or 'test'. Loss: masked L1 on valid disk pixels (target and all context frames finite, r < 0.95).
Writes outputs/phase3/runs/<model>_s<seed>/: best checkpoint, per-epoch log, run.json (git commit,
config hashes, data provenance, seed, parameters, optimiser, GPU, timings).

Robust to the laptop shutting down (it did, 2026-09-27 13:08, under full GPU load):
  * last.pt after every epoch holds model, optimiser, scheduler, RNG states, log and early-stopping
    state; a rerun of the same command resumes from it (a finished run is skipped);
  * a duty-cycle thermal controller (class Thermal) holds the GPU near --target-temp (default 75 C) with
    short sleeps between batches, and pauses at --max-temp (85 C) as a hard stop;
  * it reads the GPU temperature before EVERY batch, training and hold-out evaluation (in-process NVML, ~0.02 ms; nvidia-smi
    every 5 batches only if NVML is unavailable) and pauses at >= --max-temp until the GPU is back below
    --resume-temp; peak temperature and pause time are logged per epoch. Sampling every 5 batches through
    nvidia-smi overshot an 84 C trip point to 88-94 C (96 C with a second GPU job running): the same
    range as the shutdown. Never run another GPU job while training.
"""
import argparse
import ctypes
import hashlib
import os
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config  # noqa: E402
from suitdyn.ml import models  # noqa: E402

CFG = config.load_phase2()
CACHE = config.phase3_dir("cache")
RUNS = config.phase3_dir("runs")


class Data:
    """Memory-mapped samples; batches assembled on the fly (NaN → 0 with an explicit mask)."""

    def __init__(self, G, subset, inputs="plain"):
        meta = pd.read_parquet(CACHE / f"samples_{G}.parquet")
        rows = meta.index[meta.set == subset].values
        # The subset is loaded into RAM once and cleaned once (NaN -> 0 plus a validity mask), so a batch is
        # an index + transfer and the float32 cast happens on the GPU. Reading the memmap and cleaning per
        # batch made an epoch ~3x slower than the GPU work.
        # inputs "bg": context derotated around the static background (scripts/phase3_prepare_bg.py)
        X = np.asarray(np.load(CACHE / (f"X_{G}.npy" if inputs == "plain" else f"X_{G}_{inputs}.npy"),
                               mmap_mode="r")[rows])
        Y = np.asarray(np.load(CACHE / f"Y_{G}.npy", mmap_mode="r")[rows])
        self.valid = (np.isfinite(Y) & np.isfinite(X).all(1)).astype(np.uint8)
        self.X = np.nan_to_num(X, copy=False)
        self.Y = np.nan_to_num(Y, copy=False)
        self.h = meta.horizon.values[rows]
        self.idx = np.arange(len(rows))

    def batch(self, ids, device):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)  # noqa: E731
        return (t(self.X[ids]).float(), t(self.Y[ids]).float()[:, None], t(self.valid[ids]).float()[:, None],
                t(self.h[ids]))


class _NVML:
    """GPU temperature from the driver's NVML library, in-process (no child process per reading)."""

    def __init__(self):
        self.h = None
        try:
            lib = ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll")) \
                if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
            h = ctypes.c_void_p()
            if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0:
                self.lib, self.h = lib, h
        except Exception:
            self.h = None

    def temp(self):
        t = ctypes.c_uint()
        if self.h is None or self.lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(t)) != 0:
            return float("nan")
        return float(t.value)


NVML = _NVML()


def gpu_temp():
    t = NVML.temp()
    if t == t:
        return t
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


class Thermal:
    """Keeps the GPU near a target temperature with a smooth duty cycle, plus a hard stop.

    On/off pausing alone did not work: every restart after a pause ran the GPU at full power, and the
    die temperature jumped from below the trip point to 95-96 C before the next reading (the sensor
    updates more slowly than the die heats). Now, before every batch, the duty fraction d is lowered
    when the GPU is above `target` and raised slowly below it, and the loop sleeps
    work_time * (1/d - 1) after each batch, so the GPU runs steadily at part power. `max_temp` remains
    a hard stop (pause until below `resume_temp`, then halve d). Start low and ramp slowly: a synthetic
    load test went from 54 C to 90 C within 10 s at d = 0.5, and this laptop held ~72 C at d ~ 0.07."""

    def __init__(self, max_temp, resume_temp, target, d0=0.1, d_min=0.03):
        self.max_temp, self.resume_temp, self.target = max_temp, resume_temp, target
        self.d, self.d_min = d0, d_min
        self.last = None
        self._zero()

    def _zero(self):
        self.peak, self.paused, self.slept, self.temps, self.duties = 0.0, 0.0, 0.0, [], []

    def check(self):
        now = time.time()
        work = 0.0 if self.last is None else now - self.last
        t = gpu_temp()
        if t == t:
            self.peak = max(self.peak, t)
            self.temps.append(t)
            if t >= self.max_temp:
                torch.cuda.synchronize()
                t0 = time.time()
                while True:
                    time.sleep(5)
                    t = gpu_temp()
                    if not (t == t) or t < self.resume_temp:
                        break
                self.paused += time.time() - t0
                self.d = max(self.d_min, self.d * 0.5)
                work = 0.0
            elif t > self.target + 4:
                self.d = max(self.d_min, self.d * 0.8)
            elif t > self.target:
                self.d = max(self.d_min, self.d * 0.95)
            elif t < self.target - 2:
                self.d = min(1.0, self.d * 1.01)
        self.duties.append(self.d)
        pause = min(2.0, work * (1 / self.d - 1))
        if pause > 0:
            time.sleep(pause)
            self.slept += pause
        self.last = time.time()

    def reset(self):
        out = {"gpu_temp_peak": self.peak, "gpu_temp_mean": round(float(np.mean(self.temps)), 1) if self.temps else None,
               "duty_mean": round(float(np.mean(self.duties)), 3) if self.duties else None,
               "thermal_pause_s": round(self.paused, 1), "throttle_sleep_s": round(self.slept, 1)}
        self._zero()
        return out


def masked_l1(pred_res, x, y, m):
    target = (y - x[:, -1:]) * m
    return (torch.abs(pred_res * m - target)).sum() / m.sum().clamp(min=1)


def evaluate(model, data, mu, device, bs, thermal=None):
    model.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(data.idx), bs):
            if thermal is not None:  # the hold-out pass is ~40 full-load batches: it overshot to 90 C unchecked
                thermal.check()
            ids = data.idx[s:s + bs]
            x, y, m, h = data.batch(ids, device)
            r = model(x, m, mu, h).float()
            target = (y - x[:, -1:]) * m
            tot += float((torch.abs(r * m - target)).sum())
            n += float(m.sum())
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(models.MODELS), default="unet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", choices=["plain", "bg"], default="plain",
                    help="bg: background-aware derotation; the residual target is then relative to that B1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--grid", type=int, default=384)
    ap.add_argument("--target-temp", type=float, default=75.0, help="duty-cycle controller setpoint (C)")
    ap.add_argument("--max-temp", type=float, default=85.0, help="hard stop (C)")
    ap.add_argument("--resume-temp", type=float, default=72.0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    dev = "cuda"
    out = RUNS / (f"{a.model}_s{a.seed}" + ("" if a.inputs == "plain" else f"_{a.inputs}"))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        print(f"{out.name} already finished; skipping", flush=True)
        return
    tr, ho = Data(a.grid, "train", a.inputs), Data(a.grid, "holdout", a.inputs)
    mu = torch.from_numpy(np.load(CACHE / f"mu_{a.grid}.npy"))[None, None].to(dev)
    model = models.MODELS[a.model]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * int(np.ceil(len(tr.idx) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    thermal = Thermal(a.max_temp, a.resume_temp, a.target_temp)
    b1_holdout = None
    with torch.no_grad():
        # the holdout loss of predicting a zero residual = baseline B1
        zero = type("Z", (), {"eval": lambda s: None, "train": lambda s: None,
                              "__call__": lambda s, x, m, mu_, h: torch.zeros_like(x[:, -1:])})()
        b1_holdout = evaluate(zero, ho, mu, dev, 16, thermal)
    log, best, best_ep, bad, start = [], np.inf, -1, 0, 0
    if (out / "last.pt").exists():
        # CPU first: the RNG states must stay CPU ByteTensors; model/optimiser states move to the GPU on load
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        log, best, best_ep, bad, start = ck["log"], ck["best"], ck["best_ep"], ck["bad"], ck["epoch"] + 1
        print(f"resuming {out.name} at epoch {start}", flush=True)
    t0 = time.time() - (log[-1]["seconds"] if log else 0)
    for ep in range(start, a.epochs):
        if bad >= a.patience:
            break
        order = rng.permutation(tr.idx)
        tl, nb = 0.0, 0
        for bi, s in enumerate(range(0, len(order), a.batch)):
            if NVML.h is not None or bi % 5 == 0:
                thermal.check()
            x, y, m, h = tr.batch(np.sort(order[s:s + a.batch]), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = model(x, m, mu, h).float()
            loss = masked_l1(r, x, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss.detach())
            nb += 1
        hl = evaluate(model, ho, mu, dev, 16, thermal)
        log.append({"epoch": ep, "train_l1": tl / nb, "holdout_l1": hl, "holdout_skill_vs_B1": 1 - hl / b1_holdout,
                    "seconds": round(time.time() - t0, 1), **thermal.reset()})
        print(json.dumps(log[-1]), flush=True)
        if hl < best:
            best, best_ep, bad = hl, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        # write-then-rename so a shutdown mid-save cannot leave a corrupt checkpoint
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "log": log, "best": best, "best_ep": best_ep,
                    "bad": bad, "epoch": ep}, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")
        if bad >= a.patience:
            break
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    prep = json.loads((CACHE / "prepare_meta.json").read_text())
    run = {"model": a.model, "seed": a.seed, "params": models.n_params(model), "args": vars(a),
           "best_epoch": best_ep, "best_holdout_l1": best, "b1_holdout_l1": b1_holdout,
           "best_holdout_skill_vs_B1": 1 - best / b1_holdout, "epochs_run": len(log),
           "checkpoint_sha256": hashlib.sha256((out / "best.pt").read_bytes()).hexdigest(),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "data": {"grid": prep["grid"], "samples": prep["samples"], "store": prep["store"],
                    "prepare_git": prep.get("git"), "inputs": a.inputs,
                    "background": (json.loads((CACHE / "prepare_bg_meta.json").read_text())
                                   if a.inputs == "bg" else None)},
           "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    print(json.dumps({k: run[k] for k in ("model", "seed", "params", "best_epoch", "best_holdout_skill_vs_B1",
                                         "seconds")}, indent=1))


if __name__ == "__main__":
    main()
