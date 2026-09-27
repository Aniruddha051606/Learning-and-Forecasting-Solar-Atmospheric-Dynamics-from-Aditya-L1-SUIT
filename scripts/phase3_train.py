"""Phase 3: train one forecaster of the B1 residual.

    python scripts/phase3_train.py --model unet --seed 0 [--epochs 40] [--batch 8] [--lr 3e-4]

Train on the 'train' samples, early-stop on the 'holdout' run (masked L1 of the residual), never look at
'val' or 'test'. Loss: masked L1 on valid disk pixels (target and all context frames finite, r < 0.95).
Writes outputs/phase3/runs/<model>_s<seed>/: best checkpoint, per-epoch log, run.json (git commit,
config hashes, data provenance, seed, parameters, optimiser, GPU, timings).

Robust to the laptop shutting down (it did, 2026-09-27 13:08, under full GPU load):
  * last.pt after every epoch holds model, optimiser, scheduler, RNG states, log and early-stopping
    state; a rerun of the same command resumes from it (a finished run is skipped);
  * a thermal guard reads the GPU temperature every 20 batches and pauses at >= --max-temp until the
    GPU is back below --resume-temp; peak temperature and pause time are logged per epoch.
"""
import argparse
import hashlib
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
CACHE = config.ROOT / "outputs" / "phase3" / "cache"
RUNS = config.ROOT / "outputs" / "phase3" / "runs"


class Data:
    """Memory-mapped samples; batches assembled on the fly (NaN → 0 with an explicit mask)."""

    def __init__(self, G, subset):
        meta = pd.read_parquet(CACHE / f"samples_{G}.parquet")
        rows = meta.index[meta.set == subset].values
        # The subset is loaded into RAM once and cleaned once (NaN -> 0 plus a validity mask), so a batch is
        # an index + transfer and the float32 cast happens on the GPU. Reading the memmap and cleaning per
        # batch made an epoch ~3x slower than the GPU work.
        X = np.asarray(np.load(CACHE / f"X_{G}.npy", mmap_mode="r")[rows])
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


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


class Thermal:
    def __init__(self, max_temp, resume_temp):
        self.max_temp, self.resume_temp = max_temp, resume_temp
        self.peak, self.paused = 0.0, 0.0

    def check(self):
        t = gpu_temp()
        if t == t:
            self.peak = max(self.peak, t)
        if t == t and t >= self.max_temp:
            torch.cuda.synchronize()
            t0 = time.time()
            while True:
                time.sleep(10)
                t = gpu_temp()
                if not (t == t) or t < self.resume_temp:
                    break
            self.paused += time.time() - t0

    def reset(self):
        out = {"gpu_temp_peak": self.peak, "thermal_pause_s": round(self.paused, 1)}
        self.peak, self.paused = 0.0, 0.0
        return out


def masked_l1(pred_res, x, y, m):
    target = (y - x[:, -1:]) * m
    return (torch.abs(pred_res * m - target)).sum() / m.sum().clamp(min=1)


def evaluate(model, data, mu, device, bs):
    model.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(data.idx), bs):
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--grid", type=int, default=384)
    ap.add_argument("--max-temp", type=float, default=87.0)
    ap.add_argument("--resume-temp", type=float, default=80.0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    dev = "cuda"
    out = RUNS / f"{a.model}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        print(f"{out.name} already finished; skipping", flush=True)
        return
    tr, ho = Data(a.grid, "train"), Data(a.grid, "holdout")
    mu = torch.from_numpy(np.load(CACHE / f"mu_{a.grid}.npy"))[None, None].to(dev)
    model = models.MODELS[a.model]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * int(np.ceil(len(tr.idx) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    b1_holdout = None
    with torch.no_grad():
        # the holdout loss of predicting a zero residual = baseline B1
        zero = type("Z", (), {"eval": lambda s: None, "train": lambda s: None,
                              "__call__": lambda s, x, m, mu_, h: torch.zeros_like(x[:, -1:])})()
        b1_holdout = evaluate(zero, ho, mu, dev, 16)
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
    thermal = Thermal(a.max_temp, a.resume_temp)
    t0 = time.time() - (log[-1]["seconds"] if log else 0)
    for ep in range(start, a.epochs):
        if bad >= a.patience:
            break
        order = rng.permutation(tr.idx)
        tl, nb = 0.0, 0
        for bi, s in enumerate(range(0, len(order), a.batch)):
            if bi % 20 == 0:
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
        hl = evaluate(model, ho, mu, dev, 16)
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
                    "prepare_git": prep.get("git")},
           "seconds": round(time.time() - t0, 1), **CFG["_meta"]}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    print(json.dumps({k: run[k] for k in ("model", "seed", "params", "best_epoch", "best_holdout_skill_vs_B1",
                                         "seconds")}, indent=1))


if __name__ == "__main__":
    main()
