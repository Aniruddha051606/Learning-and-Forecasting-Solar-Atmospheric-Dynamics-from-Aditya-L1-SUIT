"""Phase 3: train one forecaster of the residual over B1 (plain or background-aware).

    python scripts/phase3_train.py --model unet --seed 0 [--inputs bg] [--epochs N] [--max-batches N]
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, paths, progress  # noqa: E402
from suitdyn.ml import data, models, thermal  # noqa: E402

CFG = config.load_dataset()
P3 = config.load_phase3()


def masked_l1(pred_res, base, y, m):
    target = (y - base) * m
    return torch.abs(pred_res * m - target).sum() / m.sum().clamp(min=1)


def run_batches(model, bank, ids, inputs, batch, th, train=False, opt=None, sched=None, clip=1.0, bf16=True,
                report=None):
    """One pass."""
    model.train(train)
    tot, n = 0.0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, s in enumerate(range(0, len(ids), batch)):
            th.check()
            b = bank.batch(ids[s:s + batch], inputs)
            x, mi = data.model_inputs(b)
            m = b["valid"].float()  # the loss mask: target and every context frame finite
            base = x[:, -1:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                r = model(x, mi, bank.mu[None, None], b["h"]).float()
            loss = masked_l1(r, base, torch.nan_to_num(b["y"]), m)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                sched.step()
            w = float(m.sum())
            tot, n = tot + float(loss.detach()) * w, n + w
            if report:
                report(bi, (len(ids) + batch - 1) // batch)
    return tot / max(n, 1)


def main():
    T, M, TH = P3["train"], P3["model"], P3["thermal"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(models.MODELS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--inputs", choices=["plain", "bg"], default=T["inputs"])
    ap.add_argument("--epochs", type=int, default=T["epochs"])
    ap.add_argument("--patience", type=int, default=T["patience"])
    ap.add_argument("--max-batches", type=int, default=0, help="cap batches per epoch (smoke tests only)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    dev = a.device
    name = f"{a.model}_{a.inputs}_s{a.seed}"
    out = paths.runs(name)
    if (out / "run.json").exists():
        print(f"{name} already finished; skipping", flush=True)
        return
    bank = data.Bank(paths.phase3("cache"), dev)
    bg_file = paths.phase3("background", f"static_bg_{bank.G}.npz")
    if a.inputs == "bg":
        data.load_background(bank, bg_file)
    tr, ho = bank.ids("train"), bank.ids("holdout")
    kw = {"unet": {"k": bank.K, "base": M["unet_base"]}, "convlstm": {"k": bank.K, "hidden": M["convlstm_hidden"]}}
    model = models.MODELS[a.model](**kw[a.model]).to(dev)
    per_epoch = int(np.ceil(len(tr) * T["epoch_fraction"] / T["batch"]))
    if a.max_batches:
        per_epoch = min(per_epoch, a.max_batches)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=T["lr"], total_steps=a.epochs * per_epoch, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    th = thermal.controller(TH, dev, sync=torch.cuda.synchronize if dev.startswith("cuda") else None)
    ho_ids = ho if not a.max_batches else ho[:a.max_batches * T["batch"]]
    zero = lambda x, m, mu, h: torch.zeros_like(x[:, -1:])  # noqa: E731  (predicting 0 = the B1 baseline)
    zero.train = zero.eval = lambda *_, **__: None
    b1_holdout = run_batches(zero, bank, ho_ids, a.inputs, 16, th, bf16=T["bf16"])
    log, best, best_ep, bad, start = [], np.inf, -1, 0, 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["cuda_rng"])
        log, best, best_ep, bad, start = ck["log"], ck["best"], ck["best_ep"], ck["bad"], ck["epoch"] + 1
        print(f"resuming {name} at epoch {start}", flush=True)
    t0 = time.time() - (log[-1]["seconds"] if log else 0)
    for ep in range(start, a.epochs):
        if bad >= a.patience:
            break
        order = rng.permutation(tr)[:per_epoch * T["batch"]]
        rep = lambda bi, nb: progress.report(  # noqa: E731
            f"train {name}", item=f"epoch {ep + 1}/{a.epochs}, batch {bi + 1}/{nb}", i=bi, n=nb, epoch=ep,
            gpu_temp=thermal.gpu_temp(), duty=round(th.d, 3), best_holdout_skill=None if not log else round(1 - best / b1_holdout, 4))
        tl = run_batches(model, bank, order, a.inputs, T["batch"], th, True, opt, sched, T["grad_clip"], T["bf16"], rep)
        # the hold-out pass is checked every eval_every epochs (and always at the last one); patience counts
        # epochs, so a check without improvement adds eval_every to it
        every = max(1, int(T.get("eval_every", 1)))
        if (ep + 1) % every and ep + 1 < a.epochs:
            log.append({"epoch": ep, "train_l1": tl, "holdout_l1": None, "holdout_skill_vs_B1": None,
                        "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1), **th.reset()})
            print(json.dumps(log[-1]), flush=True)
        else:
            hl = run_batches(model, bank, ho_ids, a.inputs, 16, th, bf16=T["bf16"])
            log.append({"epoch": ep, "train_l1": tl, "holdout_l1": hl, "holdout_skill_vs_B1": 1 - hl / b1_holdout,
                        "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1), **th.reset()})
            print(json.dumps(log[-1]), flush=True)
            if hl < best:
                best, best_ep, bad = hl, ep, 0
                tmp = out / "best.tmp"
                torch.save(model.state_dict(), tmp)
                tmp.replace(out / "best.pt")
            else:
                bad += every
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
              "log": log, "best": best, "best_ep": best_ep, "bad": bad, "epoch": ep}
        torch.save(ck, out / "last.tmp")
        (out / "last.tmp").replace(out / "last.pt")
    atomic.to_parquet(pd.DataFrame(log), out / "log.parquet")
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    run = {"name": name, "model": a.model, "seed": a.seed, "inputs": a.inputs, "params": models.n_params(model),
           "args": vars(a), "train_cfg": T, "model_cfg": M, "input_mask": "context", "best_epoch": best_ep, "best_holdout_l1": best,
           "b1_holdout_l1": b1_holdout, "best_holdout_skill_vs_B1": 1 - best / b1_holdout, "epochs_run": len(log),
           "checkpoint_sha256": hashlib.sha256((out / "best.pt").read_bytes()).hexdigest(),
           "background_sha256": hashlib.sha256(bg_file.read_bytes()).hexdigest() if a.inputs == "bg" else None,
           "data": {k: bank.meta.get(k) for k in ("grid", "samples", "store", "frames_sha256", "git")},
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
           "smoke_test": bool(a.max_batches), "seconds": round(time.time() - t0, 1), **CFG["_meta"], **P3["_meta"]}
    atomic.write_json(out / "run.json", run)
    print(json.dumps({k: run[k] for k in ("name", "params", "best_epoch", "best_holdout_skill_vs_B1", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
