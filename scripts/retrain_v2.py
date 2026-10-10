"""Addendum I: retrain without the target-validity leak, then evaluate (docs/PREREGISTRATION.md).

    python scripts/retrain_v2.py [--datasets final_offset final_centred]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from suitdyn import atomic, config, paths, pipeline  # noqa: E402

LOG = paths.OUT / "logs" / "retrain_v2"
STATE = LOG / "state.json"
PY = sys.executable
REASON = "v2 models without the target-validity leak: the registered second read (docs/PREREGISTRATION.md, Addendum I)"
KINDS = {"final_offset": ("train", "swap", "val", "newperiod", "test"), "final_centred": ("train", "swap", "val", "test")}
DEPS = {"train": None, "swap": "train", "val": "swap", "newperiod": "val", "test": "val"}


def say(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG / "chain.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def mark(key, status, **kw):
    s = load_state()
    s[key] = {"status": status, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    atomic.write_json(STATE, s)


def cmdline(pid):
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}').CommandLine"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def g_jobs():
    """pids of running Addendum G jobs (scripts/posthoc_new_period.py building its cache, not a re-score)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -like "
                        "'*posthoc_new_period.py*' -and $_.CommandLine -notlike '*--score-only*' -and $_.CommandLine -notlike "
                        "'*multiprocessing*' } | ForEach-Object { $_.ProcessId }"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def g_loaded(pid):
    """Its scoring step has started, so its models are loaded."""
    try:
        return json.loads((paths.progress() / f"{pid}.json").read_text()).get("stage") == "posthoc: new period"
    except Exception:
        return False


def run(step, args, ds, runs_dir=None, tries=3):
    env = {**os.environ, "SUITDYN_DATASET": ds, "PYTHONIOENCODING": "utf-8"}
    env.pop("SUITDYN_RUNS_DIR", None)
    if runs_dir:
        env["SUITDYN_RUNS_DIR"] = runs_dir
    for k in range(tries):
        with open(LOG / f"{step}.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)} (try {k + 1}/{tries})\n")
            f.flush()
            r = subprocess.run([PY, "-u", *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            return True
        say(f"{step}: exit code {r.returncode} (try {k + 1}/{tries}); log outputs/logs/retrain_v2/{step}.log")
        if k + 1 < tries:
            time.sleep(60)
    return False


def base(ds):
    return paths.phase3(name=ds, make=False)


def swapped(ds):
    return (base(ds) / "runs_v1_targetmask").exists() and not (base(ds) / "runs_v2").exists()


def train(ds):
    if swapped(ds):
        return True
    P3 = config.load_phase3()
    for model in P3["model"]["types"]:
        for seed in P3["train"]["seeds"]:
            name = f"{model}_{P3['train']['inputs']}_s{seed}"
            if (base(ds) / "runs_v2" / name / "run.json").exists():
                continue
            say(f"{ds}: training {name} (v2, context-only mask)")
            if not run(f"{ds}_train", ["scripts/phase3_train.py", "--model", model, "--seed", str(seed)], ds, "runs_v2"):
                return False
            say(f"{ds}: {name} finished")
    return True


def move(src, dst):
    for k in range(40):
        try:
            src.rename(dst)
            say(f"moved {src.parent.parent.name}/phase3/{src.name} -> {dst.name}")
            return
        except PermissionError:  # a reader (dashboard, paper script) has a file open: try again
            time.sleep(15)
    src.rename(dst)


def swap(ds):
    if swapped(ds):
        return True
    b = base(ds)
    for src, dst in (("runs", "runs_v1_targetmask"), ("eval", "eval_v1_targetmask"), ("posthoc", "posthoc_v1_targetmask")):
        if (b / src).exists() and not (b / dst).exists():
            move(b / src, b / dst)
    if (b / "eval_test").exists() and not (b / "eval_test_v1_targetmask").exists():
        tmp = b / "eval_test_v1_targetmask.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(b / "eval_test", tmp)
        tmp.rename(b / "eval_test_v1_targetmask")
        say(f"copied {ds}/phase3/eval_test -> eval_test_v1_targetmask (the read record stays in eval_test)")
    if (b / "runs").exists():
        raise RuntimeError(f"{b / 'runs'} still exists after keeping v1; not overwriting it")
    move(b / "runs_v2", b / "runs")
    return True


def val(ds):
    return run(f"{ds}_val", ["scripts/phase3_evaluate.py", "--split", "val"], ds, tries=2)


def newperiod(ds):
    if not (base(ds) / "newperiod" / "cache" / "samples_384.parquet").exists():
        say("newperiod: the Addendum G cache does not exist (test G did not get that far); skipped")
        return None
    ok = True
    for runs_dir, mask, tag in (("runs", "context", "v2"), ("runs_v1_targetmask", "target", "v1_targetmask"),
                                ("runs_v1_targetmask", "context", "v1_ctxmask")):
        ok &= run(f"{ds}_newperiod", ["scripts/posthoc_new_period.py", "--score-only", "--runs-dir", runs_dir,
                                      "--mask", mask, "--tag", tag], ds, tries=2)
    return ok


def test(ds):
    return run(f"{ds}_test", ["scripts/phase3_evaluate.py", "--split", "test", "--reason", REASON, "--allow-new-models"],
               ds, tries=1)


FN = {"train": train, "swap": swap, "val": val, "newperiod": newperiod, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(KINDS))
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    lock = LOG / "chain.lock"
    if lock.exists():
        pid = json.loads(lock.read_text()).get("pid")
        if pid and pid != os.getpid() and pipeline.pid_alive(pid) and "retrain_v2" in cmdline(pid):
            print(f"the chain is already running (pid {pid})")
            return
    atomic.write_json(lock, {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    pipeline.keep_awake(True)

    steps = [(ds, k) for ds in a.datasets for k in KINDS[ds]]
    say(f"chain started (pid {os.getpid()}): {', '.join(f'{d}:{k}' for d, k in steps)}")
    waiting_said = False
    try:
        while True:
            st = load_state()
            status = lambda ds, k: st.get(f"{ds}:{k}", {}).get("status")  # noqa: E731
            pending = [(ds, k) for ds, k in steps if status(ds, k) not in ("done", "failed", "skipped")]
            if not pending:
                break
            chosen, waiting = None, False
            for ds, k in pending:
                dep = DEPS[k]
                if dep and status(ds, dep) != "done":
                    continue
                if k == "test" and "newperiod" in KINDS[ds] and status(ds, "newperiod") not in ("done", "failed", "skipped"):
                    continue
                if ds == "final_offset" and k in ("swap", "newperiod"):
                    jobs = g_jobs()
                    if jobs and (k == "newperiod" or not all(g_loaded(j) for j in jobs)):
                        waiting = True
                        continue
                chosen = (ds, k)
                break
            if chosen is None:
                if waiting:
                    if not waiting_said:
                        say("waiting for test G (it must load its v1 models before they move, and finish before re-scoring)")
                        waiting_said = True
                    time.sleep(120)
                    continue
                break  # everything left depends on a failed step
            ds, k = chosen
            say(f"--- {ds}: {k}")
            t0 = time.time()
            try:
                ok = FN[k](ds)
            except Exception as e:  # noqa: BLE001
                say(f"{ds}: {k} raised {type(e).__name__}: {e}")
                ok = False
            mark(f"{ds}:{k}", "skipped" if ok is None else "done" if ok else "failed", minutes=round((time.time() - t0) / 60, 1))
            say(f"{ds}: {k} {'skipped' if ok is None else 'done' if ok else 'FAILED'} ({(time.time() - t0) / 60:.0f} min)")
        st = load_state()
        say("chain finished: " + ", ".join(f"{d}:{k}={st.get(f'{d}:{k}', {}).get('status', 'not run')}" for d, k in steps))
    finally:
        pipeline.keep_awake(False)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
