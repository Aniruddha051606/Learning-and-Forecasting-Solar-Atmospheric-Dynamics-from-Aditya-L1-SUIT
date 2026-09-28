"""The SUIT-DYN pipeline runner: one command from the raw archive to the evaluated models.

    python -m suitdyn plan   [--dataset c0]                 what would run, what is up to date, and why
    python -m suitdyn run    [--dataset c0] [--until STAGE] [--only STAGE] [--force STAGE] [--skip-archive]
                             [--smoke] [--detach]
    python -m suitdyn status [--dataset c0]                 state of every stage from the last runs

Stages (scripts in scripts/, each also runnable alone):
  archive  manifest -> frames                      frame-local, shared by every data set, incremental
  dataset  registration -> sequences -> calibration -> store -> noise_maps -> response -> noise_maps_resp
           -> samples -> background -> train:<model>:<seed> ... -> evaluate
Robustness:
  * fingerprint per stage = the stage command, the code it runs (the script and every suitdyn module it
    imports, transitively; progress/atomic/paths excluded because they cannot change a result), the config
    files it reads, the data set's exact input files (manifest SHA-256s inside its time span), and the
    fingerprints of the stages it depends on. A stage is skipped when its last run succeeded with the same
    fingerprint and its outputs exist; anything upstream that changed makes it run again.
  * crash safety: products are written atomically by the scripts; a store build that was interrupted with the
    same fingerprint resumes (--resume); training resumes from last.pt; a training run whose fingerprint
    changed is moved to runs/_superseded/ (never deleted) so it is not silently reused.
  * preflight: free disk for the stages to run, the raw archive reachable, a CUDA GPU for GPU stages, no other
    runner on the same data set (lock file with PID; a stale lock is taken over).
  * network-bound stages are retried (share hiccups); Windows is kept awake while the runner runs; every
    stage has its own log (outputs/pipeline/<name>/logs) and state (outputs/pipeline/<name>/state).
  * --smoke: a quick end-to-end check (few batches, few evaluation samples); its runs and evaluation live in
    runs_smoke/ and eval_smoke/, so a smoke test can never be taken for the real result.
  * --detach: the runner restarts itself outside this terminal (WMI Win32_Process Create with a new process
    group and no window), so it survives the session that launched it.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import atomic, config, paths, progress

ROOT = config.ROOT
PY = sys.executable
NON_SEMANTIC = {"progress", "atomic", "paths"}


@dataclass
class Stage:
    name: str
    script: str
    args: list = field(default_factory=list)
    deps: list = field(default_factory=list)
    scope: str = "dataset"          # archive | dataset
    outputs: list = field(default_factory=list)
    configs: list = field(default_factory=list)
    disk_gb: float = 0.1
    gpu: bool = False
    raw: bool = False               # reads raw FITS from the archive
    retries: int = 0
    resume_arg: str = ""


def stages(ds, smoke=False):
    P3 = config.load_phase3()
    G = 1536 // P3["samples"]["grid_factor"]
    inputs = P3["train"]["inputs"]
    runs_dir = "runs_smoke" if smoke else "runs"
    p1 = ["configs/phase1.toml"]
    d = p1 + [f"configs/datasets/{ds}.toml"]
    d3 = d + ["configs/phase3.toml"]
    resp = str(paths.phase2("response", f"response_{ds}.npz", name=ds))
    st = [
        Stage("manifest", "scripts/build_manifest.py", ["--span-of", ds], scope="archive", outputs=[paths.archive("manifest.parquet")],
              configs=p1, raw=True, retries=2),
        Stage("frames", "scripts/process_frames.py", ["--span-of", ds], deps=["manifest"], scope="archive",
              outputs=[paths.archive("frames_full.parquet")], configs=p1, raw=True, retries=2, disk_gb=0.3),
        Stage("registration", "scripts/registration_study.py", deps=["frames"],
              outputs=[paths.phase1("registration.parquet", name=ds)], configs=d, raw=True, retries=1),
        Stage("sequences", "scripts/build_sequences.py", deps=["registration"],
              outputs=[paths.sequences(f, name=ds) for f in ("frames.parquet", "windows.parquet", "test_seal.json")],
              configs=d),
        Stage("calibration", "scripts/calibrate_pattern.py", deps=["sequences"],
              outputs=[paths.calibration("nb03_pattern.npy", name=ds)], configs=d, raw=True, retries=1, disk_gb=6),
        Stage("store", "scripts/build_store.py", deps=["calibration", "sequences"],
              outputs=[paths.stores(f"{ds}.frames.parquet", name=ds), paths.stores(f"{ds}.summary.json", name=ds)],
              configs=d, raw=True, retries=2, disk_gb=12, resume_arg="--resume"),
        Stage("noise_maps", "scripts/phase2_noise_maps.py", ["--split", "train"], deps=["store"],
              outputs=[paths.phase2("noise_maps", f"noise_maps_{ds}_train_g2.npz", name=ds)], configs=d),
        Stage("response", "scripts/phase2_response.py", deps=["noise_maps"],
              outputs=[paths.phase2("response", f"response_{ds}.npz", name=ds)], configs=d),
        Stage("noise_maps_resp", "scripts/phase2_noise_maps.py", ["--split", "train", "--response", resp, "--tag", "resp"],
              deps=["response"], outputs=[paths.phase2("noise_maps", f"noise_maps_{ds}_train_g2_resp.npz", name=ds)],
              configs=d),
        Stage("samples", "scripts/phase3_prepare.py", deps=["noise_maps_resp", "response", "store"],
              outputs=[paths.phase3("cache", f, name=ds) for f in (f"frames_{G}.npy", f"samples_{G}.parquet", "prepare_meta.json")],
              configs=d3, disk_gb=2),
        Stage("background", "scripts/phase3_background.py", deps=["samples"], gpu=True,
              outputs=[paths.phase3("background", f"static_bg_{G}.npz", name=ds)], configs=d3),
    ]
    train = []
    for model in P3["model"]["types"]:
        for seed in P3["train"]["seeds"][:1] if smoke else P3["train"]["seeds"]:
            args = ["--model", model, "--seed", str(seed)] + (["--max-batches", "20", "--epochs", "2"] if smoke else [])
            name = f"train:{model}:{seed}"
            train.append(name)
            st.append(Stage(name, "scripts/phase3_train.py", args, deps=["background"], gpu=True, configs=d3,
                            outputs=[paths.phase3(runs_dir, f"{model}_{inputs}_s{seed}", "run.json", name=ds)]))
    st.append(Stage("evaluate", "scripts/phase3_evaluate.py", ["--max-samples", "64"] if smoke else [], deps=train,
                    gpu=True, configs=d3,
                    outputs=[paths.phase3("eval_smoke" if smoke else "eval", "summary.csv", name=ds)]))
    return {s.name: s for s in st}


# ------------------------------------------------------------------------------------------ fingerprints
def _module_file(mod):
    parts = mod.split(".")
    if parts[0] != "suitdyn":
        return None
    p = ROOT.joinpath(*parts)
    return p.with_suffix(".py") if p.with_suffix(".py").exists() else (p / "__init__.py" if (p / "__init__.py").exists() else None)


def code_files(script):
    """The script and every suitdyn module it imports, transitively (results-relevant code only)."""
    todo, seen = [ROOT / script], set()
    while todo:
        f = todo.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        tree = ast.parse(f.read_text(encoding="utf-8"))
        pkg = ".".join(f.relative_to(ROOT).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    up = pkg.split(".")[:len(pkg.split(".")) - node.level + 1]
                    base = ".".join(up + ([base] if base else []))
                mods = [base] + [f"{base}.{a.name}" for a in node.names]
            for m in mods:
                if m.split(".")[-1] in NON_SEMANTIC:
                    continue
                mf = _module_file(m)
                if mf:
                    todo.append(mf)
    return sorted(seen)


def dataset_inputs_hash(ds):
    """SHA-256 over the raw files inside the data set's time span (what its products are built from)."""
    man = paths.archive("manifest.parquet")
    if not man.exists():
        return "no-manifest"
    cfg = config.load_dataset(ds)
    lo, hi = config.dataset_span(cfg)
    m = pd.read_parquet(man, columns=["t", "sha256"])
    m = m[(m.t >= lo) & (m.t <= hi)]
    return hashlib.sha256("\n".join(sorted(m.sha256.dropna())).encode()).hexdigest()


def fingerprint(stage, fps, ds, smoke):
    h = hashlib.sha256()
    h.update(json.dumps([stage.script, stage.args, smoke]).encode())
    for f in code_files(stage.script):
        h.update(f.relative_to(ROOT).as_posix().encode() + f.read_bytes())
    for c in stage.configs:
        h.update((ROOT / c).read_bytes())
    if stage.scope == "dataset" and stage.name == "registration":
        h.update(dataset_inputs_hash(ds).encode())
    for d in stage.deps:
        h.update(fps[d].encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ state
def state_path(ds, stage):
    return paths.pipeline("state", stage.replace(":", "_") + ".json", name=ds)


def read_state(ds, stage):
    p = state_path(ds, stage)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def outputs_ok(stage):
    return all(Path(o).exists() for o in stage.outputs)


def plan(ds, smoke=False, force=(), only=None, until=None, frm=None, skip_archive=False):
    st = stages(ds, smoke)
    names = list(st)
    fps, rows = {}, []
    sel = set(names)
    def last_index(prefix):
        hits = [i for i, n in enumerate(names) if n == prefix or n.startswith(prefix + ":")]
        if not hits:
            raise SystemExit(f"unknown stage {prefix!r}; stages: {', '.join(names)}")
        return hits
    if until:
        sel = set(names[:max(last_index(until)) + 1])
    if frm:
        sel &= set(names[min(last_index(frm)):])
    if only:
        sel = {n for n in names if n == only or n.startswith(only + ":")}
    for n in names:
        s = st[n]
        fps[n] = fingerprint(s, fps, ds, smoke)
        prev = read_state(ds, n)
        up_to_date = prev.get("status") == "done" and prev.get("fingerprint") == fps[n] and outputs_ok(s)
        if n not in sel:
            action, why = "skip", "not selected"
        elif s.scope == "archive" and skip_archive:
            action, why = "skip", "--skip-archive"
        elif n in force or any(n.startswith(f + ":") for f in force):
            action, why = "run", "forced"
        elif s.scope == "archive":
            action, why = "run", "incremental archive refresh (only new files are processed)"
        elif up_to_date:
            action, why = "skip", "up to date"
        elif not prev:
            action, why = "run", "never run"
        elif prev.get("fingerprint") != fps[n]:
            action, why = "run", "code, config, inputs or an upstream stage changed"
        elif not outputs_ok(s):
            action, why = "run", "outputs missing"
        else:
            action, why = "run", f"last status: {prev.get('status')}"
        rows.append({"stage": n, "action": action, "why": why, "fingerprint": fps[n][:12], "gpu": s.gpu,
                     "disk_gb": s.disk_gb, "last": prev.get("status", "")})
    return st, fps, rows


# ------------------------------------------------------------------------------------------ running
def keep_awake(on):
    try:
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def pid_alive(pid):
    try:
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259
    except Exception:
        return False


def preflight(ds, st, todo):
    problems = []
    if not config.dataset_config_path(ds).exists():
        problems.append(f"no data-set config configs/datasets/{ds}.toml")
    need = sum(st[n].disk_gb for n in todo) + 5
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < need:
        problems.append(f"only {free:.0f} GB free on the project drive; the stages to run need ~{need:.0f} GB")
    if any(st[n].raw for n in todo):
        raw = config.load()["paths"]["raw_root"]
        if not os.path.isdir(raw):
            problems.append(f"raw archive not reachable: {raw}")
    if any(st[n].gpu for n in todo):
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("a GPU stage is selected but CUDA is not available")
        except Exception as e:
            problems.append(f"torch not importable: {e}")
    return problems


TRANSIENT = ("OSError", "ConnectionError", "ConnectionResetError", "TimeoutError", "BrokenPipeError",
             "BrokenProcessPool", "PermissionError", "WinError 53", "WinError 59", "WinError 64", "WinError 121",
             "network", "timed out")


def transient(log):
    """Retry only failures that can go away by themselves (the share, the network, a locked file). A code error
    such as a TypeError fails the same way every time: retrying it once cost 1.5 h (2026-09-28)."""
    tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    last = next((ln for ln in reversed(tail) if ln and not ln.startswith(" ")), "")
    return any(t in last for t in TRANSIENT) or not any("Error" in ln or "Traceback" in ln for ln in tail)


def run_stage(ds, s, fp, smoke, log_dir):
    prev = read_state(ds, s.name)
    args = list(s.args)
    if s.resume_arg and prev.get("fingerprint") == fp and prev.get("status") in ("running", "failed", "interrupted"):
        args.append(s.resume_arg)
    if s.name.startswith("train:") and prev.get("fingerprint") not in (None, fp):
        for o in s.outputs:
            run_dir = Path(o).parent
            if run_dir.exists():
                dst = run_dir.parent / "_superseded" / f"{run_dir.name}_{time.strftime('%Y%m%dT%H%M%S')}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                run_dir.rename(dst)
    env = {**os.environ, "SUITDYN_DATASET": ds, "PYTHONIOENCODING": "utf-8", **({"SUITDYN_SMOKE": "1"} if smoke else {})}
    log = log_dir / f"{s.name.replace(':', '_')}.log"
    rec = {"stage": s.name, "fingerprint": fp, "status": "running", "started": time.time(), "args": args,
           "attempts": 0, "git": config.git_state(), "smoke": smoke}
    for attempt in range(s.retries + 1):
        rec["attempts"] = attempt + 1
        atomic.write_json(state_path(ds, s.name), rec)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {s.name} attempt {attempt + 1}: "
                    f"{s.script} {' '.join(args)}\n")
            f.flush()
            r = subprocess.run([PY, str(ROOT / s.script), *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                               creationflags=0x08000000 if os.name == "nt" else 0)
        if r.returncode == 0 and outputs_ok(s):
            rec.update(status="done", ended=time.time(), seconds=round(time.time() - rec["started"], 1),
                       outputs={str(o): (Path(o).stat().st_size if Path(o).is_file() else None) for o in s.outputs})
            atomic.write_json(state_path(ds, s.name), rec)
            return True
        if attempt < s.retries and transient(log):
            time.sleep(60 * (attempt + 1))
            if s.resume_arg and s.resume_arg not in args:
                args.append(s.resume_arg)
    tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
    rec.update(status="failed", ended=time.time(), seconds=round(time.time() - rec["started"], 1),
               returncode=r.returncode, outputs_missing=[str(o) for o in s.outputs if not Path(o).exists()], log_tail=tail)
    atomic.write_json(state_path(ds, s.name), rec)
    return False


def cmd_run(a):
    ds = a.dataset
    lock = paths.pipeline("runner.lock", name=ds)
    if lock.exists():
        try:
            other = json.loads(lock.read_text())
        except Exception:
            other = {}
        if other.get("pid") and pid_alive(other["pid"]) and other["pid"] != os.getpid():
            sys.exit(f"another runner (pid {other['pid']}) is working on data set {ds}")
    atomic.write_json(lock, {"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv})
    try:
        st, fps, rows = plan(ds, a.smoke, set(a.force or []), a.only, a.until, a.frm, a.skip_archive)
        todo = [r["stage"] for r in rows if r["action"] == "run"]
        print(pd.DataFrame(rows)[["stage", "action", "why", "last"]].to_string(index=False), flush=True)
        problems = preflight(ds, st, todo)
        if problems:
            sys.exit("preflight failed:\n  " + "\n  ".join(problems))
        keep_awake(True)
        log_dir = paths.pipeline("logs", name=ds)
        t0 = time.time()
        for k, n in enumerate(todo):
            progress.report(f"pipeline {ds}", item=n, i=k, n=len(todo), every_s=0)
            print(f"[{time.strftime('%H:%M:%S')}] {n} ...", flush=True)
            if not run_stage(ds, st[n], fps[n], a.smoke, log_dir):
                s = read_state(ds, n)
                print("\n".join(s.get("log_tail", [])), flush=True)
                sys.exit(f"stage {n} FAILED after {s.get('attempts')} attempt(s); log: {log_dir / (n.replace(':', '_') + '.log')}")
            print(f"[{time.strftime('%H:%M:%S')}] {n} done ({read_state(ds, n).get('seconds')} s)", flush=True)
        progress.report(f"pipeline {ds}", item="done", i=len(todo), n=len(todo) or 1, every_s=0)
        print(f"pipeline {ds}: {len(todo)} stage(s) run in {time.time() - t0:.0f} s", flush=True)
    finally:
        keep_awake(False)
        lock.unlink(missing_ok=True)


def cmd_plan(a):
    _, _, rows = plan(a.dataset, a.smoke, set(a.force or []), a.only, a.until, a.frm, a.skip_archive)
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_status(a):
    rows = []
    for n in stages(a.dataset, a.smoke):
        s = read_state(a.dataset, n)
        rows.append({"stage": n, "status": s.get("status", "-"), "seconds": s.get("seconds"), "attempts": s.get("attempts"),
                     "ended": time.strftime("%m-%d %H:%M", time.localtime(s["ended"])) if s.get("ended") else ""})
    print(pd.DataFrame(rows).to_string(index=False))


def detach(argv):
    """Restart this command outside the current session: WMI process, new group, no window."""
    args = [a for a in argv if a != "--detach"]
    log = paths.pipeline("logs", "runner.log", name=next((args[i + 1] for i, x in enumerate(args) if x == "--dataset"),
                                                         config.DATASET))
    cmdline = f'cmd.exe /c ""{PY}" -m suitdyn {" ".join(args)} >> "{log}" 2>&1"'
    ps = ("$s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32](0x200 -bor 0x08000000)}; "
          f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{cmdline.replace(chr(39), chr(39) * 2)}'; "
          f"CurrentDirectory='{ROOT}'; ProcessStartupInformation=$s}}; \"$($r.ReturnValue) $($r.ProcessId)\"")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    code, pid = (r.stdout.strip().split() + ["?", "?"])[:2]
    print(f"detached runner: return {code}, pid {pid}; log {log}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m suitdyn")
    ap.add_argument("command", choices=["plan", "run", "status"])
    ap.add_argument("--dataset", default=config.DATASET)
    ap.add_argument("--until")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--only")
    ap.add_argument("--force", action="append")
    ap.add_argument("--skip-archive", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--detach", action="store_true")
    a = ap.parse_args(argv)
    os.environ["SUITDYN_DATASET"] = a.dataset
    config.DATASET = a.dataset
    if a.detach and a.command == "run":
        return detach(argv)
    {"plan": cmd_plan, "run": cmd_run, "status": cmd_status}[a.command](a)
