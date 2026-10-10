"""Everything after the v2 retraining, one GPU job at a time (docs/PREREGISTRATION.md, Addendum I and its
notes).

    python scripts/after_v2.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import retrain_v2 as rv  # noqa: E402
from suitdyn import atomic, pipeline  # noqa: E402

OFF, CEN = "final_offset", "final_centred"
STATE = rv.LOG / "after_v2.json"
LOG = rv.LOG / "after_v2.log"
NP = "scripts/posthoc_new_period.py"
STEPS = [("repair", OFF, [NP, "--prepare-only", "--workers", "6"]),
         ("rescore_v2", OFF, [NP, "--score-only", "--runs-dir", "runs", "--mask", "context", "--tag", "v2"]),
         ("rescore_v1_targetmask", OFF, [NP, "--score-only", "--runs-dir", "runs_v1_targetmask", "--mask", "target",
                                         "--tag", "v1_targetmask"]),
         ("rescore_v1_ctxmask", OFF, [NP, "--score-only", "--runs-dir", "runs_v1_targetmask", "--mask", "context",
                                      "--tag", "v1_ctxmask"]),
         ("maskdiff_offset", OFF, ["scripts/posthoc_mask_difference.py", "--dataset", OFF]),
         ("maskdiff_centred", CEN, ["scripts/posthoc_mask_difference.py", "--dataset", CEN])]
for ds in (OFF, CEN):
    tag = ds.split("_")[1]
    STEPS += [(f"fixed_removal_{tag}", ds, ["scripts/posthoc_fixed_removal.py", "--dataset", ds]),
              (f"shuffle_denoise_{tag}", ds, ["scripts/posthoc_shuffle_denoise.py", "--dataset", ds]),
              (f"classical_{tag}", ds, ["scripts/posthoc_classical.py", "--dataset", ds]),
              (f"state_estimate_{tag}", ds, ["scripts/posthoc_state_estimate.py", "--dataset", ds]),
              (f"flow_{tag}", ds, ["scripts/posthoc_flow_baseline.py", "--dataset", ds, "--device", "cuda"])]
STEPS += [("cross_off_on_cen", None, ["scripts/posthoc_cross_mode.py", "--train", OFF, "--on", CEN, "--device", "cuda"]),
          ("cross_cen_on_off", None, ["scripts/posthoc_cross_mode.py", "--train", CEN, "--on", OFF, "--device", "cuda"]),
          ("report", None, ["scripts/posthoc_report.py", "--datasets", f"{OFF},{CEN}"]),
          ("paper_results", None, ["scripts/paper/make_suit_paper.py", "--example-device", "cuda"]),
          ("paper_tex", None, ["scripts/paper/make_tex.py"]),
          ("paper_check", None, ["scripts/paper/check_tex.py"]),
          ("paper_claims", None, ["scripts/paper/verify_claims.py"])]


def say(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def chain_running():
    try:
        pid = json.loads((rv.LOG / "chain.lock").read_text())["pid"]
        return pipeline.pid_alive(pid) and "retrain_v2" in rv.cmdline(pid)
    except Exception:
        return False


def main():
    rv.LOG.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    lock = rv.LOG / "after_v2.lock"
    try:
        pid = json.loads(lock.read_text())["pid"]
        if pid != os.getpid() and pipeline.pid_alive(pid) and "after_v2" in rv.cmdline(pid):
            print(f"already running (pid {pid})")
            return
    except Exception:
        pass
    atomic.write_json(lock, {"pid": os.getpid()})
    pipeline.keep_awake(True)
    try:
        if chain_running():
            say("waiting for the v2 chain to finish (the GPU must be free; CPU work beside training overheats the GPU)")
            while chain_running():
                time.sleep(120)
        for name, ds, args in STEPS:
            if st.get(name, {}).get("status") == "done":
                continue
            if name == "report" and "report_v1" not in st:
                for f in ("report.json", "REPORT.md"):
                    src = ROOT / "outputs" / "tests" / f
                    dst = src.with_name(src.stem + "_v1_targetmask" + src.suffix)
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)
                st["report_v1"] = {"status": "done", "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                atomic.write_json(STATE, st)
            say(f"--- {name}: {' '.join(args)}")
            t0 = time.time()
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            env.pop("SUITDYN_RUNS_DIR", None)
            if ds:
                env["SUITDYN_DATASET"] = ds
            with open(rv.LOG / f"after_v2_{name}.log", "a", encoding="utf-8") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
                f.flush()
                rc = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            st[name] = {"status": "done" if rc == 0 else "failed", "exit": rc, "minutes": round((time.time() - t0) / 60, 1),
                        "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            atomic.write_json(STATE, st)
            say(f"{name} {'done' if rc == 0 else f'FAILED (exit {rc}; log after_v2_{name}.log)'} ({(time.time() - t0) / 60:.0f} min)")
        say("finished: " + ", ".join(f"{n}={st.get(n, {}).get('status', 'not run')}" for n, *_ in STEPS))
    finally:
        pipeline.keep_awake(False)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
