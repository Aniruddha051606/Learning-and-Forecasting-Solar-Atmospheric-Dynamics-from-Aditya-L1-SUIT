"""Run the post-hoc tests automatically as the final run's evaluations finish (started hidden by
outputs/logs/posthoc_auto.cmd; log outputs/tests/auto.log; progress outputs/tests/auto_state.json).
"""
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from suitdyn import atomic, pipeline  # noqa: E402

OUT = ROOT / "outputs" / "tests"
LOG, STATE = OUT / "auto.log", OUT / "auto_state.json"
PY = sys.executable
OFF, CEN = "final_offset", "final_centred"
REPORT = ["scripts/posthoc_report.py", "--datasets", f"{OFF},{CEN}"]
STEPS = [
    ("offset_val", [(OFF, "evaluate")], [["scripts/posthoc_flow_baseline.py", "--dataset", OFF], REPORT]),
    ("offset_test", [(OFF, "evaluate_test")], [REPORT]),
    ("centred_val", [(CEN, "evaluate")], [["scripts/posthoc_flow_baseline.py", "--dataset", CEN], REPORT]),
    ("cross_mode", [(OFF, "evaluate"), (CEN, "evaluate")],
     [["scripts/posthoc_cross_mode.py", "--train", OFF, "--on", CEN],
      ["scripts/posthoc_cross_mode.py", "--train", CEN, "--on", OFF], REPORT]),
    ("centred_test", [(CEN, "evaluate_test")], [REPORT]),
]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def chime(kind, text):
    try:
        urllib.request.urlopen("http://192.168.1.2:8765/event?" + urllib.parse.urlencode({"kind": kind, "text": text}),
                               timeout=5).close()
    except Exception:
        pass


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    log(f"posthoc auto started; already done: {[k for k, v in state.items() if v.get('status') == 'done'] or 'none'}")
    t_start = time.time()
    while any(state.get(name, {}).get("status") not in ("done", "failed") for name, *_ in STEPS):
        for name, needs, cmds in STEPS:
            if state.get(name, {}).get("status") in ("done", "failed"):
                continue
            if not all(pipeline.read_state(ds, st).get("status") == "done" for ds, st in needs):
                continue
            log(f"step {name}: starting")
            t0, ok = time.time(), True
            for cmd in cmds:
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(f"----- {' '.join(cmd)}\n")
                    f.flush()
                    rc = subprocess.call([PY] + cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                         env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
                if rc != 0:
                    log(f"step {name}: '{cmd[0]}' FAILED (exit {rc})")
                    ok = False
            state[name] = {"status": "done" if ok else "failed", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "minutes": round((time.time() - t0) / 60, 1)}
            atomic.write_json(STATE, state)
            log(f"step {name}: {state[name]['status']} in {state[name]['minutes']} min")
            if name == "centred_test":
                chime("test", "final test report ready (outputs/tests/REPORT.md)")
        if time.time() - t_start > 10 * 86400:
            log("gave up waiting after 10 days")
            return 1
        time.sleep(120)
    log("all post-hoc steps finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
