"""Live progress heartbeats for the desktop dashboard (dashboard/suitdyn_dashboard.py).

A pipeline loop calls report(stage, item=..., i=..., n=..., path=...) once per unit of work. At most
every `every_s` seconds (and always on the last item) the call writes the latest state of this process
to outputs/logs/progress/<pid>.json (atomic replace) and appends it to history.jsonl. Only the main
process of a script reports, never pool workers, so there is one writer per file.

It never raises and never slows a run: any error (full disk, locked file) is swallowed. The dashboard
is a viewer; the pipeline must behave the same with or without it.
"""
import json
import os
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "outputs" / "logs" / "progress"
_state = {"last": 0.0, "t0": None, "stage": None}


def report(stage, item=None, i=None, n=None, path=None, every_s=0.5, **extra):
    try:
        now = time.time()
        if _state["stage"] != stage:
            _state.update(stage=stage, t0=now, last=0.0)
        final = n is not None and i is not None and i + 1 >= n
        if now - _state["last"] < every_s and not final:
            return
        _state["last"] = now
        DIR.mkdir(parents=True, exist_ok=True)
        rec = {"t": now, "pid": os.getpid(), "stage": stage, "dataset": os.environ.get("SUITDYN_DATASET", "v0"),
               "item": None if item is None else str(item), "path": None if path is None else str(path),
               "i": None if i is None else int(i), "n": None if n is None else int(n),
               "stage_started": _state["t0"], **{k: v for k, v in extra.items() if v is not None}}
        line = json.dumps(rec, default=str)
        tmp = DIR / f"{os.getpid()}.tmp"
        tmp.write_text(line, encoding="utf-8")
        os.replace(tmp, DIR / f"{os.getpid()}.json")
        with open(DIR / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
