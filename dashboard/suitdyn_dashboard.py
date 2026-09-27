"""SUIT-DYN desktop dashboard: two windows, one per screen.

  Window 1  PIPELINE   every stage of the active chain (status, start, duration, exact command), the full
                       pipeline reference, the two neural networks drawn layer by layer, training runs,
                       GPU / CPU / RAM / disk, data on the share, and the chain log.
  Window 2  LIVE FEED  the SUIT file the pipeline is working on now: rendered image, header facts,
                       stage, progress, rate, ETA, and a ticker of recent files.

It only reads: heartbeats written by suitdyn/progress.py (outputs/logs/progress), chain scripts and logs
(outputs/logs/*.cmd, *.log), training runs (outputs/phase3*/runs), configs/phase1.toml, and FITS files for
the preview. It never starts, stops or changes anything.

    python dashboard/suitdyn_dashboard.py [--root PROJECT] [--selftest]
    dashboard/build_exe.cmd  ->  dashboard/dist/SUIT-DYN Dashboard.exe

Keys: F11 full screen, Esc leave full screen. Window 2 opens to the right of the primary screen
(on a second monitor when there is one).
"""
import argparse
import collections
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

# ----------------------------------------------------------------------------------------------- style
BG, PANEL, PANEL2, FG, DIM = "#0b0f17", "#121a29", "#18233a", "#e6edf7", "#8a97ad"
GOLD, GREEN, BLUE, RED, GREY, VIOLET = "#f5b700", "#3ddc84", "#4aa3ff", "#ff5c5c", "#5b6679", "#b388ff"
STATUS_COLOR = {"done": GREEN, "running": BLUE, "pending": GREY, "failed": RED, "stalled": RED}
STATUS_ICON = {"done": "✔", "running": "▶", "pending": "·", "failed": "✖", "stalled": "!"}

# ------------------------------------------------------------------------------ pipeline reference
CATALOGUE = [
    ("Runner", "Whole pipeline", "python -m suitdyn run --dataset <name> [--smoke] [--detach]",
     "every stage in order; skips what is up to date, resumes, retries"),
    ("Archive", "Manifest", "python scripts/build_manifest.py --span-of <name>", "headers + SHA-256 of the raw files"),
    ("Archive", "Per-frame measurements", "python scripts/process_frames.py --span-of <name>", "limb fits, QC, seams, motion"),
    ("Data set", "Registration", "python scripts/registration_study.py", "disk-centred, north-up; frame QC (span only)"),
    ("Data set", "Data set", "python scripts/build_sequences.py", "frame list, time splits, windows, sealed test"),
    ("Data set", "Calibration", "python scripts/calibrate_pattern.py", "detector fixed pattern from the training split"),
    ("Data set", "Store (Zarr)", "python scripts/build_store.py", "calibrated, registered 1536² frames + QC masks"),
    ("Data set", "Noise maps", "python scripts/phase2_noise_maps.py --split train", "one-frame error, pointing sensitivity"),
    ("Data set", "Pointing response", "python scripts/phase2_response.py", "first-order detector-response correction"),
    ("Data set", "Noise maps + response", "python scripts/phase2_noise_maps.py --split train --response … --tag resp",
     "trusted region"),
    ("Learning", "Samples", "python scripts/phase3_prepare.py", "frame cache + sample index (samples built on the fly)"),
    ("Learning", "Static background S", "python scripts/phase3_background.py", "what derotation must not move"),
    ("Learning", "Train UNet", "python scripts/phase3_train.py --model unet --seed N", "residual over the background-aware B1"),
    ("Learning", "Train ConvLSTM", "python scripts/phase3_train.py --model convlstm --seed N", "residual over the background-aware B1"),
    ("Learning", "Evaluation", "python scripts/phase3_evaluate.py", "strongest baselines, bootstrap, negative controls"),
]
STAGE_TO_CATALOGUE = {"manifest": "Manifest", "build_manifest": "Manifest", "frames": "Per-frame measurements",
                      "process_frames": "Per-frame measurements", "registration": "Registration",
                      "sequences": "Data set", "calibration": "Calibration", "calibrate_pattern": "Calibration",
                      "store": "Store (Zarr)", "build_store": "Store (Zarr)", "noise_maps": "Noise maps",
                      "response": "Pointing response", "samples": "Samples", "phase3_prepare": "Samples",
                      "background": "Static background S", "evaluate": "Evaluation", "pipeline": "Whole pipeline"}
PIPELINE_ORDER = ["manifest", "frames", "registration", "sequences", "calibration", "store", "noise_maps", "response",
                  "noise_maps_resp", "samples", "background", "train", "evaluate"]
SCRIPT_OF = {"manifest": "scripts/build_manifest.py", "frames": "scripts/process_frames.py",
             "registration": "scripts/registration_study.py", "sequences": "scripts/build_sequences.py",
             "calibration": "scripts/calibrate_pattern.py", "store": "scripts/build_store.py",
             "noise_maps": "scripts/phase2_noise_maps.py", "response": "scripts/phase2_response.py",
             "noise_maps_resp": "scripts/phase2_noise_maps.py", "samples": "scripts/phase3_prepare.py",
             "background": "scripts/phase3_background.py", "train": "scripts/phase3_train.py",
             "evaluate": "scripts/phase3_evaluate.py"}


# --------------------------------------------------------------------------------------- helpers
def find_root(arg=None):
    if arg:
        return Path(arg)
    starts = [Path(sys.executable).resolve().parent] if getattr(sys, "frozen", False) else []
    starts.append(Path(__file__).resolve().parent)
    for s in starts:
        for p in [s, *s.parents]:
            if (p / "configs" / "phase1.toml").exists() and (p / "suitdyn").is_dir():
                return p
    return Path(r"D:\Learning and Forecasting Solar Atmospheric Dynamics from Aditya L1SUIT")


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


def catalogue_stage(name):
    """Catalogue stage of a chain step or heartbeat stage name (longest matching script name wins)."""
    name = name or ""
    if name.startswith("noise_maps") and "resp" in name:
        return "Noise maps + response"
    if name.startswith("train"):
        return "Train ConvLSTM" if "convlstm" in name else "Train UNet"
    keys = [k for k in STAGE_TO_CATALOGUE if name.startswith(k)]
    return STAGE_TO_CATALOGUE[max(keys, key=len)] if keys else None


def fmt_dur(s):
    if s is None or s != s:
        return ""
    s = int(max(0, s))
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


TS = re.compile(r"\[\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d+))?\s*\]")


def parse_ts(line):
    m = TS.search(line)
    if not m:
        return None
    d, mo, y, h, mi, s = (int(v) for v in m.groups()[:6])
    try:
        return datetime(y, mo, d, h, mi, s).timestamp()
    except ValueError:
        return None


def tail(path, nbytes=65536):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace").splitlines()[-400:]
    except OSError:
        return []


class NVML:
    """GPU readings from the driver's NVML library (in-process, no child processes)."""

    def __init__(self):
        self.h = None
        try:
            self.lib = ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll"))
            h = ctypes.c_void_p()
            if self.lib.nvmlInit_v2() == 0 and self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0:
                self.h = h
                name = ctypes.create_string_buffer(96)
                self.lib.nvmlDeviceGetName(h, name, 96)
                self.name = name.value.decode()
        except Exception:
            self.h = None

    def read(self):
        if self.h is None:
            return {}
        out = {"name": self.name}
        t = ctypes.c_uint()
        if self.lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(t)) == 0:
            out["temp"] = t.value
        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        u = Util()
        if self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0:
            out["util"] = u.gpu
        p = ctypes.c_uint()
        if self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(p)) == 0:
            out["power_w"] = p.value / 1000
        class Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
        m = Mem()
        if self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)) == 0:
            out["mem_used_gb"], out["mem_total_gb"] = m.used / 1e9, m.total / 1e9
        return out


class CPU:
    def __init__(self):
        self.prev = self._times()

    @staticmethod
    def _times():
        try:
            idle, kern, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong())
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            return idle.value, kern.value + user.value
        except Exception:
            return None

    def usage(self):
        now = self._times()
        if not now or not self.prev:
            return None
        di, dt = now[0] - self.prev[0], now[1] - self.prev[1]
        self.prev = now
        return 100.0 * (1 - di / dt) if dt > 0 else None


def ram():
    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    try:
        m = MS()
        m.len = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return (m.total - m.avail) / 1e9, m.total / 1e9
    except Exception:
        return None


# ------------------------------------------------------------------------------------------ FITS
def read_fits(path):
    """Primary HDU of an uncompressed FITS file: (header dict, float32 image). SUIT Level-1 is int16 + BZERO."""
    hdr = {}
    with open(path, "rb") as f:
        end = False
        while not end:
            block = f.read(2880)
            if len(block) < 2880:
                raise ValueError("truncated header")
            for k in range(36):
                card = block[80 * k:80 * (k + 1)].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    end = True
                    break
                if card[8:10] == "= ":
                    v = card[10:]
                    if v.lstrip().startswith("'"):
                        v = v.split("'")[1].strip() if v.count("'") >= 2 else v
                    else:
                        v = v.split("/")[0].strip()
                        try:
                            v = float(v) if any(c in v for c in ".Ee") else int(v)
                        except ValueError:
                            pass
                    hdr[key] = v
        bitpix, n1, n2 = int(hdr["BITPIX"]), int(hdr["NAXIS1"]), int(hdr.get("NAXIS2", 1))
        dt = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}[bitpix]
        raw = np.frombuffer(f.read(n1 * n2 * abs(bitpix) // 8), dt).reshape(n2, n1)
    img = raw.astype(np.float32) * float(hdr.get("BSCALE", 1.0)) + float(hdr.get("BZERO", 0.0))
    img[img <= -2700] = np.nan  # Level-1 clipped / fill value (int16 floor with BZERO 30000)
    return hdr, img


def _lut():
    # black -> deep red -> orange -> gold -> white (a "Mg II k" look)
    stops = [(0.0, (0, 0, 0)), (0.3, (110, 10, 10)), (0.55, (220, 90, 10)), (0.8, (250, 190, 40)), (1.0, (255, 250, 230))]
    x = np.linspace(0, 1, 256)
    out = np.zeros((256, 3))
    for c in range(3):
        out[:, c] = np.interp(x, [s[0] for s in stops], [s[1][c] for s in stops])
    return out.astype(np.uint8)


LUT = _lut()


def render(img, size):
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    h, w = img.shape
    f = max(1, int(max(h, w) // size))
    if f > 1:
        hh, ww = h // f * f, w // f * f
        img = np.nanmean(img[:hh, :ww].reshape(hh // f, f, ww // f, f), axis=(1, 3))
    v = img[np.isfinite(img)]
    if v.size == 0:
        return Image.new("RGB", (size, size))
    lo, hi = np.percentile(v, [2, 99.8])
    x = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    x = np.arcsinh(6 * x) / np.arcsinh(6)
    rgb = LUT[(x * 255).astype(np.uint8)][::-1]  # FITS row 0 is the bottom
    im = Image.fromarray(rgb, "RGB")
    im.thumbnail((size, size))
    return im


# ------------------------------------------------------------------------------------- collector
class Collector:
    """Background thread: reads everything the windows show into a snapshot (the UI thread only reads it)."""

    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.snap = {}
        self.nvml, self.cpu = NVML(), CPU()
        self.rates = collections.defaultdict(lambda: collections.deque(maxlen=240))
        self.index, self.days, self.index_t = {}, {}, 0.0
        self.newest = []  # newest NB03 full-disk frames on the share (name, path, observation time)
        self.procs, self.procs_t = [], 0.0
        self.preview = {"key": None, "image": None, "hdr": None, "error": None}
        self.want_preview = None
        cfg = {}
        try:
            cfg = tomllib.loads((root / "configs" / "phase1.toml").read_text(encoding="utf-8"))
        except Exception:
            pass
        self.raw_roots = [p for p in (cfg.get("paths", {}).get("raw_root"), cfg.get("paths", {}).get("local_copy_root")) if p]
        for fn in (self._loop, self._index_loop, self._preview_loop):
            threading.Thread(target=fn, daemon=True).start()

    # -- archive index (file name -> path) and files per day, names only
    def _index_loop(self):
        while True:
            idx, days = {}, collections.Counter()
            for r in self.raw_roots:
                try:
                    for dp, _, fn in os.walk(r):
                        for f in fn:
                            if f.endswith(".fits"):
                                idx.setdefault(f, os.path.join(dp, f))
                                if r == self.raw_roots[0]:
                                    m = re.search(r"(20\d\d-\d\d-\d\d)T", f)
                                    days[(m.group(1) if m else "?", "NB03" in f)] += 1
                except OSError:
                    pass
            newest = []
            stamp = re.compile(r"(20\d\d-\d\d-\d\dT\d\d\.\d\d\.\d\d)")
            nb = sorted((m.group(1), f) for f in idx if "NB03" in f and (m := stamp.search(f)))
            for ts, f in reversed(nb[-400:]):
                try:
                    if os.path.getsize(idx[f]) > 8_000_000:  # 2048² binned full disk (ROIs are smaller)
                        newest.append((f, idx[f], ts.replace(".", ":").replace("T", " ")))
                except OSError:
                    pass
                if len(newest) >= 12:
                    break
            with self.lock:
                self.index, self.days, self.index_t, self.newest = idx, dict(days), time.time(), newest
            time.sleep(300)

    def _preview_loop(self):
        while True:
            want = self.want_preview
            if want and want[0] != self.preview["key"]:
                name, path = want
                path = path if path and os.path.exists(path) else self.index.get(os.path.basename(name or ""))
                try:
                    if not path:
                        raise FileNotFoundError("not in the archive index yet")
                    hdr, img = read_fits(path)
                    im = render(img, 900)
                    self.preview = {"key": name, "image": im, "hdr": hdr, "path": path, "error": None, "t": time.time()}
                except Exception as e:
                    self.preview = {"key": name, "image": None, "hdr": None, "path": path, "error": str(e)}
            time.sleep(0.8)

    def _processes(self):
        if time.time() - self.procs_t < 10:
            return self.procs
        try:
            cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='cmd.exe'\" | "
                   "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                               timeout=20, creationflags=0x08000000)
            rows = json.loads(r.stdout) if r.stdout.strip() else []
            rows = rows if isinstance(rows, list) else [rows]
            self.procs = [{"pid": p["ProcessId"], "name": p["Name"], "cmd": p.get("CommandLine") or ""} for p in rows
                          if "scripts" in (p.get("CommandLine") or "") or ".cmd" in (p.get("CommandLine") or "")]
        except Exception:
            pass
        self.procs_t = time.time()
        return self.procs

    def _heartbeats(self):
        d = self.root / "outputs" / "logs" / "progress"
        beats = []
        for p in d.glob("*.json") if d.exists() else []:
            try:
                b = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            b["alive"] = pid_alive(b.get("pid", 0))
            b["age"] = time.time() - b.get("t", 0)
            if b["alive"] or b["age"] < 30:
                q = self.rates[(b.get("pid"), b.get("stage"))]
                if not q or q[-1][0] != b["t"]:
                    q.append((b["t"], b.get("i") or 0))
                beats.append(b)
        for b in beats:
            q = [x for x in self.rates[(b.get("pid"), b.get("stage"))] if x[0] > b["t"] - 90]
            rate = (q[-1][1] - q[0][1]) / (q[-1][0] - q[0][0]) if len(q) > 1 and q[-1][0] > q[0][0] else None
            b["rate"] = rate
            if rate and b.get("n") is not None and b.get("i") is not None:
                b["eta"] = (b["n"] - b["i"] - 1) / rate
        beats.sort(key=lambda b: (not b["alive"], -b.get("t", 0)))
        hist = []
        for line in tail(d / "history.jsonl", 200000)[-300:]:
            try:
                hist.append(json.loads(line))
            except Exception:
                pass
        return beats, hist

    def _chains(self):
        chains = []
        for cmdf in sorted((self.root / "outputs" / "logs").glob("*.cmd")):
            lines = cmdf.read_text(encoding="utf-8", errors="replace").splitlines()
            steps, log, dataset = [], None, "v0"
            for k, ln in enumerate(lines):
                m = re.search(r"echo .*START (.+?) >> (\S+)", ln)
                if ln.strip().lower().startswith("set suitdyn_dataset="):
                    dataset = ln.split("=", 1)[1].strip()
                if m:
                    log = m.group(2)
                    nxt = lines[k + 1] if k + 1 < len(lines) else ""
                    cmd = re.sub(r'^"[^"]*python\.exe"\s*', "python ", nxt.split(" >> ")[0]).strip()
                    steps.append({"name": m.group(1).strip(), "cmd": cmd, "dataset": dataset, "status": "pending"})
            if not steps:
                continue
            logp = self.root / log if log else None
            loglines = tail(logp, 400000) if logp else []
            starts, end, endstate = {}, None, None
            for ln in loglines:
                t = parse_ts(ln)
                m = re.search(r"\] START (.+?)\s*$", ln)
                if m:
                    starts[m.group(1).strip()] = t
                elif "CHAIN DONE" in ln:
                    end, endstate = t, "done"
                elif "CHAIN FAILED" in ln:
                    end, endstate = t, "failed"
            started = [s for s in steps if s["name"] in starts]
            for k, s in enumerate(steps):
                if s["name"] not in starts:
                    continue
                s["start"] = starts[s["name"]]
                later = [starts[x["name"]] for x in steps[k + 1:] if x["name"] in starts]
                if later:
                    s["status"], s["end"] = "done", later[0]
                elif endstate:
                    s["status"], s["end"] = endstate, end
                else:
                    s["status"] = "running"
            alive = any(cmdf.name in p["cmd"] or cmdf.stem in p["cmd"] for p in self.procs)
            state = endstate or ("running" if started else "waiting")
            if state == "running" and not alive and self.procs_t:
                state = "stalled"
                for s in steps:
                    if s["status"] == "running":
                        s["status"] = "stalled"
            chains.append({"name": cmdf.stem, "file": str(cmdf), "log": str(logp) if logp else None, "steps": steps,
                           "state": state, "log_tail": loglines[-60:],
                           "started": min((s.get("start") for s in started if s.get("start")), default=None),
                           "mtime": max(cmdf.stat().st_mtime, logp.stat().st_mtime if logp and logp.exists() else 0)})
        chains.sort(key=lambda c: -c["mtime"])
        return chains

    def _pipelines(self):
        """The runner's per-stage state (outputs/pipeline/<data set>/state/*.json) in the chain format."""
        out = []
        base = self.root / "outputs" / "pipeline"
        for dsd in sorted(base.glob("*")) if base.exists() else []:
            sd = dsd / "state"
            states = {}
            for f in sd.glob("*.json") if sd.exists() else []:
                try:
                    j = json.loads(f.read_text(encoding="utf-8"))
                    states[j.get("stage", f.stem)] = (j, f.stat().st_mtime)
                except Exception:
                    pass
            lock = dsd / "runner.lock"
            alive = False
            if lock.exists():
                try:
                    alive = pid_alive(json.loads(lock.read_text()).get("pid", 0))
                except Exception:
                    pass
            if not states and not alive:
                continue
            try:
                p3 = tomllib.loads((self.root / "configs" / "phase3.toml").read_text(encoding="utf-8"))
                trains = [f"train:{m}:{sd_}" for m in p3["model"]["types"] for sd_ in p3["train"]["seeds"]]
            except Exception:
                trains = []
            names = [n for n in PIPELINE_ORDER if n != "train"]
            names = names[:names.index("evaluate")] + sorted(set(trains) | {k for k in states if k.startswith("train:")}) + ["evaluate"]
            steps, running_log = [], None
            for n in names:
                j, mt = states.get(n, ({}, 0))
                st = j.get("status", "pending")
                if st == "running" and not alive:
                    st = "stalled"
                st = st if st in STATUS_COLOR else ("failed" if st == "interrupted" else "pending")
                key = n.split(":")[0]
                cmd = f"python {SCRIPT_OF.get(key, '?')} {' '.join(j.get('args', []))}".strip()
                steps.append({"name": n, "cmd": cmd, "dataset": dsd.name, "status": st, "start": j.get("started"),
                              "end": j.get("ended"), "attempts": j.get("attempts")})
                if st in ("running", "stalled", "failed") and running_log is None:
                    running_log = dsd / "logs" / f"{n.replace(':', '_')}.log"
            log = running_log if running_log and running_log.exists() else dsd / "logs" / "runner.log"
            state = ("running" if alive else "failed" if any(x["status"] in ("failed", "stalled") for x in steps)
                     else "done" if steps and steps[-1]["status"] == "done" else "stopped")
            out.append({"name": f"pipeline {dsd.name}", "file": str(dsd), "log": str(log), "steps": steps, "state": state,
                        "log_tail": tail(log, 60000)[-60:] if log.exists() else [],
                        "started": min((x["start"] for x in steps if x.get("start")), default=None),
                        "mtime": max([m for _, m in states.values()] + [lock.stat().st_mtime if lock.exists() else 0])})
        return out

    def _runs(self):
        runs = []
        for rd in sorted(self.root.glob("outputs/datasets/*/phase3/runs*/*")):
            if not rd.is_dir() or rd.name.startswith("_"):
                continue
            ds = rd.parent.parent.parent.name + (" (smoke)" if rd.parent.name.endswith("smoke") else "")
            r = {"run": rd.name, "dataset": ds, "status": "pending"}
            if (rd / "run.json").exists():
                try:
                    j = json.loads((rd / "run.json").read_text(encoding="utf-8"))
                    r.update(model=j.get("model"), seed=j.get("seed"), params=j.get("params"),
                             inputs=(j.get("args") or {}).get("inputs", "plain"), best_epoch=j.get("best_epoch"),
                             epochs=j.get("epochs_run"), skill=j.get("best_holdout_skill_vs_B1"), status="done",
                             seconds=j.get("seconds"))
                except Exception:
                    pass
            logf = rd.parent.parent.parent.parent.parent / "pipeline" / rd.parent.parent.parent.name / "logs" / \
                f"train_{r.get('model', rd.name.split('_')[0])}_{r.get('seed', rd.name.rsplit('s', 1)[-1])}.log"
            last = None
            for ln in reversed(tail(logf, 20000)):
                if ln.startswith("{") and '"epoch"' in ln:
                    try:
                        last = json.loads(ln)
                        break
                    except Exception:
                        pass
            if last:
                r["last_epoch"] = last
                if r["status"] != "done":
                    r["status"], r["epochs"] = "partial", last.get("epoch", 0) + 1
                    r["skill"] = last.get("holdout_skill_vs_B1")
                r["gpu_peak"] = last.get("gpu_temp_peak")
            elif (rd / "last.pt").exists() and r["status"] != "done":
                r["status"] = "partial"
            runs.append(r)
        return runs

    def _loop(self):
        while True:
            try:
                snap = {"t": time.time(), "gpu": self.nvml.read(), "cpu": self.cpu.usage(), "ram": ram()}
                try:
                    du = shutil.disk_usage(str(self.root))
                    snap["disk"] = (du.free / 1e9, du.total / 1e9)
                except OSError:
                    pass
                snap["procs"] = self._processes()
                snap["beats"], snap["history"] = self._heartbeats()
                snap["chains"] = sorted(self._pipelines() + self._chains(), key=lambda c: -c["mtime"])
                snap["runs"] = self._runs()
                with self.lock:
                    snap["days"], snap["index_n"], snap["index_t"] = dict(self.days), len(self.index), self.index_t
                meta = self.root / "outputs" / "phase1" / "manifest_meta.json"
                if meta.exists():
                    try:
                        j = json.loads(meta.read_text(encoding="utf-8"))
                        snap["manifest"] = {k: j.get(k) for k in ("files", "rows_from_archive", "rows_from_local_copy",
                                                                   "seconds", "manifest_sha256")}
                        snap["manifest"]["mtime"] = meta.stat().st_mtime
                    except Exception:
                        pass
                live = [b for b in snap["beats"] if b["alive"] and (b.get("item") or "").lower().endswith(".fits")]
                snap["feed_mode"] = "live" if live else "archive"
                if live:
                    self.want_preview = (live[0]["item"], live[0].get("path"))
                elif self.newest:
                    # nothing reports a file: cycle through the newest full-disk frames on the share (labelled)
                    k = int(time.time() // 20) % len(self.newest)
                    self.want_preview = self.newest[k][:2]
                snap["newest"] = list(self.newest)
                self.snap = snap
            except Exception as e:  # never let the collector die
                self.snap = {**self.snap, "error": repr(e)}
            time.sleep(2)


# ------------------------------------------------------------------------------------------- UI
def label(parent, text="", size=10, color=FG, bold=False, bg=PANEL, **kw):
    return tk.Label(parent, text=text, fg=color, bg=bg, font=("Segoe UI", size, "bold" if bold else "normal"), **kw)


def panel(parent, title, **kw):
    f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#22304a", **kw)
    if title:
        label(f, title.upper(), 9, DIM, True).pack(anchor="w", padx=10, pady=(8, 2))
    return f


def fullscreen_keys(win):
    win.bind("<F11>", lambda e: win.attributes("-fullscreen", not win.attributes("-fullscreen")))
    win.bind("<Escape>", lambda e: win.attributes("-fullscreen", False))


def draw_networks(cv, w):
    """UNetSmall and ConvLSTM, drawn from suitdyn/ml/models.py (channels, grid sizes, parameter counts)."""
    cv.delete("all")

    def box(x, y, bw, bh, text, sub="", color=BLUE, fill=PANEL2):
        cv.create_rectangle(x, y, x + bw, y + bh, outline=color, fill=fill, width=1.5)
        cv.create_text(x + bw / 2, y + bh / 2 - (6 if sub else 0), text=text, fill=FG, font=("Segoe UI", 9, "bold"))
        if sub:
            cv.create_text(x + bw / 2, y + bh / 2 + 9, text=sub, fill=DIM, font=("Segoe UI", 8))
        return (x, y, bw, bh)

    def arrow(x0, y0, x1, y1, color=DIM, dash=None):
        cv.create_line(x0, y0, x1, y1, fill=color, arrow="last", width=1.4, dash=dash)

    y = 8
    cv.create_text(10, y, anchor="nw", text="UNetSmall — 212,273 parameters (frames as channels)", fill=GOLD,
                   font=("Segoe UI", 11, "bold"))
    y += 26
    bw, bh, gap = (w - 60) / 9, 44, 6
    xs = [10 + k * (bw + gap) for k in range(9)]
    levels = [0, 1, 2, 3, 2, 1, 0]
    names = [("e1", "16 ch · 384²"), ("e2", "32 ch · 192²"), ("e3", "64 ch · 96²"), ("mid", "64 ch · 48²"),
             ("d3", "32 ch · 96²"), ("d2", "16 ch · 192²"), ("d1", "16 ch · 384²")]
    box(xs[0], y + 10, bw, bh + 30, "input", "8 ch · 384²", GOLD)
    cv.create_text(xs[0] + bw / 2, y + bh + 55, text="4 Δ frames, last−1,\nmask, μ, horizon", fill=DIM,
                   font=("Segoe UI", 7), justify="center")
    pos = []
    for k, ((n, sub), lv) in enumerate(zip(names, levels)):
        bx, by = xs[k + 1], y + 10 + lv * 34
        pos.append(box(bx, by, bw, bh, n, sub, VIOLET if n == "mid" else BLUE))
    arrow(xs[0] + bw, y + 10 + bh / 2, pos[0][0], pos[0][1] + bh / 2)
    for a, b in zip(pos[:-1], pos[1:]):
        arrow(a[0] + a[2], a[1] + bh / 2, b[0], b[1] + bh / 2)
    for a, b in ((0, 6), (1, 5), (2, 4)):
        pa, pb = pos[a], pos[b]
        cv.create_line(pa[0] + bw / 2, pa[1], pa[0] + bw / 2, pa[1] - 8 - 4 * a, pb[0] + bw / 2, pa[1] - 8 - 4 * a,
                       pb[0] + bw / 2, pb[1], fill=GREEN, dash=(3, 3), arrow="last")
    ox = xs[8]
    box(ox, y + 10, bw, bh, "1×1 conv", "residual (zero-init)", GOLD)
    arrow(pos[-1][0] + bw, pos[-1][1] + bh / 2, ox, y + 10 + bh / 2)
    box(ox, y + 70, bw, bh, "+ B1", "forecast t+H", GREEN)
    arrow(ox + bw / 2, y + 10 + bh, ox + bw / 2, y + 70)
    cv.create_text(10, y + 170, anchor="nw", fill=DIM, font=("Segoe UI", 8),
                   text="blocks: 2× (conv3×3 → GroupNorm → GELU) · avg-pool down · bilinear up · skip concat (green)")

    y += 196
    cv.create_text(10, y, anchor="nw", text="ConvLSTM — 103,969 parameters (frames as a sequence)", fill=GOLD,
                   font=("Segoe UI", 11, "bold"))
    y += 26
    bw2 = (w - 60) / 7
    xs = [10 + k * (bw2 + 8) for k in range(7)]
    box(xs[0], y, bw2, bh, "step k = 1…5", "4 ch · 384²", GOLD)
    box(xs[1], y, bw2, bh, "encoder", "stride-2 · 32 ch · 192²")
    box(xs[2], y, bw2, bh, "ConvLSTM cell", "hidden 32 · 3×3 gates", VIOLET)
    cv.create_arc(xs[2] + bw2 / 2 - 18, y - 22, xs[2] + bw2 / 2 + 18, y + 6, start=0, extent=300, style="arc",
                  outline=VIOLET, width=1.5)
    cv.create_text(xs[2] + bw2 / 2 + 30, y - 14, text="h, c", fill=VIOLET, font=("Segoe UI", 8))
    box(xs[3], y, bw2, bh, "upsample", "h → 384²")
    box(xs[4], y, bw2, bh, "decoder", "+4 ch · 2× conv 32")
    box(xs[5], y, bw2, bh, "1×1 conv", "residual (zero-init)", GOLD)
    box(xs[6], y, bw2, bh, "+ B1", "forecast t+H", GREEN)
    for k in range(6):
        arrow(xs[k] + bw2, y + bh / 2, xs[k + 1], y + bh / 2)
    cv.create_text(10, y + bh + 14, anchor="nw", fill=DIM, font=("Segoe UI", 8),
                   text="per step: frame−last (or last−1), mask, μ, horizon · the hidden state carries the 5 context frames")
    y += bh + 40
    cv.create_text(10, y, anchor="nw", fill=FG, font=("Segoe UI", 9), justify="left",
                   text="Both predict the residual over B1 (derotated persistence; or background-aware B1 with --inputs bg).\n"
                        "Output 0 = B1 exactly · loss: masked L1 on valid disk pixels · AdamW 3e-4, OneCycle, bf16 · "
                        "early stop on the hold-out run\nThermal: duty-cycle controller, target 75 °C, hard stop 85 °C "
                        "· never two GPU jobs at once")


class PipelineWindow:
    def __init__(self, tkroot, col):
        self.col = col
        w = self.w = tkroot
        w.title("SUIT-DYN · Pipeline control")
        w.configure(bg=BG)
        w.geometry("1680x1000+0+0")
        fullscreen_keys(w)
        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        label(top, "☀ SUIT-DYN  ·  MISSION CONTROL", 18, GOLD, True, bg=BG).pack(side="left")
        label(top, "  Aditya-L1 / SUIT · NB03 Mg II k 279.6 nm · forecasting the chromosphere", 11, DIM, bg=BG).pack(side="left")
        self.clock = label(top, "", 14, FG, True, bg=BG)
        self.clock.pack(side="right")

        strip = tk.Frame(w, bg=BG)
        strip.pack(fill="x", padx=12, pady=4)
        self.tiles = {}
        for key, title in (("gpu", "GPU"), ("gpuload", "GPU LOAD"), ("cpu", "CPU"), ("ram", "RAM"), ("disk", "DISK D:"),
                           ("data", "DATA ON SHARE"), ("chain", "ACTIVE CHAIN")):
            f = panel(strip, title)
            f.pack(side="left", fill="both", expand=True, padx=4)
            v = label(f, "…", 16, FG, True)
            v.pack(anchor="w", padx=10)
            s = label(f, "", 8, DIM)
            s.pack(anchor="w", padx=10, pady=(0, 8))
            self.tiles[key] = (v, s)

        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=4)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg=BG, width=760)
        right.pack(side="right", fill="both", padx=(8, 0))
        right.pack_propagate(False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("D.Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=24,
                        font=("Segoe UI", 9), borderwidth=0)
        style.configure("D.Treeview.Heading", background=PANEL2, foreground=DIM, font=("Segoe UI", 9, "bold"),
                        borderwidth=0)
        style.map("D.Treeview", background=[("selected", "#243554")])
        style.configure("D.Horizontal.TProgressbar", troughcolor=PANEL2, background=GOLD, borderwidth=0, thickness=18)

        p1 = panel(left, "Active chain — steps, status, exact commands")
        p1.pack(fill="both", expand=True)
        self.chain_title = label(p1, "", 10, FG, True)
        self.chain_title.pack(anchor="w", padx=10)
        cols = ("status", "step", "dataset", "start", "duration", "command")
        self.tv = ttk.Treeview(p1, columns=cols, show="headings", style="D.Treeview", height=13)
        for c, wd in zip(cols, (90, 190, 60, 80, 90, 520)):
            self.tv.heading(c, text=c.upper())
            self.tv.column(c, width=wd, anchor="w", stretch=c == "command")
        for s, c in STATUS_COLOR.items():
            self.tv.tag_configure(s, foreground=c)
        self.tv.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        p2 = panel(left, "Full pipeline reference")
        p2.pack(fill="both", expand=True, pady=(8, 0))
        cols2 = ("phase", "stage", "status", "command", "what")
        self.tv2 = ttk.Treeview(p2, columns=cols2, show="headings", style="D.Treeview", height=10)
        for c, wd in zip(cols2, (70, 180, 80, 420, 330)):
            self.tv2.heading(c, text=c.upper())
            self.tv2.column(c, width=wd, anchor="w", stretch=c in ("command", "what"))
        for s, c in STATUS_COLOR.items():
            self.tv2.tag_configure(s, foreground=c)
        self.tv2.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        p3 = panel(left, "Chain log (tail)")
        p3.pack(fill="both", expand=False, pady=(8, 0))
        self.log = tk.Text(p3, height=9, bg="#0d1320", fg="#b9c6db", font=("Consolas", 9), relief="flat",
                           insertbackground=FG)
        self.log.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        p4 = panel(right, "Neural networks")
        p4.pack(fill="x")
        self.cv = tk.Canvas(p4, bg=PANEL, height=440, highlightthickness=0)
        self.cv.pack(fill="x", padx=6, pady=6)
        self.cv.bind("<Configure>", lambda e: draw_networks(self.cv, e.width))

        p5 = panel(right, "Training runs (Phase 3)")
        p5.pack(fill="both", expand=True, pady=(8, 0))
        cols3 = ("run", "data", "inputs", "status", "epochs", "best", "skill", "gpu peak")
        self.tv3 = ttk.Treeview(p5, columns=cols3, show="headings", style="D.Treeview", height=8)
        for c, wd in zip(cols3, (130, 50, 55, 70, 60, 50, 70, 70)):
            self.tv3.heading(c, text=c.upper())
            self.tv3.column(c, width=wd, anchor="w")
        for s, c in (("done", GREEN), ("partial", BLUE), ("pending", GREY)):
            self.tv3.tag_configure(s, foreground=c)
        self.tv3.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        self.train_live = label(p5, "", 9, BLUE)
        self.train_live.pack(anchor="w", padx=10, pady=(0, 8))

    def refresh(self, s):
        self.clock.config(text=datetime.now().strftime("%a %d %b  %H:%M:%S"))
        g = s.get("gpu") or {}
        t = g.get("temp")
        self.tiles["gpu"][0].config(text=f"{t} °C" if t is not None else "n/a",
                                    fg=RED if (t or 0) >= 85 else GOLD if (t or 0) >= 76 else GREEN)
        self.tiles["gpu"][1].config(text=g.get("name", ""))
        self.tiles["gpuload"][0].config(text=f"{g.get('util', 0)} %" if g else "n/a")
        self.tiles["gpuload"][1].config(text=f"{g.get('power_w', 0):.0f} W · {g.get('mem_used_gb', 0):.1f}/"
                                             f"{g.get('mem_total_gb', 0):.1f} GB" if g else "")
        c = s.get("cpu")
        self.tiles["cpu"][0].config(text=f"{c:.0f} %" if c is not None else "…")
        r = s.get("ram")
        if r:
            self.tiles["ram"][0].config(text=f"{r[0]:.1f} GB")
            self.tiles["ram"][1].config(text=f"of {r[1]:.1f} GB")
        d = s.get("disk")
        if d:
            self.tiles["disk"][0].config(text=f"{d[0]:.0f} GB free", fg=RED if d[0] < 15 else FG)
            self.tiles["disk"][1].config(text=f"of {d[1]:.0f} GB")
        days = s.get("days") or {}
        if days:
            n_all = sum(v for (dd, nb), v in days.items())
            n_nb = sum(v for (dd, nb), v in days.items() if nb)
            ds = sorted({dd for dd, _ in days})
            self.tiles["data"][0].config(text=f"{n_all:,} files")
            self.tiles["data"][1].config(text=f"{ds[0]} → {ds[-1]} · {len(ds)} days · NB03 {n_nb:,}")
        else:
            self.tiles["data"][0].config(text="indexing…")
        chains = s.get("chains") or []
        if chains:
            ch = chains[0]
            col = {"done": GREEN, "failed": RED, "stalled": RED, "running": BLUE}.get(ch["state"], DIM)
            self.tiles["chain"][0].config(text=ch["state"].upper(), fg=col)
            el = (time.time() if ch["state"] == "running" else max((st.get("end") or 0) for st in ch["steps"]) or time.time()) \
                - (ch.get("started") or time.time())
            self.tiles["chain"][1].config(text=f"{ch['name']} · {fmt_dur(el)}")
            self.chain_title.config(text=f"{ch['name']}   ({ch['file']})")
            self.tv.delete(*self.tv.get_children())
            for st in ch["steps"]:
                dur = (st.get("end") or time.time()) - st["start"] if st.get("start") else None
                self.tv.insert("", "end", tags=(st["status"],), values=(
                    f"{STATUS_ICON[st['status']]} {st['status']}", st["name"], st["dataset"],
                    datetime.fromtimestamp(st["start"]).strftime("%H:%M:%S") if st.get("start") else "",
                    fmt_dur(dur), st["cmd"]))
            self.log.delete("1.0", "end")
            self.log.insert("end", "\n".join(ch["log_tail"][-40:]))
            self.log.see("end")
        # reference: status of each catalogue stage from the active chain and live heartbeats
        status = {}
        for ch in chains[:1]:
            for st in ch["steps"]:
                cat = catalogue_stage(st["name"])
                if cat:
                    status[cat] = st["status"]
        for b in s.get("beats") or []:
            cat = catalogue_stage(b.get("stage"))
            if cat and b["alive"]:
                status[cat] = "running"
            if (b.get("stage") or "").startswith("train") and b["alive"]:
                status["Train UNet" if "unet" in b["stage"] else "Train ConvLSTM"] = "running"
        self.tv2.delete(*self.tv2.get_children())
        for ph, name, cmd, what in CATALOGUE:
            st = status.get(name, "")
            self.tv2.insert("", "end", tags=(st,) if st else (), values=(
                ph, name, f"{STATUS_ICON[st]} {st}" if st else "", cmd, what))
        self.tv3.delete(*self.tv3.get_children())
        for r in s.get("runs") or []:
            sk = r.get("skill")
            self.tv3.insert("", "end", tags=(r["status"],), values=(
                r["run"], r["dataset"], r.get("inputs", ""), r["status"], r.get("epochs", ""), r.get("best_epoch", ""),
                f"{sk:.3f}" if isinstance(sk, (int, float)) else "", r.get("gpu_peak", "")))
        tr = [b for b in s.get("beats") or [] if (b.get("stage") or "").startswith("train") and b["alive"]]
        if tr:
            b = tr[0]
            self.train_live.config(text=f"▶ {b['stage']} · {b.get('item', '')} · GPU {b.get('gpu_temp', '?')} °C · "
                                        f"duty {b.get('duty', '?')} · best hold-out skill {b.get('best_holdout_skill', '–')}")
        else:
            self.train_live.config(text="no training running")


class FeedWindow:
    def __init__(self, tkroot, col):
        self.col = col
        w = self.w = tk.Toplevel(tkroot)
        w.title("SUIT-DYN · Live feed")
        w.configure(bg=BG)
        sw = tkroot.winfo_screenwidth()
        try:  # a second monitor exists when the virtual desktop is wider than the primary screen
            virtual = ctypes.windll.user32.GetSystemMetrics(78)
        except Exception:
            virtual = sw
        w.geometry(f"1680x1000+{sw}+0" if virtual > sw + 100 else "1500x950+60+40")
        fullscreen_keys(w)
        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=14, pady=(10, 2))
        label(top, "◉ LIVE FEED", 18, RED, True, bg=BG).pack(side="left")
        self.stage = label(top, "", 16, GOLD, True, bg=BG)
        self.stage.pack(side="left", padx=18)
        self.clock = label(top, "", 14, FG, True, bg=BG)
        self.clock.pack(side="right")

        pr = tk.Frame(w, bg=BG)
        pr.pack(fill="x", padx=14, pady=4)
        self.bar = ttk.Progressbar(pr, style="D.Horizontal.TProgressbar", maximum=1000)
        self.bar.pack(fill="x")
        self.prog = label(pr, "", 12, FG, bg=BG)
        self.prog.pack(anchor="w", pady=(4, 0))

        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=6)
        imgp = panel(body, "")
        imgp.pack(side="left", fill="both", expand=True)
        self.fname = label(imgp, "", 13, FG, True)
        self.fname.pack(anchor="w", padx=12, pady=(10, 0))
        self.fpath = label(imgp, "", 8, DIM)
        self.fpath.pack(anchor="w", padx=12)
        self.img = tk.Label(imgp, bg="#000000")
        self.img.pack(fill="both", expand=True, padx=12, pady=12)
        side = tk.Frame(body, bg=BG, width=560)
        side.pack(side="right", fill="both", padx=(10, 0))
        side.pack_propagate(False)
        hp = panel(side, "Frame header")
        hp.pack(fill="x")
        self.hdr = tk.Text(hp, height=16, bg=PANEL, fg=FG, font=("Consolas", 10), relief="flat")
        self.hdr.pack(fill="x", padx=10, pady=(2, 10))
        ap = panel(side, "Other live processes")
        ap.pack(fill="x", pady=(8, 0))
        self.others = label(ap, "", 9, FG, justify="left", wraplength=520)
        self.others.pack(anchor="w", padx=10, pady=(0, 10))
        tp = panel(side, "Recent files")
        tp.pack(fill="both", expand=True, pady=(8, 0))
        self.ticker = tk.Listbox(tp, bg=PANEL, fg="#c9d4e6", font=("Consolas", 9), relief="flat", highlightthickness=0,
                                 selectbackground="#243554", activestyle="none")
        self.ticker.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self._shown = None

    def refresh(self, s):
        self.clock.config(text=datetime.now().strftime("%H:%M:%S"))
        beats = s.get("beats") or []
        live = [b for b in beats if b["alive"]]
        chains = s.get("chains") or []
        if live:
            b = live[0]
            self.stage.config(text=f"{b.get('stage')}   ·   data set {b.get('dataset')}")
            i, n = b.get("i"), b.get("n")
            if i is not None and n:
                self.bar["value"] = 1000 * (i + 1) / n
                rate = b.get("rate")
                eta = b.get("eta")
                started = b.get("stage_started")
                self.prog.config(text=f"{i + 1:,} / {n:,}   ({100 * (i + 1) / n:.1f} %)     "
                                      f"{(f'{rate:.2f} items/s' if rate else 'rate …')}     "
                                      f"ETA {fmt_dur(eta) if eta else '…'}     elapsed {fmt_dur(time.time() - started) if started else ''}")
            else:
                self.bar["value"] = 0
                self.prog.config(text=b.get("item") or "")
            extra = {k: v for k, v in b.items() if k in ("horizon", "set", "epoch", "gpu_temp", "duty",
                                                         "best_holdout_skill")}
            if extra:
                self.prog.config(text=self.prog.cget("text") + "     " + "  ".join(f"{k}={v}" for k, v in extra.items()))
            others = [f"▶ {o.get('stage')} · {o.get('item') or ''}" for o in live[1:]]
            self.others.config(text="\n".join(others) or "none")
        else:
            running = [st for ch in chains[:1] for st in ch["steps"] if st["status"] == "running"]
            since = f" since {datetime.fromtimestamp(running[0]['start']).strftime('%H:%M')}" if running and running[0].get("start") else ""
            self.stage.config(text=(f"{running[0]['name']}{since}  ·  this step does not report files — "
                                    f"showing the newest frames on the share" if running
                                    else "idle — showing the newest frames on the share"))
            self.bar["value"] = 0
            self.prog.config(text=f"chain {chains[0]['name']}: {chains[0]['state']}" if chains else "")
            self.others.config(text="none")
        pv = self.col.preview
        if pv.get("key") and pv.get("key") != self._shown:
            self._shown = pv["key"]
            mode = "LIVE · " if (self.col.snap or {}).get("feed_mode") == "live" else "ARCHIVE (newest on share) · "
            self.fname.config(text=mode + pv["key"])
            self.fpath.config(text=pv.get("path") or "")
            if pv.get("image") is not None:
                box = max(300, min(self.img.winfo_width(), self.img.winfo_height()) - 10)
                im = pv["image"].copy()
                im.thumbnail((box, box))
                self._photo = ImageTk.PhotoImage(im)
                self.img.config(image=self._photo, text="")
            else:
                self.img.config(image="", text=f"no preview: {pv.get('error')}", fg=DIM, font=("Segoe UI", 12))
            h = pv.get("hdr") or {}
            keys = ["DATE-OBS", "FTR_NAME", "OBS_MODE", "IMG_TYPE", "ROI_FF", "NAXIS1", "NAXIS2", "CMD_EXPT",
                    "MEAS_EXP", "CRPIX1", "CRPIX2", "CROTA2", "RSUN_OBS", "HGLT_OBS", "F_VER", "FLAT_CF", "SCAT_CF"]
            lines = [f"{k:<9} {h[k]}" for k in keys if k in h]
            if "CRPIX1" in h:
                mode = "centred" if float(h["CRPIX1"]) < 1150 * int(h.get("NAXIS1", 2048)) / 2048 else "offset"
                lines.append(f"{'POINTING':<9} {mode}")
            self.hdr.delete("1.0", "end")
            self.hdr.insert("end", "\n".join(lines))
        hist = s.get("history") or []
        if not hist and s.get("newest"):
            self.ticker.delete(0, "end")
            self.ticker.insert("end", "newest NB03 full-disk frames on the share:")
            for name, _, ts in s["newest"]:
                self.ticker.insert("end", f"{ts}  {name}")
            return
        rows, seen = [], set()
        for hrec in reversed(hist):
            it = hrec.get("item") or ""
            if it in seen:
                continue
            seen.add(it)
            rows.append(f"{datetime.fromtimestamp(hrec.get('t', 0)).strftime('%H:%M:%S')}  {hrec.get('stage', '')[:26]:<26} {it}")
            if len(rows) >= 60:
                break
        self.ticker.delete(0, "end")
        for r in rows:
            self.ticker.insert("end", r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--selftest", action="store_true", help="collect once, render one preview, write a report, exit")
    a = ap.parse_args()
    root = find_root(a.root)
    col = Collector(root)
    if a.selftest:
        t0 = time.time()
        while (not col.snap or not col.index) and time.time() - t0 < 60:
            time.sleep(1)
        s = col.snap
        some = next(iter(col.index.items()), (None, None))
        if some[1]:
            col.want_preview = some
            t1 = time.time()
            while col.preview.get("key") != some[0] and time.time() - t1 < 30:
                time.sleep(0.5)
        rep = {"root": str(root), "gpu": s.get("gpu"), "chains": [(c["name"], c["state"], len(c["steps"])) for c in s.get("chains", [])],
               "runs": len(s.get("runs", [])), "beats": len(s.get("beats", [])), "index_files": len(col.index),
               "preview": {"file": col.preview.get("key"), "ok": col.preview.get("image") is not None,
                           "error": col.preview.get("error")}, "error": s.get("error")}
        out = root / "outputs" / "logs" / "dashboard_selftest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        print(json.dumps(rep, indent=1, default=str))
        return
    tkroot = tk.Tk()
    pw = PipelineWindow(tkroot, col)
    fw = FeedWindow(tkroot, col)

    def tick():
        s = col.snap
        if s:
            try:
                pw.refresh(s)
                fw.refresh(s)
            except Exception as e:  # keep the UI alive whatever a refresh hits
                pw.clock.config(text=f"refresh error: {e}")
        tkroot.after(1500, tick)

    tkroot.after(500, tick)
    tkroot.mainloop()


if __name__ == "__main__":
    main()
