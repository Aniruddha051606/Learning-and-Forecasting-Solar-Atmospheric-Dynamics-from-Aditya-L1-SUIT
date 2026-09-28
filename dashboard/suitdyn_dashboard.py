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
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=600))  # 20 min at 2 s
        self.thumbs = collections.deque(maxlen=8)
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
                    th = im.copy()
                    th.thumbnail((170, 170))
                    self.thumbs.append((name, th, hdr.get("DATE-OBS", "")))
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
            smoke = any(j.get("smoke") for j, _ in states.values())
            try:
                smoke = smoke or "--smoke" in json.loads(lock.read_text()).get("argv", []) if lock.exists() else smoke
            except Exception:
                pass
            try:
                p3 = tomllib.loads((self.root / "configs" / "phase3.toml").read_text(encoding="utf-8"))
                seeds = p3["train"]["seeds"][:1] if smoke else p3["train"]["seeds"]
                trains = [f"train:{m}:{sd_}" for m in p3["model"]["types"] for sd_ in seeds]
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
            lt = tail(log, 60000) if log.exists() else []
            heads = [k for k, ln in enumerate(lt) if ln.startswith("===== ")]
            if heads and log.name != "runner.log":
                lt = lt[heads[-1]:]  # the current attempt only
            out.append({"name": f"pipeline {dsd.name}" + (" (smoke)" if smoke else ""), "file": str(dsd), "log": str(log), "steps": steps, "state": state,
                        "log_tail": lt[-60:],
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
            last, curve = None, []
            for ln in tail(logf, 400000):
                if ln.startswith("{") and '"epoch"' in ln:
                    try:
                        j = json.loads(ln)
                    except Exception:
                        continue
                    if curve and j.get("epoch", 0) <= curve[-1][0]:
                        curve = []  # a restarted run: keep the latest attempt
                    curve.append((j.get("epoch", 0), j.get("holdout_skill_vs_B1"), j.get("train_l1")))
                    last = j
            r["curve"] = curve
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
                now = snap["t"]
                for key, val in (("gpu_temp", snap["gpu"].get("temp")), ("gpu_util", snap["gpu"].get("util")),
                                 ("cpu", snap["cpu"]), ("ram", (snap["ram"] or (None, 1))[0])):
                    if val is not None:
                        self.hist[key].append((now, float(val)))
                snap["hist"] = {k: list(v) for k, v in self.hist.items()}
                snap["thumbs"] = list(self.thumbs)
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
                meta = self.root / "outputs" / "archive" / "manifest_meta.json"
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
BORDER, INK = "#223050", "#0d1320"
F_UI, F_MONO = "Segoe UI", "Consolas"
SERIES = [GOLD, BLUE, GREEN, VIOLET, "#ff9f43", "#48dbfb", RED, "#c8d6e5"]
SHORT = {"manifest": "Manifest", "frames": "Frames", "registration": "Registration", "sequences": "Splits",
         "calibration": "Calibration", "store": "Store", "noise_maps": "Noise maps", "response": "Response",
         "noise_maps_resp": "Trusted", "samples": "Samples", "background": "Background", "train": "Train",
         "evaluate": "Evaluate"}
GROUP = {"manifest": "ARCHIVE", "frames": "ARCHIVE", "samples": "LEARNING", "background": "LEARNING",
         "train": "LEARNING", "evaluate": "LEARNING"}


def _mix(c1, c2, a):
    """c1 blended over c2 with weight a (Tk has no alpha)."""
    x = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    y = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(a * p + (1 - a) * q):02x}" for p, q in zip(x, y))


def label(parent, text="", size=10, color=FG, bold=False, bg=PANEL, **kw):
    return tk.Label(parent, text=text, fg=color, bg=bg, font=(F_UI, size, "bold" if bold else "normal"), **kw)


def card(parent, title=None, accent=GOLD):
    outer = tk.Frame(parent, bg=BORDER)
    inner = tk.Frame(outer, bg=PANEL)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    if title:
        hdr = tk.Frame(inner, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(9, 2))
        tk.Frame(hdr, bg=accent, width=3, height=14).pack(side="left", padx=(0, 8))
        label(hdr, title.upper(), 9, DIM, True).pack(side="left")
        inner.hdr = hdr
    return outer, inner


def panel(parent, title, **kw):  # kept for the network drawing helpers
    return card(parent, title)[1]


def round_rect(cv, x0, y0, x1, y1, r=10, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1, x0, y1,
           x0, y1 - r, x0, y0 + r, x0, y0]
    return cv.create_polygon(pts, smooth=True, **kw)


def sparkline(cv, series, color, vmin=None, vmax=None):
    cv.delete("all")
    w, h = cv.winfo_width(), cv.winfo_height()
    if w < 20 or len(series) < 2:
        return
    t, v = [p[0] for p in series], [p[1] for p in series]
    lo = min(v) if vmin is None else vmin
    hi = max(v) if vmax is None else vmax
    hi = hi if hi - lo > 1e-9 else lo + 1
    t0, t1 = t[0], max(t[-1], t[0] + 1)
    pts = []
    for ti, vi in zip(t, v):
        pts += [2 + (w - 4) * (ti - t0) / (t1 - t0), h - 3 - (h - 6) * (min(max(vi, lo), hi) - lo) / (hi - lo)]
    cv.create_polygon([2, h - 2] + pts + [pts[-2], h - 2], fill=_mix(color, PANEL, 0.22), outline="")
    cv.create_line(pts, fill=color, width=1.6)


def fullscreen_keys(win):
    win.bind("<F11>", lambda e: win.attributes("-fullscreen", not win.attributes("-fullscreen")))
    win.bind("<Escape>", lambda e: win.attributes("-fullscreen", False))


def style_ttk():
    st = ttk.Style()
    st.theme_use("clam")
    st.configure("D.Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=25,
                 font=(F_UI, 9), borderwidth=0)
    st.configure("D.Treeview.Heading", background=PANEL2, foreground=DIM, font=(F_UI, 9, "bold"), borderwidth=0,
                 relief="flat")
    st.map("D.Treeview", background=[("selected", "#243554")])
    st.configure("Gold.Horizontal.TProgressbar", troughcolor=INK, background=GOLD, borderwidth=0, thickness=14,
                 lightcolor=GOLD, darkcolor=GOLD)
    st.configure("D.TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=FG, arrowcolor=GOLD,
                 bordercolor=BORDER, lightcolor=PANEL2, darkcolor=PANEL2)
    st.map("D.TCombobox", fieldbackground=[("readonly", PANEL2)], foreground=[("readonly", FG)],
           selectbackground=[("readonly", PANEL2)], selectforeground=[("readonly", GOLD)])
    return st


def tree(parent, cols, widths, height, stretch=()):
    tv = ttk.Treeview(parent, columns=cols, show="headings", style="D.Treeview", height=height)
    for c, wd in zip(cols, widths):
        tv.heading(c, text=c.upper())
        tv.column(c, width=wd, anchor="w", stretch=c in stretch)
    for sname, colr in STATUS_COLOR.items():
        tv.tag_configure(sname, foreground=colr)
    tv.tag_configure("partial", foreground=BLUE)
    return tv


def group_steps(steps):
    """The stage-flow nodes: every stage, with the training runs merged into one 'train' node."""
    nodes, trains = [], [s for s in steps if s["name"].startswith("train:")]
    for s in steps:
        if s["name"].startswith("train:"):
            if not any(n["key"] == "train" for n in nodes):
                sts = [t["status"] for t in trains]
                st = ("failed" if any(x in ("failed", "stalled") for x in sts) else "running" if "running" in sts
                      else "done" if sts and all(x == "done" for x in sts) else "pending")
                starts = [t["start"] for t in trains if t.get("start")]
                ends = [t.get("end") for t in trains if t.get("end")]
                nodes.append({"key": "train", "status": st, "label": f"Train {sum(x == 'done' for x in sts)}/{len(sts)}",
                              "start": min(starts) if starts else None,
                              "end": max(ends) if ends and st == "done" else None})
            continue
        nodes.append({"key": s["name"], "status": s["status"], "label": SHORT.get(s["name"], s["name"]),
                      "start": s.get("start"), "end": s.get("end")})
    return nodes


def draw_flow(cv, steps, pulse):
    cv.delete("all")
    w, h = cv.winfo_width(), cv.winfo_height()
    nodes = group_steps(steps)
    if w < 100 or not nodes:
        cv.create_text(w / 2, h / 2, text="no pipeline run yet — python -m suitdyn run --dataset <name>", fill=DIM,
                       font=(F_UI, 11))
        return
    n = len(nodes)
    gap = (w - 30) / n
    bw, bh, y0 = min(118, gap - 12), 46, 44
    xs = [15 + gap * k + (gap - bw) / 2 for k in range(n)]
    # group bands
    k = 0
    while k < n:
        g = GROUP.get(nodes[k]["key"], "DATA SET")
        j = k
        while j + 1 < n and GROUP.get(nodes[j + 1]["key"], "DATA SET") == g:
            j += 1
        cv.create_line(xs[k], 26, xs[j] + bw, 26, fill=_mix(GOLD, PANEL, 0.45), width=2)
        cv.create_text(xs[k], 14, text=g, anchor="w", fill=DIM, font=(F_UI, 8, "bold"))
        k = j + 1
    for k in range(n - 1):
        colr = GREEN if nodes[k]["status"] == "done" else BORDER
        cv.create_line(xs[k] + bw, y0 + bh / 2, xs[k + 1], y0 + bh / 2, fill=colr, width=2)
    for k, nd in enumerate(nodes):
        colr = STATUS_COLOR.get(nd["status"], GREY)
        running = nd["status"] == "running"
        round_rect(cv, xs[k], y0, xs[k] + bw, y0 + bh, 12, fill=_mix(colr, PANEL, 0.18 if nd["status"] != "pending" else 0.06),
                   outline=(GOLD if pulse else colr) if running else colr, width=3 if running and pulse else 1.5)
        cv.create_text(xs[k] + bw / 2, y0 + 17, text=nd["label"], fill=FG, font=(F_UI, 9, "bold"))
        cv.create_text(xs[k] + bw / 2, y0 + 33, text=f"{STATUS_ICON.get(nd['status'], '')} {nd['status']}", fill=colr,
                       font=(F_UI, 8))
        if nd.get("start"):
            dur = (nd.get("end") or time.time()) - nd["start"]
            cv.create_text(xs[k] + bw / 2, y0 + bh + 13, text=fmt_dur(dur), fill=DIM, font=(F_UI, 8))


def draw_curves(cv, runs):
    cv.delete("all")
    w, h = cv.winfo_width(), cv.winfo_height()
    runs = [r for r in runs if r.get("curve")][-6:]
    if w < 60 or not runs:
        cv.create_text(w / 2, h / 2, text="no training curve yet", fill=DIM, font=(F_UI, 10))
        return
    L, R, T, B = 44, 12, 12, 26
    vals = [c[1] for r in runs for c in r["curve"] if c[1] is not None]
    lo, hi = min(vals + [0.0]), max(vals + [0.05])
    emax = max(c[0] for r in runs for c in r["curve"]) + 1
    X = lambda e: L + (w - L - R) * e / max(emax - 1, 1)  # noqa: E731
    Y = lambda v: h - B - (h - T - B) * (v - lo) / (hi - lo + 1e-9)  # noqa: E731
    for f in (0, 0.5, 1):
        v = lo + f * (hi - lo)
        cv.create_line(L, Y(v), w - R, Y(v), fill=BORDER)
        cv.create_text(L - 6, Y(v), text=f"{100 * v:.0f}%", anchor="e", fill=DIM, font=(F_UI, 8))
    cv.create_text(L, h - 8, text="epoch", anchor="w", fill=DIM, font=(F_UI, 8))
    cv.create_text(w - R, h - 8, text="hold-out skill vs its B1", anchor="e", fill=DIM, font=(F_UI, 8))
    for k, r in enumerate(runs):
        pts = [(X(e), Y(v)) for e, v, _ in r["curve"] if v is not None]
        colr = SERIES[k % len(SERIES)]
        if len(pts) > 1:
            cv.create_line([c for p in pts for c in p], fill=colr, width=2)
        for x, y in pts:
            cv.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill=colr, outline="")
        if pts:
            cv.create_text(pts[-1][0] - 4, pts[-1][1] - 9, text=f"{r['run']} ({r['dataset']})", anchor="e", fill=colr,
                           font=(F_UI, 8))


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

    y += 190
    cv.create_text(10, y, anchor="nw", text="ConvLSTM — 103,969 parameters (frames as a sequence)", fill=GOLD,
                   font=("Segoe UI", 11, "bold"))
    y += 46
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
    TILES = (("gpu_temp", "GPU temperature", GOLD), ("gpu_util", "GPU load", BLUE), ("cpu", "CPU", GREEN),
             ("ram", "Memory", VIOLET), ("disk", "Disk D:", "#48dbfb"), ("data", "Data on the share", "#ff9f43"))

    def __init__(self, tkroot, col):
        self.col, self.pulse, self.sel = col, False, None
        w = self.w = tkroot
        w.title("SUIT-DYN · Pipeline control")
        w.configure(bg=BG)
        w.geometry("1720x1020+0+0")
        w.minsize(1280, 820)
        fullscreen_keys(w)
        style_ttk()

        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=16, pady=(12, 6))
        label(top, "☀", 24, GOLD, bg=BG).pack(side="left")
        tt = tk.Frame(top, bg=BG)
        tt.pack(side="left", padx=(8, 0))
        label(tt, "SUIT-DYN  MISSION CONTROL", 17, FG, True, bg=BG).pack(anchor="w")
        label(tt, "Aditya-L1 / SUIT · NB03 Mg II k 279.6 nm · forecasting the chromosphere", 9, DIM, bg=BG).pack(anchor="w")
        self.clock = label(top, "", 16, FG, True, bg=BG)
        self.clock.pack(side="right")
        self.pill = tk.Label(top, text="IDLE", bg=GREY, fg=BG, font=(F_UI, 10, "bold"), padx=14, pady=4)
        self.pill.pack(side="right", padx=14)
        self.dsvar = tk.StringVar()
        self.dsbox = ttk.Combobox(top, textvariable=self.dsvar, state="readonly", width=22, style="D.TCombobox",
                                  font=(F_UI, 10))
        self.dsbox.pack(side="right", padx=6)
        self.dsbox.bind("<<ComboboxSelected>>", lambda e: setattr(self, "sel", self.dsvar.get()))
        label(top, "pipeline", 9, DIM, bg=BG).pack(side="right")

        o, flow = card(w, "Stage flow", GOLD)
        o.pack(fill="x", padx=16, pady=4)
        self.flow = tk.Canvas(flow, bg=PANEL, height=118, highlightthickness=0)
        self.flow.pack(fill="x", padx=10, pady=(0, 8))
        self.overall = label(flow.hdr, "", 9, DIM)
        self.overall.pack(side="right")

        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(4, 12))
        body.columnconfigure(0, weight=13)
        body.columnconfigure(1, weight=9)
        body.rowconfigure(0, weight=1)
        left, right = tk.Frame(body, bg=BG), tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        # current stage
        o, cur = card(left, "Now running", BLUE)
        o.pack(fill="x")
        self.cur_name = label(cur, "", 15, FG, True)
        self.cur_name.pack(anchor="w", padx=14)
        self.cur_cmd = tk.Label(cur, text="", fg=DIM, bg=PANEL, font=(F_MONO, 9))
        self.cur_cmd.pack(anchor="w", padx=14)
        self.cur_item = label(cur, "", 10, FG)
        self.cur_item.pack(anchor="w", padx=14, pady=(6, 2))
        self.cur_bar = ttk.Progressbar(cur, style="Gold.Horizontal.TProgressbar", maximum=1000)
        self.cur_bar.pack(fill="x", padx=14)
        self.cur_stats = label(cur, "", 9, DIM)
        self.cur_stats.pack(anchor="w", padx=14, pady=(3, 10))

        o, st = card(left, "Stages", GREEN)
        o.pack(fill="both", expand=True, pady=(8, 0))
        self.tv = tree(st, ("stage", "status", "started", "duration", "tries", "command"), (140, 90, 70, 75, 45, 360), 11,
                       ("command",))
        self.tv.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        o, lg = card(left, "Log", GREY)
        o.pack(fill="x", pady=(8, 0))
        self.log = tk.Text(lg, height=10, bg=INK, fg="#b9c6db", font=(F_MONO, 9), relief="flat", wrap="none",
                           insertbackground=FG)
        self.log.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log.tag_configure("err", foreground=RED)
        self.log.tag_configure("ok", foreground=GREEN)
        self.log.tag_configure("head", foreground=GOLD)

        # right column: system, networks, training
        o, sysc = card(right, "System", VIOLET)
        o.pack(fill="x")
        grid = tk.Frame(sysc, bg=PANEL)
        grid.pack(fill="x", padx=8, pady=(2, 8))
        self.tiles = {}
        for k, (key, title, colr) in enumerate(self.TILES):
            f = tk.Frame(grid, bg=PANEL2, highlightthickness=0)
            f.grid(row=k // 3, column=k % 3, sticky="nsew", padx=4, pady=4)
            grid.columnconfigure(k % 3, weight=1)
            label(f, title, 8, DIM, bg=PANEL2).pack(anchor="w", padx=10, pady=(6, 0))
            v = label(f, "…", 17, colr, True, bg=PANEL2)
            v.pack(anchor="w", padx=10)
            sub = label(f, "", 8, DIM, bg=PANEL2)
            sub.pack(anchor="w", padx=10)
            cv = tk.Canvas(f, bg=PANEL2, height=30, highlightthickness=0)
            cv.pack(fill="x", padx=6, pady=(0, 6))
            self.tiles[key] = (v, sub, cv, colr)

        o, nn = card(right, "Neural networks", GOLD)
        o.pack(fill="x", pady=(8, 0))
        self.cv = tk.Canvas(nn, bg=PANEL, height=430, highlightthickness=0)
        self.cv.pack(fill="x", padx=6, pady=(0, 6))
        self.cv.bind("<Configure>", lambda e: draw_networks(self.cv, e.width))

        o, trc = card(right, "Training", BLUE)
        o.pack(fill="both", expand=True, pady=(8, 0))
        self.curves = tk.Canvas(trc, bg=PANEL, height=150, highlightthickness=0)
        self.curves.pack(fill="x", padx=8)
        self.tv3 = tree(trc, ("run", "data", "inputs", "status", "epochs", "best", "skill", "peak °C"),
                        (120, 90, 50, 65, 55, 45, 60, 60), 5)
        self.tv3.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        self.train_live = label(trc, "", 9, BLUE)
        self.train_live.pack(anchor="w", padx=12, pady=(0, 8))

    def _chain(self, s):
        chains = s.get("chains") or []
        names = [c["name"] for c in chains]
        if list(self.dsbox["values"]) != names:
            self.dsbox["values"] = names
        if self.sel not in names:
            self.sel = names[0] if names else None
        if self.sel and self.dsvar.get() != self.sel:
            self.dsvar.set(self.sel)
        return next((c for c in chains if c["name"] == self.sel), None)

    def refresh(self, s):
        self.pulse = not self.pulse
        self.clock.config(text=datetime.now().strftime("%a %d %b  %H:%M:%S"))
        hist = s.get("hist") or {}
        g = s.get("gpu") or {}
        t = g.get("temp")
        v, sub, cv, colr = self.tiles["gpu_temp"]
        v.config(text=f"{t} °C" if t is not None else "n/a", fg=RED if (t or 0) >= 85 else GOLD if (t or 0) >= 76 else GREEN)
        sub.config(text=g.get("name", "").replace("NVIDIA GeForce ", ""))
        sparkline(cv, hist.get("gpu_temp", []), colr, 40, 95)
        v, sub, cv, colr = self.tiles["gpu_util"]
        v.config(text=f"{g.get('util', 0)} %" if g else "n/a")
        sub.config(text=f"{g.get('power_w', 0):.0f} W · {g.get('mem_used_gb', 0):.1f}/{g.get('mem_total_gb', 0):.1f} GB" if g else "")
        sparkline(cv, hist.get("gpu_util", []), colr, 0, 100)
        c = s.get("cpu")
        v, sub, cv, colr = self.tiles["cpu"]
        v.config(text=f"{c:.0f} %" if c is not None else "…")
        npr = len(s.get("procs") or [])
        sub.config(text=f"{npr} pipeline process{'es' if npr != 1 else ''}")
        sparkline(cv, hist.get("cpu", []), colr, 0, 100)
        r = s.get("ram")
        v, sub, cv, colr = self.tiles["ram"]
        if r:
            v.config(text=f"{r[0]:.1f} GB")
            sub.config(text=f"of {r[1]:.1f} GB")
            sparkline(cv, hist.get("ram", []), colr, 0, r[1])
        d = s.get("disk")
        v, sub, cv, colr = self.tiles["disk"]
        if d:
            v.config(text=f"{d[0]:.0f} GB free", fg=RED if d[0] < 15 else colr)
            sub.config(text=f"of {d[1]:.0f} GB")
            cv.delete("all")
            ww = cv.winfo_width()
            cv.create_rectangle(4, 10, ww - 4, 20, fill=INK, outline="")
            cv.create_rectangle(4, 10, 4 + (ww - 8) * (1 - d[0] / d[1]), 20, fill=colr, outline="")
        days = s.get("days") or {}
        v, sub, cv, colr = self.tiles["data"]
        if days:
            ds = sorted({dd for dd, _ in days})
            v.config(text=f"{sum(days.values()):,} files")
            sub.config(text=f"{ds[0][5:]} → {ds[-1][5:]} · {len(ds)} days")
            cv.delete("all")
            per = [sum(v_ for (dd, _), v_ in days.items() if dd == day) for day in ds]
            ww, top_ = cv.winfo_width(), max(per)
            bwid = (ww - 8) / max(len(per), 1)
            for k, n_ in enumerate(per):
                cv.create_rectangle(4 + k * bwid + 1, 28 - 24 * n_ / top_, 4 + (k + 1) * bwid - 1, 28, fill=colr, outline="")
        else:
            v.config(text="indexing…")

        ch = self._chain(s)
        steps = ch["steps"] if ch else []
        draw_flow(self.flow, steps, self.pulse)
        state = ch["state"] if ch else "idle"
        pc = {"running": BLUE, "done": GREEN, "failed": RED, "stalled": RED}.get(state, GREY)
        self.pill.config(text=state.upper(), bg=pc)
        ndone = sum(x["status"] == "done" for x in steps)
        self.overall.config(text=f"{ndone}/{len(steps)} stages done" + (
            f" · started {datetime.fromtimestamp(ch['started']).strftime('%d %b %H:%M')}" if ch and ch.get("started") else ""))

        # now running
        run = next((x for x in steps if x["status"] in ("running", "stalled")), None) or \
            next((x for x in steps if x["status"] == "failed"), None)
        ds_name = (ch or {}).get("file", "").replace("\\", "/").rsplit("/", 1)[-1]
        beats = [b for b in s.get("beats") or [] if b["alive"] and b.get("dataset") == ds_name
                 and not (b.get("stage") or "").startswith("pipeline")]
        if run:
            self.cur_name.config(text=f"{STATUS_ICON[run['status']]} {run['name']}", fg=STATUS_COLOR[run["status"]])
            self.cur_cmd.config(text=run["cmd"])
        else:
            self.cur_name.config(text="nothing running" if state != "done" else "✔ pipeline complete", fg=DIM if state != "done" else GREEN)
            self.cur_cmd.config(text="")
        if beats:
            b = beats[0]
            i, n = b.get("i"), b.get("n")
            self.cur_item.config(text=f"{b.get('stage')} · {b.get('item') or ''}")
            self.cur_bar["value"] = 1000 * (i + 1) / n if i is not None and n else 0
            rate, eta = b.get("rate"), b.get("eta")
            self.cur_stats.config(text=(f"{i + 1:,} / {n:,}  ({100 * (i + 1) / n:.1f} %)   " if i is not None and n else "")
                                  + (f"{rate:.2f} items/s   " if rate else "") + (f"ETA {fmt_dur(eta)}   " if eta else "")
                                  + (f"elapsed {fmt_dur(time.time() - run['start'])}" if run and run.get("start") else ""))
        else:
            self.cur_item.config(text="(this step reports no per-item progress)" if run and run["status"] == "running" else "")
            self.cur_bar["value"] = 0
            self.cur_stats.config(text=f"elapsed {fmt_dur(time.time() - run['start'])}" if run and run.get("start") and
                                  run["status"] == "running" else "")

        self.tv.delete(*self.tv.get_children())
        for x in steps:
            dur = (x.get("end") or time.time()) - x["start"] if x.get("start") else None
            self.tv.insert("", "end", tags=(x["status"],), values=(
                x["name"], f"{STATUS_ICON.get(x['status'], '')} {x['status']}",
                datetime.fromtimestamp(x["start"]).strftime("%H:%M:%S") if x.get("start") else "",
                fmt_dur(dur) if x["status"] != "pending" else "", x.get("attempts") or "", x["cmd"]))
        self.log.delete("1.0", "end")
        for ln in (ch or {}).get("log_tail", [])[-60:]:
            tag = "err" if any(k in ln for k in ("Error", "Traceback", "FAILED", "failed")) else \
                "ok" if " done" in ln or "passed" in ln else "head" if ln.startswith("=====") else ""
            self.log.insert("end", ln + "\n", tag)
        self.log.see("end")

        runs = s.get("runs") or []
        draw_curves(self.curves, runs)
        self.tv3.delete(*self.tv3.get_children())
        for r in runs[-12:]:
            sk = r.get("skill")
            self.tv3.insert("", "end", tags=(r["status"],), values=(
                r["run"], r["dataset"], r.get("inputs", ""), r["status"], r.get("epochs", ""), r.get("best_epoch", ""),
                f"{100 * sk:.1f} %" if isinstance(sk, (int, float)) else "", r.get("gpu_peak", "")))
        tr = [b for b in s.get("beats") or [] if (b.get("stage") or "").startswith("train") and b["alive"]]
        self.train_live.config(text=(f"▶ {tr[0]['stage']} · {tr[0].get('item', '')} · GPU {tr[0].get('gpu_temp', '?')} °C · "
                                     f"duty {tr[0].get('duty', '?')}") if tr else "no training running")


class FeedWindow:
    def __init__(self, tkroot, col):
        self.col, self.blink, self._shown, self._thumb_keys = col, False, None, None
        w = self.w = tk.Toplevel(tkroot)
        w.title("SUIT-DYN · Live feed")
        w.configure(bg=BG)
        sw = tkroot.winfo_screenwidth()
        try:
            virtual = ctypes.windll.user32.GetSystemMetrics(78)
        except Exception:
            virtual = sw
        w.geometry(f"1720x1020+{sw}+0" if virtual > sw + 100 else "1560x980+60+30")
        w.minsize(1200, 760)
        fullscreen_keys(w)

        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=16, pady=(12, 6))
        self.dot = tk.Canvas(top, width=22, height=22, bg=BG, highlightthickness=0)
        self.dot.pack(side="left")
        self.mode = label(top, "LIVE FEED", 17, FG, True, bg=BG)
        self.mode.pack(side="left", padx=(6, 18))
        self.stage = label(top, "", 13, GOLD, True, bg=BG)
        self.stage.pack(side="left")
        self.clock = label(top, "", 16, FG, True, bg=BG)
        self.clock.pack(side="right")

        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(4, 12))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        o, imgc = card(body, "SUIT frame", GOLD)
        o.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.img = tk.Label(imgc, bg="#000000")
        self.img.pack(fill="both", expand=True, padx=10, pady=(4, 4))
        self.caption = label(imgc, "", 11, FG, True, anchor="w", justify="left")
        self.caption.pack(anchor="w", padx=14)
        self.caption2 = label(imgc, "", 9, DIM, anchor="w", justify="left")
        self.caption2.pack(anchor="w", padx=14, pady=(0, 10))

        side = tk.Frame(body, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        o, pc = card(side, "Progress", BLUE)
        o.pack(fill="x")
        row = tk.Frame(pc, bg=PANEL)
        row.pack(fill="x", padx=14)
        self.pct = label(row, "–", 34, FG, True)
        self.pct.pack(side="left")
        self.count = label(row, "", 11, DIM)
        self.count.pack(side="left", padx=12, pady=(14, 0))
        self.bar = ttk.Progressbar(pc, style="Gold.Horizontal.TProgressbar", maximum=1000)
        self.bar.pack(fill="x", padx=14, pady=(4, 2))
        self.prog = label(pc, "", 10, DIM)
        self.prog.pack(anchor="w", padx=14, pady=(2, 10))

        o, hp = card(side, "Frame header", VIOLET)
        o.pack(fill="x", pady=(8, 0))
        self.hdr = tk.Text(hp, height=11, bg=PANEL, fg=FG, font=(F_MONO, 10), relief="flat")
        self.hdr.pack(fill="x", padx=12, pady=(2, 10))
        self.hdr.tag_configure("k", foreground=DIM)

        o, fs = card(side, "Filmstrip", GREEN)
        o.pack(fill="x", pady=(8, 0))
        self.strip = tk.Canvas(fs, bg=PANEL, height=190, highlightthickness=0)
        self.strip.pack(fill="x", padx=10, pady=(2, 8))

        o, tp = card(side, "Recent files", GREY)
        o.pack(fill="both", expand=True, pady=(8, 0))
        self.ticker = tk.Listbox(tp, bg=PANEL, fg="#c9d4e6", font=(F_MONO, 9), relief="flat", highlightthickness=0,
                                 selectbackground="#243554", activestyle="none", borderwidth=0)
        self.ticker.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    def refresh(self, s):
        self.blink = not self.blink
        self.clock.config(text=datetime.now().strftime("%H:%M:%S"))
        live = [b for b in s.get("beats") or [] if b["alive"] and not (b.get("stage") or "").startswith("pipeline")]
        is_live = bool(live) and s.get("feed_mode") == "live"
        self.dot.delete("all")
        self.dot.create_oval(3, 3, 19, 19, fill=(RED if self.blink else "#7a2020") if is_live else GREY, outline="")
        self.mode.config(text="LIVE FEED" if is_live else "ARCHIVE", fg=FG if is_live else DIM)
        chains = s.get("chains") or []
        if live:
            b = live[0]
            self.stage.config(text=f"{b.get('stage')}  ·  data set {b.get('dataset')}")
            i, n = b.get("i"), b.get("n")
            if i is not None and n:
                self.bar["value"] = 1000 * (i + 1) / n
                self.pct.config(text=f"{100 * (i + 1) / n:.0f}%")
                self.count.config(text=f"{i + 1:,} of {n:,}")
            rate, eta, started = b.get("rate"), b.get("eta"), b.get("stage_started")
            extra = "  ".join(f"{k}={b[k]}" for k in ("horizon", "set", "gpu_temp", "duty") if k in b)
            self.prog.config(text="   ".join(x for x in (f"{rate:.2f} items/s" if rate else "", f"ETA {fmt_dur(eta)}" if eta else "",
                                                       f"elapsed {fmt_dur(time.time() - started)}" if started else "", extra) if x))
        else:
            running = [st for ch in chains[:1] for st in ch["steps"] if st["status"] == "running"]
            self.stage.config(text=(f"{running[0]['name']} is running (no per-file report) — newest frames on the share"
                                    if running else "idle — newest frames on the share"))
            self.bar["value"] = 0
            self.pct.config(text="–")
            self.count.config(text="")
            self.prog.config(text=f"{chains[0]['name']}: {chains[0]['state']}" if chains else "")
        pv = self.col.preview
        if pv.get("key") and pv.get("key") != self._shown:
            self._shown = pv["key"]
            h = pv.get("hdr") or {}
            if pv.get("image") is not None:
                box = max(360, min(self.img.winfo_width(), self.img.winfo_height()) - 8)
                im = pv["image"].copy()
                im.thumbnail((box, box))
                self._photo = ImageTk.PhotoImage(im)
                self.img.config(image=self._photo, text="")
            else:
                self.img.config(image="", text=f"no preview: {pv.get('error')}", fg=DIM, font=(F_UI, 12))
            mode = ""
            if "CRPIX1" in h:
                mode = "centred" if float(h["CRPIX1"]) < 1150 * int(h.get("NAXIS1", 2048)) / 2048 else "offset"
            self.caption.config(text=(("● LIVE  " if is_live else "ARCHIVE  ") + pv["key"]))
            self.caption2.config(text=f"{h.get('DATE-OBS', '')}   {h.get('FTR_NAME', '')}   {h.get('NAXIS1', '')}×"
                                      f"{h.get('NAXIS2', '')}   {h.get('ROI_FF', '')}   pointing {mode}   {pv.get('path') or ''}")
            keys = ["DATE-OBS", "FTR_NAME", "OBS_MODE", "ROI_FF", "NAXIS1", "CMD_EXPT", "MEAS_EXP", "CRPIX1", "CRPIX2",
                    "CROTA2", "RSUN_OBS", "HGLT_OBS", "F_VER", "FLAT_CF", "SCAT_CF"]
            self.hdr.delete("1.0", "end")
            for k in keys:
                if k in h:
                    self.hdr.insert("end", f"{k:<9} ", "k")
                    self.hdr.insert("end", f"{h[k]}\n")
        thumbs = s.get("thumbs") or []
        tkeys = [t[0] for t in thumbs]
        if tkeys != self._thumb_keys:
            self._thumb_keys = tkeys
            self.strip.delete("all")
            self._thumb_photos = []
            wdt = self.strip.winfo_width()
            per_row = 4
            size = max(60, int((wdt - 10) / per_row) - 8)
            for k, (name, im, date) in enumerate(reversed(thumbs)):
                if k >= 2 * per_row:
                    break
                ph = ImageTk.PhotoImage(im.resize((size, size)))
                self._thumb_photos.append(ph)
                x, y = 4 + (k % per_row) * (size + 8), 2 + (k // per_row) * (size + 20)
                self.strip.create_image(x, y, image=ph, anchor="nw")
                self.strip.create_text(x, y + size + 2, text=str(date)[11:19], anchor="nw", fill=DIM, font=(F_UI, 8))
            self.strip.config(height=2 * (size + 20) + 4)
        hist = s.get("history") or []
        self.ticker.delete(0, "end")
        if not hist and s.get("newest"):
            self.ticker.insert("end", "newest NB03 full-disk frames on the share:")
            for name, _, ts in s["newest"]:
                self.ticker.insert("end", f"{ts}  {name}")
            return
        seen = set()
        for hrec in reversed(hist):
            it = hrec.get("item") or ""
            if it in seen or (hrec.get("stage") or "").startswith("pipeline"):
                continue
            seen.add(it)
            self.ticker.insert("end", f"{datetime.fromtimestamp(hrec.get('t', 0)).strftime('%H:%M:%S')}  "
                                      f"{hrec.get('stage', '')[:18]:<18} {it}")
            if len(seen) >= 80:
                break


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
