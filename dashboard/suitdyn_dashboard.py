"""SUIT-DYN desktop dashboard: two windows, one per screen.

    python dashboard/suitdyn_dashboard.py [--root PROJECT] [--selftest]
    dashboard/build_exe.cmd  ->  dashboard/dist/SUIT-DYN Dashboard.exe
"""
import argparse
import collections
import ctypes
import json
import math
import os
import random
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
BG, PANEL, PANEL2, FG, DIM = "#050505", "#111111", "#1b1b1b", "#ececec", "#8c8c8c"
GOLD, GREEN, BLUE, RED, GREY, VIOLET = "#d4d4d4", "#7cc47f", "#b0b0b0", "#e06c6c", "#4a4a4a", "#9e9e9e"
STATUS_COLOR = {"done": GREEN, "running": BLUE, "pending": GREY, "failed": RED, "stalled": RED, "waiting": GOLD,
                "skipped": GREY}
STATUS_ICON = {"done": "✔", "running": "▶", "pending": "·", "failed": "✖", "stalled": "!", "waiting": "⏸",
               "skipped": "–"}

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
                  "noise_maps_resp", "samples", "background", "train", "evaluate", "evaluate_test"]
SCRIPT_OF = {"manifest": "scripts/build_manifest.py", "frames": "scripts/process_frames.py",
             "registration": "scripts/registration_study.py", "sequences": "scripts/build_sequences.py",
             "calibration": "scripts/calibrate_pattern.py", "store": "scripts/build_store.py",
             "noise_maps": "scripts/phase2_noise_maps.py", "response": "scripts/phase2_response.py",
             "noise_maps_resp": "scripts/phase2_noise_maps.py", "samples": "scripts/phase3_prepare.py",
             "background": "scripts/phase3_background.py", "train": "scripts/phase3_train.py",
             "evaluate": "scripts/phase3_evaluate.py", "evaluate_test": "scripts/phase3_evaluate.py --split test"}


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


def pid_alive(pid, since=None):
    """True if process pid is running and, when since (Unix time) is given, was created no later than since."""
    try:
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p  # a 64-bit handle; the default int return can truncate it
        h = k32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        h = ctypes.c_void_p(h)
        try:
            code = ctypes.c_ulong()
            if not (k32.GetExitCodeProcess(h, ctypes.byref(code)) and code.value == 259):
                return False
            if since is None:
                return True
            times = [ctypes.c_ulonglong() for _ in range(4)]  # creation, exit, kernel, user (FILETIME)
            if not k32.GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
                return True  # cannot tell: running is the best information there is
            return times[0].value / 1e7 - 11644473600 <= float(since) + 5
        finally:
            k32.CloseHandle(h)
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


def fmt_long(s):
    """fmt_dur with days for long waits."""
    if s is None or s != s:
        return ""
    return f"{int(s // 86400)}d {int(s % 86400 // 3600)}h" if s >= 86400 else fmt_dur(s)


def read_csv(path):
    """A CSV as dicts, numbers converted (the exe has no pandas)."""
    import csv
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def ens_table(rows, skill="skill_vs_strongest", lo="lo", hi="hi", region="disk"):
    """{ensemble method: {horizon: (skill, lo, hi)}} from a summary CSV."""
    out = {}
    for r in rows:
        m = r.get("method")
        if r.get("region") == region and isinstance(m, str) and "-ens" in m:
            out.setdefault(m, {})[int(r["horizon"])] = (r.get(skill), r.get(lo), r.get(hi))
    return out


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
def _read_header(f):
    """The primary header from an open FITS file (left positioned at the data)."""
    hdr, end = {}, False
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
    return hdr


def pointing_of(hdr):
    """'centred' or 'offset' from the disc centre in the header (the rule the feed caption uses), else None."""
    if "CRPIX1" not in hdr:
        return None
    return "centred" if float(hdr["CRPIX1"]) < 1150 * int(hdr.get("NAXIS1", 2048)) / 2048 else "offset"


def dataset_scope(root, ds):
    """(first, last) observation-time stamps as in SUIT file names ('2026-09-18T04.59.00') and the pointing
    mode of data set `ds`, from configs/datasets/<ds>.toml; None if it cannot be read.
    """
    try:
        cfg = tomllib.loads((root / "configs" / "datasets" / f"{ds}.toml").read_text(encoding="utf-8"))
        ends = [t for span in cfg["split"].values() if isinstance(span, list) for t in span]
        stamp = lambda t: str(t).strip().replace(" ", "T").replace(":", ".")[:19]
        return (stamp(min(ends)), stamp(max(ends))), cfg.get("dataset", {}).get("pointing_mode")
    except Exception:
        return None


def read_fits(path):
    """Primary HDU of an uncompressed FITS file: (header dict, float32 image)."""
    with open(path, "rb") as f:
        hdr = _read_header(f)
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
        self.newest_label = "newest frames on the share"
        self.active_ds = None  # the data set a live runner is working on (the archive feed shows its frames)
        self._pointing = {}  # file name -> 'centred' / 'offset' / None, from its header
        self.procs, self.procs_t = [], 0.0
        self._rtc, self._csvc, self._p3c = {}, {}, (None, {})  # caches: v2 training logs, CSVs, phase3.toml
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
        t0 = time.time()
        while "chains" not in self.snap and time.time() - t0 < 20:  # know the running data set before the first pick
            time.sleep(0.5)
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
            stamp = re.compile(r"(20\d\d-\d\d-\d\dT\d\d\.\d\d\.\d\d)")
            nb = sorted((m.group(1), f) for f in idx if "NB03" in f and (m := stamp.search(f)))
            ds = self.active_ds
            scope = dataset_scope(self.root, ds) if ds else None
            newest, label = [], "newest frames on the share"
            if scope:  # a runner is live: its own data set's newest frames, in its pointing
                (lo, hi), mode = scope
                inside = [x for x in nb if lo <= x[0] <= hi]
                newest = self._pick(idx, inside[::max(1, len(inside) // 40)], mode)  # spread over the whole span
                label = f"frames of data set {ds} ({lo[:10]} to {hi[:10]}{', ' + mode if mode else ''})"
            if not newest:
                newest, label = self._pick(idx, nb, None), "newest frames on the share"
            with self.lock:
                self.index, self.days, self.index_t = idx, dict(days), time.time()
                self.newest, self.newest_label = newest, label
            for _ in range(60):  # re-index every 5 min, or at once when the running data set changes
                time.sleep(5)
                if self.active_ds != ds:
                    break

    def _pick(self, idx, nb, mode, want=12, tries=400):
        """Newest-first full-disk frames from `nb` [(stamp, name)], only in pointing `mode` when one is
        given.
        """
        out = []
        for ts, f in reversed(nb[-tries:]):
            try:
                if os.path.getsize(idx[f]) <= 8_000_000:  # 2048² binned full disk (ROIs are smaller)
                    continue
                if mode:
                    if f not in self._pointing:
                        with open(idx[f], "rb") as fh:
                            self._pointing[f] = pointing_of(_read_header(fh))
                    if self._pointing[f] != mode:
                        continue
                out.append((f, idx[f], ts.replace(".", ":").replace("T", " ")))
            except (OSError, ValueError):
                pass
            if len(out) >= want:
                break
        return out

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
            b["alive"] = pid_alive(b.get("pid", 0), since=b.get("t", 0))  # the writer existed when it wrote
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
                    alive = pid_alive(json.loads(lock.read_text()).get("pid", 0), since=lock.stat().st_mtime)
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
            names = names[:names.index("evaluate")] + sorted(set(trains) | {k for k in states if k.startswith("train:")}) +                 ["evaluate"] + (["evaluate_test"] if "evaluate_test" in states else [])
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
            if not rd.is_dir() or rd.name.startswith("_") or not any(rd.iterdir()):
                continue
            dsn = rd.parent.parent.parent.name
            ds = dsn + (" (smoke)" if rd.parent.name.endswith("smoke") else "")
            v2 = rd.parent.name == "runs_v2"
            r = {"run": rd.name, "dataset": ds, "status": "pending"}
            if (rd / "run.json").exists():
                try:
                    j = json.loads((rd / "run.json").read_text(encoding="utf-8"))
                    v2 = v2 or j.get("input_mask") == "context"
                    r.update(model=j.get("model"), seed=j.get("seed"), params=j.get("params"),
                             inputs=(j.get("args") or {}).get("inputs", "plain"), best_epoch=j.get("best_epoch"),
                             epochs=j.get("epochs_run"), skill=j.get("best_holdout_skill_vs_B1"), status="done",
                             seconds=j.get("seconds"))
                except Exception:
                    pass
            r["v2"] = v2
            if not rd.parent.name.endswith("smoke"):
                r["dataset"] = ds.replace("final_", "") + (" v2" if v2 else " v1")
            logf = rd.parent.parent.parent.parent.parent / "pipeline" / rd.parent.parent.parent.name / "logs" / \
                f"train_{r.get('model', rd.name.split('_')[0])}_{r.get('seed', rd.name.rsplit('s', 1)[-1])}.log"
            last, curve = None, []
            if v2:  # the retraining chain logs every run of a data set into one file
                ep = self._rt_curves(dsn).get(rd.name, [])
                curve = [(j.get("epoch", 0), j.get("holdout_skill_vs_B1"), j.get("train_l1")) for j in ep]
                last = ep[-1] if ep else None
            for ln in [] if v2 else tail(logf, 400000):
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
        runs.sort(key=lambda r: r.get("v2", False))  # the v2 runs last: the tables and curves show the newest
        return runs

    # -- the v2 retraining (docs/PREREGISTRATION.md Addenda H and I; scripts/retrain_v2.py)
    def _p3(self):
        f = self.root / "configs" / "phase3.toml"
        try:
            mt = f.stat().st_mtime
            if mt != self._p3c[0]:
                self._p3c = (mt, tomllib.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
        return self._p3c[1]

    def _csv(self, path):
        try:
            mt = path.stat().st_mtime
        except OSError:
            return []
        c = self._csvc.get(path)
        if not c or c[0] != mt:
            c = self._csvc[path] = (mt, read_csv(path))
        return c[1]

    def _rt_curves(self, ds):
        """The chain's training log of a data set -> {run: [epoch records]} (the latest attempt of each
        epoch).
        """
        p = self.root / "outputs" / "logs" / "retrain_v2" / f"{ds}_train.log"
        try:
            mt = p.stat().st_mtime
        except OSError:
            return {}
        c = self._rtc.get(ds)
        if c and c[0] == mt:
            return c[1]
        inputs = (self._p3().get("train") or {}).get("inputs", "bg")
        out, cur = {}, None
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"--model (\w+) --seed (\d+)", ln) if ln.startswith("===") else None
            if m:
                cur = f"{m.group(1)}_{inputs}_s{m.group(2)}"
            elif cur and ln.startswith("{") and '"epoch"' in ln:
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                cv = out.setdefault(cur, [])
                while cv and cv[-1].get("epoch", 0) >= j.get("epoch", 0):  # a resumed attempt repeats epochs
                    cv.pop()
                cv.append(j)
        self._rtc[ds] = (mt, out)
        return out

    def _test_g(self, beats):
        """Test G (Addendum G): the old models on the 9 new days; it builds the cache the v2 models are
        scored on.
        """
        log = self.root / "outputs" / "logs" / "new_period.log"
        lines = tail(log, 30000)
        phase, detail, frac = ("not started", "", None) if not lines else ("starting", "", None)
        for ln in lines:
            st = ln.strip()
            if "frames to register" in st:
                phase, detail, frac = "registering frames (reads every new FITS file from the share)", st, None
            elif st.startswith("registration check"):
                phase, detail = "frame selection", st
            elif st.startswith("frame selection"):
                phase, detail = "building the frame cache", st
            elif (m := re.match(r"frame cache (\d+)/(\d+)", st)):
                phase, detail, frac = "building the frame cache", st, int(m.group(1)) / max(int(m.group(2)), 1)
            elif re.match(r"\d+ windows:", st):
                phase, detail, frac = "scoring the old (v1) models", st, None
            elif st.startswith("new period (28 Sep"):
                phase, detail, frac = "done", "cache built; v1 models scored (input-only mask)", 1.0
            elif st.startswith("Traceback") or "Error:" in st:
                phase, detail = "error", st
            elif st.startswith("[stopped") and "cache" in st:  # stopped on purpose once its cache was built
                try:
                    rep_ = json.loads((self.root / "outputs/logs/retrain_v2/after_v2.json").read_text()).get("repair", {})
                except Exception:
                    rep_ = {}
                if rep_.get("status") == "done":
                    phase, detail, frac = "done: frame cache repaired and complete", "scored on all 9 days by after_v2.py", 1.0
                else:
                    phase, detail, frac = ("cache incomplete: 5056 of 6453 frames lost to share read errors",
                                           "the after-training job (after_v2.py) rebuilds them, then scores all 9 days", 0.22)
        b = next((b for b in beats if b["alive"] and b.get("stage") == "posthoc: new period"), None)
        if b and b.get("n"):
            frac, detail = (b.get("i", 0) + 1) / b["n"], f"scoring {b.get('item', '')} of {b['n']}"
        alive = any("posthoc_new_period.py" in p["cmd"] and "--score-only" not in p["cmd"] for p in self.procs)
        cache = (self.root / "outputs/datasets/final_offset/phase3/newperiod/cache/samples_384.parquet").exists()
        try:
            started = log.stat().st_ctime
        except OSError:
            started = None
        return {"phase": phase, "detail": detail, "frac": frac, "alive": alive, "cache": cache, "started": started}

    def _after_steps(self, base):
        """scripts/after_v2.py as grouped steps: status from after_v2.json, start/end and running from
        after_v2.log.
        """
        try:
            st = json.loads((base / "after_v2.json").read_text(encoding="utf-8"))
        except Exception:
            st = {}
        alive = False
        try:
            lk = base / "after_v2.lock"
            alive = pid_alive(json.loads(lk.read_text())["pid"], since=lk.stat().st_mtime)
        except Exception:
            pass
        starts, ends, cur, waiting = {}, {}, None, False
        for ln in tail(base / "after_v2.log", 200000):
            m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$", ln)
            if not m:
                continue
            t, msg = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(), m.group(2)
            if (mm := re.match(r"--- (\w+):", msg)):
                starts.setdefault(mm.group(1), t)
                cur, waiting = mm.group(1), False
            elif (mm := re.match(r"(\w+) (done|FAILED)", msg)):
                ends[mm.group(1)] = t
                cur = None
            elif msg.startswith("waiting for the v2 chain"):
                waiting = True
        out = []
        for lab, prefixes, what, n_steps in AFTER:
            names = [n for n in set(st) | set(starts) if n.startswith(prefixes)]
            n_done = sum(st.get(n, {}).get("status") == "done" for n in names)
            n_fail = sum(st.get(n, {}).get("status") == "failed" for n in names)
            if alive and cur and cur.startswith(prefixes):
                s_ = "running"
            elif n_done + n_fail >= n_steps:
                s_ = "failed" if n_fail else "done"
            elif n_done or n_fail:
                s_ = "running" if alive else "stopped"
            elif alive and waiting and not out:
                s_ = "waiting"
            else:
                s_ = "pending"
            ts = [starts[n] for n in names if n in starts]
            te = [ends[n] for n in names if n in ends]
            out.append({"name": f"after · {lab}", "key": f"after:{lab}", "ds": "after", "kind": lab, "dataset": "",
                        "cmd": what, "status": s_, "start": min(ts) if ts else None,
                        "end": max(te) if te and s_ in ("done", "failed") else None, "attempts": n_fail or None})
        return out

    def _retrain(self, beats):
        base = self.root / "outputs" / "logs" / "retrain_v2"
        logp = base / "chain.log"
        if not logp.exists():
            return None
        try:
            state = json.loads((base / "state.json").read_text(encoding="utf-8"))
        except Exception:
            state = {}
        alive, lk = False, base / "chain.lock"
        try:
            alive = pid_alive(json.loads(lk.read_text())["pid"], since=lk.stat().st_mtime)
        except Exception:
            pass
        lines = logp.read_text(encoding="utf-8", errors="replace").splitlines()
        starts, ends, fails, wait_t = {}, {}, collections.Counter(), None
        for ln in lines:
            m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$", ln)
            if not m:
                continue
            t, msg = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(), m.group(2)
            if (mm := re.match(r"--- (\w+): (\w+)$", msg)):
                starts.setdefault(f"{mm.group(1)}:{mm.group(2)}", t)
            elif (mm := re.match(r"(\w+): (\w+) (done|FAILED|skipped) \(", msg)):
                ends[f"{mm.group(1)}:{mm.group(2)}"] = t
            elif (mm := re.match(r"(\w+)_(train|val|newperiod|test): exit code", msg)):
                fails[f"{mm.group(1)}:{mm.group(2)}"] += 1
            elif msg.startswith("waiting for test G"):
                wait_t = t
        p3 = self._p3()
        steps, cur_ds = [], None
        for ds, kinds in RT_KINDS:
            prev = None
            for k in kinds:
                key = f"{ds}:{k}"
                st = (state.get(key) or {}).get("status")
                if st not in ("done", "failed", "skipped"):
                    dep_done = prev is None or (state.get(f"{ds}:{prev}") or {}).get("status") == "done"
                    if key in starts and key not in ends:
                        st = "running" if alive else "stalled"
                    elif alive and dep_done and wait_t and ds == "final_offset" and k in ("swap", "newperiod") \
                            and wait_t >= ends.get(f"{ds}:{prev}", 0):
                        st = "waiting"
                    else:
                        st = "pending"
                steps.append({"name": f"{RT_SHORT[ds]} · {RT_LABEL[k]}", "key": key, "ds": ds, "kind": k, "dataset": ds,
                              "cmd": RT_WHAT[k], "status": st, "start": starts.get(key), "end": ends.get(key),
                              "attempts": fails[key] + 1 if fails[key] else None})
                if st in ("running", "stalled") and cur_ds is None:
                    cur_ds = ds
                prev = k
        sts = [x["status"] for x in steps]
        cstate = ("done" if all(x in ("done", "skipped") for x in sts) else "failed" if "failed" in sts and not alive
                  else "running" if alive and "running" in sts else "waiting" if alive else "stopped")
        steps += self._after_steps(base)
        if cstate == "done" and any(x["status"] not in ("done", "skipped") for x in steps):
            ast = [x["status"] for x in steps if x["ds"] == "after"]
            cstate = "running" if "running" in ast else "waiting" if "waiting" in ast else \
                "failed" if "failed" in ast else "stopped"

        # the 12 runs
        tr = p3.get("train") or {}
        types, seeds = (p3.get("model") or {}).get("types", ["unet", "convlstm"]), tr.get("seeds", [0, 1, 2])
        inputs, nmax = tr.get("inputs", "bg"), tr.get("epochs", 100)
        runs, spe = [], {}
        for ds, _ in RT_KINDS:
            ph = self.root / "outputs" / "datasets" / ds / "phase3"
            curves = self._rt_curves(ds)
            for m in types:
                for sd in seeds:
                    name = f"{m}_{inputs}_s{sd}"
                    rj, v1 = None, None
                    for d in (ph / "runs_v2" / name, ph / "runs" / name, ph / "runs_v1_targetmask" / name):
                        try:
                            j = json.loads((d / "run.json").read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        if j.get("input_mask") == "context":
                            rj = rj or j
                        else:
                            v1 = v1 or j
                    ep = curves.get(name, [])
                    beat = next((b for b in beats if b["alive"] and b.get("stage") == f"train {name}"
                                 and b.get("dataset") == ds), None)
                    status = ("done" if rj else "running" if beat else
                              "partial" if (ph / "runs_v2" / name / "last.pt").exists() else "pending")
                    hs = [(j.get("epoch", 0), j["holdout_skill_vs_B1"]) for j in ep if j.get("holdout_skill_vs_B1") is not None]
                    secs = [j.get("seconds") for j in ep if j.get("seconds") is not None]
                    dd = [b_ - a_ for a_, b_ in zip(secs, secs[1:]) if b_ > a_][-6:]
                    r = {"ds": ds, "run": name, "model": m, "seed": sd, "status": status, "dataset": f"{RT_SHORT[ds]} v2",
                         "epochs": rj.get("epochs_run") if rj else (ep[-1].get("epoch", 0) + 1 if ep else 0),
                         "best_epoch": rj.get("best_epoch") if rj else (max(hs, key=lambda x: x[1])[0] if hs else None),
                         "skill": rj.get("best_holdout_skill_vs_B1") if rj else (max(x[1] for x in hs) if hs else None),
                         "v1_epochs": v1.get("epochs_run") if v1 else None,
                         "v1_skill": v1.get("best_holdout_skill_vs_B1") if v1 else None,
                         "spe": sorted(dd)[len(dd) // 2] if dd else None,
                         "curve": [(j.get("epoch", 0), j.get("holdout_skill_vs_B1"), j.get("train_l1")) for j in ep],
                         "item": beat.get("item") if beat else None, "gpu": beat.get("gpu_temp") if beat else None,
                         "duty": beat.get("duty") if beat else None,
                         "v1_spe": v1["seconds"] / max(v1.get("epochs_run") or 1, 1) if v1 and v1.get("seconds") else None}
                    if r["spe"]:
                        spe.setdefault((ds, m), r["spe"])
                    runs.append(r)
        # time left: as many epochs as the v1 run of the same name needed (early stopping), at the current pace
        v1r = {}
        for ds, _ in RT_KINDS:
            for m in types:
                xs = [r["v1_spe"] for r in runs if r["ds"] == ds and r["model"] == m and r["v1_spe"]]
                v1r[(ds, m)] = sum(xs) / len(xs) if xs else None
        total, known = 0.0, True
        for r in runs:
            if r["status"] == "done":
                r["eta"] = 0
                continue
            pace = r["spe"] or spe.get((r["ds"], r["model"]))
            if pace is None:  # not started: scale a measured pace by the v1 epoch-time ratio
                for (ds2, m2), p2 in spe.items():
                    a_, b_ = v1r.get((r["ds"], r["model"])), v1r.get((ds2, m2))
                    if a_ and b_:
                        pace = p2 * a_ / b_
                        break
            r["want"] = min(nmax, r["v1_epochs"] or nmax)
            left = max(r["want"] - r["epochs"], 2 if r["status"] in ("running", "partial") else 0)
            r["eta"] = left * pace if pace else None
            if r["eta"] is None:
                known = False
            else:
                total += r["eta"]

        # results (CSV) as they arrive
        res, minutes = [], {}
        for ds, _ in RT_KINDS:
            ph = self.root / "outputs" / "datasets" / ds / "phase3"
            swapped = (ph / "runs_v1_targetmask").exists() and not (ph / "runs_v2").exists()
            v1_eval = ph / "eval_v1_targetmask" / "summary.csv" if (ph / "eval_v1_targetmask").exists() else \
                (ph / "eval" / "summary.csv" if not swapped else None)
            mc = next((ph / d / "mask_check_summary.csv" for d in ("posthoc", "posthoc_v1_targetmask")
                       if (ph / d / "mask_check_summary.csv").exists()), None)
            sets = []
            if v1_eval:
                rows = self._csv(v1_eval)
                for r_ in rows:
                    if r_.get("region") == "disk" and isinstance(r_.get("minutes"), float):
                        minutes.setdefault((ds, int(r_["horizon"])), r_["minutes"])
                sets.append(("val · v1 old mask", ens_table(rows, lo="lo_strongest", hi="hi_strongest"), ""))
            if mc:
                tb = ens_table(self._csv(mc))
                sets.append(("val · v1 input-only", tb, "-ctxmask"))
                res.append({"ds": ds, "label": "leak", "table": tb})
            if swapped and (ph / "eval" / "summary.csv").exists():
                sets.append(("val · v2", ens_table(self._csv(ph / "eval" / "summary.csv"), lo="lo_strongest",
                                                   hi="hi_strongest"), ""))
            npd = ph / "newperiod"
            for f, lab in (("summary_v2.csv", "9 new days · v2"), ("summary_v1_targetmask.csv", "9 new days · v1 old mask"),
                           ("summary_v1_ctxmask.csv", "9 new days · v1 input-only")):
                if (npd / f).exists():
                    sets.append((lab, ens_table(self._csv(npd / f)), ""))
            if not (npd / "summary_v1_ctxmask.csv").exists() and (npd / "summary.csv").exists():
                sets.append(("9 new days · v1 input-only", ens_table(self._csv(npd / "summary.csv")), ""))
            v1_test = ph / "eval_test_v1_targetmask" / "summary.csv" if (ph / "eval_test_v1_targetmask").exists() else \
                (ph / "eval_test" / "summary.csv" if not swapped else None)
            if v1_test and v1_test.exists():
                sets.append(("test · v1 (1st read)", ens_table(self._csv(v1_test), lo="lo_strongest", hi="hi_strongest"), ""))
            if (state.get(f"{ds}:test") or {}).get("status") == "done":
                sets.append(("test · v2 (2nd read)", ens_table(self._csv(ph / "eval_test" / "summary.csv"),
                                                             lo="lo_strongest", hi="hi_strongest"), ""))
            for lab, tb, sfx in sets:
                for m in types:
                    row = tb.get(f"{m}_{inputs}-ens{sfx}")
                    if row:
                        res.append({"ds": ds, "label": lab, "model": m, "row": row})
        lt = lines[-30:]
        cur = next((x for x in steps if x["status"] in ("running", "stalled")), None)
        if cur and cur["ds"] == "after":
            lt = lines[-12:] + ["----- after_v2.log -----"] + tail(base / "after_v2.log", 20000)[-14:]
            cur = None
        if cur:
            sl = tail(base / f"{cur['ds']}_{cur['kind']}.log", 20000)
            sl = [x for x in sl if x.strip() and "Warning" not in x and "warnings.warn" not in x][-8:]
            if sl:
                lt = lt + [f"----- {cur['ds']}_{cur['kind']}.log -----"] + sl
        chain = {"name": "retrain v2 (leak fix)", "file": str(self.root / "outputs" / "datasets" / (cur_ds or "final_offset")),
                 "log": str(logp), "steps": steps, "state": cstate, "log_tail": lt,
                 "started": min(starts.values(), default=None), "mtime": logp.stat().st_mtime}
        return {"state": cstate, "alive": alive, "steps": steps, "runs": runs, "eta": total if known else None,
                "eta_partial": total, "thermal": p3.get("thermal") or {}, "g": self._test_g(beats), "results": res,
                "minutes": minutes, "log_tail": lt, "cur_ds": cur_ds, "chain": chain}

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
                rt = self._retrain(snap["beats"])
                snap["retrain"] = rt
                snap["chains"] = sorted(self._pipelines() + self._chains() + ([rt["chain"]] if rt else []),
                                        key=lambda c: -c["mtime"])
                self.active_ds = next((Path(c["file"]).name for c in snap["chains"]
                                       if c["name"].startswith("pipeline ") and c["state"] == "running"), None) or \
                    (rt["cur_ds"] if rt and rt["alive"] else None)
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
                snap["newest"], snap["newest_label"] = list(self.newest), self.newest_label
                self.snap = snap
            except Exception as e:  # never let the collector die
                self.snap = {**self.snap, "error": repr(e)}
            time.sleep(2)


# ------------------------------------------------------------------------------------------- UI
BORDER, INK = "#2b2b2b", "#0a0a0a"
F_UI, F_MONO = "Segoe UI", "Consolas"
SERIES = [GOLD, BLUE, GREEN, VIOLET, "#dedede", "#c2c2c2", RED, "#cccccc"]
SHORT = {"manifest": "Manifest", "frames": "Frames", "registration": "Registration", "sequences": "Splits",
         "calibration": "Calibration", "store": "Store", "noise_maps": "Noise maps", "response": "Response",
         "noise_maps_resp": "Trusted", "samples": "Samples", "background": "Background", "train": "Train",
         "evaluate": "Evaluate", "evaluate_test": "Test (once)"}
GROUP = {"manifest": "ARCHIVE", "frames": "ARCHIVE", "samples": "LEARNING", "background": "LEARNING",
         "train": "LEARNING", "evaluate": "LEARNING", "evaluate_test": "LEARNING"}

# the v2 retraining chain (scripts/retrain_v2.py; docs/PREREGISTRATION.md Addendum I)
RT_KINDS = (("final_offset", ("train", "swap", "val", "newperiod", "test")),
            ("final_centred", ("train", "swap", "val", "test")))
RT_SHORT = {"final_offset": "offset", "final_centred": "centred"}
RT_LABEL = {"train": "Train 6 runs", "swap": "Keep v1", "val": "Validation", "newperiod": "9 new days",
            "test": "Test (2nd read)"}
RT_WHAT = {"train": "python scripts/phase3_train.py --model {unet, convlstm} --seed {0, 1, 2}  (into phase3/runs_v2)",
           "swap": "old models and results kept as *_v1_targetmask; runs_v2 -> runs",
           "val": "python scripts/phase3_evaluate.py --split val",
           "newperiod": "python scripts/posthoc_new_period.py --score-only  (v2; v1 old mask; v1 input-only)",
           "test": "python scripts/phase3_evaluate.py --split test --allow-new-models  (the registered 2nd read)"}
for _ds, _ks in RT_KINDS:
    for _k in _ks:
        SHORT[f"{RT_SHORT[_ds]} · {RT_LABEL[_k]}"] = RT_LABEL[_k]
        GROUP[f"{RT_SHORT[_ds]} · {RT_LABEL[_k]}"] = f"{RT_SHORT[_ds].upper()} · V2"
# scripts/after_v2.py, grouped: (label, step-name prefixes, what)
AFTER = (("9-day repair", ("repair",), "posthoc_new_period.py --prepare-only: rebuild the frames lost to share read errors", 1),
         ("9-day scoring", ("rescore_",), "v2, v1 old mask, v1 input-only on all 9 days (posthoc_new_period.py --score-only)", 3),
         ("Leak size", ("maskdiff_",), "posthoc_mask_difference.py, both pointings", 2),
         ("Analyses C-F", ("fixed_removal_", "shuffle_denoise_", "classical_", "state_estimate_", "flow_", "cross_"),
          "fixed map, transplant/denoise, classical, state estimate, optical flow, cross-pointing (v2, validation)", 12),
         ("Report", ("report",), "posthoc_report.py: P1-P3, C1-C2 for v2", 1),
         ("Paper", ("paper_",), "make_suit_paper.py, make_tex.py, check_tex.py", 3))
for _lab, _p, _w, _n in AFTER:
    SHORT[f"after · {_lab}"] = _lab
    GROUP[f"after · {_lab}"] = "AFTER TRAINING"


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
    st.map("D.Treeview", background=[("selected", "#333333")])
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
    try:
        th = tomllib.loads((find_root() / "configs" / "phase3.toml").read_text(encoding="utf-8")).get("thermal", {})
        therm = (f"duty-cycle controller, target {th['target_c']:.0f} °C, pause at {th['max_c']:.0f} °C"
                 if th.get("enabled", True) else "controller off (temperature only logged)")
    except Exception:
        therm = "duty-cycle controller (configs/phase3.toml [thermal])"
    cv.create_text(10, y, anchor="nw", fill=FG, font=("Segoe UI", 9), justify="left",
                   text="Both predict the residual over B1 (derotated persistence; or background-aware B1 with --inputs bg).\n"
                        "Output 0 = B1 exactly · loss: masked L1 on valid disk pixels · AdamW 3e-4, OneCycle, bf16 · "
                        "early stop on the hold-out run\nmask = valid pixels of the input frames only (v2) · "
                        f"thermal: {therm}")


class NetAnim:
    """A fully connected network (input, three hidden layers, output) animated from the run's heartbeat."""
    LAYERS = (8, 9, 9, 9, 4)
    NAMES = ("input layer", "hidden layer 1", "hidden layer 2", "hidden layer 3", "output layer")
    EDGE0, EDGE1 = "#161616", "#3c3c3c"  # weak ..
    FWD, BWD = "#f4f4f4", "#9a9a9a"  # forward signal, backward gradient
    FPS = 20

    def __init__(self, parent, col, height=220):
        self.col = col
        self.cv = tk.Canvas(parent, bg=PANEL, height=height, highlightthickness=0)
        self.cv.pack(fill="x", padx=8, pady=(0, 6))
        self.rng = random.Random(7)
        self.wts = [[[self.rng.uniform(-1, 1) for _ in range(b)] for _ in range(a)]
                    for a, b in zip(self.LAYERS, self.LAYERS[1:])]
        self.edges, self.nodes, self.shown, self.cap = {}, {}, {}, None
        self.phase, self.last = 0.0, time.time()
        self.cv.bind("<Configure>", lambda e: self._build(e.width, e.height))
        self.cv.after(500, self._tick)

    def _build(self, w, h):
        cv, L = self.cv, self.LAYERS
        cv.delete("all")
        self.edges, self.nodes, self.shown = {}, {}, {}
        top, bot, m = 26, 36, max(L)
        r = max(4.0, min(9.0, (h - top - bot) / (m * 2.6)))
        step = (h - top - bot - 2 * r) / (m - 1)
        xs = [44 + k * (w - 120) / (len(L) - 1) for k in range(len(L))]
        ys = [[top + r + (m - n) * step / 2 + i * step for i in range(n)] for n in L]
        for g in range(len(L) - 1):
            for i, y0 in enumerate(ys[g]):
                for j, y1 in enumerate(ys[g + 1]):
                    self.edges[(g, i, j)] = cv.create_line(xs[g] + r, y0, xs[g + 1] - r, y1, fill=self.EDGE0)
        for l, x in enumerate(xs):
            for i, y in enumerate(ys[l]):
                self.nodes[(l, i)] = cv.create_oval(x - r, y - r, x + r, y + r, outline=GREY, fill=PANEL2, width=1.2)
            cv.create_text(x, 11, text=self.NAMES[l], fill=DIM, font=(F_UI, 8))
        for y in ys[-1]:
            cv.create_line(xs[-1] + r + 3, y, xs[-1] + r + 32, y, fill=GREY, arrow="last")
        self.cap = cv.create_text(10, h - 4, anchor="sw", text="", fill=DIM, font=(F_UI, 8), justify="left")

    def _state(self):
        beats = [b for b in (self.col.snap.get("beats") or []) if b.get("alive")]
        tr = next((b for b in beats if (b.get("stage") or "").startswith("train")), None)
        ev = next((b for b in beats if (b.get("stage") or "").startswith("evaluate")), None)
        b = tr or ev
        if not b:
            return "idle", None, None
        rate = b.get("rate")
        return ("train" if tr else "infer"), (min(6.0, max(0.8, 1 / rate)) if rate else 2.4), b

    def _tick(self):
        try:
            if self.edges and self.cv.winfo_viewable():
                self._frame()
        except Exception:  # never let the animation break the window
            pass
        self.cv.after(1000 // self.FPS, self._tick)

    def _frame(self):
        mode, period, b = self._state()
        now = time.time()
        dt, self.last = min(now - self.last, 0.5), now
        gaps = len(self.LAYERS) - 1
        pos, glow = None, self.FWD
        if mode != "idle":
            self.phase += dt / period
            if self.phase >= 1:
                self.phase %= 1
                if mode == "train":
                    self._learn()
            if mode == "train":  # first half: forward 0 -> gaps; second half: backward gaps -> 0
                fwd = self.phase < 0.5
                pos = 2 * self.phase * gaps if fwd else (2 - 2 * self.phase) * gaps
                glow = self.FWD if fwd else self.BWD
            else:
                pos = self.phase * gaps
        for (g, i, j), item in self.edges.items():
            w = abs(self.wts[g][i][j])
            c = _mix(self.EDGE1, self.EDGE0, round(w * 8) / 8)
            if pos is not None:
                a = round(math.exp(-((pos - g - 0.5) / 0.5) ** 2) * (0.35 + 0.65 * w) * 10) / 10
                if a > 0:
                    c = _mix(glow, c, a)
            if self.shown.get(item) != c:
                self.cv.itemconfig(item, fill=c)
                self.shown[item] = c
        for (l, i), item in self.nodes.items():
            a = 0.0 if pos is None else round(math.exp(-((pos - l) / 0.4) ** 2) * 10) / 10
            key = (GREY, PANEL2) if a == 0 else (_mix(glow, GREY, a), _mix(glow, PANEL2, 0.8 * a))
            if self.shown.get(item) != key:
                self.cv.itemconfig(item, outline=key[0], fill=key[1])
                self.shown[item] = key
        if mode == "train":
            t = f"training {(b.get('stage') or '')[6:]} · {b.get('item') or ''}\n→ forward pass   ← backward pass (gradients)"
        elif mode == "infer":
            t = f"{b.get('stage')}\nforward passes only (no learning)"
        else:
            t = "idle · no model running"
        if self.shown.get(self.cap) != t:
            self.cv.itemconfig(self.cap, text=t)
            self.shown[self.cap] = t

    def _learn(self):
        """After each batch the weights move a little: the edge shading slowly changes, as training does."""
        for gap in self.wts:
            for row in gap:
                for j in range(len(row)):
                    row[j] = max(-1.0, min(1.0, row[j] + self.rng.gauss(0, 0.05)))


class PipelineWindow:
    TILES =(("gpu_temp", "GPU temperature", GOLD), ("gpu_util", "GPU load", BLUE), ("cpu", "CPU", GREEN),
             ("ram", "Memory", VIOLET), ("disk", "Disk D:", "#c2c2c2"), ("data", "Data on the share", "#dedede"))

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
        lg = logo_label(top)
        if lg:
            lg.pack(side="right", padx=(12, 0))
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
        self.log = tk.Text(lg, height=10, bg=INK, fg="#bdbdbd", font=(F_MONO, 9), relief="flat", wrap="none",
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
        lg = logo_label(top)
        if lg:
            lg.pack(side="right", padx=(12, 0))
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

        o, nc = card(side, "Neural network · schematic", GOLD)
        o.pack(fill="x", pady=(8, 0))
        self.net = NetAnim(nc, col)

        o, hp = card(side, "Frame header", VIOLET)
        o.pack(fill="x", pady=(8, 0))
        self.hdr = tk.Text(hp, height=11, bg=PANEL, fg=FG, font=(F_MONO, 10), relief="flat")
        self.hdr.pack(fill="x", padx=12, pady=(2, 10))
        self.hdr.tag_configure("k", foreground=DIM)

        o, fs = card(side, "Filmstrip", GREEN)
        o.pack(fill="x", pady=(8, 0))
        self.strip = tk.Canvas(fs, bg=PANEL, height=110, highlightthickness=0)
        self.strip.pack(fill="x", padx=10, pady=(2, 8))

        o, tp = card(side, "Recent files", GREY)
        o.pack(fill="both", expand=True, pady=(8, 0))
        self.ticker = tk.Listbox(tp, bg=PANEL, fg="#cfcfcf", font=(F_MONO, 9), relief="flat", highlightthickness=0,
                                 selectbackground="#333333", activestyle="none", borderwidth=0)
        self.ticker.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    def refresh(self, s):
        self.blink = not self.blink
        self.clock.config(text=datetime.now().strftime("%H:%M:%S"))
        live = [b for b in s.get("beats") or [] if b["alive"] and not (b.get("stage") or "").startswith("pipeline")]
        is_live = bool(live) and s.get("feed_mode") == "live"
        self.dot.delete("all")
        self.dot.create_oval(3, 3, 19, 19, fill=(RED if self.blink else "#5c2a2a") if is_live else GREY, outline="")
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
            what = s.get("newest_label") or "newest frames on the share"
            self.stage.config(text=(f"{running[0]['name']} is running (no per-file report) — {what}"
                                    if running else f"idle — {what}"))
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
            mode = pointing_of(h) or ""
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
                if k >= per_row:
                    break
                ph = ImageTk.PhotoImage(im.resize((size, size)))
                self._thumb_photos.append(ph)
                x, y = 4 + (k % per_row) * (size + 8), 2 + (k // per_row) * (size + 20)
                self.strip.create_image(x, y, image=ph, anchor="nw")
                self.strip.create_text(x, y + size + 2, text=str(date)[11:19], anchor="nw", fill=DIM, font=(F_UI, 8))
            self.strip.config(height=size + 24)
        hist = s.get("history") or []
        self.ticker.delete(0, "end")
        if not hist and s.get("newest"):
            self.ticker.insert("end", f"NB03 full-disk {s.get('newest_label') or 'newest frames on the share'}:")
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


# ------------------------------------------------------------------------------------- logo + results
def resource(*parts):
    """A file shipped with the dashboard (inside the exe when frozen by PyInstaller)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).joinpath(*parts)


_LOGOS = {}


def logo_label(parent, height=44, bg=BG):
    """The ISRO logo (dashboard/assets/isro_logo.png) for a header corner, or None if the file is absent."""
    p = next((c for c in (resource("assets", "isro_logo.png"), find_root() / "dashboard" / "assets" / "isro_logo.png")
              if c.exists()), None)
    if p is None:
        return None
    try:
        if height not in _LOGOS:
            im = Image.open(p).convert("RGBA")
            im.thumbnail((height * 4, height))
            _LOGOS[height] = ImageTk.PhotoImage(im)
        return tk.Label(parent, image=_LOGOS[height], bg=bg)
    except Exception:
        return None


def pct(x, signed=True):
    return "–" if x is None else (f"{100 * x:+.2f}%" if signed else f"{100 * x:.2f}%")


class ResultsWindow:
    """Test results against the pre-registered criteria (docs/PREREGISTRATION.md), from
    outputs/tests/report.json (scripts/posthoc_report.py; the 'Rebuild report' button runs it).
    """

    def __init__(self, tkroot, root):
        self.root, self.mtime, self.proc = root, None, None
        w = self.w = tk.Toplevel(tkroot)
        w.title("SUIT-DYN · Results & tests")
        w.configure(bg=BG)
        w.geometry("1640x980+40+40")
        w.minsize(1200, 760)
        fullscreen_keys(w)
        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=16, pady=(12, 6))
        lg = logo_label(top)
        if lg:
            lg.pack(side="right", padx=(12, 0))
        tt = tk.Frame(top, bg=BG)
        tt.pack(side="left")
        label(tt, "RESULTS & TESTS", 17, FG, True, bg=BG).pack(anchor="w")
        label(tt, "pre-registered criteria: docs/PREREGISTRATION.md · report: outputs/tests/report.json", 9, DIM,
              bg=BG).pack(anchor="w")
        self.btn = tk.Button(top, text="Rebuild report", command=self.rebuild, bg=PANEL2, fg=FG, relief="flat",
                             activebackground=GREY, activeforeground=FG, font=(F_UI, 10, "bold"), padx=14, pady=4)
        self.btn.pack(side="right", padx=8)
        self.gen = label(top, "", 10, DIM, bg=BG)
        self.gen.pack(side="right", padx=8)

        o, hc = card(w, "Headline", GOLD)
        o.pack(fill="x", padx=16, pady=4)
        self.head = label(hc, "", 11, FG, anchor="w", justify="left")
        self.head.pack(fill="x", padx=14, pady=(0, 10))

        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=4)
        self.cols = {}
        for k, ds in enumerate(("final_offset", "final_centred")):
            body.columnconfigure(k, weight=1)
            o, c = card(body, ds.replace("final_", "") + " pointing", GOLD)
            o.grid(row=0, column=k, sticky="nsew", padx=(0 if k == 0 else 8, 0))
            st = label(c, "", 9, DIM, anchor="w")
            st.pack(fill="x", padx=14)
            cv = tk.Canvas(c, bg=PANEL, height=230, highlightthickness=0)
            cv.pack(fill="x", padx=10, pady=(4, 4))
            tv = tree(c, ("model", "horizon", "skill", "95% CI", "P1", "day CI", "P2"), (120, 70, 80, 150, 40, 150, 40), 8,
                      stretch=("95% CI", "day CI"))
            tv.pack(fill="x", padx=10)
            tv.tag_configure("yes", foreground=GREEN)
            tv.tag_configure("no", foreground=DIM)
            ctl = label(c, "", 9, FG, anchor="w", justify="left")
            ctl.pack(fill="x", padx=14, pady=(8, 2))
            sec = label(c, "", 9, DIM, anchor="w", justify="left")
            sec.pack(fill="x", padx=14, pady=(2, 10))
            self.cols[ds] = (st, cv, tv, ctl, sec)
        body.rowconfigure(0, weight=1)
        o, sc = card(w, "Solar origin (P3): skill in both pointing modes", GOLD)
        o.pack(fill="x", padx=16, pady=(4, 12))
        self.solar = tk.Label(sc, text="", fg=FG, bg=PANEL, anchor="w", justify="left", font=(F_MONO, 9))
        self.solar.pack(fill="x", padx=14, pady=(0, 10))

    def rebuild(self):
        import shutil
        py = shutil.which("python") or shutil.which("py")
        if not py or (self.proc and self.proc.poll() is None):
            return
        self.proc = subprocess.Popen([py, str(self.root / "scripts" / "posthoc_report.py")], cwd=str(self.root),
                                     creationflags=0x08000000, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.btn.config(text="Building…")
        self.w.after(3000, lambda: self.btn.config(text="Rebuild report"))

    def chart(self, cv, rows, split):
        cv.delete("all")
        W, H = max(cv.winfo_width(), 500), int(cv["height"])
        if not rows:
            cv.create_text(W / 2, H / 2, text="no results yet", fill=DIM, font=(F_UI, 11))
            return
        vals = [v for r in rows for v in (r.get("lo"), r.get("hi"), r.get("skill")) if v is not None]
        top, bot = max(0.01, max(vals)), min(-0.01, min(vals))
        y0, y1 = 22, H - 30

        def Y(v):
            return y1 - (v - bot) / (top - bot) * (y1 - y0)
        cv.create_line(40, Y(0), W - 10, Y(0), fill=GREY)
        cv.create_text(36, Y(0), text="0", fill=DIM, anchor="e", font=(F_UI, 8))
        cv.create_text(36, Y(top), text=pct(top), fill=DIM, anchor="e", font=(F_UI, 8))
        cv.create_text(36, Y(bot), text=pct(bot), fill=DIM, anchor="e", font=(F_UI, 8))
        hs = sorted({r["horizon"] for r in rows})
        ms = sorted({r["method"] for r in rows})
        gw = (W - 60) / max(len(hs), 1)
        bw = gw / (len(ms) + 1)
        shades = {m: c for m, c in zip(ms, ("#d4d4d4", "#7a7a7a", "#a8a8a8"))}
        for i, h in enumerate(hs):
            x0 = 50 + i * gw
            mins = next((r["minutes"] for r in rows if r["horizon"] == h and r.get("minutes")), None)
            cv.create_text(x0 + gw / 2 - bw / 2, H - 12, text=f"{mins / 60:.1f} h" if mins and mins >= 90 else f"{mins:.0f} min" if mins else h,
                           fill=DIM, font=(F_UI, 9))
            for j, m in enumerate(ms):
                r = next((r for r in rows if r["horizon"] == h and r["method"] == m), None)
                if not r or r.get("skill") is None:
                    continue
                xa = x0 + j * bw
                fill = shades[m] if r.get("P1") else _mix(shades[m], PANEL, 0.35)
                cv.create_rectangle(xa, Y(max(r["skill"], 0)), xa + bw * 0.8, Y(min(r["skill"], 0)), fill=fill, outline="")
                if r.get("lo") is not None and r.get("hi") is not None:
                    xm = xa + bw * 0.4
                    cv.create_line(xm, Y(r["lo"]), xm, Y(r["hi"]), fill=FG)
                    cv.create_line(xm - 4, Y(r["lo"]), xm + 4, Y(r["lo"]), fill=FG)
                    cv.create_line(xm - 4, Y(r["hi"]), xm + 4, Y(r["hi"]), fill=FG)
        for j, m in enumerate(ms):
            cv.create_rectangle(50 + j * 170, 6, 62 + j * 170, 16, fill=shades[m], outline="")
            cv.create_text(68 + j * 170, 11, text=m, fill=FG, anchor="w", font=(F_UI, 8))
        cv.create_text(W - 10, 11, text=f"{split} · skill vs the strongest baseline (disk, 95% CI)", fill=DIM, anchor="e",
                       font=(F_UI, 8))

    def refresh(self):
        p = self.root / "outputs" / "tests" / "report.json"
        try:
            mt = p.stat().st_mtime
        except OSError:
            self.head.config(text="No report yet: it is built after the evaluations (press 'Rebuild report', or run\n"
                                  "python scripts/posthoc_report.py). The offset results arrive after its evaluation stage.")
            return
        if mt == self.mtime:
            return
        self.mtime = mt
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            self.head.config(text=f"report unreadable: {e}")
            return
        self.gen.config(text=f"report {rep.get('generated', '')}")
        warn = ["⚠ v1 models (validity mask included the target): overstated per Addendum H. The v2 results arrive in "
                "the 'Retraining v2' window."] if (self.root / "outputs" / "logs" / "retrain_v2" / "chain.log").exists() else []
        self.head.config(text="\n".join(warn + rep.get("headline", [])))
        for ds, (st, cv, tv, ctl, sec) in self.cols.items():
            b = rep.get("datasets", {}).get(ds)
            if not b:
                st.config(text="not in the report")
                continue
            s = b["status"]
            st.config(text="   ".join(f"{k}: {'✔' if v else '·'}" for k, v in s.items()))
            split = "test" if b["test"] else "val"
            rows = b[split]
            self.w.update_idletasks()
            self.chart(cv, rows, split)
            tv.delete(*tv.get_children())
            for r in rows:
                tv.insert("", "end", values=(r["method"], f"{r['minutes']:.0f} min" if r.get("minutes") else r["horizon"],
                                             pct(r["skill"]), f"[{pct(r['lo'])}, {pct(r['hi'])}]", "yes" if r.get("P1") else "no",
                                             f"[{pct(r.get('lo_day'))}, {pct(r.get('hi_day'))}]" if "lo_day" in r else "–",
                                             ("–" if r.get("P2") is None else "yes" if r["P2"] else "no")),
                          tags=("yes" if r.get("P1") else "no",))
            c = b["controls"]
            verdict = lambda v: "pending" if v is None else ("PASS" if v else "FAIL")  # noqa: E731
            shuf = [x["skill"] for x in c.get("shuffle", []) if x.get("skill") is not None]
            ctl.config(text=f"C1 frozen Sun: {verdict(c.get('C1'))}     C2 corotation: {verdict(c.get('C2'))}"
                            f" (max fixed share {pct(c.get('max_fixed_share'), False)})     shuffle median: "
                            f"{pct(float(np.median(shuf))) if shuf else '–'}",
                       fg=RED if c.get("C1") is False or c.get("C2") is False else FG)
            fl = [x for x in b.get("flow", []) if x["method"] == "OF-adv" and x["reference"] == "B1-avg-bgS"]
            cr = b.get("cross", [])
            sec.config(text=("S1 optical flow vs strongest: " + "  ".join(f"{x['minutes']:.0f}m {pct(x['skill'])}" for x in
                                                                       sorted(fl, key=lambda x: x["horizon"])) if fl else "S1 optical flow: pending")
                       + "\n" + ("S2 transferred models: " + "  ".join(f"{x['method']} {x['minutes']:.0f}m {pct(x['skill'])}"
                                                                        for x in cr[:6]) if cr else "S2 cross-mode: pending"))
        sol = rep.get("solar", [])
        if sol:
            lines = [f"{'model':<18}{'horizon':>9}{'offset':>11}{'centred':>11}   P3"]
            for r in sol:
                lines.append(f"{r['method']:<18}{r['minutes']:>7.0f} m{pct(r.get('final_offset')):>11}"
                             f"{pct(r.get('final_centred')):>11}   {'yes' if r['P3'] else 'no'}")
            self.solar.config(text="\n".join(lines))
        else:
            self.solar.config(text="needs the sealed test of both pointing modes")


class RetrainWindow:
    """Window 4: the v2 retraining without the target-validity leak (docs/PREREGISTRATION.md Addenda H and
    I).
    """
    HZ = (20, 40, 80, 160)
    NAME = {"unet": "U-Net", "convlstm": "ConvLSTM"}

    def __init__(self, tkroot):
        self.pulse, self._res_sig = False, None
        w = self.w = tk.Toplevel(tkroot)
        w.title("SUIT-DYN · Retraining v2 (leak fix)")
        w.configure(bg=BG)
        w.geometry("1680x1000+24+24")
        w.minsize(1280, 800)
        fullscreen_keys(w)
        st = ttk.Style()
        st.configure("S.Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=21,
                     font=(F_UI, 9), borderwidth=0)
        st.configure("S.Treeview.Heading", background=PANEL2, foreground=DIM, font=(F_UI, 8, "bold"), borderwidth=0,
                     relief="flat")
        st.map("S.Treeview", background=[("selected", "#333333")])

        top = tk.Frame(w, bg=BG)
        top.pack(fill="x", padx=16, pady=(12, 4))
        tt = tk.Frame(top, bg=BG)
        tt.pack(side="left")
        label(tt, "RETRAINING v2 · LEAK FIX", 17, FG, True, bg=BG).pack(anchor="w")
        label(tt, "all 12 models retrained with a validity mask built from the input frames only · "
                  "docs/PREREGISTRATION.md, Addenda H and I", 9, DIM, bg=BG).pack(anchor="w")
        lg = logo_label(top)
        if lg:
            lg.pack(side="right", padx=(12, 0))
        self.clock = label(top, "", 16, FG, True, bg=BG)
        self.clock.pack(side="right")
        self.pill = tk.Label(top, text="…", bg=GREY, fg=BG, font=(F_UI, 10, "bold"), padx=14, pady=4)
        self.pill.pack(side="right", padx=14)
        self.eta = label(top, "", 11, GOLD, True, bg=BG, justify="right")
        self.eta.pack(side="right", padx=10)

        o, fl = card(w, "The plan, in order (scripts/retrain_v2.py)", GOLD)
        o.pack(fill="x", padx=16, pady=4)
        self.flow = tk.Canvas(fl, bg=PANEL, height=112, highlightthickness=0)
        self.flow.pack(fill="x", padx=10, pady=(0, 4))
        self.overall = label(fl.hdr, "", 9, DIM)
        self.overall.pack(side="right")

        body = tk.Frame(w, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(4, 12))
        body.columnconfigure(0, weight=10)
        body.columnconfigure(1, weight=11)
        body.rowconfigure(0, weight=1)
        left, right = tk.Frame(body, bg=BG), tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        o, why = card(left, "Why: the leak (Addendum H)", RED)
        o.pack(fill="x")
        label(why, "The models were also given a map of which pixels of the TARGET frame are valid: a hint from the "
                   "future (about 0.5 % of the disk pixels in almost every window). Scored without that hint on the "
                   "validation split (skill vs the strongest baseline, %, old mask → input-only):", 9, FG,
              justify="left", wraplength=700, anchor="w").pack(fill="x", padx=14, pady=(0, 4))
        self.leak = self._tree(why, ("model", "h20", "h40", "h80", "h160"), (150, 120, 120, 120, 120), 4)
        self.leak.tag_configure("bad", foreground=RED)
        self.leak.tag_configure("ok", foreground=GREEN)
        self.leak.pack(fill="x", padx=10)
        label(why, "Fix: the mask now comes from the input frames only. The old (v1) models and results are kept as "
                   "*_v1_targetmask; claims come from v2 only.", 9, DIM, justify="left", wraplength=700,
              anchor="w").pack(fill="x", padx=14, pady=(4, 8))

        o, stc = card(left, "Steps", GREEN)
        o.pack(fill="x", pady=(8, 0))
        self.steps = self._tree(stc, ("step", "status", "started", "duration", "tries", "what"),
                                (140, 85, 95, 70, 40, 360), 9, stretch=("what",))
        self.steps.pack(fill="x", padx=10, pady=(2, 8))

        o, gc = card(left, "Test G: the old models on the 9 new days (28 Sep - 6 Oct)", BLUE)
        o.pack(fill="x", pady=(8, 0))
        self.g_phase = label(gc, "", 12, FG, True, anchor="w")
        self.g_phase.pack(fill="x", padx=14)
        self.g_bar = ttk.Progressbar(gc, style="Gold.Horizontal.TProgressbar", maximum=1000)
        self.g_bar.pack(fill="x", padx=14, pady=(4, 2))
        self.g_detail = label(gc, "", 9, DIM, anchor="w", justify="left", wraplength=700)
        self.g_detail.pack(fill="x", padx=14, pady=(0, 8))

        o, lgc = card(left, "Chain log (outputs/logs/retrain_v2)", GREY)
        o.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(lgc, height=6, bg=INK, fg="#bdbdbd", font=(F_MONO, 9), relief="flat", wrap="none")
        self.log.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self.log.tag_configure("err", foreground=RED)
        self.log.tag_configure("ok", foreground=GREEN)
        self.log.tag_configure("head", foreground=GOLD)

        o, gpu = card(right, "GPU and the thermal limits", VIOLET)
        o.pack(fill="x")
        row = tk.Frame(gpu, bg=PANEL)
        row.pack(fill="x", padx=14)
        self.temp = label(row, "…", 24, GOLD, True)
        self.temp.pack(side="left")
        info = tk.Frame(row, bg=PANEL)
        info.pack(side="left", padx=16)
        self.gpu1 = label(info, "", 10, FG, anchor="w")
        self.gpu1.pack(anchor="w")
        self.gpu2 = label(info, "", 9, DIM, anchor="w")
        self.gpu2.pack(anchor="w")
        self.spark = tk.Canvas(gpu, bg=PANEL, height=34, highlightthickness=0)
        self.spark.pack(fill="x", padx=10, pady=(2, 8))

        o, trc = card(right, "Training runs (v2)", BLUE)
        o.pack(fill="x", pady=(8, 0))
        self.curves = tk.Canvas(trc, bg=PANEL, height=104, highlightthickness=0)
        self.curves.pack(fill="x", padx=8)
        self.runs = self._tree(trc, ("data", "run", "status", "epoch", "best", "skill", "v1 skill", "min/ep", "left"),
                               (70, 100, 75, 80, 45, 70, 70, 60, 80), 12)
        self.runs.pack(fill="x", padx=8, pady=(4, 2))
        self.live = label(trc, "", 9, BLUE, anchor="w")
        self.live.pack(fill="x", padx=12, pady=(0, 8))

        o, rsc = card(right, "Results as they arrive (disk, seed ensembles, skill vs the strongest baseline, %, 95% CI)", GOLD)
        o.pack(fill="both", expand=True, pady=(8, 0))
        self.res = self._tree(rsc, ("model", "result", "h20", "h40", "h80", "h160"), (115, 165, 120, 120, 120, 120), 7)
        self.res.tag_configure("v2", foreground=GOLD)
        self.res.tag_configure("v1", foreground=DIM)
        self.res.pack(fill="both", expand=True, padx=8, pady=(2, 8))

    def _tree(self, parent, cols, widths, height, stretch=()):
        tv = tree(parent, cols, widths, height, stretch)
        tv.configure(style="S.Treeview")
        return tv

    @staticmethod
    def _mins(m):
        return "" if not m else f"≈{m:.0f} min" if m < 90 else f"≈{m / 60:.1f} h"

    def _headings(self, rt):
        for tv in (self.leak, self.res):
            for h in self.HZ:
                m = rt["minutes"].get(("final_offset", h))
                tv.heading(f"h{h}", text=f"+{h} FR {self._mins(m)}".upper() if m else f"+{h} FRAMES")

    def refresh(self, s):
        self.pulse = not self.pulse
        self.clock.config(text=datetime.now().strftime("%a %d %b  %H:%M:%S"))
        rt = s.get("retrain")
        if not rt:
            self.overall.config(text="the retraining chain has not started yet (outputs/logs/retrain_v2/chain.log)")
            return
        state = rt["state"]
        self.pill.config(text="STOPPED · double-click 'Resume SUIT-DYN run'" if state == "stopped" else state.upper(),
                         bg={"running": BLUE, "done": GREEN, "failed": RED, "stopped": RED, "waiting": GOLD}.get(state, GREY))
        runs = rt["runs"]
        nd = sum(r["status"] == "done" for r in runs)
        if nd == len(runs):
            self.eta.config(text=f"training finished ({nd}/{len(runs)} runs)")
        elif rt.get("eta") is not None:
            fin = datetime.fromtimestamp(time.time() + rt["eta"]).strftime("%a %d %b %H:%M")
            self.eta.config(text=f"training left ≈ {fmt_long(rt['eta'])}  (≈ {fin})\n{nd}/{len(runs)} runs done · "
                                 "then the evaluations (~2-3 h)")
        else:
            self.eta.config(text=f"training left ≥ {fmt_long(rt['eta_partial'])} (measuring)\n{nd}/{len(runs)} runs done")
        draw_flow(self.flow, rt["steps"], self.pulse)
        nds = sum(x["status"] in ("done", "skipped") for x in rt["steps"])
        started = rt["chain"].get("started")
        self.overall.config(text=f"{nds}/{len(rt['steps'])} steps done" +
                            (f" · started {datetime.fromtimestamp(started).strftime('%d %b %H:%M')}" if started else ""))
        self._headings(rt)

        # leak table
        self.leak.delete(*self.leak.get_children())
        for e in rt["results"]:
            if e["label"] != "leak":
                continue
            for m in ("unet", "convlstm"):
                old = next((v for k, v in e["table"].items() if k.startswith(m + "_") and k.endswith("-ens")), None)
                new = next((v for k, v in e["table"].items() if k.startswith(m + "_") and k.endswith("-ens-ctxmask")), None)
                if not old or not new:
                    continue
                cells, drop = [], 0.0
                for h in self.HZ:
                    a, b = (old.get(h) or (None,))[0], (new.get(h) or (None,))[0]
                    cells.append(f"{100 * a:+.1f} → {100 * b:+.1f}" if a is not None and b is not None else "")
                    if a is not None and b is not None:
                        drop = max(drop, a - b)
                self.leak.insert("", "end", values=(f"{RT_SHORT[e['ds']]} {self.NAME[m]}", *cells),
                                 tags=("bad" if drop > 0.01 else "ok",))

        # steps
        self.steps.delete(*self.steps.get_children())
        for x in rt["steps"]:
            dur = (x.get("end") or time.time()) - x["start"] if x.get("start") else None
            self.steps.insert("", "end", tags=(x["status"],), values=(
                x["name"], f"{STATUS_ICON.get(x['status'], '')} {x['status']}",
                datetime.fromtimestamp(x["start"]).strftime("%d %b %H:%M") if x.get("start") else "",
                fmt_long(dur) if x["status"] not in ("pending", "waiting") else "", x.get("attempts") or "", x["cmd"]))

        # test G
        g = rt["g"]
        done = g["phase"].startswith("done")
        colr = GREEN if done else RED if g["phase"] == "error" or not g["alive"] else BLUE
        self.g_phase.config(text=(("✔ " if done else "▶ " if g["alive"] else "■ stopped: ") + g["phase"]), fg=colr)
        self.g_bar["value"] = 1000 * (g["frac"] or 0)
        sw = next((x for x in rt["steps"] if x["key"] == "final_offset:swap"), {})
        self.g_detail.config(text=(g["detail"] + "\n" if g["detail"] else "") +
                             (f"started {datetime.fromtimestamp(g['started']).strftime('%d %b %H:%M')} · " if g["started"] else "") +
                             ("its frame cache is ready for the v2 scoring" if g["cache"] else
                              "it builds the frame cache that the v2 models are scored on") +
                             (" · the offset swap is waiting for it" if sw.get("status") == "waiting" else ""))

        # GPU
        th, gg = rt["thermal"], s.get("gpu") or {}
        t = gg.get("temp")
        tgt, mx, rs = th.get("target_c"), th.get("max_c"), th.get("resume_c")
        on = th.get("enabled", True)
        self.temp.config(text=f"{t} °C" if t is not None else "n/a",
                         fg=RED if t is not None and mx and t >= mx else GOLD if t is not None and tgt and t >= tgt - 5 else GREEN)
        live = next((r for r in runs if r["status"] == "running"), None)
        self.gpu1.config(text=(f"load {gg.get('util', '?')} % · {gg.get('power_w', 0):.0f} W" +
                               (f" · controller duty {live['duty']:.2f}" if live and live.get("duty") is not None else "")))
        self.gpu2.config(text=(f"controller on: target {tgt:.0f} °C · pause at {mx:.0f} °C until below {rs:.0f} °C · "
                               "the GPU slows itself at 97 °C" if on and tgt else "controller OFF: full power, temperature only logged"))
        hist = (s.get("hist") or {}).get("gpu_temp", [])
        sparkline(self.spark, hist, VIOLET, 40, 100)
        if tgt and self.spark.winfo_width() > 20:
            h_ = self.spark.winfo_height()
            for v, c_ in ((tgt, GOLD), (mx, RED)):
                if v:
                    y = h_ - 3 - (h_ - 6) * (min(max(v, 40), 100) - 40) / 60
                    self.spark.create_line(2, y, self.spark.winfo_width() - 2, y, fill=c_, dash=(3, 3))
                    self.spark.create_text(self.spark.winfo_width() - 4, y - 6, text=f"{v:.0f}", fill=c_, anchor="e",
                                           font=(F_UI, 7))

        # runs
        draw_curves(self.curves, [r for r in runs if r["curve"]])
        self.runs.delete(*self.runs.get_children())
        for r in runs:
            sk, v1 = r.get("skill"), r.get("v1_skill")
            want = r.get("want")
            self.runs.insert("", "end", tags=(r["status"],), values=(
                RT_SHORT[r["ds"]], f"{self.NAME.get(r['model'], r['model'])} s{r['seed']}", r["status"],
                (f"{r['epochs']}" + (f" / ~{want}" if want and r["status"] != "done" else "")) if r["epochs"] or want else "",
                r.get("best_epoch") if r.get("best_epoch") is not None else "",
                f"{100 * sk:+.1f} %" if isinstance(sk, (int, float)) else "",
                f"{100 * v1:+.1f} %" if isinstance(v1, (int, float)) else "",
                f"{r['spe'] / 60:.1f}" if r.get("spe") else "",
                "" if r["status"] == "done" else fmt_long(r["eta"]) if r.get("eta") is not None else "?"))
        self.live.config(text=(f"▶ {live['run']} ({RT_SHORT[live['ds']]}) · {live.get('item') or ''} · GPU "
                               f"{live.get('gpu', '?')} °C" if live else
                               "no training running" + (" (the chain is stopped)" if state == "stopped" else "")))

        # results
        sig = tuple((e["ds"], e["label"], e.get("model"), tuple(sorted((e.get("row") or {}).items())))
                    for e in rt["results"] if e["label"] != "leak")
        if sig != self._res_sig:
            self._res_sig = sig
            self.res.delete(*self.res.get_children())
            for e in rt["results"]:
                if e["label"] == "leak":
                    continue
                cells = []
                for h in self.HZ:
                    v = e["row"].get(h)
                    if not v or v[0] is None:
                        cells.append("")
                    elif isinstance(v[1], float) and isinstance(v[2], float):
                        cells.append(f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]")
                    else:
                        cells.append(f"{100 * v[0]:+.1f}")
                self.res.insert("", "end", values=(f"{RT_SHORT[e['ds']]} {self.NAME.get(e['model'], e['model'])}",
                                                   e["label"], *cells), tags=("v2" if "v2" in e["label"] else "v1",))

        self.log.delete("1.0", "end")
        for ln in rt["log_tail"][-60:]:
            tag = "err" if any(k in ln for k in ("Error", "Traceback", "FAILED", "exit code")) else \
                "ok" if (" done" in ln or "finished" in ln) else "head" if ln.startswith(("---", "=====")) or " --- " in ln else ""
            self.log.insert("end", ln + "\n", tag)
        self.log.see("end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--selftest", action="store_true", help="collect once, render one preview, write a report, exit")
    a = ap.parse_args()
    # above-normal priority: pipeline stages keep all cores busy (store at 96 % CPU), and a starved window is
    # flagged "not responding" and closed by Windows (2026-09-29 02:32).
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit pseudo-handle: the default int return truncates it
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass
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
        rt = s.get("retrain")
        if rt:
            rep["retrain"] = {"state": rt["state"], "steps": {x["key"]: x["status"] for x in rt["steps"]},
                              "runs": {f"{r['ds']}:{r['run']}": (r["status"], r["epochs"], r.get("eta")) for r in rt["runs"]},
                              "eta_h": round(rt["eta"] / 3600, 1) if rt.get("eta") else None, "test_g": rt["g"],
                              "results": [(e["ds"], e["label"], e.get("model")) for e in rt["results"]]}
        out = root / "outputs" / "logs" / "dashboard_selftest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        print(json.dumps(rep, indent=1, default=str))
        return
    tkroot = tk.Tk()
    pw = PipelineWindow(tkroot, col)
    fw = FeedWindow(tkroot, col)
    rw = ResultsWindow(tkroot, root)
    rtw = RetrainWindow(tkroot)

    def tick():
        s = col.snap
        if s:
            try:
                pw.refresh(s)
                fw.refresh(s)
            except Exception as e:  # keep the UI alive whatever a refresh hits
                pw.clock.config(text=f"refresh error: {e}")
        if s:
            try:
                rtw.refresh(s)
            except Exception as e:
                rtw.overall.config(text=f"refresh error: {e}")
        try:
            rw.refresh()  # reads report.json only when it changed
        except Exception as e:
            rw.head.config(text=f"results refresh error: {e}")
        tkroot.after(1500, tick)

    tkroot.after(500, tick)
    tkroot.mainloop()


if __name__ == "__main__":
    main()
