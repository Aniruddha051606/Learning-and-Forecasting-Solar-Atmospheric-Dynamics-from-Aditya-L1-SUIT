"""Laptop side of the SUIT-DYN buzzer: follows a pipeline run and tells the buzzer service on the Ubuntu data
machine (tools/buzzer/suitdyn_buzzer.py) when to sound.

    python tools/buzzer/buzz_watch.py --dataset final_offset,final_centred --detach
    python tools/buzzer/buzz_watch.py --test                 (one test beep, to check the connection)
    python tools/buzzer/buzz_watch.py --rate-only --detach   (only the speed ticker, no run watching)
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from suitdyn import paths  # noqa: E402

DEFAULT_HOST = "192.168.1.2"  # the Ubuntu data machine on the local network (on Tailscale: 100.85.244.64, "issdc")
PY = sys.executable

RE_FINISHED = re.compile(r"^pipeline (\S+): \d+ stage\(s\) run")
RE_STAGE_DONE = re.compile(r"^\[\d\d:\d\d:\d\d\] (\S+) done \(")
RE_STAGE_START = re.compile(r"^\[\d\d:\d\d:\d\d\] (\S+) \.\.\.$")
RE_DATASET = re.compile(r"^===== data set (\S+) =====")
RE_FAILED = re.compile(r"stage (\S+) FAILED|preflight failed|another runner \(pid")


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


class Sender:
    """HTTP to the buzzer service; important events are queued until delivered."""

    def __init__(self, host, port, retry_s=24 * 3600, timeout=5):
        self.base, self.retry_s, self.timeout, self.queue = f"http://{host}:{port}", retry_s, timeout, []
        self.reachable = None

    def _get(self, path, params):
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return json.loads(r.read() or b"{}")

    def heartbeat(self, text):
        try:
            self._get("/heartbeat", {"text": text})
            ok, err = True, ""
        except Exception as e:
            ok, err = False, str(e)
        if ok != self.reachable:  # log changes only, not every cycle
            say(f"buzzer service reachable at {self.base}" if ok else f"buzzer service NOT reachable at {self.base}: {err}")
            self.reachable = ok
        return ok

    def pulse(self):
        """The run is alive: best effort, never queued."""
        try:
            urllib.request.urlopen(f"{self.base}/pulse", timeout=1.5).close()
            return True
        except Exception:
            return False

    def rate(self, mbps):
        """Speed-ticker update: best effort, never queued (a stale rate is worthless)."""
        try:
            url = f"{self.base}/rate?{urllib.parse.urlencode({'mbps': f'{mbps:.2f}'})}"
            urllib.request.urlopen(url, timeout=1.5).close()
            return True
        except Exception:
            return False

    def event(self, path, params):
        self.queue.append((time.time(), path, params))
        self.flush()

    def flush(self):
        keep = []
        for t, path, params in self.queue:
            try:
                self._get(path, params)
                say(f"sent {path} {params}")
            except Exception as e:
                if time.time() - t < self.retry_s:
                    keep.append((t, path, params))
                else:
                    say(f"gave up on {path} {params}: {e}")
        self.queue = keep
        return not keep


class SmbRate:
    """MB/s of the SMB traffic with one host's shares: Windows counter "Data Bytes/sec" of "SMB Client
    Shares", summed over that host's share instances (e.g. //192.168.1.2/data), read through PDH.
    """

    PDH_FMT_DOUBLE, PDH_MORE_DATA = 0x00000200, 0x800007D2

    def __init__(self, host):
        import ctypes
        from ctypes import wintypes
        self.ct, self.host = ctypes, host.lower()

        class Value(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

        class Item(ctypes.Structure):
            _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", Value)]

        self.Item = Item
        pdh = self.pdh = ctypes.windll.pdh
        H, P = ctypes.c_void_p, ctypes.POINTER
        pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, P(H)]
        pdh.PdhAddEnglishCounterW.argtypes = [H, ctypes.c_wchar_p, ctypes.c_size_t, P(H)]
        pdh.PdhCollectQueryData.argtypes = [H]
        pdh.PdhGetFormattedCounterArrayW.argtypes = [H, wintypes.DWORD, P(wintypes.DWORD), P(wintypes.DWORD), ctypes.c_void_p]
        for f in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhGetFormattedCounterArrayW"):
            getattr(pdh, f).restype = wintypes.DWORD
        self.query, self.counter = H(), H()
        self._check(pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)), "PdhOpenQuery")
        self._check(pdh.PdhAddEnglishCounterW(self.query, "\\SMB Client Shares(*)\\Data Bytes/sec", 0,
                                              ctypes.byref(self.counter)), "PdhAddEnglishCounter")
        pdh.PdhCollectQueryData(self.query)  # a rate needs two samples: the first read() is relative to this one

    @staticmethod
    def _check(status, what):
        if status:
            raise OSError(f"{what} failed: 0x{status:08X}")

    def read(self):
        ct, pdh = self.ct, self.pdh
        from ctypes import wintypes
        self._check(pdh.PdhCollectQueryData(self.query), "PdhCollectQueryData")
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        st = pdh.PdhGetFormattedCounterArrayW(self.counter, self.PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), None)
        if st != self.PDH_MORE_DATA:
            return 0.0  # no share connected (no instance)
        buf = (ct.c_byte * size.value)()
        self._check(pdh.PdhGetFormattedCounterArrayW(self.counter, self.PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count),
                                                     ct.cast(buf, ct.c_void_p)), "PdhGetFormattedCounterArray")
        items = ct.cast(buf, ct.POINTER(self.Item))
        total = 0.0
        for i in range(count.value):
            name = (items[i].szName or "").lower().lstrip("\\")
            if name.startswith(self.host + "\\") and items[i].FmtValue.CStatus in (0, 1):  # valid / new data
                total += items[i].FmtValue.doubleValue
        return total / 1e6


def read_rate(meter):
    try:
        return meter.read()
    except Exception:
        return None


def runner_start(lines, first_ds):
    """Index of the line that starts the current runner invocation (the log is appended across runs)."""
    for i in range(len(lines) - 1, -1, -1):
        if RE_DATASET.match(lines[i]) and RE_DATASET.match(lines[i]).group(1) == first_ds:
            return i
    return 0


def read_lines(path, pos):
    """New complete lines of a text file from byte offset pos; returns (lines, new pos)."""
    try:
        with open(path, "rb") as f:
            f.seek(pos)
            data = f.read()
    except OSError:
        return [], pos
    end = data.rfind(b"\n") + 1
    text = data[:end].decode("utf-8", "replace")
    return [ln.rstrip("\r") for ln in text.split("\n")[:-1]], pos + end


class Watch:
    """Log-and-process state machine; step() is one cycle (tests drive it directly)."""

    def __init__(self, log, datasets, pid, alive, sender, stage_chirp=False):
        self.log, self.datasets, self.pid, self.alive, self.send = Path(log), list(datasets), pid, alive, sender
        self.stage_chirp, self.finished, self.current, self.over = stage_chirp, set(), "starting", None
        lines, self.pos = read_lines(self.log, 0)
        # what the current invocation already did before the watcher started is known, not announced
        for ln in lines[runner_start(lines, self.datasets[0]):]:
            self._parse(ln, announce=False)

    def _parse(self, ln, announce=True):
        m = RE_DATASET.match(ln)
        if m:
            self.ds = m.group(1)
            return
        m = RE_STAGE_START.match(ln)
        if m:
            self.current = f"{getattr(self, 'ds', '?')}: {m.group(1)}"
            return
        m = RE_STAGE_DONE.match(ln)
        if m and announce and self.stage_chirp:
            self.send.event("/event", {"kind": "stage", "text": f"{getattr(self, 'ds', '?')}: {m.group(1)} done"})
        m = RE_FINISHED.match(ln)
        if m:
            self.finished.add(m.group(1))
            if announce:
                self.send.event("/event", {"kind": "dataset_done", "text": f"{m.group(1)} finished"})
            return
        if RE_FAILED.search(ln):
            self.over = ("fail", ln.strip()[:250])

    def step(self):
        """One cycle."""
        alive = self.alive(self.pid)  # before reading: lines written before the exit are then all seen
        lines, self.pos = read_lines(self.log, self.pos)
        for ln in lines:
            self._parse(ln)
        if self.over is None and set(self.datasets) <= self.finished:
            self.over = ("done", "all data sets finished: " + ", ".join(self.datasets))
        if self.over is None and not alive:
            self.over = ("fail", f"runner (pid {self.pid}) stopped before finishing; last: {self.current}")
        if self.over:
            kind, text = self.over
            self.send.event("/event", {"kind": kind, "text": text})
            self.send.event("/disarm", {})
            return self.over
        self.send.heartbeat(self.current)
        self.send.flush()
        return None


def runner_pid(datasets, wait_s):
    """pid of the runner working on these data sets (from a runner.lock), waiting up to wait_s for one."""
    t0 = time.time()
    while True:
        for ds in datasets:
            lock = paths.pipeline("runner.lock", name=ds, make=False)
            try:
                return int(json.loads(lock.read_text())["pid"])
            except (OSError, ValueError, KeyError):
                pass
        if time.time() - t0 > wait_s:
            return None
        time.sleep(10)


def pid_alive(pid):
    from suitdyn.pipeline import pid_alive as alive
    return alive(pid)


def detach(argv):
    """Restart this command outside the current session (WMI process, new group, no window)."""
    args = [a for a in argv if a != "--detach"]
    log = paths.OUT / "logs" / "buzz_watch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(f'"{a}"' if " " in a else a for a in args)
    cmdline = f'cmd.exe /c ""{PY}" "{Path(__file__).resolve()}" {quoted} >> "{log}" 2>&1"'
    ps = ("$s = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32](0x200 -bor 0x08000000)}; "
          f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{cmdline.replace(chr(39), chr(39) * 2)}'; "
          f"CurrentDirectory='{ROOT}'; ProcessStartupInformation=$s}}; \"$($r.ReturnValue) $($r.ProcessId)\"")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    code, pid = (r.stdout.strip().split() + ["?", "?"])[:2]
    print(f"detached buzzer watcher: return {code}, pid {pid}; log {log}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final_offset,final_centred")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--share-host", default="192.168.1.2", help="host whose SMB share traffic drives the speed ticker")
    ap.add_argument("--rate-every", type=float, default=2, help="seconds between speed-ticker updates")
    ap.add_argument("--rate-only", action="store_true", help="only the speed ticker (runs until stopped)")
    ap.add_argument("--no-rate", action="store_true", help="no speed-ticker updates")
    ap.add_argument("--no-pulse", action="store_true", help="no once-a-second 'run alive' pulses")
    ap.add_argument("--every", type=float, default=60, help="seconds between heartbeats")
    ap.add_argument("--retry-h", type=float, default=24)
    ap.add_argument("--stage-chirp", action="store_true")
    ap.add_argument("--log", help="runner log (default: the first data set's logs/runner.log)")
    ap.add_argument("--pid", type=int, help="runner pid (default: from runner.lock)")
    ap.add_argument("--test", action="store_true", help="send one test beep and exit")
    ap.add_argument("--detach", action="store_true")
    a = ap.parse_args(argv)
    send = Sender(a.host, a.port, a.retry_h * 3600)
    if a.test:
        try:
            send._get("/event", {"kind": "test", "text": "test from the laptop"})
            print("delivered: the buzzer should beep twice. Status:", json.dumps(send._get("/", {}), indent=1))
        except Exception as e:
            sys.exit(f"buzzer service not reachable at {send.base}: {e}")
        return
    if a.detach:
        return detach(argv)
    meter = None
    if not a.no_rate:
        try:
            meter = SmbRate(a.share_host)
        except Exception as e:
            say(f"no share-traffic meter ({e}); the speed ticker gets no data")
    if a.rate_only:
        if meter is None:
            sys.exit("--rate-only needs the share-traffic meter")
        say(f"speed ticker only: SMB traffic with {a.share_host} -> {send.base} every {a.rate_every} s")
        while True:
            mbps = read_rate(meter)
            if mbps is not None:
                send.rate(mbps)
            time.sleep(a.rate_every)
    datasets = [d.strip() for d in a.dataset.split(",") if d.strip()]
    log = Path(a.log) if a.log else paths.pipeline("logs", "runner.log", name=datasets[0], make=False)
    pid = a.pid or runner_pid(datasets, wait_s=1800)
    if pid is None:
        send.event("/event", {"kind": "fail", "text": f"no runner found for {a.dataset}"})
        sys.exit(f"no runner.lock for {a.dataset} within 30 min")
    say(f"watching runner pid {pid}, log {log}, data sets {datasets}; buzzer {send.base}")
    w = Watch(log, datasets, pid, pid_alive, send, a.stage_chirp)
    say(f"already finished: {sorted(w.finished) or 'none'}; now: {w.current}")
    next_step = 0.0
    while True:
        if meter is not None:
            mbps = read_rate(meter)
            if mbps is not None:
                send.rate(mbps)
        if time.time() >= next_step:
            over = w.step()
            if over:
                say(f"run over: {over}")
                break
            next_step = time.time() + a.every
        if not a.no_pulse and pid_alive(pid):
            send.pulse()
        time.sleep(a.rate_every if (meter is not None or not a.no_pulse) else a.every)
    t0 = time.time()
    while not send.flush() and time.time() - t0 < a.retry_h * 3600:
        time.sleep(60)


if __name__ == "__main__":
    main()
