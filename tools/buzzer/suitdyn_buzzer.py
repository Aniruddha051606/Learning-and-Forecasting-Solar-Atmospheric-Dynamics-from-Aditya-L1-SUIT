#!/usr/bin/env python3
"""SUIT-DYN buzzer: sounds the PC speaker (motherboard buzzer) of the Ubuntu data machine when the training
laptop reports a pipeline event.
"""
import argparse
import glob
import ipaddress
import json
import os
import random
import struct
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

EV_SND, SND_TONE = 0x12, 0x02

# (frequency Hz, milliseconds); frequency 0 is a pause
PATTERNS = {
    "test": [(1000, 150), (0, 100), (1500, 150)],
    "stage": [(1800, 60)],
    "back": [(1200, 100), (0, 80), (1200, 100)],
    "dataset_done": [(880, 150), (0, 60), (1175, 150), (0, 60), (1760, 350)],
    "done": [(880, 150), (1175, 150), (1480, 150), (1760, 450), (0, 400)] * 2,
    "fail": [(1500, 250), (0, 100), (700, 250), (0, 100)] * 4,
    "lost": [(500, 700), (0, 500)] * 4,
}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def find_pcspkr():
    """/dev/input/eventN of the kernel's "PC Speaker" input device, or None."""
    for p in sorted(glob.glob("/dev/input/by-path/*pcspkr*event*")):
        return p
    for d in sorted(glob.glob("/sys/class/input/event*")):
        try:
            with open(os.path.join(d, "device", "name")) as f:
                if f.read().strip() == "PC Speaker":
                    return "/dev/input/" + os.path.basename(d)
        except OSError:
            continue
    return None


class PcSpeaker:
    """Tones through the PC-speaker input device (struct input_event: timeval, type, code, value)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.path = None

    def _device(self):
        self.path = self.path or find_pcspkr()
        if self.path is None:
            raise OSError("no PC Speaker input device (is the pcspkr module loaded? sudo modprobe pcspkr)")
        return self.path

    def _write(self, fd, freq):
        os.write(fd, struct.pack("llHHi", 0, 0, EV_SND, SND_TONE, int(freq)))

    def click(self, freq=2000, ms=5):
        """One short tick or beep of the ticker; skipped while an alarm pattern is playing."""
        if not self.lock.acquire(blocking=False):
            return
        try:
            try:
                fd = os.open(self._device(), os.O_WRONLY)
            except OSError:
                self.path = None
                raise
            try:
                self._write(fd, freq)
                time.sleep(ms / 1000)
            finally:
                try:
                    self._write(fd, 0)
                finally:
                    os.close(fd)
        finally:
            self.lock.release()

    def play(self, pattern):
        self.path = None  # re-detected for every pattern (the device can change after a module reload)
        path = self._device()
        with self.lock:
            fd = os.open(path, os.O_WRONLY)
            try:
                for freq, ms in pattern:
                    self._write(fd, freq)
                    time.sleep(ms / 1000)
            finally:
                try:
                    self._write(fd, 0)
                finally:
                    os.close(fd)


class LogSpeaker:
    """No sound: records what would have been played (tests, machines without a speaker)."""

    def __init__(self):
        self.played, self.clicks, self.click_tones = [], 0, []

    def play(self, pattern):
        self.played.append(pattern)

    def click(self, freq=2000, ms=5):
        self.clicks += 1
        self.click_tones.append((freq, ms))


TAILSCALE = ipaddress.ip_network("100.64.0.0/10")
MODES = ("pulse", "speed", "off")
PULSE_TONE = (1000, 40)  # the once-a-second beep of the pulse mode (Hz, ms); speed clicks are 2000 Hz, 5 ms


def allowed(addr):
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_private or ip.is_loopback or (ip.version == 4 and ip in TAILSCALE)


class Buzzer:
    """Event and dead-man logic, independent of HTTP (tick() is driven by a thread or by tests)."""

    def __init__(self, speaker, timeout_s=900, repeat_s=600, repeats=3, clock=time.time, tick_mbps=5.0, tick_max=25.0,
                 tick_min_mbps=0.5, ticker_on=True, rate_stale_s=10.0, mode="pulse", pulse_ttl_s=10.0):
        self.speaker, self.timeout_s, self.repeat_s, self.repeats, self.clock = speaker, timeout_s, repeat_s, repeats, clock
        self.tick_mbps, self.tick_max, self.tick_min_mbps, self.rate_stale_s = tick_mbps, tick_max, tick_min_mbps, rate_stale_s
        self.ticker_on, self.rate, self.rate_t, self.click_error = ticker_on, 0.0, None, None
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.pulse_ttl_s, self.pulse_t = mode, pulse_ttl_s, None
        self.lock = threading.Lock()
        self.armed, self.last_hb, self.last_text, self.lost_raised = False, None, "", False
        self.alarm, self.alarm_left, self.next_alarm = None, 0, 0.0
        self.events = deque(maxlen=50)

    def _note(self, kind, text):
        self.events.append({"t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock())), "kind": kind,
                            "text": text})
        log(f"{kind}: {text}")

    def _sound(self, kind):
        try:
            self.speaker.play(PATTERNS[kind])
        except Exception as e:  # a missing speaker must not stop the service
            log(f"could not sound '{kind}': {e}")

    def _sound_async(self, kind):
        threading.Thread(target=self._sound, args=(kind,), daemon=True).start()

    def _raise(self, kind):
        self.alarm, self.alarm_left, self.next_alarm = kind, self.repeats, self.clock()

    def event(self, kind, text=""):
        if kind not in PATTERNS:
            raise ValueError(f"unknown kind {kind!r}; one of {sorted(PATTERNS)}")
        with self.lock:
            self._note(kind, text)
            if kind in ("done", "fail"):
                self.armed, self.pulse_t = False, None  # the run is over: no heartbeats, no pulse beeps
            if kind == "fail":
                self._raise("fail")
                return
        self._sound_async(kind)

    def heartbeat(self, text=""):
        back = False
        with self.lock:
            self.armed, self.last_hb, self.last_text = True, self.clock(), text
            if self.lost_raised:
                self.lost_raised, back = False, True
                if self.alarm == "lost":
                    self.alarm, self.alarm_left = None, 0
                self._note("back", text)
        if back:
            self._sound_async("back")

    def disarm(self):
        with self.lock:
            self.armed, self.lost_raised, self.pulse_t = False, False, None
            if self.alarm == "lost":
                self.alarm, self.alarm_left = None, 0
            self._note("disarm", "")

    def ack(self):
        with self.lock:
            self.alarm, self.alarm_left = None, 0
            self._note("ack", "")

    def set_rate(self, mbps):
        mbps = float(mbps)
        if not mbps >= 0:  # also refuses NaN
            raise ValueError(f"bad rate {mbps!r}")
        with self.lock:
            self.rate, self.rate_t = mbps, self.clock()

    def set_ticker(self, on):
        with self.lock:
            self.ticker_on = bool(on)
            self._note("ticker", "on" if on else "off")

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.lock:
            self.mode = mode
            self._note("mode", mode)

    def pulse(self):
        with self.lock:
            self.pulse_t = self.clock()

    def pulse_due(self):
        """True while the pulse mode should beep: unmuted and the run's pulse is fresh."""
        with self.lock:
            return (self.mode == "pulse" and self.ticker_on and self.pulse_t is not None
                    and self.clock() - self.pulse_t <= self.pulse_ttl_s)

    def ticks_per_s(self):
        """Mean click rate of the speed ticker now (0 unless in speed mode, and when muted, stale or idle)."""
        with self.lock:
            if (self.mode != "speed" or not self.ticker_on or self.rate_t is None
                    or self.clock() - self.rate_t > self.rate_stale_s or self.rate < self.tick_min_mbps):
                return 0.0
            return min(self.rate / self.tick_mbps, self.tick_max)

    def click(self, freq=2000, ms=5):
        try:
            self.speaker.click(freq, ms)
            self.click_error = None
        except Exception as e:
            if self.click_error != str(e):  # log a speaker problem once, not 25 times a second
                log(f"ticker click failed: {e}")
            self.click_error = str(e)

    def tick(self):
        """Raise 'lost' when heartbeats stop; sound a due repeating alarm."""
        now = self.clock()
        with self.lock:
            if self.armed and not self.lost_raised and now - self.last_hb > self.timeout_s:
                self.lost_raised = True
                self._note("lost", f"no heartbeat for {(now - self.last_hb) / 60:.0f} min (last: {self.last_text})")
                self._raise("lost")
            if not (self.alarm and self.alarm_left > 0 and now >= self.next_alarm):
                return None
            kind = self.alarm
            self.alarm_left -= 1
            self.next_alarm = now + self.repeat_s
            if self.alarm_left == 0:
                self.alarm = None
        self._sound(kind)
        return kind

    def status(self):
        with self.lock:
            return {"armed": self.armed, "seconds_since_heartbeat": None if self.last_hb is None else
                    round(self.clock() - self.last_hb), "last": self.last_text, "alarm": self.alarm,
                    "alarm_repeats_left": self.alarm_left, "speaker": find_pcspkr(), "ticker_on": self.ticker_on,
                    "mode": self.mode, "pulse_age_s": None if self.pulse_t is None else round(self.clock() - self.pulse_t, 1),
                    "rate_mbps": self.rate, "rate_age_s": None if self.rate_t is None else round(self.clock() - self.rate_t, 1),
                    "events": list(self.events)[-15:]}


def handler_for(buzzer):
    class Handler(BaseHTTPRequestHandler):
        def _params(self):
            u = urlparse(self.path)
            p = {k: v[-1] for k, v in parse_qs(u.query).items()}
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                body = self.rfile.read(min(n, 65536)).decode("utf-8", "replace")
                try:
                    p.update({k: str(v) for k, v in json.loads(body).items()})
                except (ValueError, AttributeError):
                    p.update({k: v[-1] for k, v in parse_qs(body).items()})
            return u.path.rstrip("/") or "/", p

        def _reply(self, code, obj):
            data = json.dumps(obj, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self):
            if not allowed(self.client_address[0]):
                return self._reply(403, {"error": "LAN and Tailscale clients only"})
            path, p = self._params()
            text = p.get("text", "")[:300]
            try:
                if path == "/":
                    return self._reply(200, buzzer.status())
                if path == "/event":
                    buzzer.event(p.get("kind", ""), text)
                elif path == "/heartbeat":
                    buzzer.heartbeat(text)
                elif path == "/disarm":
                    buzzer.disarm()
                elif path == "/ack":
                    buzzer.ack()
                elif path == "/rate":
                    buzzer.set_rate(p.get("mbps", ""))
                elif path == "/ticker":
                    buzzer.set_ticker(p.get("on", "1").lower() not in ("0", "off", "false", "no"))
                elif path == "/pulse":
                    buzzer.pulse()
                elif path == "/mode":
                    buzzer.set_mode(p.get("m", ""))
                else:
                    return self._reply(404, {"error": "unknown path", "paths": ["/", "/event", "/heartbeat", "/disarm",
                                                                                 "/ack", "/rate", "/ticker", "/pulse", "/mode"]})
            except ValueError as e:
                return self._reply(400, {"error": str(e)})
            self._reply(200, {"ok": True})

        do_GET = do_POST = _handle

        def log_message(self, fmt, *args):  # heartbeats would flood the journal
            pass

    return Handler


def serve(buzzer, host, port, tick_s=1.0, pulse_s=1.0):
    srv = ThreadingHTTPServer((host, port), handler_for(buzzer))
    srv.daemon_threads = True

    def ticker():
        while True:
            buzzer.tick()
            time.sleep(tick_s)

    def speed_ticker(slot_s=0.01, rng=random.Random()):
        # Poisson clicks: in each short slot, a click with probability rate x slot (random gaps, like a Geiger
        # counter)
        last = time.monotonic()
        while True:
            time.sleep(slot_s)
            now = time.monotonic()
            if rng.random() < buzzer.ticks_per_s() * (now - last):
                buzzer.click()
            last = now

    def pulse_beeper():
        # one beep per pulse_s on a fixed schedule (no drift), while the run's pulse is fresh
        nxt = time.monotonic()
        while True:
            nxt += pulse_s
            if buzzer.pulse_due():
                buzzer.click(*PULSE_TONE)
            time.sleep(max(0.0, nxt - time.monotonic()))

    threading.Thread(target=ticker, daemon=True).start()
    threading.Thread(target=speed_ticker, daemon=True).start()
    threading.Thread(target=pulse_beeper, daemon=True).start()
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--timeout-min", type=float, default=15)
    s.add_argument("--repeat-min", type=float, default=10)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--speaker", choices=["pc", "none"], default="pc")
    s.add_argument("--tick-mbps", type=float, default=5.0, help="MB/s per click per second")
    s.add_argument("--tick-max", type=float, default=25.0, help="most clicks per second")
    s.add_argument("--tick-min-mbps", type=float, default=0.5, help="silent below this rate")
    s.add_argument("--no-ticker", action="store_true", help="start with the ticker muted")
    s.add_argument("--mode", choices=MODES, default="pulse", help="ticker: 1 beep/s while the run lives, speed clicks, or off")
    p = sub.add_parser("play")
    p.add_argument("kind", choices=sorted(PATTERNS))
    a = ap.parse_args(argv)
    if a.cmd == "play":
        PcSpeaker().play(PATTERNS[a.kind])
        return
    spk = PcSpeaker() if a.speaker == "pc" else LogSpeaker()
    buzzer = Buzzer(spk, a.timeout_min * 60, a.repeat_min * 60, a.repeats, tick_mbps=a.tick_mbps, tick_max=a.tick_max,
                    tick_min_mbps=a.tick_min_mbps, ticker_on=not a.no_ticker, mode=a.mode)
    srv = serve(buzzer, a.host, a.port)
    log(f"listening on {a.host}:{a.port}; speaker device: {find_pcspkr() or 'NOT FOUND'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
