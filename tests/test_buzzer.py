"""Buzzer: the Ubuntu-side service logic (events, dead-man switch, HTTP) and the laptop-side log watcher,
against each other over real HTTP on localhost with a silent speaker.
"""
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "buzzer"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bz = _load("suitdyn_buzzer")
bw = _load("buzz_watch")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _kinds(spk):
    inv = {tuple(v): k for k, v in bz.PATTERNS.items()}
    return [inv[tuple(p)] for p in spk.played]


def test_dead_man_switch_and_fail_repeats():
    clk, spk = Clock(), bz.LogSpeaker()
    b = bz.Buzzer(spk, timeout_s=900, repeat_s=600, repeats=3, clock=clk)
    assert b.tick() is None                      # not armed before the first heartbeat
    b.heartbeat("final_offset: store")
    clk.t += 800
    assert b.tick() is None
    clk.t += 200                                 # 1000 s without a heartbeat
    assert b.tick() == "lost"
    clk.t += 60
    assert b.tick() is None                      # repeats wait repeat_s
    clk.t += 600
    assert b.tick() == "lost"
    b.heartbeat("final_offset: store")           # laptop back: alarm cleared
    assert b.alarm is None and not b.lost_raised
    b.event("fail", "stage store FAILED")
    assert not b.armed
    assert b.tick() == "fail"
    b.ack()
    clk.t += 10_000
    assert b.tick() is None                      # acknowledged, and disarmed by the failure
    with pytest.raises(ValueError):
        b.event("nonsense")


def test_allowed_clients():
    assert bz.allowed("192.168.1.10") and bz.allowed("100.91.1.37") and bz.allowed("127.0.0.1")
    assert bz.allowed("::ffff:192.168.1.10")
    assert not bz.allowed("8.8.8.8") and not bz.allowed("not an ip")


@pytest.fixture
def service():
    spk = bz.LogSpeaker()
    b = bz.Buzzer(spk, timeout_s=3600)
    srv = bz.serve(b, "127.0.0.1", 0, tick_s=0.05)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield b, spk, srv.server_address[1]
    srv.shutdown()


def _wait(cond, s=3.0):
    import time
    t0 = time.time()
    while time.time() - t0 < s:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_watcher_against_service(service, tmp_path):
    b, spk, port = service
    log = tmp_path / "runner.log"
    old = ("===== data set final_offset =====\n[10:00:00] manifest ...\n[10:01:00] manifest done (60 s)\n"
           "pipeline final_offset: 1 stage(s) run in 60 s\n")  # an earlier invocation: ignored
    log.write_text(old + "===== data set final_offset =====\n[20:30:08] manifest ...\n", encoding="utf-8")
    alive = {"v": True}
    w = bw.Watch(log, ["final_offset", "final_centred"], 1234, lambda pid: alive["v"],
                 bw.Sender("127.0.0.1", port), stage_chirp=True)
    assert w.finished == set() and w.current == "final_offset: manifest"
    assert w.step() is None
    assert b.armed and b.last_text == "final_offset: manifest"
    with open(log, "a", encoding="utf-8") as f:
        f.write("[21:10:00] manifest done (2400 s)\n[21:10:00] frames ...\n")
    assert w.step() is None and w.current == "final_offset: frames"
    with open(log, "a", encoding="utf-8") as f:
        f.write("[23:00:00] frames done (1 s)\npipeline final_offset: 19 stage(s) run in 99 s\n"
                "===== data set final_centred =====\n[23:00:01] manifest ...\n[23:00:05] manifest done (4 s)\n"
                "pipeline final_centred: 19 stage(s) run in 9 s\n")
    over = w.step()
    assert over[0] == "done"
    assert _wait(lambda: not b.armed and len(spk.played) >= 5)
    kinds = _kinds(spk)
    assert kinds.count("stage") == 3 and kinds.count("dataset_done") == 2 and "done" in kinds
    status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read())
    assert status["events"][-1]["kind"] == "disarm"


def test_watcher_reports_a_failure_and_a_dead_runner(service, tmp_path):
    b, spk, port = service
    log = tmp_path / "runner.log"
    log.write_text("===== data set final_offset =====\n[20:30:08] store ...\n", encoding="utf-8")
    alive = {"v": True}
    w = bw.Watch(log, ["final_offset"], 1, lambda pid: alive["v"], bw.Sender("127.0.0.1", port))
    assert w.step() is None
    with open(log, "a", encoding="utf-8") as f:
        f.write("stage store FAILED after 3 attempt(s); log: x\n")
    assert w.step()[0] == "fail"
    assert _wait(lambda: "fail" in _kinds(spk))
    # a runner that dies without a FAILED line (power loss, kill) is a failure too
    log2 = tmp_path / "runner2.log"
    log2.write_text("===== data set final_offset =====\n[20:30:08] train:unet:0 ...\n", encoding="utf-8")
    w2 = bw.Watch(log2, ["final_offset"], 2, lambda pid: False, bw.Sender("127.0.0.1", port))
    kind, text = w2.step()
    assert kind == "fail" and "train:unet:0" in text


def test_undeliverable_events_are_kept():
    s = bw.Sender("127.0.0.1", 9, timeout=0.5)   # nothing listens on port 9
    s.event("/event", {"kind": "done"})
    assert len(s.queue) == 1 and not s.heartbeat("x")


def test_speed_ticker_rate():
    clk = Clock()
    b = bz.Buzzer(bz.LogSpeaker(), clock=clk, tick_mbps=5, tick_max=25, tick_min_mbps=0.5, rate_stale_s=10,
                  mode="speed")
    assert b.ticks_per_s() == 0                  # no rate reported yet
    b.set_rate(50)
    assert b.ticks_per_s() == 10                 # one click a second per 5 MB/s
    b.set_rate(400)
    assert b.ticks_per_s() == 25                 # capped
    b.set_rate(0.2)
    assert b.ticks_per_s() == 0                  # idle link: silent
    b.set_rate(50)
    clk.t += 11
    assert b.ticks_per_s() == 0                  # stale (laptop stopped reporting): silent
    b.set_rate(50)
    b.set_ticker(False)
    assert b.ticks_per_s() == 0
    with pytest.raises(ValueError):
        b.set_rate("nan")


def test_speed_ticker_over_http(service):
    b, spk, port = service
    get = lambda q: json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{q}", timeout=5).read())  # noqa: E731
    get("/mode?m=speed")                                      # pulse is the default mode
    assert bw.Sender("127.0.0.1", port).rate(100.0)          # 20 clicks a second
    assert _wait(lambda: spk.clicks >= 4, 3)
    st = get("/")
    assert st["rate_mbps"] == 100.0 and st["ticker_on"] is True
    get("/ticker?on=0")
    n = spk.clicks
    import time
    time.sleep(0.3)
    assert spk.clicks <= n + 1 and get("/")["ticker_on"] is False
    assert not bw.Sender("127.0.0.1", 9).rate(5.0)            # unreachable: reported, never raised


def test_pulse_mode_logic():
    clk = Clock()
    b = bz.Buzzer(bz.LogSpeaker(), clock=clk)
    assert b.mode == "pulse" and not b.pulse_due()   # default mode; silent until the run pulses
    b.set_rate(50)
    assert b.ticks_per_s() == 0                      # no speed clicks in pulse mode
    b.pulse()
    assert b.pulse_due()
    clk.t += 9
    assert b.pulse_due()
    clk.t += 2                                       # 11 s without a pulse: the run stopped
    assert not b.pulse_due()
    b.pulse()
    b.set_ticker(False)
    assert not b.pulse_due()                         # muted
    b.set_ticker(True)
    b.set_mode("off")
    assert not b.pulse_due()
    b.set_mode("pulse")
    assert b.pulse_due()
    b.event("done", "all finished")
    assert not b.pulse_due()                         # stops at once when the run ends
    b.pulse()
    b.disarm()
    assert not b.pulse_due()
    with pytest.raises(ValueError):
        b.set_mode("loud")


def test_pulse_over_http(service):
    b, spk, port = service
    get = lambda q: json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{q}", timeout=5).read())  # noqa: E731
    assert bw.Sender("127.0.0.1", port).pulse()
    assert _wait(lambda: spk.clicks >= 2, 3.5)                # one beep a second
    assert set(spk.click_tones) == {bz.PULSE_TONE}
    st = get("/")
    assert st["mode"] == "pulse" and st["pulse_age_s"] is not None
    with pytest.raises(urllib.error.HTTPError):               # 400: unknown mode
        get("/mode?m=bad")
    assert not bw.Sender("127.0.0.1", 9).pulse()              # unreachable: reported, never raised


@pytest.mark.skipif(sys.platform != "win32", reason="Windows performance counters")
def test_smb_rate_meter():
    m = bw.SmbRate("10.255.255.254")                          # no share on this host
    assert m.read() == 0.0
