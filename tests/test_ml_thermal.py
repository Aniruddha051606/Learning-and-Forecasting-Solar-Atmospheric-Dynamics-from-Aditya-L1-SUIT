from suitdyn.ml.thermal import Thermal


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_duty_falls_above_target_and_rises_below():
    clk = Clock()
    temps = iter([80.0] * 30 + [60.0] * 400)
    th = Thermal(max_temp=85, resume_temp=72, target=75, d0=0.5, sensor=lambda: next(temps), sleep=clk.sleep, clock=clk)
    for _ in range(30):
        clk.t += 0.1  # one batch of work
        th.check()
    low = th.d
    assert low < 0.5 * 0.95 ** 20
    for _ in range(400):
        clk.t += 0.1
        th.check()
    assert th.d > low


def test_sleep_follows_duty():
    clk = Clock()
    th = Thermal(85, 72, 75, d0=0.2, sensor=lambda: 74.0, sleep=clk.sleep, clock=clk)
    th.check()                      # first call: no work measured yet
    clk.t += 1.0                    # one second of work
    before = clk.t
    th.check()
    assert abs((clk.t - before) - min(2.0, 1.0 * (1 / 0.2 - 1))) < 1e-9


def test_hard_stop_pauses_until_cool_and_halves_duty():
    clk = Clock()
    seq = iter([90.0, 88.0, 80.0, 70.0])
    synced = []
    th = Thermal(85, 72, 75, d0=0.4, sensor=lambda: next(seq), sleep=clk.sleep, clock=clk, sync=lambda: synced.append(1))
    th.check()
    assert synced and th.d == 0.2
    stats = th.reset()
    assert stats["gpu_temp_peak"] == 90.0 and stats["thermal_pause_s"] == 15.0
