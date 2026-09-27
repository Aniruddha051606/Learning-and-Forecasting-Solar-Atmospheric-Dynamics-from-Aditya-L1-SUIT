"""GPU thermal safety for training on a laptop that shut down once under full load (2026-09-27 13:08).

gpu_temp() reads the driver's NVML library in-process (~0.02 ms; nvidia-smi only as a fallback).

Thermal holds the GPU near a target temperature with a smooth duty cycle, plus a hard stop. On/off pausing
alone did not work: every restart ran the GPU at full power and the die jumped from below the trip point to
95-96 C before the next reading. Before every batch, the duty fraction d is lowered when the GPU is above
`target` and raised slowly below it, and the loop sleeps work_time * (1/d - 1), so the GPU runs steadily at
part power. `max_temp` is a hard stop: pause until below `resume_temp`, then halve d. It starts low and
ramps slowly: a synthetic load went from 54 C to 90 C within 10 s at d = 0.5; this laptop holds ~65-72 C at
d ~ 0.07-0.10. Call check() before every training AND evaluation batch (the unchecked evaluation pass once
overshot to 90 C). The sensor, clock and sleep are injectable so the logic is unit-tested without a GPU.
"""
import ctypes
import os
import subprocess
import time

import numpy as np


class _NVML:
    def __init__(self):
        self.h = None
        try:
            lib = ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll")) \
                if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
            h = ctypes.c_void_p()
            if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0:
                self.lib, self.h = lib, h
        except Exception:
            self.h = None

    def temp(self):
        t = ctypes.c_uint()
        if self.h is None or self.lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(t)) != 0:
            return float("nan")
        return float(t.value)


NVML = _NVML()


def gpu_temp():
    t = NVML.temp()
    if t == t:
        return t
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


class Thermal:
    def __init__(self, max_temp, resume_temp, target, d0=0.1, d_min=0.03, sensor=gpu_temp, sleep=time.sleep,
                 clock=time.time, sync=None):
        self.max_temp, self.resume_temp, self.target = max_temp, resume_temp, target
        self.d, self.d_min = d0, d_min
        self.sensor, self.sleep, self.clock, self.sync = sensor, sleep, clock, sync
        self.last = None
        self._zero()

    def _zero(self):
        self.peak, self.paused, self.slept, self.temps, self.duties = 0.0, 0.0, 0.0, [], []

    def check(self):
        now = self.clock()
        work = 0.0 if self.last is None else now - self.last
        t = self.sensor()
        if t == t:
            self.peak = max(self.peak, t)
            self.temps.append(t)
            if t >= self.max_temp:
                if self.sync:
                    self.sync()
                t0 = self.clock()
                while True:
                    self.sleep(5)
                    t = self.sensor()
                    if not (t == t) or t < self.resume_temp:
                        break
                self.paused += self.clock() - t0
                self.d = max(self.d_min, self.d * 0.5)
                work = 0.0
            elif t > self.target + 4:
                self.d = max(self.d_min, self.d * 0.8)
            elif t > self.target:
                self.d = max(self.d_min, self.d * 0.95)
            elif t < self.target - 2:
                self.d = min(1.0, self.d * 1.01)
        self.duties.append(self.d)
        pause = min(2.0, work * (1 / self.d - 1))
        if pause > 0:
            self.sleep(pause)
            self.slept += pause
        self.last = self.clock()

    def reset(self):
        out = {"gpu_temp_peak": self.peak, "gpu_temp_mean": round(float(np.mean(self.temps)), 1) if self.temps else None,
               "duty_mean": round(float(np.mean(self.duties)), 3) if self.duties else None,
               "thermal_pause_s": round(self.paused, 1), "throttle_sleep_s": round(self.slept, 1)}
        self._zero()
        return out
