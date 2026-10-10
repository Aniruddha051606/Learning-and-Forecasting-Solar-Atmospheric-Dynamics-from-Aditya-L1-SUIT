"""GPU thermal safety for training on a laptop that shut down once under full load (2026-09-27 13:08)."""
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


class NoThermal:
    """Stand-in on the CPU (tests): nothing to throttle."""

    d = 1.0

    def check(self):
        pass

    def reset(self):
        return {}


class Monitor:
    """[thermal] enabled = false: no throttling and no pauses; the GPU temperature is still read and logged."""

    d = 1.0

    def __init__(self, sensor=gpu_temp):
        self.sensor = sensor
        self._zero()

    def _zero(self):
        self.peak, self.temps = 0.0, []

    def check(self):
        t = self.sensor()
        if t == t:
            self.peak = max(self.peak, t)
            self.temps.append(t)

    def reset(self):
        out = {"gpu_temp_peak": self.peak, "gpu_temp_mean": round(float(np.mean(self.temps)), 1) if self.temps else None,
               "duty_mean": 1.0, "thermal_pause_s": 0.0, "throttle_sleep_s": 0.0}
        self._zero()
        return out


def controller(cfg, device, sync=None):
    """The duty-cycle controller for CUDA work, a no-op elsewhere."""
    if not str(device).startswith("cuda"):
        return NoThermal()
    if not cfg.get("enabled", True):
        return Monitor()
    return Thermal(cfg["max_c"], cfg["resume_c"], cfg["target_c"], sync=sync)


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
