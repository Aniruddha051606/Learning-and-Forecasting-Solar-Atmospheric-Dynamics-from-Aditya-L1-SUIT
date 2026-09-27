"""Draws the dashboard icon (a gold Sun with a darker limb) as dashboard/build/suitdyn.ico."""
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(__file__).resolve().parent / "build"
out.mkdir(exist_ok=True)
n = 256
y, x = np.mgrid[:n, :n]
r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n * 0.42)
mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))
rgba = np.zeros((n, n, 4), np.uint8)
disk = r < 1
rgba[..., 0] = np.where(disk, 150 + 105 * mu, 0)
rgba[..., 1] = np.where(disk, 60 + 150 * mu ** 1.4, 0)
rgba[..., 2] = np.where(disk, 10 + 40 * mu ** 3, 0)
rgba[..., 3] = np.where(disk, 255, 0)
Image.fromarray(rgba, "RGBA").save(out / "suitdyn.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(out / "suitdyn.ico")
