"""Time-lapse video of every NB03 (Mg II k 279.6 nm) full-disk frame, disk-centred, at 60 frames per second.

    python tools/make_nb03_video.py [--fps 60] [--size 1080] [--workers 3] [--limit N] [--out path.mp4]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from suitdyn import paths, progress  # noqa: E402

VMAX, GAMMA = 3.0, 0.75  # display: intensity / disk-interior median in [0, VMAX], then gamma (plage unsaturated)


def lut():
    import matplotlib
    rgb = (matplotlib.colormaps["afmhot"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()  # OpenCV wants BGR


LUT = None
SIZE = 1080


def init(size):
    global LUT, SIZE
    LUT, SIZE = lut(), size
    try:  # stay out of the way of the final run
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000040)  # IDLE_PRIORITY_CLASS
    except Exception:
        pass


def render(job):
    i, n, path, name, t, x0, y0, R, mode = job
    from astropy.io import fits
    try:
        img = np.asarray(fits.getdata(path, memmap=False), dtype=np.float32)
    except Exception as e:
        return i, None, f"{name}: {e}"
    img = np.nan_to_num(img, nan=0.0)
    # disk-interior median (rho < 0.7), on a subsample
    yy, xx = np.mgrid[0:img.shape[0]:4, 0:img.shape[1]:4]
    inner = (xx - x0) ** 2 + (yy - y0) ** 2 < (0.7 * R) ** 2
    med = float(np.median(img[::4, ::4][inner])) if inner.any() else float(np.median(img))
    v = np.clip(img / max(med, 1e-6), 0, VMAX) / VMAX
    u8 = (255 * v ** GAMMA).astype(np.uint8)
    s = 0.46 * SIZE / R  # fixed output disk radius, centred, rows flipped (FITS row 0 is the bottom)
    M = np.float32([[s, 0, SIZE / 2 - s * x0], [0, -s, SIZE / 2 + s * y0]])
    out = cv2.warpAffine(u8, M, (SIZE, SIZE), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame = LUT[out]
    f = cv2.FONT_HERSHEY_SIMPLEX
    sc = SIZE / 1080
    txt = lambda s_, xy, size=0.8, col=(235, 235, 235): cv2.putText(  # noqa: E731
        frame, s_, (int(xy[0] * sc), int(xy[1] * sc)), f, size * sc, col, max(1, int(2 * sc)), cv2.LINE_AA)
    txt("Aditya-L1 SUIT  NB03  Mg II k 279.6 nm", (24, 44))
    txt(pd.Timestamp(t).strftime("%Y-%m-%d  %H:%M:%S UT"), (24, 84), 0.9, (120, 220, 255))
    txt(f"{mode} pointing", (24, SIZE / sc - 50), 0.7, (180, 200, 255) if mode == "centred" else (150, 200, 150))
    txt(f"frame {i + 1:,} / {n:,}", (24, SIZE / sc - 20), 0.6, (170, 170, 170))
    return i, frame, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (a quick test)")
    ap.add_argument("--dataset", default="final_offset", help="whose registration provides centres and QC")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    init(a.size)
    reg = pd.read_parquet(paths.phase1("registration.parquet", name=a.dataset, make=False))
    reg = reg[(reg.frame == "full_binned") & reg.qc_usable.fillna(False).astype(bool)]
    man = pd.read_parquet(paths.archive("manifest.parquet"), columns=["file", "path"])
    df = reg.merge(man, on="file").sort_values("t").reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)
    n = len(df)
    out = Path(a.out) if a.out else ROOT / "outputs" / "videos" / (
        f"SUIT_NB03_fulldisk_{df.t.min():%Y%m%d}-{df.t.max():%Y%m%d}_{int(a.fps)}fps{'_test' if a.limit else ''}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.size, a.size))
    if not w.isOpened():
        sys.exit("cannot open the video writer (mp4v)")
    print(f"{n:,} frames {df.t.min()} .. {df.t.max()} -> {out} ({n / a.fps:.1f} s at {a.fps:g} fps)", flush=True)
    jobs = [(i, n, r.path, r.file, r.t, r.reg_x0, r.reg_y0, r.reg_R, r.pointing_mode) for i, r in enumerate(df.itertuples())]
    t0, bad = time.time(), []
    with Pool(a.workers, initializer=init, initargs=(a.size,)) as pool:
        for i, frame, err in pool.imap(render, jobs, chunksize=2):
            if frame is None:
                bad.append(err)
                continue
            w.write(frame)
            progress.report("video nb03", item=jobs[i][3], i=i, n=n, path=jobs[i][2], every_s=2)
            if i % 500 == 0:
                print(f"{i + 1:,}/{n:,}  {(i + 1) / (time.time() - t0):.1f} frames/s", flush=True)
    w.release()
    tmp.replace(out)
    if a.limit:  # a still for checking the look
        cap = cv2.VideoCapture(str(out))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, n // 2))
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(str(out.with_suffix(".png")), fr)
    info = (f"{n - len(bad):,} frames written, {len(bad)} unreadable, {time.time() - t0:.0f} s, "
            f"{out.stat().st_size / 1e6:.0f} MB: {out}")
    print(info, *bad[:20], sep="\n", flush=True)
    (out.with_suffix(".txt")).write_text(info + "\n" + "\n".join(bad) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
