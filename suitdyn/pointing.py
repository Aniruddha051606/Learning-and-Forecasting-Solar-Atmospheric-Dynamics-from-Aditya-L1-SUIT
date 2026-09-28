"""Pointing modes: where the Sun sits on the detector, grouped into distinct clusters.

The Level-1 large-scale response (vignetting, scatter) depends strongly on where the Sun is on the CCD
(PHASE3 §4b: the offset pointing has a factor-3 east-west gradient, the centred one almost none). A data
set must therefore never mix pointings silently. `mode()` keeps the coarse label used so far (centred /
offset); `clusters()` separates every distinct pointing, so a new pointing appearing in the archive (the
spacecraft pointing changed once already, on 2026-09-23 ~05:00 UT) becomes its own cluster instead of
being merged into "offset". The oscillation (±10 px) and slow drift stay inside one cluster.
"""
import numpy as np


def mode(x0, y0, size):
    """'centred' when the disk centre is within 200 (binned) px of the detector centre, else 'offset'."""
    c = (size - 1) / 2
    return np.where(np.hypot(np.asarray(x0) - c, np.asarray(y0) - c) < 200 * size / 2048, "centred", "offset")


def clusters(x0, y0, size=2048, radius=150.0):
    """Cluster label per frame from its disk centre (x0, y0) in pixels of a `size` frame.

    Frames are taken in the given order (use time order); a frame joins the nearest existing cluster whose
    running median centre is within `radius` (2048-px units, scaled to `size`), else it starts a new one.
    Labels are '<mode>@<x>,<y>' with the cluster's median centre in 2048-px units rounded to 10 px, so the
    same pointing gets the same name in every data set."""
    x0, y0 = np.asarray(x0, float), np.asarray(y0, float)
    s = 2048.0 / size
    rad = radius / s
    centres, members, lab = [], [], np.empty(len(x0), dtype=object)
    for i, (x, y) in enumerate(zip(x0, y0)):
        if not (np.isfinite(x) and np.isfinite(y)):
            lab[i] = None
            continue
        d = [np.hypot(x - cx, y - cy) for cx, cy in centres]
        k = int(np.argmin(d)) if d and min(d) < rad else -1
        if k < 0:
            centres.append((x, y))
            members.append([])
            k = len(centres) - 1
        members[k].append(i)
        mx, my = np.median(x0[members[k]]), np.median(y0[members[k]])
        centres[k] = (mx, my)
        lab[i] = k
    names = []
    for k, (cx, cy) in enumerate(centres):
        m = str(mode(cx, cy, size))
        names.append(f"{m}@{int(round(cx * s / 10) * 10)},{int(round(cy * s / 10) * 10)}")
    return np.array([names[k] if k is not None else None for k in lab], dtype=object)
