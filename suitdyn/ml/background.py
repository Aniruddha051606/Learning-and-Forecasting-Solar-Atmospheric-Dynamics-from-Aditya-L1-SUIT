"""The static background S: what derotation must NOT move (PHASE3 §3.1, §4b, §4c)."""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .. import baselines


def mean_residuals(bank, ids, clip=0.25, batch=16, report=None, tick=None):
    """M(H) for each horizon over the given samples (clipped mean; NaN where fewer than half the pairs of
    that horizon are valid), plus the median context times and B0 of those pairs.
    """
    G = bank.G
    acc = {}
    for n0 in range(0, len(ids), batch):
        if tick:  # the thermal controller (GPU work)
            tick()
        b = bank.batch(ids[n0:n0 + batch], "plain")
        r = b["y"][:, 0] - b["x_plain"].mean(1)
        ok = torch.isfinite(r) & (r.abs() < clip)
        for h in torch.unique(b["h"]).tolist():
            m = b["h"] == h
            s, c, n = acc.get(h, (torch.zeros(G, G, device=r.device), torch.zeros(G, G, device=r.device), 0))
            acc[h] = (s + torch.where(ok[m], r[m], 0).sum(0), c + ok[m].sum(0), n + int(m.sum()))
        if report:
            report(n0, len(ids))
    out, geo = {}, {}
    for h, (s, c, n) in acc.items():
        out[h] = torch.where(c >= 0.5 * n, s / c.clamp(min=1), torch.full_like(s, np.nan)).cpu().numpy()
        sel = ids[bank.horizon[ids] == h]
        geo[h] = (np.median(bank.dt[sel], 0), float(np.median(bank.b0[sel])))
    return out, geo


def rot_matrix(G, r_ref, b0, dt, pos):
    """Sparse bilinear derotation on the disk pixels (rows: target, cols: source); rows whose source is off
    the disk are all-zero and flagged invalid.
    """
    (rows, cols), ok = baselines.derotation_coords(G, r_ref, b0, dt)
    tgt = np.flatnonzero(pos.ravel() >= 0)
    rr, cc = rows.ravel()[tgt], cols.ravel()[tgt]
    r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
    fr, fc = rr - r0, cc - c0
    good = ok.ravel()[tgt] & (r0 >= 0) & (c0 >= 0) & (r0 + 1 < G) & (c0 + 1 < G)
    I, J, V = [], [], []
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        src = np.full(len(tgt), -1)
        src[good] = pos[r0[good] + dr, c0[good] + dc]
        good &= (src >= 0) | (w < 1e-9)
        I.append(np.arange(len(tgt)))
        J.append(np.maximum(src, 0))
        V.append(np.where(src >= 0, w, 0.0))
    n = len(tgt)
    return sparse.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(n, n)), good


def horizon_operator(G, r_ref, dts, b0, pos):
    mats = [rot_matrix(G, r_ref, b0, float(dt), pos) for dt in dts]
    return sum(m[0] for m in mats) / len(mats), np.logical_and.reduce([m[1] for m in mats])


def gradient_matrix(pos):
    n = int((pos >= 0).sum())
    I, J, V, k = [], [], [], 0
    for a, b in ((pos[:, :-1], pos[:, 1:]), (pos[:-1, :], pos[1:, :])):
        m = (a >= 0) & (b >= 0)
        cnt = int(m.sum())
        I += [np.arange(k, k + cnt)] * 2
        J += [a[m], b[m]]
        V += [np.ones(cnt), -np.ones(cnt)]
        k += cnt
    return sparse.csr_matrix((np.concatenate(V), (np.concatenate(I), np.concatenate(J))), shape=(k, n))


def disk_index(G, r_ref, rho_max):
    c = (G - 1) / 2
    v, u = np.indices((G, G))
    disk = np.hypot(u - c, v - c) / r_ref < rho_max
    pos = np.full((G, G), -1)
    pos[disk] = np.arange(int(disk.sum()))
    return pos


def solve(M, ops, pos, lam, x0=None, iter_lim=4000, prior=None, shrink=0.0):
    """S on the disk pixels from {h: M(h)} and {h: (Rbar_h, valid_rows)}; lam weights the gradient penalty."""
    G = pos.shape[0]
    Dg = gradient_matrix(pos)
    blocks, rhs = [], []
    for h, Mh in M.items():
        Rbar, good = ops[h]
        m = Mh.ravel()[pos.ravel() >= 0]
        use = good & np.isfinite(m)
        blocks.append((sparse.identity(Rbar.shape[0], format="csr") - Rbar)[use])
        rhs.append(m[use])
    extra, extra_b = [np.sqrt(lam) * Dg], [np.zeros(Dg.shape[0])]
    if prior is not None and shrink > 0:
        n = int((pos >= 0).sum())
        extra.append(np.sqrt(shrink) * sparse.identity(n, format="csr"))
        extra_b.append(np.sqrt(shrink) * np.nan_to_num(np.asarray(prior)[pos >= 0]))
    A = sparse.vstack(blocks + extra).tocsr()
    b = np.concatenate(rhs + extra_b)
    Dg = sparse.vstack(extra).tocsr()  # rows after the data rows (for the fit rms below)
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=iter_lim, x0=x0)
    S = np.full((G, G), np.nan, np.float32)
    S[pos >= 0] = sol[0]
    nd = len(b) - Dg.shape[0]
    fit_rms = float(np.sqrt(np.mean((A[:nd] @ sol[0] - b[:nd]) ** 2))) if nd else float("nan")
    return S, sol[0], {"iterations": int(sol[2]), "istop": int(sol[1]), "fit_rms": fit_rms}


def pointing_groups(px, py, bin_px, min_pairs):
    """Group samples by their target pointing (2048-px units): bins of bin_px; bins with fewer than min_pairs
    samples are merged into the nearest large bin.
    """
    px, py = np.asarray(px, float), np.asarray(py, float)
    if np.ptp(px) < bin_px and np.ptp(py) < bin_px:
        return np.zeros(len(px), np.int64), np.array([[np.median(px), np.median(py)]])
    keys = np.stack([np.floor(px / bin_px), np.floor(py / bin_px)], 1)
    uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    big = np.flatnonzero(cnt >= min_pairs)
    if len(big) == 0:
        return np.zeros(len(px), np.int64), np.array([[np.median(px), np.median(py)]])
    cen = np.array([[np.median(px[inv == k]), np.median(py[inv == k])] for k in range(len(uniq))])
    to_big = {k: (k if k in set(big) else big[np.argmin(np.hypot(*(cen[big] - cen[k]).T))]) for k in range(len(uniq))}
    g_old = np.array([to_big[k] for k in inv])
    ids = {k: i for i, k in enumerate(sorted(set(g_old)))}
    g = np.array([ids[k] for k in g_old], np.int64)
    centres = np.array([[np.median(px[g == i]), np.median(py[g == i])] for i in range(len(ids))])
    return g, centres


def score(bank, ids, S, batch=16, rho_max=None, tick=None):
    """Median per-horizon MAE of B1-avg and of B1-avg-bgS (= mean_k [rot_k(F_k - S) + S]) on the samples,
    inside rho < rho_max when given (the outer ring is poorly constrained: foreshortening, sources off the
    disk).
    """
    if isinstance(S, tuple):
        bank.set_background_groups(*S)
    else:
        bank.set_background(S)
    inner = None if rho_max is None else bank.mu > float(np.sqrt(1 - rho_max ** 2))
    per = {}
    for n0 in range(0, len(ids), batch):
        if tick:
            tick()
        b = bank.batch(ids[n0:n0 + batch], "plain")
        A, Ab, y = b["x_plain"].mean(1), b["x_bg"].mean(1), b["y"][:, 0]
        v = torch.isfinite(A) & torch.isfinite(Ab) & torch.isfinite(y)
        if inner is not None:
            v = v & inner
        for j in range(len(A)):
            d = per.setdefault(int(b["h"][j]), {"B1-avg": [], "B1-avg-bgS": []})
            d["B1-avg"].append(float((A[j] - y[j]).abs()[v[j]].mean()))
            d["B1-avg-bgS"].append(float((Ab[j] - y[j]).abs()[v[j]].mean()))
    return {h: {k: float(np.median(x)) for k, x in d.items()} for h, d in per.items()}
